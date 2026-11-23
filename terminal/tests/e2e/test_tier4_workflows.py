"""Tier 4 E2E Real-World End-User Workflow Tests (5 scenarios)."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

TERMINAL_ROOT = Path(__file__).resolve().parents[2]
if str(TERMINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TERMINAL_ROOT))

from tests.e2e.harness import (  # noqa: E402
    IsolatedWorkspace,
    WorkspaceSnapshotGuard,
    assert_dir_unchanged,
    call_http,
    dir_snapshot,
    root_clutter_entries,
)


class _Tier4Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._repo_guard = WorkspaceSnapshotGuard()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._repo_guard.assert_unchanged(cls.__name__)


class TestTier4Workflows(_Tier4Base):
    """T4-01 .. T4-05: Full end-user application workflows."""

    def test_t4_01_multi_model_workstation_setup_three_models(self):
        import polystrata
        from serve import models as serve_models
        from tools.strata_mcp import Strata, Tools

        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            for fam, mod in (("qwen", "Q2_0"), ("coder", "IQ1_M"), ("flint", "Q8_0")):
                code, out, err = ws.run_setup(
                    ["--family", fam, "--model", mod, "--yes", "--no-start", "--vision", "no"]
                )
                self.assertEqual(code, 0, f"setup {fam}/{mod} failed: {out}\n{err}")

            self.assertEqual(root_clutter_entries(ws.repo), [])
            self.assertEqual(root_clutter_entries(ws.install), [])

            expected = {
                "q2_0": "qwen3.8-flash-next-q2_0",
                "coder-iq1_m": "qwen3.8-flash-next-coder-iq1_m",
                "flint-q8_0": "flint-40b-a2.5b-q8_0",
            }
            expected_ids = set(expected)
            expected_names = set(expected.values())

            ccode, cout, cerr = ws.run_cli(["models", "list", "--installed", "--output", "json"])
            self.assertEqual(ccode, 0, cerr)
            cli_models = json.loads(cout)["models"]
            self.assertEqual({m["id"] for m in cli_models}, expected_ids)
            self.assertEqual({m["name"] for m in cli_models}, expected_names)
            for m in cli_models:
                self.assertEqual(m["name"], expected[m["id"]])
                self.assertEqual(Path(m["source_config"]).parent.name, "configs")

            pcode, pout, perr = ws.run_polystrata(["--list-models"])
            self.assertEqual(pcode, 0, perr)
            for mname in expected_names:
                self.assertIn(mname, pout)

            discovered = serve_models.discover(ws.install)
            self.assertEqual({e["id"] for e in discovered}, expected_names)
            for tag, mname in expected.items():
                expected_cfg = (ws.install / "configs" / f"strata-{tag}.json").resolve()
                by_entry = polystrata.entry_of(tag)
                by_tag = serve_models.find(discovered, tag)
                by_name = serve_models.find(discovered, mname)
                self.assertIsNotNone(by_tag, f"serve registry failed to find tag {tag!r}")
                self.assertIsNotNone(by_name, f"serve registry failed to find name {mname!r}")
                for entry in (by_entry, by_tag, by_name):
                    self.assertEqual(entry["id"], mname)
                    self.assertEqual(entry["status"], "available")
                    self.assertEqual(Path(entry["config"]).resolve(), expected_cfg)

            mcp_status = Tools(Strata(ws.install, ws.terminal_here)).strata_status(hardware=False)
            mcp_models = mcp_status["installed"]["models"]
            self.assertEqual({m["model"] for m in mcp_models}, expected_ids)
            self.assertEqual({m["model_name"] for m in mcp_models}, expected_names)
            for m in mcp_models:
                self.assertEqual(m["model_name"], expected[m["model"]])
                self.assertIs(m["ready"], True)

            self.assertEqual(len(list((ws.install / "scripts").glob("run-*.*"))), 3)
            self.assertEqual(root_clutter_entries(ws.repo), [])
            self.assertEqual(root_clutter_entries(ws.install), [])

    def test_t4_02_cluttered_legacy_root_upgrade_and_server_verify(self):
        import setup

        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            ws.make_installed_model(
                "q2_0",
                install_dir=ws.repo,
                layout="legacy",
                shared_settings={"temperature": 0.35},
                create_log=True,
                create_script=True,
                write_bak=True,
            )
            ws.make_installed_model(
                "iq3_xxs",
                install_dir=ws.repo,
                layout="legacy",
                create_log=True,
                create_script=True,
            )

            moved = setup.move_over_old_layout()
            self.assertTrue(moved)
            self.assertEqual(root_clutter_entries(ws.repo), [])
            self.assertEqual(root_clutter_entries(ws.install), [])

            cfg_q2 = ws.install / "configs" / "strata-q2_0.json"
            dcode, dout, derr = ws.run_cli(["doctor", "--profile", str(cfg_q2), "--output", "json"])
            self.assertEqual(dcode, 0, f"doctor failed: {dout}\n{derr}")
            self.assertTrue(json.loads(dout)["ok"])

            icode, iout, ierr = ws.run_cli(["models", "inspect", "q2_0", "--output", "json"])
            self.assertEqual(icode, 0, ierr)
            self.assertEqual(json.loads(iout)["model"]["component_status"], "present")

            with ws.start_mock_service(cfg_q2) as svc:
                st_h, _h1, _b1, health = call_http(svc, "GET", "/health")
                self.assertEqual(st_h, 200)
                self.assertEqual(health["status"], "ok")

                st_s, _h2, _b2, settings = call_http(svc, "GET", "/settings")
                self.assertEqual(st_s, 200)
                self.assertEqual(settings["defaults"].get("temperature"), 0.35)

                st_c, _h3, _b3, chat = call_http(
                    svc,
                    "POST",
                    "/v1/chat/completions",
                    {"model": "q2_0", "messages": [{"role": "user", "content": "Hello"}], "stream": False},
                )
                self.assertEqual(st_c, 200, chat)
                self.assertTrue(chat.get("choices"))

    def test_t4_03_full_model_lifecycle_setup_edit_profile_serve_remove(self):
        from tools.strata_mcp import Strata, Tools

        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            code, out, err = ws.run_setup(
                ["--family", "qwen", "--model", "Q2_0", "--yes", "--no-start", "--vision", "no"]
            )
            self.assertEqual(code, 0, f"setup failed: {out}\n{err}")

            cfg_path = ws.install / "configs" / "strata-q2_0.json"
            self.assertTrue(cfg_path.is_file())

            c1, _o1, e1 = ws.run_cli(["profile", "create", "prod", "--from-config", str(cfg_path), "--output", "json"])
            self.assertEqual(c1, 0, e1)
            c2, _o2, e2 = ws.run_cli(["profile", "set", "prod", "parallel=4", "--output", "json"])
            self.assertEqual(c2, 0, e2)

            with ws.start_mock_service(cfg_path) as svc:
                st1, _h1, _b1, cfg_res = call_http(svc, "POST", "/config", {"set": {"reasoning_budget_tokens": 512}})
                self.assertEqual(st1, 200, cfg_res)
                st2, _h2, _b2, ss_res = call_http(svc, "POST", "/settings", {"defaults": {"temperature": 0.3}})
                self.assertEqual(st2, 200, ss_res)

            self.assertTrue((ws.install / "configs" / "strata-q2_0.json.bak").is_file())
            self.assertTrue((ws.install / "configs" / "strata-q2_0.shared-settings.json").is_file())

            # Restart mock server and verify persistence
            with ws.start_mock_service(cfg_path) as svc2:
                st_g1, _hg1, _bg1, cfg_view = call_http(svc2, "GET", "/config")
                self.assertEqual(st_g1, 200)
                budget_item = next(k for k in cfg_view["keys"] if k["key"] == "reasoning_budget_tokens")
                self.assertEqual(budget_item["value"], 512)

                st_g2, _hg2, _bg2, ss_view = call_http(svc2, "GET", "/settings")
                self.assertEqual(st_g2, 200)
                self.assertEqual(ss_view["defaults"].get("temperature"), 0.3)

            # Remove model via MCP and confirm clean workspace
            res = Tools(Strata(ws.install, ws.terminal_here)).strata_remove(model="q2_0", confirm=True)
            self.assertEqual(res.get("failed"), [], res)
            self.assertEqual(list((ws.install / "configs").glob("strata-q2_0*")), [])
            self.assertEqual(list((ws.install / "logs").glob("strata-q2_0*")) if (ws.install / "logs").is_dir() else [], [])
            self.assertEqual(list((ws.install / "scripts").glob("run-q2_0*")) if (ws.install / "scripts").is_dir() else [], [])
            self.assertEqual(root_clutter_entries(ws.repo), [])
            self.assertEqual(root_clutter_entries(ws.install), [])

    def test_t4_04_external_install_dir_and_legacy_remembered_coexistence(self):
        import setup

        with IsolatedWorkspace() as ws:
            folder_a = ws.install
            folder_b = ws.base / "legacy-remembered-install"
            folder_b.mkdir()
            ws.make_installed_model("mod-a", install_dir=folder_a, layout="configs")
            leg_b = ws.make_installed_model("mod-b", install_dir=folder_b, layout="legacy")
            snap_b = dir_snapshot(folder_b)

            settings_file = setup.settings_path()
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            settings_file.write_text(json.dumps({"installs": [str(folder_b)]}), encoding="utf-8")

            with mock.patch.dict(os.environ):
                os.environ.pop("STRATA_INSTALL_DIR", None)
                from strata_cli import cli as cli_mod

                with mock.patch.object(cli_mod, "ROOT", folder_a):
                    code, out, err = ws.run_cli(["models", "list", "--installed", "--output", "json"])
                    self.assertEqual(code, 0, err)
                    ids = {m["id"] for m in json.loads(out)["models"]}
                    self.assertEqual(ids, {"mod-a", "mod-b"})

                    ic_a, _oa, ea = ws.run_cli(["models", "inspect", "mod-a", "--output", "json"])
                    self.assertEqual(ic_a, 0, ea)
                    ic_b, _ob, eb = ws.run_cli(["models", "inspect", "mod-b", "--output", "json"])
                    self.assertEqual(ic_b, 0, eb)

                    pcode, _pout, perr = ws.run_cli(
                        ["serve", "--profile", str(leg_b["cfg"]), "--plan", "--output", "json"]
                    )
                    self.assertEqual(pcode, 0, perr)

            assert_dir_unchanged(self, folder_b, snap_b, label="legacy remembered folder B must remain untouched")
            self.assertEqual(root_clutter_entries(folder_a), [])

    def test_t4_05_multi_model_hot_switch_and_legacy_config_restart(self):
        with IsolatedWorkspace() as ws:
            alpha = ws.make_installed_model("alpha", layout="configs", extra_cfg={"switch_on_request": True})
            beta = ws.make_installed_model("beta", layout="configs")

            with ws.start_mock_service(alpha["cfg"], extra_args=("--switch-on-request",)) as svc:
                st_v1, _h1, _b1, v1 = call_http(svc, "GET", "/v1/models")
                self.assertEqual(st_v1, 200)
                self.assertTrue({"alpha", "beta"}.issubset({m["id"] for m in v1["data"]}))

                st_sw, _h2, _b2, sw = call_http(svc, "POST", "/api/models/load", {"model": "beta"})
                self.assertEqual(st_sw, 200, sw)
                self.assertEqual(sw["model"]["id"], "beta")

                st_set, _h3, _b3, ss = call_http(svc, "POST", "/settings", {"defaults": {"temperature": 0.25}})
                self.assertEqual(st_set, 200, ss)

            # Restart directly via legacy `--config <home>/configs/strata-beta.json`
            with ws.start_mock_service(beta["cfg"]) as svc_beta:
                st_g, _h4, _b4, settings = call_http(svc_beta, "GET", "/settings")
                self.assertEqual(st_g, 200)
                self.assertTrue(settings["shared"])
                self.assertEqual(settings["defaults"].get("temperature"), 0.25)

            self.assertTrue((ws.install / "configs" / "strata-beta.shared-settings.json").is_file())
            self.assertEqual(root_clutter_entries(ws.install), [])


if __name__ == "__main__":
    unittest.main()
