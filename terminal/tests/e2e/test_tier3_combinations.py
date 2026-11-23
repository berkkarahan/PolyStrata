"""Tier 3 E2E Cross-Feature Pairwise Combination Tests (7 tests)."""
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
    call_http,
    root_clutter_entries,
)


class _Tier3Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._repo_guard = WorkspaceSnapshotGuard()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._repo_guard.assert_unchanged(cls.__name__)


class TestTier3Combinations(_Tier3Base):
    """T3-01 .. T3-07: Pairwise combinations across R1, R2, R3, and R4."""

    def test_t3_01_setup_launcher_cli_mcp_parity_and_root_cleanliness(self):
        import polystrata
        from tools.strata_mcp import Strata, Tools

        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            for fam, mod in (("qwen", "Q2_0"), ("qwen", "IQ3_XXS")):
                code, out, err = ws.run_setup(
                    ["--family", fam, "--model", mod, "--yes", "--no-start", "--vision", "no"]
                )
                self.assertEqual(code, 0, f"setup {fam}/{mod} failed: {out}\n{err}")

            self.assertEqual(root_clutter_entries(ws.repo), [])
            self.assertEqual(root_clutter_entries(ws.install), [])

            pcode, pout, perr = ws.run_polystrata(["--list-models"])
            self.assertEqual(pcode, 0, perr)
            self.assertIn("q2_0", pout)
            self.assertIn("iq3_xxs", pout)
            self.assertEqual(Path(polystrata.entry_of("q2_0")["config"]).parent.name, "configs")

            ccode, cout, cerr = ws.run_cli(["models", "list", "--installed", "--output", "json"])
            self.assertEqual(ccode, 0, cerr)
            cli_ids = sorted(m["id"] for m in json.loads(cout)["models"])
            self.assertEqual(cli_ids, ["iq3_xxs", "q2_0"])

            mcp_status = Tools(Strata(ws.install, ws.terminal_here)).strata_status(hardware=False)
            mcp_ids = sorted(m["model"] for m in mcp_status["installed"]["models"])
            self.assertEqual(mcp_ids, ["iq3_xxs", "q2_0"])

            self.assertEqual(len(list((ws.install / "scripts").glob("run-*.*"))), 2)
            self.assertEqual(len(list((ws.install / "logs").glob("strata-*.log"))), 0)

    def test_t3_02_migrate_then_server_switch_on_request_with_shared_settings(self):
        import setup
        from serve import models as srv_models

        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            ws.make_installed_model(
                "q2_0",
                install_dir=ws.repo,
                layout="legacy",
                shared_settings={"temperature": 0.2},
            )
            ws.make_installed_model(
                "iq3_xxs",
                install_dir=ws.repo,
                layout="legacy",
                shared_settings={"temperature": 0.9},
            )
            (ws.repo / "models.json").write_text(
                json.dumps({"q2_0": {"display_name": "Custom Root Name", "aliases": ["daily"]}}),
                encoding="utf-8",
            )

            moved = setup.move_over_old_layout()
            self.assertTrue(moved)
            self.assertEqual(root_clutter_entries(ws.repo), [])
            self.assertEqual(root_clutter_entries(ws.install), [])
            self.assertFalse((ws.repo / "models.json").exists())
            self.assertTrue((ws.install / "configs" / "models.json").is_file())

            discovered = {m["id"]: m for m in srv_models.discover(ws.install)}
            self.assertEqual(discovered["q2_0"]["display_name"], "Custom Root Name")
            self.assertIn("daily", discovered["q2_0"]["aliases"])

            cfg_q2 = ws.install / "configs" / "strata-q2_0.json"
            self.assertTrue(cfg_q2.is_file())

            with ws.start_mock_service(cfg_q2, extra_args=("--switch-on-request",)) as svc:
                st_v1, _h1, _b1, v1 = call_http(svc, "GET", "/v1/models")
                self.assertEqual(st_v1, 200)
                self.assertTrue({"q2_0", "iq3_xxs"}.issubset({m["id"] for m in v1["data"]}))

                st_api, _h2, _b2, api_m = call_http(svc, "GET", "/api/models")
                self.assertEqual(st_api, 200)
                self.assertEqual(sorted(m["id"] for m in api_m["models"]), ["iq3_xxs", "q2_0"])

                st_load, _h3, _b3, loaded = call_http(svc, "POST", "/api/models/load", {"model": "iq3_xxs"})
                self.assertEqual(st_load, 200, loaded)

                st_set, _h4, _b4, settings = call_http(svc, "GET", "/settings")
                self.assertEqual(st_set, 200)
                self.assertEqual(settings["defaults"].get("temperature"), 0.9)

    def test_t3_03_configs_and_explicit_external_config_coexistence(self):
        with IsolatedWorkspace() as ws:
            ws.make_installed_model("alpha", layout="configs")
            ext_dir = ws.base / "external_cfgs"
            ext_dir.mkdir()
            ext = ws.make_installed_model(
                "ext",
                install_dir=ws.install,
                layout=str(ext_dir),
                filename_override="custom.json",
            )

            pcode, pout, perr = ws.run_cli(["serve", "--profile", str(ext["cfg"]), "--plan", "--output", "json"])
            self.assertEqual(pcode, 0, perr)
            self.assertEqual(Path(json.loads(pout)["plan"]["source_config"]).resolve(), ext["cfg"].resolve())

            with ws.start_mock_service(ext["cfg"]) as svc:
                st, _h, _b, resp = call_http(svc, "POST", "/config", {"set": {"fit_max_tokens": False}})
                self.assertEqual(st, 200, resp)

            self.assertTrue((ext_dir / "custom.json.bak").is_file())
            self.assertFalse((ws.install / "configs" / "custom.json.bak").exists())
            self.assertEqual(root_clutter_entries(ws.install), [])

    def test_t3_04_models_move_inspect_profile_serve_plan(self):
        with IsolatedWorkspace() as ws:
            m = ws.make_installed_model("q2_0", layout="configs")
            new_data = ws.base / "new-data-root"
            new_data.mkdir()

            mcode, mout, merr = ws.run_cli(
                ["models", "move", str(new_data), "--from", str(ws.data), "--install-dir", str(ws.install), "--output", "json"]
            )
            self.assertEqual(mcode, 0, f"models move failed: {mout}\n{merr}")

            icode, iout, ierr = ws.run_cli(["models", "inspect", "q2_0", "--output", "json"])
            self.assertEqual(icode, 0, ierr)
            inspected = json.loads(iout)
            self.assertEqual(inspected["model"]["component_status"], "present")

            ccode, _cout, cerr = ws.run_cli(["profile", "create", "moved", "--from-config", str(m["cfg"]), "--output", "json"])
            self.assertEqual(ccode, 0, cerr)
            pcode, pout, perr = ws.run_cli(["serve", "--profile", "moved", "--plan", "--output", "json"])
            self.assertEqual(pcode, 0, perr)
            plan = json.loads(pout)["plan"]
            self.assertTrue(any(str(new_data) in str(arg) for arg in plan["engine_argv"]))
            self.assertEqual(root_clutter_entries(ws.install), [])

    def test_t3_05_sidecars_mcp_describe_and_remove_first_model(self):
        from tools.strata_mcp import Strata, Tools

        with IsolatedWorkspace() as ws:
            m1 = ws.make_installed_model("one", layout="configs", create_log=True, create_script=True)
            m2 = ws.make_installed_model("two", layout="configs", create_log=True, create_script=True)
            legacy_cfg = ws.install / "strata-one.json"
            legacy_cfg.write_text(m1["cfg"].read_text(encoding="utf-8"), encoding="utf-8")
            legacy_warm = ws.install / "strata-one.warm-start"
            legacy_warm.write_text("old-ws", encoding="utf-8")
            overlay_path = ws.install / "configs" / "models.json"
            overlay_data = {"one": {"display_name": "One Overlay"}, "two": {"display_name": "Two Overlay"}}
            overlay_path.write_text(json.dumps(overlay_data), encoding="utf-8")

            with ws.start_mock_service(m1["cfg"]) as svc:
                call_http(svc, "POST", "/settings", {"defaults": {"temperature": 0.4}})
                call_http(svc, "POST", "/config", {"set": {"fit_max_tokens": False}})

            self.assertTrue((ws.install / "configs" / "strata-one.shared-settings.json").is_file())
            self.assertTrue((ws.install / "configs" / "strata-one.json.bak").is_file())

            s = Strata(ws.install, ws.terminal_here)
            t = Tools(s)
            desc = s.describe_config(m1["cfg"])
            self.assertEqual(desc["start_script"], m1["script"].name)

            res = t.strata_remove(model="one", confirm=True)
            self.assertEqual(res.get("failed"), [], res)
            self.assertFalse(m1["cfg"].exists())
            self.assertFalse(m1["log"].exists())
            self.assertFalse(m1["script"].exists())
            self.assertFalse(legacy_cfg.exists())
            self.assertFalse(legacy_warm.exists())
            self.assertFalse((ws.install / "configs" / "strata-one.shared-settings.json").exists())
            self.assertFalse((ws.install / "configs" / "strata-one.json.bak").exists())
            self.assertTrue(overlay_path.is_file())
            self.assertEqual(json.loads(overlay_path.read_text(encoding="utf-8")), overlay_data)

            self.assertTrue(m2["cfg"].is_file())
            self.assertTrue(m2["log"].is_file())
            self.assertTrue(m2["script"].is_file())
            self.assertEqual(root_clutter_entries(ws.install), [])

            ccode, cout, cerr = ws.run_cli(["models", "list", "--installed", "--output", "json"])
            self.assertEqual(ccode, 0, cerr)
            cli_ids = [m["id"] for m in json.loads(cout)["models"]]
            self.assertEqual(cli_ids, ["two"])

            pcode, pout, perr = ws.run_polystrata(["--list-models"])
            self.assertEqual(pcode, 0, perr)
            self.assertIn("two", pout)
            self.assertNotIn("one", pout.split())

    def test_t3_06_app_mode_isolation_vs_terminal_remembered_legacy(self):
        import setup
        from strata_control import discovery

        with IsolatedWorkspace(control_mode=True) as ws:
            with mock.patch.dict(os.environ):
                os.environ.pop("STRATA_COMPONENTS_DIR", None)
                app_dir = discovery.app_install_dir()
                app_settings = discovery.app_settings_file()
                old_dir = ws.base / "remembered-legacy"
                app_dir.mkdir(parents=True, exist_ok=True)
                old_dir.mkdir(parents=True, exist_ok=True)

                ws.make_installed_model("appmod", install_dir=app_dir, layout="configs")
                ws.make_installed_model("oldmod", install_dir=old_dir, layout="legacy")

                settings_file = setup.settings_path()
                settings_file.parent.mkdir(parents=True, exist_ok=True)
                settings_file.write_text(json.dumps({"installs": [str(old_dir)]}), encoding="utf-8")

                # Terminal mode: STRATA_SETTINGS_FILE unset, STRATA_INSTALL_DIR unset -> discovers both
                os.environ.pop("STRATA_INSTALL_DIR", None)
                os.environ.pop("STRATA_SETTINGS_FILE", None)
                code_t, out_t, err_t = ws.run_cli(["models", "list", "--installed", "--output", "json"])
                self.assertEqual(code_t, 0, err_t)
                ids_t = {m["id"] for m in json.loads(out_t)["models"]}
                self.assertEqual(ids_t, {"appmod", "oldmod"}, f"terminal should discover both, got {ids_t}")

                # App mode: STRATA_SETTINGS_FILE set -> isolated to app install only
                app_settings.parent.mkdir(parents=True, exist_ok=True)
                app_settings.write_text(json.dumps({"installs": []}), encoding="utf-8")
                os.environ["STRATA_SETTINGS_FILE"] = str(app_settings)
                os.environ.pop("STRATA_INSTALL_DIR", None)
                code_a, out_a, err_a = ws.run_cli(["models", "list", "--installed", "--output", "json"])
                self.assertEqual(code_a, 0, err_a)
                ids_a = {m["id"] for m in json.loads(out_a)["models"]}
                self.assertEqual(ids_a, {"appmod"})

                self.assertEqual(root_clutter_entries(app_dir), [])
                self.assertEqual(root_clutter_entries(ws.terminal_here), [])

    def test_t3_07_cfg_paths_repair_and_ram_budget_move_in_configs(self):
        from strata_cli import ram_budget_move
        from strata_control import cfg_paths

        with IsolatedWorkspace() as ws:
            cur_ctrl = ws.components / "control" / "0.2.20"
            cur_ctrl.mkdir(parents=True)
            (cur_ctrl / "helper.txt").write_text("ok", encoding="utf-8")
            (ws.components / "control" / "active.json").write_text(json.dumps({"folder": "0.2.20"}), encoding="utf-8")

            m = ws.make_installed_model(
                "q4",
                layout="configs",
                extra_cfg={
                    "args": [
                        "--native", str(ws.data / "models" / "Q4" / "Qwen3.8-Flash-Next-q4-00001-of-00001.gguf"),
                        "--pack", str(ws.data / "packs" / "q4"),
                        "--resident-budget-gib", "40",
                        "--expert-profile", str(ws.components / "control" / "0.1.0" / "helper.txt"),
                        "--max-context", "32768",
                    ]
                },
            )
            fake_setup = mock.MagicMock()
            fake_setup.migrate_run_configs.return_value = [m["cfg"].name]

            moved = ram_budget_move.run(folders=[ws.install], setup_module=fake_setup, say=lambda _t: None)
            self.assertTrue(moved.get(str(ws.install)), f"ram_budget_move did not detect configs/ file: {moved}")

            notes, problems = cfg_paths.repair([m["cfg"]])
            self.assertEqual(problems, [])
            self.assertTrue(notes)
            repaired = json.loads(m["cfg"].read_text(encoding="utf-8"))
            self.assertTrue(any("0.2.20" in str(a) for a in repaired["args"]))
            self.assertEqual(root_clutter_entries(ws.install), [])


if __name__ == "__main__":
    unittest.main()
