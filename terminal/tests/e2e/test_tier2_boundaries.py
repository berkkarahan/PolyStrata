"""Tier 2 E2E Boundary & Corner Case Tests (R1–R4, 20 tests)."""
from __future__ import annotations

import hashlib
import json
import os
import stat
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
    dir_snapshot,
    make_gguf,
    root_clutter_entries,
)


class _Tier2Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._repo_guard = WorkspaceSnapshotGuard()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._repo_guard.assert_unchanged(cls.__name__)


class TestTier2R1Boundaries(_Tier2Base):
    """T2-R1-01 .. T2-R1-05: R1 boundary & corner cases."""

    def test_t2_r1_01_dotted_and_mixed_case_model_tags(self):
        import setup

        with IsolatedWorkspace(control_mode=True) as ws:
            for tag in ("DeepSeek-V4.1-Test", "Qwen3.6_UD-Q3_K_M"):
                norm = tag.lower()
                cfg_path = ws.install / "configs" / f"strata-{norm}.json"
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg = {
                    "exe": "engine",
                    "tokenizer": "tok",
                    "args": ["--max-context", "32768"],
                    "log": str(ws.install / "logs" / f"strata-{norm}.log"),
                    "port": 8080,
                }
                setup.write_setup_config(cfg_path, cfg)
                script = setup.write_run_script(tag, cfg_path, 8080, open_browser=False)
                self.assertEqual(cfg_path.name, f"strata-{norm}.json")
                self.assertEqual(script.parent.resolve(), (ws.install / "scripts").resolve())
                self.assertTrue(script.name.startswith(f"run-{norm}."))

            self.assertEqual(root_clutter_entries(ws.install), [])

    def test_t2_r1_02_appendlog_creates_absent_logs_directory(self):
        from serve import appendlog

        with IsolatedWorkspace() as ws:
            logs_dir = ws.install / "logs"
            self.assertFalse(logs_dir.exists())
            target = logs_dir / "strata-x.log"
            with appendlog.open_append(target) as fh:
                fh.write("line-1\n")
            with appendlog.open_append(target) as fh:
                fh.write("line-2\n")
            self.assertTrue(logs_dir.is_dir())
            text = target.read_text(encoding="utf-8")
            self.assertIn("line-1", text)
            self.assertIn("line-2", text)
            self.assertEqual(root_clutter_entries(ws.install), [])

    def test_t2_r1_03_empty_and_null_settings_unlink_sidecar(self):
        with IsolatedWorkspace() as ws:
            m = ws.make_installed_model("x", layout="configs")
            sidecar = ws.install / "configs" / "strata-x.shared-settings.json"
            with ws.start_mock_service(m["cfg"]) as svc:
                st1, _h1, _b1, js1 = call_http(svc, "POST", "/settings", {"defaults": {"temperature": 0.6}})
                self.assertEqual(st1, 200, js1)
                self.assertTrue(sidecar.is_file())

                st2, _h2, _b2, js2 = call_http(svc, "POST", "/settings", {"defaults": {}})
                self.assertEqual(st2, 200, js2)
                self.assertFalse(js2["shared"])
                self.assertFalse(sidecar.exists(), "empty {} settings must unlink sidecar")

                call_http(svc, "POST", "/settings", {"defaults": {"top_p": 0.9}})
                self.assertTrue(sidecar.is_file())
                st3, _h3, _b3, js3 = call_http(svc, "POST", "/settings", {"defaults": None})
                self.assertEqual(st3, 200, js3)
                self.assertFalse(sidecar.exists(), "null settings must unlink sidecar")

                st4, _h4, _b4, js4 = call_http(svc, "GET", "/settings")
                self.assertEqual(st4, 200)
                self.assertEqual(js4, {"shared": False, "defaults": {}})

    def test_t2_r1_04_twelve_models_stress_root_cleanliness(self):
        from serve import models as serve_models

        with IsolatedWorkspace() as ws:
            tags = [f"model-{i:02d}" for i in range(12)]
            for tag in tags:
                ws.make_installed_model(tag, layout="configs", create_log=True, create_script=True)

            self.assertEqual(root_clutter_entries(ws.install), [])
            self.assertEqual(len(list((ws.install / "configs").glob("strata-*.json"))), 12)
            self.assertEqual(len(list((ws.install / "logs").glob("strata-*.log"))), 12)
            self.assertEqual(len(list((ws.install / "scripts").glob("run-*.*"))), 12)

            code, out, err = ws.run_cli(["models", "list", "--installed", "--output", "json"])
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(out)["model_count"], 12)
            self.assertEqual(len(serve_models.discover(ws.install)), 12)

    def test_t2_r1_05_invalid_config_post_leaves_no_bak_or_tmp(self):
        with IsolatedWorkspace() as ws:
            m = ws.make_installed_model("x", layout="configs")
            before_bytes = m["cfg"].read_bytes()
            with ws.start_mock_service(m["cfg"]) as svc:
                for payload in ({"set": {"unknown_key": 1}}, {"set": {"sampling.top_p": 5.0}}):
                    st, _h, _b, _js = call_http(svc, "POST", "/config", payload)
                    self.assertEqual(st, 400)

            self.assertEqual(m["cfg"].read_bytes(), before_bytes)
            self.assertFalse((ws.install / "configs" / "strata-x.json.bak").exists())
            self.assertFalse(list((ws.install / "configs").glob("*.tmp")))
            self.assertEqual(root_clutter_entries(ws.install), [])


class TestTier2R2Boundaries(_Tier2Base):
    """T2-R2-01 .. T2-R2-05: R2 boundary & corner cases."""

    def test_t2_r2_01_utf8_bom_config_in_configs(self):
        from serve import models as serve_models
        from strata_control import models as ctrl_models, profiles

        with IsolatedWorkspace() as ws:
            m = ws.make_installed_model("bom", layout="configs")
            raw_no_bom = m["cfg"].read_bytes()
            raw_bom = b"\xef\xbb\xbf" + raw_no_bom
            m["cfg"].write_bytes(raw_bom)
            expected_sha = hashlib.sha256(raw_bom).hexdigest()

            listed = ctrl_models.list_installed()
            self.assertEqual(listed["model_count"], 1)
            self.assertEqual(listed["models"][0]["source_sha256"], expected_sha)
            inspected = ctrl_models.inspect("bom")
            self.assertEqual(inspected["model"]["source_sha256"], expected_sha)

            srv_found = serve_models.discover(ws.install)
            self.assertEqual([x["id"] for x in srv_found], ["bom"])

            loaded = profiles.read_run_config(m["cfg"])
            self.assertEqual(loaded["model_name"], "bom")

            with ws.start_mock_service(m["cfg"]) as svc:
                st, _h, _b, health = call_http(svc, "GET", "/health")
                self.assertEqual(st, 200)
                self.assertEqual(health["status"], "ok")

    def test_t2_r2_02_malformed_empty_oversized_and_symlink_configs(self):
        from serve import models as serve_models

        with IsolatedWorkspace() as ws:
            ws.make_installed_model("ok", layout="configs")
            cfg_dir = ws.install / "configs"
            (cfg_dir / "strata-bad.json").write_text("{broken", encoding="utf-8")
            (cfg_dir / "strata-empty.json").write_bytes(b"")
            (cfg_dir / "strata-big.json").write_bytes(b'{"a": "' + b"x" * (1024 * 1024 + 64) + b'"}')

            has_symlink = False
            try:
                (cfg_dir / "strata-link.json").symlink_to(cfg_dir / "strata-ok.json")
                has_symlink = True
            except OSError:
                has_symlink = False

            code, out, err = ws.run_cli(["models", "list", "--installed", "--output", "json"])
            self.assertEqual(code, 0, err)
            items = {m["id"]: m for m in json.loads(out)["models"]}
            self.assertEqual(items["ok"]["component_status"], "present")
            for bad_id in ("bad", "empty", "big"):
                self.assertEqual(items[bad_id]["component_status"], "invalid")
            if has_symlink:
                self.assertEqual(items["link"]["component_status"], "invalid")

            icode, _iout, _ierr = ws.run_cli(["models", "inspect", "bad", "--output", "json"])
            self.assertEqual(icode, 5)

            srv_list = serve_models.discover(ws.install)
            self.assertEqual([m["id"] for m in srv_list], ["ok"])

    def test_t2_r2_03_duplicate_id_configs_beats_legacy_root(self):
        from serve import models as serve_models
        from tools.strata_mcp import Strata

        with IsolatedWorkspace() as ws:
            ws.make_installed_model("dup", layout="configs", context=65536)
            ws.make_installed_model("dup", layout="legacy", context=8192)

            code, out, err = ws.run_cli(["models", "list", "--installed", "--output", "json"])
            self.assertEqual(code, 0, err)
            listed = json.loads(out)["models"]
            self.assertEqual(len(listed), 1, f"duplicate model ID returned twice: {listed}")
            self.assertEqual(listed[0]["context_tokens"], 65536)
            self.assertEqual(Path(listed[0]["source_config"]).parent.name, "configs")

            srv_list = serve_models.discover(ws.install)
            self.assertEqual(len(srv_list), 1)
            self.assertEqual(srv_list[0]["context"], 65536)

            mcp_cfgs = Strata(ws.install, ws.terminal_here).configs()
            self.assertEqual(len(mcp_cfgs), 1)
            self.assertEqual(mcp_cfgs[0].parent.name, "configs")

    def test_t2_r2_04_shared_settings_never_treated_as_model_config(self):
        from serve import models as serve_models
        from strata_cli import ram_budget_move
        from strata_control import models_move, tune_apply
        from tools.strata_mcp import Strata

        with IsolatedWorkspace() as ws:
            cfg_dir = ws.install / "configs"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "strata-orphan.shared-settings.json").write_text(
                json.dumps({"temperature": 0.5, "args": ["--resident-budget-gib", "40"]}),
                encoding="utf-8",
            )
            ws.make_installed_model("q2_0", layout="configs", shared_settings={"temperature": 0.7})

            code, out, err = ws.run_cli(["models", "list", "--installed", "--output", "json"])
            self.assertEqual(code, 0, err)
            self.assertEqual([m["id"] for m in json.loads(out)["models"]], ["q2_0"])
            self.assertEqual([m["id"] for m in serve_models.discover(ws.install)], ["q2_0"])
            self.assertEqual(len(Strata(ws.install, ws.terminal_here).configs()), 1)
            self.assertEqual(len(models_move.config_files(ws.install)), 1)
            self.assertEqual(len(tune_apply.find_config_files(ws.install)), 1)
            self.assertFalse(ram_budget_move._needs_setup(ws.install))

    def test_t2_r2_05_invalid_model_id_and_invalid_install_dir(self):
        with IsolatedWorkspace() as ws:
            for bad_id in ("../etc/passwd", "a..b", ".leading"):
                code, _out, _err = ws.run_cli(["models", "inspect", bad_id, "--output", "json"])
                self.assertEqual(code, 2, f"expected exit 2 for invalid model ID {bad_id!r}, got {code}")

            c_rel, _o1, _e1 = ws.run_cli(["models", "list", "--install-dir", "relative/path", "--output", "json"])
            self.assertEqual(c_rel, 5)
            c_missing, _o2, _e2 = ws.run_cli(
                ["models", "list", "--install-dir", str(ws.base / "nonexistent-dir"), "--output", "json"]
            )
            self.assertEqual(c_missing, 5)


class TestTier2R3Boundaries(_Tier2Base):
    """T2-R3-01 .. T2-R3-05: R3 boundary & corner cases."""

    def test_t2_r3_01_windows_crlf_bat_and_posix_lf_sh_rules(self):
        import setup

        with IsolatedWorkspace(control_mode=True) as ws:
            cfg_path = ws.install / "configs" / "strata-q2_0.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text("{}", encoding="utf-8")

            with mock.patch.object(setup, "WIN", True):
                bat = setup.write_run_script("q2_0", cfg_path, 8080, open_browser=False)
                raw_bat = bat.read_bytes()
                self.assertEqual(bat.parent.resolve(), (ws.install / "scripts").resolve())
                self.assertEqual(bat.suffix, ".bat")
                self.assertIn(b"\r\n", raw_bat)
                self.assertNotIn(b"\r\r\n", raw_bat)
                self.assertNotIn(b"\n", raw_bat.replace(b"\r\n", b""))
                self.assertIn(b"if errorlevel 1 pause", raw_bat)

            with mock.patch.object(setup, "WIN", False):
                sh = setup.write_run_script("q2_0", cfg_path, 8080, open_browser=False)
                raw_sh = sh.read_bytes()
                self.assertEqual(sh.parent.resolve(), (ws.install / "scripts").resolve())
                self.assertEqual(sh.suffix, ".sh")
                self.assertNotIn(b"\r", raw_sh)
                self.assertTrue(raw_sh.startswith(b"#!/bin/sh\n"))
                self.assertIn(b"\nexec ", raw_sh)
                if os.name != "nt":
                    self.assertTrue(bool(sh.stat().st_mode & stat.S_IXUSR))

    def test_t2_r3_02_paths_with_spaces_and_unicode(self):
        import setup

        with IsolatedWorkspace(suffix_label="Poly Strata Тест", control_mode=True) as ws:
            m = ws.make_installed_model("uni", layout="configs")
            script = setup.write_run_script("uni", m["cfg"], 8080, open_browser=False)
            self.assertEqual(script.parent.resolve(), (ws.install / "scripts").resolve())
            self.assertIn(f'"{m["cfg"]}"', script.read_text(encoding="utf-8"))

            code, out, err = ws.run_cli(["models", "list", "--installed", "--output", "json"])
            self.assertEqual(code, 0, err)
            self.assertEqual([x["id"] for x in json.loads(out)["models"]], ["uni"])

            with ws.start_mock_service(m["cfg"]) as svc:
                st, _h, _b, js = call_http(svc, "GET", "/health")
                self.assertEqual(st, 200)
                self.assertEqual(js["status"], "ok")

    def test_t2_r3_03_unified_launcher_ambiguous_partial_vs_exact(self):
        with IsolatedWorkspace() as ws:
            m1 = ws.make_installed_model("qwen-q2_0", layout="configs")
            ws.make_installed_model("unsloth-ud-q2_0", layout="configs")

            code_amb, out_amb, err_amb = ws.run_polystrata(["--model", "q2_0"])
            self.assertNotEqual(code_amb, 0)
            combined = out_amb + err_amb
            self.assertIn("qwen-q2_0", combined)
            self.assertIn("unsloth-ud-q2_0", combined)

            import polystrata

            resolved = polystrata.entry_of("qwen-q2_0")
            self.assertEqual(Path(resolved["config"]).resolve(), m1["cfg"].resolve())

    def test_t2_r3_04_unified_launcher_zero_models_installed(self):
        with IsolatedWorkspace() as ws:
            (ws.install / "configs").mkdir(parents=True, exist_ok=True)
            code_l, out_l, _err_l = ws.run_polystrata(["--list-models"])
            self.assertEqual(code_l, 1)
            self.assertIn("setup", out_l.lower())

            code_m, out_m, err_m = ws.run_polystrata(["--model", "anything"])
            self.assertNotEqual(code_m, 0)
            self.assertNotIn("Traceback", out_m + err_m)

    def test_t2_r3_05_mcp_removal_plan_protects_shared_and_external_files(self):
        from tools.strata_mcp import Strata, Tools

        with IsolatedWorkspace() as ws:
            ext_dir = ws.base / "external_user_models"
            ext_gguf = make_gguf(ext_dir / "user-own.gguf")
            shared_mtp = ws.data / "mtp" / "shared"
            (shared_mtp / "rt").mkdir(parents=True, exist_ok=True)
            shared_experts = shared_mtp / "rt" / "experts.bin"
            shared_experts.write_bytes(b"shared-mtp")

            m1 = ws.make_installed_model(
                "m1",
                layout="configs",
                shared_settings={"temperature": 0.5},
                create_log=True,
                create_script=True,
                write_bak=True,
            )
            # Point m1 to external GGUF and shared MTP folder
            c1 = json.loads(m1["cfg"].read_text(encoding="utf-8"))
            c1["args"] = ["--native", str(ext_gguf), "--pack", str(m1["pack"]), "--mtp", str(shared_mtp), "--max-context", "32768"]
            m1["cfg"].write_text(json.dumps(c1, indent=1), encoding="utf-8")

            m2 = ws.make_installed_model("m2", layout="configs", create_log=True, create_script=True)
            c2 = json.loads(m2["cfg"].read_text(encoding="utf-8"))
            c2["args"] = ["--native", str(m2["gguf"]), "--pack", str(m2["pack"]), "--mtp", str(shared_mtp), "--max-context", "32768"]
            m2["cfg"].write_text(json.dumps(c2, indent=1), encoding="utf-8")

            s = Strata(ws.install, ws.terminal_here)
            t = Tools(s)
            plan = s.removal_plan(m1["cfg"])
            remove_paths = {Path(x["path"]).resolve() for x in plan["remove"]}
            for expected_rm in (
                m1["cfg"],
                m1["cfg"].with_name("strata-m1.json.bak"),
                m1["cfg"].with_name("strata-m1.shared-settings.json"),
                m1["log"],
                m1["script"],
            ):
                self.assertIn(expected_rm.resolve(), remove_paths, f"expected {expected_rm} in removal_plan")

            res = t.strata_remove(model="m1", confirm=True)
            self.assertEqual(res.get("failed"), [], res)
            self.assertFalse(m1["cfg"].exists())
            self.assertFalse(m1["log"].exists())
            self.assertFalse(m1["script"].exists())
            self.assertFalse((ws.install / "configs" / "strata-m1.shared-settings.json").exists())
            self.assertTrue(ext_gguf.is_file(), "external user GGUF must never be deleted")
            self.assertTrue(shared_experts.is_file(), "shared MTP experts must be preserved for m2")
            self.assertTrue(m2["cfg"].is_file())


class TestTier2R4Boundaries(_Tier2Base):
    """T2-R4-01 .. T2-R4-05: R4 boundary & corner cases."""

    def test_t2_r4_01_idempotent_re_migration(self):
        import setup

        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            ws.make_installed_model(
                "q2_0",
                install_dir=ws.repo,
                layout="legacy",
                shared_settings={"temperature": 0.4},
                create_log=True,
                create_script=True,
                write_bak=True,
            )
            first = setup.move_over_old_layout()
            self.assertTrue(first)
            snap_after_first = dir_snapshot(ws.install)

            second = setup.move_over_old_layout()
            self.assertEqual(second, [])
            snap_after_second = dir_snapshot(ws.install)
            self.assertEqual(snap_after_second, snap_after_first)
            self.assertEqual(root_clutter_entries(ws.repo), [])
            self.assertEqual(root_clutter_entries(ws.install), [])

    def test_t2_r4_02_migration_destination_collision_never_overwrites(self):
        import setup

        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            new_m = ws.make_installed_model("q2_0", install_dir=ws.install, layout="configs", context=65536)
            new_bytes = new_m["cfg"].read_bytes()
            ws.make_installed_model("q2_0", install_dir=ws.repo, layout="legacy", context=8192)

            setup.move_over_old_layout()
            self.assertEqual(new_m["cfg"].read_bytes(), new_bytes, "migration must not overwrite existing configs/ entry")
            self.assertEqual(root_clutter_entries(ws.install), [], "install home root must stay clean on collision")
            self.assertFalse((ws.install / "strata-q2_0.json").exists(), "colliding legacy file must not land in install root")

    def test_t2_r4_03_migration_preserves_custom_external_log_and_unknown_keys(self):
        import setup

        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            ext_log = ws.base / "custom_logs" / "my.log"
            ext_log.parent.mkdir(parents=True, exist_ok=True)
            ext_log.write_text("existing\n", encoding="utf-8")
            ws.make_installed_model(
                "q2_0",
                install_dir=ws.repo,
                layout="legacy",
                extra_cfg={"log": str(ext_log), "custom_key": {"a": 1}, "aliases": ["fast"]},
            )
            setup.move_over_old_layout()
            migrated_cfg = ws.install / "configs" / "strata-q2_0.json"
            self.assertTrue(migrated_cfg.is_file())
            data = json.loads(migrated_cfg.read_text(encoding="utf-8-sig"))
            self.assertEqual(Path(data["log"]).resolve(), ext_log.resolve())
            self.assertEqual(data["custom_key"], {"a": 1})
            self.assertEqual(data["aliases"], ["fast"])

    def test_t2_r4_04_migration_corrupted_legacy_json_does_not_crash_or_delete(self):
        import setup

        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            ws.make_installed_model("q2_0", install_dir=ws.repo, layout="legacy")
            broken = ws.repo / "strata-broken.json"
            broken.write_text("{not json", encoding="utf-8")

            setup.move_over_old_layout()
            self.assertTrue((ws.install / "configs" / "strata-q2_0.json").is_file())
            # Corrupted file must not be silently deleted/lost
            self.assertTrue(
                broken.exists() or (ws.install / "configs" / "strata-broken.json").exists(),
                "corrupted legacy config must not be deleted during migration",
            )

    def test_t2_r4_05_migrated_shared_settings_with_unknown_key(self):
        import setup

        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            ws.make_installed_model(
                "q2_0",
                install_dir=ws.repo,
                layout="legacy",
                shared_settings={"temperature": 0.8, "future_flag": True},
            )
            setup.move_over_old_layout()
            migrated_cfg = ws.install / "configs" / "strata-q2_0.json"
            migrated_ss = ws.install / "configs" / "strata-q2_0.shared-settings.json"
            self.assertTrue(migrated_cfg.is_file())
            self.assertTrue(migrated_ss.is_file())

            with ws.start_mock_service(migrated_cfg) as svc:
                st, _h, _b, js = call_http(svc, "GET", "/settings")
                self.assertEqual(st, 200)
                self.assertTrue(js["shared"])
                self.assertEqual(js["defaults"]["temperature"], 0.8)


if __name__ == "__main__":
    unittest.main()
