"""Tier 1 E2E Feature Coverage Tests (R1–R4, 20 tests).

Opaque-box tests exercising public PolyStrata entrypoints inside isolated
temporary workspaces without modifying the repository checkout.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
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
    make_fake_engine,
    make_fake_engine_component,
    make_gguf,
    make_tokenizer,
    root_clutter_entries,
)


class _Tier1Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._repo_guard = WorkspaceSnapshotGuard()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._repo_guard.assert_unchanged(cls.__name__)


class TestTier1R1RootCleanliness(_Tier1Base):
    """T1-R1-01 .. T1-R1-05: Relocate per-model generated files out of root."""

    def test_t1_r1_01_multi_model_setup_zero_root_clutter(self):
        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            for fam, mod in (("qwen", "Q2_0"), ("coder", "IQ1_M")):
                code, out, err = ws.run_setup(
                    ["--family", fam, "--model", mod, "--yes", "--no-start", "--vision", "no"]
                )
                self.assertEqual(code, 0, f"setup {fam}/{mod} failed: {out}\n{err}")

            self.assertEqual(root_clutter_entries(ws.repo), [], "repo root must have zero per-model clutter")
            self.assertEqual(root_clutter_entries(ws.install), [], "install home root must have zero per-model clutter")

            cfgs = sorted((ws.install / "configs").glob("strata-*.json"))
            self.assertEqual(len(cfgs), 2, f"expected 2 configs in configs/, got {cfgs}")
            for cfg_path in cfgs:
                data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
                log_p = Path(data["log"])
                self.assertEqual(log_p.parent.resolve(), (ws.install / "logs").resolve())

            scripts = list((ws.install / "scripts").glob("run-*.*")) if (ws.install / "scripts").is_dir() else []
            self.assertEqual(len(scripts), 2, f"expected 2 scripts in scripts/, got {scripts}")

    def test_t1_r1_02_poly_setup_home_and_subfolder_placement(self):
        with IsolatedWorkspace(repo_layout=False, control_mode=True) as ws:
            # home() == ws.install != setup.ROOT (ws.terminal_here)
            code, out, err = ws.run_setup(
                ["--family", "flint", "--model", "Q8_0", "--yes", "--no-start", "--vision", "no"]
            )
            self.assertEqual(code, 0, f"poly setup failed: {out}\n{err}")

            self.assertEqual(root_clutter_entries(ws.terminal_here), [], "setup.ROOT must have zero per-model clutter")
            self.assertEqual(root_clutter_entries(ws.install), [], "home() root must have zero per-model clutter")
            cfg_path = ws.install / "configs" / "strata-flint-q8_0.json"
            self.assertTrue(cfg_path.is_file(), f"expected poly config at {cfg_path}")
            cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(Path(cfg["log"]).parent.resolve(), (ws.install / "logs").resolve())
            ext = "bat" if os.name == "nt" else "sh"
            self.assertTrue((ws.install / "scripts" / f"run-flint-q8_0.{ext}").is_file())

    def test_t1_r1_03_server_sidecars_colocated_in_configs(self):
        with IsolatedWorkspace() as ws:
            m = ws.make_installed_model("q2_0", layout="configs")
            with ws.start_mock_service(m["cfg"]) as svc:
                st, _h, _b, js = call_http(svc, "POST", "/settings", {"defaults": {"temperature": 0.4}})
                self.assertEqual(st, 200, js)
                self.assertTrue(js["shared"])

                st2, _h2, _b2, js2 = call_http(svc, "POST", "/config", {"set": {"reasoning_budget_tokens": 256}})
                self.assertEqual(st2, 200, js2)
                self.assertIn("reasoning_budget_tokens", js2["changed"])

            self.assertEqual(root_clutter_entries(ws.install), [])
            self.assertTrue((ws.install / "configs" / "strata-q2_0.shared-settings.json").is_file())
            self.assertTrue((ws.install / "configs" / "strata-q2_0.json.bak").is_file())

    def test_t1_r1_04_appendlog_creates_logs_dir_and_writes_in_logs(self):
        from serve import appendlog

        with IsolatedWorkspace() as ws:
            logs_dir = ws.install / "logs"
            self.assertFalse(logs_dir.exists())
            log_file = logs_dir / "strata-q2_0.log"
            with appendlog.open_append(log_file) as fh:
                fh.write("hello-appendlog\n")
                fh.flush()
            self.assertTrue(logs_dir.is_dir(), "appendlog.open_append must create missing logs/ parent dir")
            self.assertTrue(log_file.is_file())
            self.assertIn("hello-appendlog", log_file.read_text(encoding="utf-8"))
            self.assertEqual(root_clutter_entries(ws.install), [])

    def test_t1_r1_05_setup_backup_warm_start_and_tool_media_containment(self):
        import atexit
        import setup
        from serve import mcp as mcp_mod

        with IsolatedWorkspace(control_mode=True) as ws:
            cfg_dir = ws.install / "configs"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = cfg_dir / "strata-q2_0.json"
            base_cfg = {
                "exe": str(make_fake_engine(ws.install / "engine")),
                "tokenizer": str(make_tokenizer(ws.data / "packs" / "q2_0" / "tokenizer")),
                "args": ["--max-context", "32768"],
                "port": 8080,
                "warm_start": True,
                "mcp_servers": {"fake": {"command": "echo"}},
                "mcp": {"image_results": True},
            }
            setup.write_setup_config(cfg_path, dict(base_cfg))
            mod_cfg = {**base_cfg, "args": ["--max-context", "65536"]}
            setup.write_setup_config(cfg_path, mod_cfg)

            self.assertTrue((cfg_dir / "strata-q2_0.json.bak").is_file())
            with mock.patch.object(mcp_mod.McpHub, "start", lambda self, wait=False: None), \
                    mock.patch.object(atexit, "register", lambda *a, **k: None), \
                    ws.start_mock_service(cfg_path) as svc:
                self.assertEqual(svc.warm_path().parent.resolve(), cfg_dir.resolve())
                self.assertEqual(svc.pin_path().parent.resolve(), cfg_dir.resolve())
                self.assertIsNotNone(svc.mcp)
                self.assertEqual(
                    Path(svc.mcp.media_dir).resolve(),
                    cfg_path.with_suffix(".tool-media").resolve(),
                )
                self.assertEqual(Path(svc.mcp.media_dir).parent.resolve(), cfg_dir.resolve())
            self.assertEqual(root_clutter_entries(ws.install), [])


class TestTier1R2StorageAndDiscovery(_Tier1Base):
    """T1-R2-01 .. T1-R2-05: Structured JSON storage & cross-reader consistency."""

    def test_t1_r2_01_structured_json_crud_and_arbitrary_keys_preserved(self):
        import setup

        with IsolatedWorkspace(control_mode=True) as ws:
            custom = {
                "mcp_servers": {"demo": {"command": "echo"}},
                "before_load": ["echo hi"],
                "sampling": {"min_p": 0.05},
                "aliases": ["fast-q2"],
                "user_custom": {"note": "keep-me"},
            }
            m = ws.make_installed_model("q2_0", layout="configs", extra_cfg=custom)

            with ws.start_mock_service(m["cfg"]) as svc:
                st, _h, _b, js = call_http(svc, "POST", "/config", {"set": {"fit_max_tokens": False}})
                self.assertEqual(st, 200, js)

            # Re-run setup.write_setup_config and verify custom keys survive carry_over
            updated_base = json.loads(m["cfg"].read_text(encoding="utf-8-sig"))
            new_cfg = {
                "exe": updated_base["exe"],
                "tokenizer": updated_base["tokenizer"],
                "args": ["--native", updated_base["args"][1], "--pack", updated_base["args"][3], "--max-context", "65536"],
                "port": 8080,
            }
            setup.write_setup_config(m["cfg"], new_cfg)
            persisted = json.loads(m["cfg"].read_text(encoding="utf-8"))
            self.assertEqual(persisted["mcp_servers"], custom["mcp_servers"])
            self.assertEqual(persisted["before_load"], custom["before_load"])
            self.assertEqual(persisted["user_custom"], custom["user_custom"])
            self.assertEqual(persisted["aliases"], custom["aliases"])
            self.assertEqual(persisted.get("fit_max_tokens"), False)
            self.assertEqual(persisted.get("args_written"), persisted["args"])

    def test_t1_r2_02_cross_reader_discovery_parity_in_configs(self):
        from serve import models as serve_models
        from tools.strata_mcp import Strata

        with IsolatedWorkspace() as ws:
            m1 = ws.make_installed_model("q2_0", layout="configs", context=32768)
            m2 = ws.make_installed_model("iq3_xxs", layout="configs", context=65536)

            code, out, err = ws.run_cli(["models", "list", "--installed", "--output", "json"])
            self.assertEqual(code, 0, err)
            cli_models = json.loads(out)["models"]
            cli_ids = sorted(x["id"] for x in cli_models)
            self.assertEqual(cli_ids, ["iq3_xxs", "q2_0"])
            for x in cli_models:
                self.assertEqual(Path(x["source_config"]).parent.name, "configs")

            srv_list = serve_models.discover(ws.install)
            self.assertEqual(sorted(x["id"] for x in srv_list), ["iq3_xxs", "q2_0"])
            self.assertEqual(
                {Path(x["config"]).resolve() for x in srv_list},
                {m1["cfg"].resolve(), m2["cfg"].resolve()},
            )

            pcode, pout, perr = ws.run_polystrata(["--list-models"])
            self.assertEqual(pcode, 0, perr)
            self.assertIn("q2_0", pout)
            self.assertIn("iq3_xxs", pout)

            mcp = Strata(ws.install, ws.terminal_here)
            mcp_cfgs = mcp.configs()
            self.assertEqual(
                {p.resolve() for p in mcp_cfgs},
                {m1["cfg"].resolve(), m2["cfg"].resolve()},
            )

    def test_t1_r2_03_strict_read_only_discovery_contract(self):
        from serve import models as serve_models
        from strata_control import discovery, models as ctrl_models

        with IsolatedWorkspace() as ws:
            m = ws.make_installed_model("q2_0", layout="configs")
            before = dir_snapshot(ws.install)

            places = discovery.places(here=ws.install)
            self.assertTrue(places)
            listed = ctrl_models.list_installed()
            self.assertEqual(listed["model_count"], 1)
            inspected = ctrl_models.inspect("q2_0")
            self.assertEqual(inspected["model"]["id"], "q2_0")
            self.assertEqual(len(serve_models.discover(ws.install)), 1)
            code, _out, err = ws.run_cli(["serve", "--profile", str(m["cfg"]), "--plan", "--output", "json"])
            self.assertEqual(code, 0, err)

            assert_dir_unchanged(self, ws.install, before, label="read-only discovery")

    def test_t1_r2_04_service_models_and_switch_on_request_in_configs(self):
        with IsolatedWorkspace() as ws:
            a = ws.make_installed_model("alpha", layout="configs", extra_cfg={"switch_on_request": True})
            ws.make_installed_model("beta", layout="configs")

            with ws.start_mock_service(a["cfg"], extra_args=("--switch-on-request",)) as svc:
                st, _h, _b, api_models = call_http(svc, "GET", "/api/models")
                self.assertEqual(st, 200, api_models)
                self.assertEqual(api_models["current"], "alpha")
                self.assertEqual(sorted(m["id"] for m in api_models["models"]), ["alpha", "beta"])

                st2, _h2, _b2, v1_models = call_http(svc, "GET", "/v1/models")
                self.assertEqual(st2, 200, v1_models)
                ids = {m["id"] for m in v1_models["data"]}
                self.assertTrue({"alpha", "beta"}.issubset(ids), f"expected alpha and beta in /v1/models, got {ids}")

    def test_t1_r2_05_write_path_tools_update_configs(self):
        from strata_control import cfg_paths, models_move, tune_apply

        with IsolatedWorkspace() as ws:
            m = ws.make_installed_model("q2_0", layout="configs")
            new_data = ws.base / "moved-data"
            new_data.mkdir()

            self.assertEqual([p.resolve() for p in models_move.config_files(ws.install)], [m["cfg"].resolve()])
            self.assertEqual([p.resolve() for p in tune_apply.find_config_files(ws.install)], [m["cfg"].resolve()])

            code, out, err = ws.run_cli(
                ["models", "move", str(new_data), "--from", str(ws.data), "--install-dir", str(ws.install), "--output", "json"]
            )
            self.assertEqual(code, 0, f"models move failed: {out}\n{err}")
            moved_cfg = json.loads(m["cfg"].read_text(encoding="utf-8"))
            self.assertIn(str(new_data), moved_cfg["args"][1])

            # Test cfg_paths.repair updates missing old component version path inside configs/
            cur_ctrl = ws.components / "control" / "0.2.20"
            cur_ctrl.mkdir(parents=True)
            (cur_ctrl / "active_marker.txt").write_text("ok", encoding="utf-8")
            (ws.components / "control" / "active.json").write_text(json.dumps({"folder": "0.2.20"}), encoding="utf-8")
            cfg_data = json.loads(m["cfg"].read_text(encoding="utf-8"))
            cfg_data["args"].extend(["--expert-profile", str(ws.components / "control" / "0.1.0" / "active_marker.txt")])
            m["cfg"].write_text(json.dumps(cfg_data, indent=1), encoding="utf-8")

            notes, problems = cfg_paths.repair([m["cfg"]])
            self.assertEqual(problems, [])
            self.assertTrue(notes)
            repaired = json.loads(m["cfg"].read_text(encoding="utf-8"))
            self.assertTrue(any("0.2.20" in str(a) for a in repaired["args"]))
            self.assertEqual(root_clutter_entries(ws.install), [])


class TestTier1R3StartupAndLauncher(_Tier1Base):
    """T1-R3-01 .. T1-R3-05: Relocated scripts & unified CLI launcher."""

    def test_t1_r3_01_write_run_script_places_in_scripts_dir(self):
        import setup

        with IsolatedWorkspace(control_mode=True) as ws:
            cfg_path = ws.install / "configs" / "strata-q2_0.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text("{}", encoding="utf-8")

            script = setup.write_run_script("q2_0", cfg_path, 8080, open_browser=False)
            self.assertEqual(script.parent.resolve(), (ws.install / "scripts").resolve())
            self.assertTrue(script.is_file())
            self.assertEqual(root_clutter_entries(ws.install), [])
            text = script.read_text(encoding="utf-8")
            self.assertIn(str(cfg_path), text)
            self.assertIn(str(setup.ROOT), text)

    def test_t1_r3_02_generated_script_executes_and_targets_configs(self):
        import setup

        with IsolatedWorkspace(control_mode=True) as ws:
            cfg_path = ws.install / "configs" / "strata-q2_0.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text("{}", encoding="utf-8")

            s_closed = setup.write_run_script("q2_0", cfg_path, 8080, open_browser=False)
            text_closed = s_closed.read_text(encoding="utf-8")
            self.assertIn("--engine", text_closed)
            self.assertIn(str(cfg_path), text_closed)
            self.assertIn("--port", text_closed)
            self.assertIn("8080", text_closed)
            self.assertNotIn("--open", text_closed)

            s_open = setup.write_run_script("q2_0", cfg_path, 8080, open_browser=True)
            text_open = s_open.read_text(encoding="utf-8")
            self.assertIn("--open", text_open)

            # Execute generated script with a stubbed serve/server.py inside ws.terminal_here
            stub_srv = ws.terminal_here / "serve" / "server.py"
            stub_srv.parent.mkdir(parents=True, exist_ok=True)
            marker = ws.base / "script_args.json"
            stub_srv.write_text(
                f"import json, sys; open({str(marker)!r}, 'w').write(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            cmd = ["cmd.exe", "/c", str(s_open)] if os.name == "nt" else ["sh", str(s_open)]
            rc, out = ws.run_subprocess_to_file(cmd, cwd=ws.base)
            self.assertEqual(rc, 0, f"script execution failed: {out}")
            argv = json.loads(marker.read_text(encoding="utf-8"))
            self.assertIn("--config", argv)
            self.assertEqual(Path(argv[argv.index("--config") + 1]).resolve(), cfg_path.resolve())
            self.assertIn("--open", argv)

    def test_t1_r3_03_unified_cli_list_models_without_root_scripts(self):
        with IsolatedWorkspace() as ws:
            ws.make_installed_model("q2_0", layout="configs", create_script=False)
            ws.make_installed_model("iq3_xxs", layout="configs", create_script=False)
            self.assertEqual(root_clutter_entries(ws.install), [])

            code, out, err = ws.run_polystrata(["--list-models"])
            self.assertEqual(code, 0, err)
            self.assertIn("q2_0", out)
            self.assertIn("iq3_xxs", out)

            ccode, cout, cerr = ws.run_cli(["models", "list", "--installed", "--output", "json"])
            self.assertEqual(ccode, 0, cerr)
            self.assertEqual(sorted(m["id"] for m in json.loads(cout)["models"]), ["iq3_xxs", "q2_0"])

    def test_t1_r3_04_unified_cli_model_resolution_and_launch(self):
        import polystrata

        with IsolatedWorkspace() as ws:
            m = ws.make_installed_model("q2_0", layout="configs", create_script=False)
            entry = polystrata.entry_of("q2_0")
            self.assertEqual(Path(entry["config"]).resolve(), m["cfg"].resolve())

            # Unified launcher via `polystrata.py` or `strata` CLI without root run-* scripts
            code, out, err = ws.run_cli(["serve", "--profile", str(m["cfg"]), "--plan", "--output", "json"])
            self.assertEqual(code, 0, err)
            plan = json.loads(out)["plan"]
            self.assertEqual(Path(plan["source_config"]).resolve(), m["cfg"].resolve())

            # Also verify polystrata unified launch path works without requiring root run-* scripts
            with mock.patch.object(polystrata, "Server") as mock_srv, \
                    mock.patch.object(polystrata.terminal, "chat", return_value=0):
                pcode, _pout, perr = ws.run_polystrata(["--model", "q2_0"])
                self.assertEqual(pcode, 0, perr)
                mock_srv.assert_called_once()
                self.assertEqual(Path(mock_srv.call_args.args[0]["config"]).resolve(), m["cfg"].resolve())

    def test_t1_r3_05_opt_out_no_scripts_and_mcp_start(self):
        from tools import strata_mcp

        with IsolatedWorkspace(control_mode=True) as ws:
            _comp_dir, comp_backend = make_fake_engine_component(ws.components)
            with mock.patch.dict(os.environ, {"STRATA_ENGINE_BACKEND": comp_backend}):
                # Verify setup exposes an opt-out flag to skip per-model script generation
                help_code, help_out, help_err = ws.run_setup(["--help"], mock_move_over=True)
                help_all = help_out + help_err
                opt_flag = next(
                    (f for f in ("--no-scripts", "--no-run-script", "--no-run-scripts", "--skip-scripts") if f in help_all),
                    None,
                )
                self.assertIsNotNone(opt_flag, f"R3 requires a script opt-out flag in setup.py --help (exit {help_code})")

                code, out, err = ws.run_setup(
                    ["--family", "qwen", "--model", "Q2_0", "--yes", "--no-start", "--vision", "no", opt_flag]
                )
                self.assertEqual(code, 0, f"setup with {opt_flag} failed: {out}\n{err}")
                self.assertEqual(root_clutter_entries(ws.install), [])
                scripts = list((ws.install / "scripts").glob("run-*")) if (ws.install / "scripts").is_dir() else []
                self.assertEqual(scripts, [], f"expected zero run-* scripts with {opt_flag}, got {scripts}")

                s = strata_mcp.Strata(ws.install, ws.terminal_here)
                s.python_override = sys.executable
                cfg_path = s.find_config("q2_0")
                desc = s.describe_config(cfg_path)
                self.assertIsNone(desc["start_script"])
                self.assertTrue(desc["ready"], desc)

                # Verify strata_start builds --config pointing into configs/ without needing run-* script
                tools = strata_mcp.Tools(s)
                with mock.patch.object(strata_mcp, "port_free", return_value=True), \
                        mock.patch.object(strata_mcp, "spawn_detached") as mock_spawn, \
                        mock.patch.object(strata_mcp, "proc_identity", return_value="fake-ident"), \
                        mock.patch.object(tools, "wait_ready", return_value={"summary": "ready"}):
                    mock_spawn.return_value = mock.MagicMock(pid=12345)
                    res = tools.strata_start(model="q2_0", port=18181, wait_seconds=1)
                    self.assertEqual(res["summary"], "ready")
                    spawned_cmd = mock_spawn.call_args.args[0]
                    self.assertIn("--config", spawned_cmd)
                    self.assertEqual(Path(spawned_cmd[spawned_cmd.index("--config") + 1]).resolve(), cfg_path.resolve())


class TestTier1R4MigrationAndCompat(_Tier1Base):
    """T1-R4-01 .. T1-R4-05: Migration, backward compatibility & isolation."""

    def test_t1_r4_01_legacy_repo_root_and_home_migration(self):
        import setup

        with IsolatedWorkspace(repo_layout=True, install_as_here=True) as ws:
            # Place legacy flat files in simulated repo root (ROOT.parent)
            m = ws.make_installed_model(
                "q2_0",
                install_dir=ws.repo,
                layout="legacy",
                shared_settings={"temperature": 0.65},
                create_log=True,
                create_script=True,
                write_bak=True,
            )
            # Also verify script initially points at repo/serve/server.py as pre-0.2.16 clones did
            ext = "bat" if os.name == "nt" else "sh"
            self.assertIn(str(ws.repo / "serve" / "server.py"), m["script"].read_text(encoding="utf-8"))

            moved = setup.move_over_old_layout()
            self.assertTrue(moved, "move_over_old_layout() must migrate legacy repo-root files")
            self.assertEqual(root_clutter_entries(ws.repo), [], "repo root must be clean after migration")
            self.assertEqual(root_clutter_entries(ws.install), [], "install home root must be clean after migration")

            new_cfg = ws.install / "configs" / "strata-q2_0.json"
            new_ss = ws.install / "configs" / "strata-q2_0.shared-settings.json"
            new_bak = ws.install / "configs" / "strata-q2_0.json.bak"
            new_log = ws.install / "logs" / "strata-q2_0.log"
            new_script = ws.install / "scripts" / f"run-q2_0.{ext}"

            for p in (new_cfg, new_ss, new_bak, new_log, new_script):
                self.assertTrue(p.is_file(), f"expected migrated file at {p}")

            cfg_data = json.loads(new_cfg.read_text(encoding="utf-8-sig"))
            self.assertEqual(Path(cfg_data["log"]).resolve(), new_log.resolve())
            new_script_text = new_script.read_text(encoding="utf-8")
            self.assertIn(str(new_cfg), new_script_text)
            self.assertIn(str(ws.install / "serve" / "server.py"), new_script_text)
            self.assertNotIn(str(ws.repo / "serve" / "server.py"), new_script_text)
            self.assertEqual(json.loads(new_ss.read_text(encoding="utf-8")), {"temperature": 0.65})

    def test_t1_r4_02_read_only_legacy_flat_fallback_discovery(self):
        from serve import models as serve_models

        with IsolatedWorkspace() as ws:
            legacy_dir = ws.base / "legacy_unmigrated"
            legacy_dir.mkdir()
            ws.make_installed_model("legacy", install_dir=legacy_dir, layout="legacy")
            before = dir_snapshot(legacy_dir)

            with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(legacy_dir)}):
                code, out, err = ws.run_cli(["models", "list", "--installed", "--install-dir", str(legacy_dir), "--output", "json"])
                self.assertEqual(code, 0, err)
                listed = json.loads(out)
                self.assertEqual([m["id"] for m in listed["models"]], ["legacy"])

                icode, iout, ierr = ws.run_cli(["models", "inspect", "legacy", "--install-dir", str(legacy_dir), "--output", "json"])
                self.assertEqual(icode, 0, ierr)
                self.assertTrue(json.loads(iout)["components"]["ok"])

                srv_list = serve_models.discover(legacy_dir)
                self.assertEqual([m["id"] for m in srv_list], ["legacy"])

                pcode, _pout, perr = ws.run_cli(["serve", "--profile", str(legacy_dir / "strata-legacy.json"), "--plan", "--output", "json"])
                self.assertEqual(pcode, 0, perr)

            assert_dir_unchanged(self, legacy_dir, before, label="legacy flat discovery must be read-only")

    def test_t1_r4_03_explicit_config_backward_compatibility(self):
        with IsolatedWorkspace() as ws:
            custom_dir = ws.base / "arbitrary_user_dir"
            custom_dir.mkdir()
            m = ws.make_installed_model(
                "custom",
                install_dir=ws.install,
                layout=str(custom_dir),
                filename_override="my-custom.json",
            )
            cfg_path = m["cfg"]

            pcode, pout, perr = ws.run_cli(["serve", "--profile", str(cfg_path), "--plan", "--output", "json"])
            self.assertEqual(pcode, 0, perr)
            self.assertEqual(Path(json.loads(pout)["plan"]["source_config"]).resolve(), cfg_path.resolve())

            with ws.start_mock_service(cfg_path) as svc:
                st, _h, _b, health = call_http(svc, "GET", "/health")
                self.assertEqual(st, 200)
                self.assertEqual(health["status"], "ok")

                st_s, _hs, _bs, ss = call_http(svc, "POST", "/settings", {"defaults": {"temperature": 0.5}})
                self.assertEqual(st_s, 200, ss)
                self.assertTrue((custom_dir / "my-custom.shared-settings.json").is_file())

                st_c, _hc, _bc, cfg_resp = call_http(svc, "POST", "/config", {"set": {"fit_max_tokens": False}})
                self.assertEqual(st_c, 200, cfg_resp)
                self.assertTrue((custom_dir / "my-custom.json.bak").is_file())

    def test_t1_r4_04_models_json_overrides_and_profile_compat(self):
        from serve import models as serve_models

        with IsolatedWorkspace() as ws:
            m = ws.make_installed_model("q2_0", layout="configs")
            (ws.install / "models.json").write_text(
                json.dumps({"q2_0": {"display_name": "Custom Q2 Display", "notes": "fast daily"}}),
                encoding="utf-8",
            )
            found = serve_models.discover(ws.install)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["display_name"], "Custom Q2 Display")
            self.assertEqual(found[0]["notes"], "fast daily")

            c1, _o1, e1 = ws.run_cli(["profile", "create", "p1", "--from-config", str(m["cfg"]), "--output", "json"])
            self.assertEqual(c1, 0, e1)
            c2, o2, e2 = ws.run_cli(["serve", "--profile", "p1", "--plan", "--output", "json"])
            self.assertEqual(c2, 0, e2)
            self.assertEqual(Path(json.loads(o2)["plan"]["source_config"]).resolve(), m["cfg"].resolve())

    def test_t1_r4_05_workspace_isolation_and_repo_guard(self):
        from tools.strata_mcp import Strata

        # Negative self-test: prove WorkspaceSnapshotGuard detects mutations in a guarded temp target
        with tempfile.TemporaryDirectory() as tmp_guard_dir:
            g_repo = Path(tmp_guard_dir) / "repo"
            g_term = g_repo / "terminal"
            g_term.mkdir(parents=True)
            guard = WorkspaceSnapshotGuard(repo_root=g_repo, terminal_root=g_term)
            (g_term / "strata-leak.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(AssertionError):
                guard.assert_unchanged("negative self-test")

        # Real operations across isolated workspace must leave real repo untouched
        with IsolatedWorkspace() as ws:
            m = ws.make_installed_model("q2_0", layout="configs")
            ws.run_cli(["models", "list", "--installed", "--output", "json"])
            ws.run_polystrata(["--list-models"])
            Strata(ws.install, ws.terminal_here).configs()
            with ws.start_mock_service(m["cfg"]) as svc:
                call_http(svc, "GET", "/health")
        self._repo_guard.assert_unchanged("after T1-R4-05 operations")


if __name__ == "__main__":
    unittest.main()
