"""Milestone M2 unit tests: setup writers, server/runtime, R3 scripts/launcher, MCP & tools."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

TERMINAL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TERMINAL_ROOT.parent
if str(TERMINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TERMINAL_ROOT))
if str(TERMINAL_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(TERMINAL_ROOT / "tools"))

from strata_control import layout  # noqa: E402


class TestF3SetupAndAdoptionWriters(unittest.TestCase):
    def test_write_setup_config_carries_over_from_unmigrated_legacy_config(self):
        import setup

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            legacy = home / "strata-q2_0.json"
            legacy.write_text(
                json.dumps({"exe": "old", "args": ["--max-context", "8192"], "sampling": {"min_p": 0.05}}),
                encoding="utf-8-sig",
            )
            cfg_path = layout.config_path(home, "Q2_0")
            new_cfg = {"exe": "new", "args": ["--max-context", "32768"], "port": 8080}
            buf = io.StringIO()
            with mock.patch.object(setup, "ROOT", home), mock.patch.object(setup, "_CONTROL_HOME", None), \
                    contextlib.redirect_stdout(buf):
                setup.write_setup_config(cfg_path, new_cfg)
            self.assertTrue(cfg_path.is_file())
            written = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(written["sampling"], {"min_p": 0.05})
            self.assertFalse((home / "configs" / "strata-q2_0.json.bak").exists())

    def test_poly_setup_control_mode_writes_under_control_home_only(self):
        import setup

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            code_root = base / "code_root"
            ctrl_home = base / "ctrl_home"
            data_dir = base / "data"
            code_root.mkdir()
            ctrl_home.mkdir()
            fam = setup.FAMILIES["flint"]
            size = "Q8_0"
            tag = fam["tag"] + size
            models_dir = base / "models"
            model_dir = models_dir / tag
            model_dir.mkdir(parents=True)
            gguf = model_dir / setup.model_file(fam, size, 1)
            gguf.write_bytes(b"GGUF")
            pack = data_dir / "packs" / tag.lower()
            (pack / "tokenizer").mkdir(parents=True)
            (pack / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
            eng = ctrl_home / setup.ENGINE_POLY_DIR
            eng.mkdir(parents=True)
            (eng / setup.POLY_EXE).write_bytes(b"")

            a = argparse.Namespace(
                family="flint", model=size, context=32768, kv=None, kv_streaming="auto",
                rope_scaling=None, rope_scale=None, low_ram="auto", resident_budget_gib=None,
                vram_reserve_mib=None, draft_vocab=None, layer_split=None, calibrate=False,
                experimental_speed_projection=None, vision=None, vision_tokens=None,
                parallel=None, models_dir=str(models_dir), gguf_dir=None, host=None,
                api_key=None, browser=None, no_start=True, no_scripts=False, yes=True,
                gpu=None, _custom_gguf_shards=None, warm_start=None, build=False,
            )
            gpu = {"index": 0, "count": 1, "vram_gb": 24.0, "name": "Test RTX", "arch": "89"}
            common_patches = (
                mock.patch.object(setup, "ROOT", code_root),
                mock.patch.object(setup, "_CONTROL_HOME", ctrl_home),
                mock.patch.object(setup, "CONTROL", True),
                mock.patch.object(setup, "data_folder_for", lambda _a: (data_dir, [])),
                mock.patch.object(setup, "free_gb", lambda _p: 500.0),
                mock.patch.object(setup, "pip_install", lambda *x, **k: None),
                mock.patch.object(setup, "get_llama_cpp", lambda: None),
                mock.patch.object(setup, "get_poly_prebuilt", lambda *x, **k: eng),
                mock.patch.object(setup, "build_engine_poly", lambda *x, **k: eng),
                mock.patch.object(setup, "engine_lib_dirs", lambda _e: []),
                mock.patch.object(setup, "download", side_effect=AssertionError("no download")),
                mock.patch.object(setup, "done", return_value=True),
                mock.patch.object(setup, "whole_shard", return_value=True),
                mock.patch.object(setup, "check_shards", lambda _s: None),
                mock.patch.object(setup, "verify_poly_download", lambda *x, **k: None),
                mock.patch.object(setup, "run", lambda *x, **k: None),
                mock.patch.object(setup, "poly_saved_conv", return_value=0),
                mock.patch.object(setup, "say"),
                mock.patch.object(setup, "ok"),
                mock.patch.object(setup, "step"),
                mock.patch.object(setup, "warn"),
            )
            with contextlib.ExitStack() as st:
                for p in common_patches:
                    st.enter_context(p)
                rc = setup.poly_setup(a, "flint", gpu, False, 13, None, 64.0, data_dir, [], 8080, None)

            self.assertEqual(rc, 0)
            self.assertEqual(list(code_root.glob("strata-*")), [])
            self.assertEqual(list(code_root.glob("run-*")), [])
            self.assertEqual(list(ctrl_home.glob("strata-*")), [])
            self.assertEqual(list(ctrl_home.glob("run-*")), [])
            cfg_file = ctrl_home / "configs" / f"strata-{tag.lower()}.json"
            self.assertTrue(cfg_file.is_file())
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            self.assertEqual(Path(cfg["log"]).resolve(), (ctrl_home / "logs" / f"strata-{tag.lower()}.log").resolve())
            ext = "bat" if setup.WIN else "sh"
            self.assertTrue((ctrl_home / "scripts" / f"run-{tag.lower()}.{ext}").is_file())

            # Re-run with --no-scripts: script is not regenerated when deleted
            (ctrl_home / "scripts" / f"run-{tag.lower()}.{ext}").unlink()
            a.no_scripts = True
            with contextlib.ExitStack() as st:
                for p in common_patches:
                    st.enter_context(p)
                rc2 = setup.poly_setup(a, "flint", gpu, False, 13, None, 64.0, data_dir, [], 8080, None)
            self.assertEqual(rc2, 0)
            self.assertFalse((ctrl_home / "scripts" / f"run-{tag.lower()}.{ext}").exists())

    def test_adopt_prepared_files_writes_to_configs_and_logs(self):
        from strata_cli.launch import adopt, runner, settings as S

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            (src / "data").mkdir(parents=True)
            (src / "data" / "expert-profile.bin").write_bytes(b"p")
            data = root / "data"
            pack = data / "packs" / "swift-iq3_xxs"
            (pack / "tokenizer").mkdir(parents=True)
            (pack / "experts.bin").write_bytes(b"e")
            model_dir = root / "models" / "swift-iq3_xxs"
            model_dir.mkdir(parents=True)
            s1 = model_dir / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
            s2 = model_dir / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"
            s1.write_bytes(b"1")
            s2.write_bytes(b"2")
            install_root = root / "install"
            install_root.mkdir()

            cat = mock.Mock()
            cat.src = src
            cat.root = install_root
            cat.data_dir.return_value = data
            cat.tag_of.return_value = "swift-iq3_xxs"
            cat.installed_config.return_value = None
            cat.tables.return_value = ({}, {"swift": {"tag": "swift-", "name": "swift-1.5"}}, [], "test")
            cat.computer.return_value = mock.Mock(main_vram_gb=24.0)

            s = S.LaunchSettings(family="swift", model="IQ3_XXS", context=65536, model_path=str(model_dir))
            r = runner.Runner(s, catalog=cat)
            with mock.patch("strata_control.engine_component.detect_backend", return_value=("sycl", "detected")), \
                    mock.patch.object(adopt, "intel_driver", return_value="xe"):
                out_path = r.adopt_prepared_files("", chosen=True)

            self.assertEqual(out_path, install_root / "configs" / "strata-swift-iq3_xxs.json")
            self.assertTrue(out_path.is_file())
            cfg = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(Path(cfg["log"]).resolve(), (install_root / "logs" / "strata-swift-iq3_xxs.log").resolve())
            self.assertEqual(list(install_root.glob("strata-*")), [])

    def test_write_config_and_migrate_run_configs_cleanup_tmp_on_error(self):
        import setup

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            cfg_dir = home / "configs"
            cfg_dir.mkdir()
            cfg_p = cfg_dir / "strata-test.json"
            cfg_p.write_text(json.dumps({"exe": "old"}), encoding="utf-8")

            orig_write_text = Path.write_text

            def partial_enospc(self_path, data, *args, **kwargs):
                if self_path.name.endswith(".tmp"):
                    orig_write_text(self_path, '{"partial":', *args, **kwargs)
                    raise OSError(28, "No space left on device")
                return orig_write_text(self_path, data, *args, **kwargs)

            with mock.patch.object(Path, "write_text", partial_enospc):
                with self.assertRaises(OSError):
                    setup.write_config(cfg_p, {"exe": "new"})
            self.assertEqual([p.name for p in cfg_dir.iterdir()], ["strata-test.json"])
            self.assertEqual(json.loads(cfg_p.read_text(encoding="utf-8")), {"exe": "old"})

            with mock.patch.object(setup.atomic, "replace", side_effect=OSError(13, "Permission denied")):
                with self.assertRaises(OSError):
                    setup.write_config(cfg_p, {"exe": "new"})
            self.assertEqual([p.name for p in cfg_dir.iterdir()], ["strata-test.json"])
            self.assertEqual(json.loads(cfg_p.read_text(encoding="utf-8")), {"exe": "old"})

            mig_p = cfg_dir / "strata-unsloth.json"
            mig_p.write_text(json.dumps({"exe": "old", "args": ["--resident-budget-gib", "40"]}), encoding="utf-8")
            with mock.patch.object(setup, "migrate_ram_budget", return_value={"exe": "new"}), \
                    mock.patch.object(setup.atomic, "replace", side_effect=OSError(28, "No space left on device")):
                moved = setup.migrate_run_configs(home)
            self.assertEqual(moved, [])
            self.assertFalse(any(p.name.endswith(".tmp") for p in cfg_dir.iterdir()))

    def test_setup_intel_no_scripts_alchemist_warn_and_s_root_replace(self):
        engine_root = REPO_ROOT / "engine"
        if str(engine_root) not in sys.path:
            sys.path.insert(0, str(engine_root))
        import setup
        from sycl import setup_intel

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            cfg_p = home / "configs" / "strata-q2_0.json"
            cfg_p.parent.mkdir(parents=True)
            cfg_p.write_text(
                json.dumps({"exe": "e", "args": ["--max-context", "32768", "--kv", "int8"], "gpu": 0}),
                encoding="utf-8",
            )
            stub = home / "reader"
            stub.mkdir()
            (stub / "BUILD.json").write_text("{}", encoding="utf-8")
            exe = home / "build-sycl-aot" / "strata"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"")

            saved = {k: getattr(setup, k) for k in (
                "say", "warn", "gpus", "amd_gpus", "amd_problem", "hip_vision",
                "build_engine_hip", "hipblaslt_table", "ram_gb", "mem_have",
                "bench_tips", "write_setup_config", "write_run_script", "start",
            )}
            warnings: list[str] = []
            says: list[str] = []

            def fake_main():
                setup.write_setup_config(cfg_p, json.loads(cfg_p.read_text(encoding="utf-8")))
                return 0

            try:
                with mock.patch.object(setup, "ROOT", home), \
                        mock.patch.object(setup, "_CONTROL_HOME", None), \
                        mock.patch.object(setup, "WIN", False), \
                        mock.patch.object(setup, "ram_gb", return_value=16.0), \
                        mock.patch.object(setup, "warn", side_effect=lambda m="": warnings.append(str(m))), \
                        mock.patch.object(setup, "say", side_effect=lambda m="": says.append(str(m))), \
                        mock.patch.object(setup, "main", side_effect=fake_main), \
                        mock.patch.object(setup_intel, "ROOT", home), \
                        mock.patch.object(setup_intel, "MOUNT", home), \
                        mock.patch.object(setup_intel, "READER_DIR", stub), \
                        mock.patch.object(setup_intel, "sycl_engine", return_value=(exe, None)), \
                        mock.patch.object(setup_intel, "sycl_version", return_value="0.2.20"), \
                        mock.patch.object(setup_intel, "intel_devices", return_value=(
                            [{"vram_gb": 16.0, "driver": "i915", "devid": "56a0", "name": "Arc A770"}], [],
                        )), \
                        mock.patch.object(sys, "argv", ["setup_intel.py"]):
                    with self.assertRaises(SystemExit) as cm:
                        setup_intel.install(["--no-scripts", "--yes", "--no-start"])
                    self.assertEqual(cm.exception.code, 0)
                    self.assertTrue(any("Q2_0 on this Arc" in w for w in warnings), warnings)

                    script = setup.write_run_script("Q2_0", cfg_p, 8080, open_browser=False)
                    script_text = script.read_text(encoding="utf-8")
                    self.assertIn(str(setup_intel.SERVER), script_text)
                    self.assertNotIn(str(home / "serve" / "server.py"), script_text)
            finally:
                for k, v in saved.items():
                    setattr(setup, k, v)


class TestF4ServerAndLogs(unittest.TestCase):
    def test_appendlog_creates_parent_logs_dir(self):
        from serve import appendlog

        with tempfile.TemporaryDirectory() as td:
            log_file = Path(td) / "logs" / "nested" / "strata-q2_0.log"
            self.assertFalse(log_file.parent.exists())
            with appendlog.open_append(log_file) as fh:
                fh.write("first\n")
            with appendlog.open_append(log_file) as fh:
                fh.write("second\n")
            self.assertEqual(log_file.read_text(encoding="utf-8"), "first\nsecond\n")

    def test_serve_models_caps_huge_max_context_digits(self):
        from serve import models as serve_models

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "exe").write_bytes(b"")
            (root / "tok").mkdir()
            (root / "tok" / "vocab.json").write_text("{}", encoding="utf-8")
            cfg = root / "configs" / "strata-huge.json"
            cfg.parent.mkdir()
            cfg.write_text(
                json.dumps({"exe": str(root / "exe"), "tokenizer": str(root / "tok"), "args": ["--max-context", "9" * 5000]}),
                encoding="utf-8",
            )
            entry = serve_models.describe(cfg)
            self.assertIsNotNone(entry)
            self.assertIsNone(entry["context"])

    def test_server_main_models_root_shared_settings_and_sidecars(self):
        from serve import mcp as mcp_mod
        from serve import server as srv_mod

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            cfg_dir = home / "configs"
            cfg_dir.mkdir()
            from strata_tokenizer import BYTE_TO_UNICODE
            tok_dir = home / "tok"
            tok_dir.mkdir()
            tokens = [BYTE_TO_UNICODE[b] for b in range(256)] + ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
            (tok_dir / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}), encoding="utf-8")
            (tok_dir / "merges.json").write_text("[]", encoding="utf-8")
            (tok_dir / "token_type.json").write_text(json.dumps([1] * 256 + [3, 3, 4, 4]), encoding="utf-8")
            (tok_dir / "tokenizer.json").write_text(
                json.dumps({"pre": "qwen35", "special_ids": {"tokenizer.ggml.eos_token_id": 257}}),
                encoding="utf-8",
            )
            alpha_cfg = cfg_dir / "strata-alpha.json"
            alpha_cfg.write_text(
                json.dumps({"exe": "mock", "tokenizer": str(home / "tok"), "args": ["--max-context", "32768"],
                            "model_name": "alpha", "port": 8080, "warm_start": True, "mcp_servers": {"x": {"command": "echo"}}}),
                encoding="utf-8",
            )
            legacy_cfg = home / "strata-beta.json"
            legacy_cfg.write_text(
                json.dumps({"exe": "mock", "tokenizer": str(home / "tok"), "args": ["--max-context", "32768"],
                            "model_name": "beta", "port": 8081}),
                encoding="utf-8",
            )
            shared_p = cfg_dir / "strata-alpha.shared-settings.json"
            shared_p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"temperature": 0.4}).encode("utf-8"))

            captured = {}

            class FakeHttpd:
                server_address = ("127.0.0.1", 18080)

                def shutdown(self):
                    pass

            def fake_serve(svc, host="127.0.0.1", port=8080):
                captured["svc"] = svc
                return FakeHttpd()

            with mock.patch.object(srv_mod, "serve", fake_serve), \
                    mock.patch.object(mcp_mod.McpHub, "start", lambda self, wait=False: None), \
                    mock.patch.object(srv_mod, "resolve_image_results", lambda *a, **k: True), \
                    mock.patch("atexit.register"), \
                    mock.patch.object(srv_mod.Service, "start_warm"), \
                    mock.patch.object(srv_mod.Service, "start_idle_unload"), \
                    mock.patch.object(srv_mod.Service, "start_vision_idle"), \
                    mock.patch.object(srv_mod.time, "sleep", side_effect=KeyboardInterrupt), \
                    contextlib.redirect_stdout(io.StringIO()):
                srv_mod.main(["--engine", "mock", "--config", str(alpha_cfg), "--port", "0"])

            svc = captured["svc"]
            self.assertEqual(Path(svc.models_root).resolve(), home.resolve())
            self.assertEqual(Path(svc.shared_path).resolve(), shared_p.resolve())
            self.assertEqual(svc.shared, {"temperature": 0.4})
            self.assertEqual(svc.warm_path().parent.resolve(), cfg_dir.resolve())
            self.assertEqual(Path(svc.mcp.media_dir).resolve(), alpha_cfg.with_suffix(".tool-media").resolve())
            self.assertEqual(sorted(m["id"] for m in svc.models()), ["alpha", "beta"])

            svc.set_shared({"top_p": 0.9})
            self.assertEqual(json.loads(shared_p.read_text(encoding="utf-8")), {"top_p": 0.9})
            svc.set_shared({})
            self.assertFalse(shared_p.exists())

    def test_hip_env_guard_uses_logs_tmp_when_log_missing(self):
        from serve import server as srv_mod

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "install"
            root.mkdir()
            bad_tmp = str(Path(td) / "nonexistent_dir_12345")
            env = {"TEMP": bad_tmp, "TMP": bad_tmp, "TMPDIR": bad_tmp}
            cfg = {"cwd": str(root)}
            with mock.patch.object(srv_mod, "ROOT", root):
                said = srv_mod.hip_env_guard(env, cfg)
            self.assertTrue(said)
            self.assertEqual(Path(env["TEMP"]).resolve(), (root / "logs" / "tmp").resolve())
            self.assertFalse((Path(td) / "tmp").exists())


class TestF5LauncherAndScripts(unittest.TestCase):
    def test_write_run_script_places_in_scripts_with_exact_newlines(self):
        import setup

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            cfg = home / "configs" / "strata-q2_0.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text("{}", encoding="utf-8")
            with mock.patch.object(setup, "ROOT", home), mock.patch.object(setup, "_CONTROL_HOME", None):
                with mock.patch.object(setup, "WIN", True):
                    bat = setup.write_run_script("Q2_0", cfg, 8080, open_browser=False)
                    self.assertEqual(bat.parent, home / "scripts")
                    raw_bat = bat.read_bytes()
                    self.assertNotIn(b"\r\r\n", raw_bat)
                    self.assertIn(b"\r\n", raw_bat)
                    self.assertIn(str(cfg).encode("utf-8"), raw_bat)
                with mock.patch.object(setup, "WIN", False):
                    sh = setup.write_run_script("Q2_0", cfg, 8080, open_browser=True)
                    self.assertEqual(sh.parent, home / "scripts")
                    raw_sh = sh.read_bytes()
                    self.assertNotIn(b"\r", raw_sh)
                    self.assertIn(b"--open", raw_sh)

    def test_polystrata_launcher_list_start_and_errors(self):
        import polystrata

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            buf = io.StringIO()
            with mock.patch.object(polystrata, "ROOT", home), contextlib.redirect_stdout(buf):
                rc_empty = polystrata.list_models(None, "")
            self.assertEqual(rc_empty, 1)
            self.assertIn("setup", buf.getvalue())

            (home / "exe").write_bytes(b"")
            from strata_tokenizer import BYTE_TO_UNICODE
            tok_dir = home / "tok"
            tok_dir.mkdir()
            tokens = [BYTE_TO_UNICODE[b] for b in range(256)] + ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
            (tok_dir / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}), encoding="utf-8")
            (tok_dir / "merges.json").write_text("[]", encoding="utf-8")
            (tok_dir / "token_type.json").write_text(json.dumps([1] * 256 + [3, 3, 4, 4]), encoding="utf-8")
            (tok_dir / "tokenizer.json").write_text(
                json.dumps({"pre": "qwen35", "special_ids": {"tokenizer.ggml.eos_token_id": 257}}),
                encoding="utf-8",
            )
            cfg = home / "configs" / "strata-q2_0.json"
            cfg.parent.mkdir()
            cfg.write_text(
                json.dumps({"exe": str(home / "exe"), "tokenizer": str(home / "tok"),
                            "args": ["--max-context", "32768"], "model_name": "q2_0", "port": 8199}),
                encoding="utf-8",
            )
            with mock.patch.object(polystrata, "ROOT", home), \
                    mock.patch.object(polystrata.terminal, "api", return_value=(0, {})), \
                    contextlib.redirect_stdout(io.StringIO()) as list_out:
                rc = polystrata.list_models(None, "")
            self.assertEqual(rc, 0)
            self.assertIn("q2_0", list_out.getvalue())

            with mock.patch.object(polystrata, "ROOT", home):
                with self.assertRaises(SystemExit) as bad:
                    polystrata.entry_of("missing-model")
                self.assertNotEqual(bad.exception.code, 0)

            with mock.patch.object(polystrata, "ROOT", home), \
                    mock.patch.object(polystrata.subprocess, "call", return_value=0) as call_mock, \
                    mock.patch.object(sys, "argv", ["polystrata.py", "start", "q2_0", "--open"]):
                rc_start = polystrata.main()
            self.assertEqual(rc_start, 0)
            cmd = call_mock.call_args.args[0]
            self.assertIn("--config", cmd)
            self.assertEqual(Path(cmd[cmd.index("--config") + 1]).resolve(), cfg.resolve())
            self.assertIn("--open", cmd)

    def test_strata_cli_serve_model_and_run_plan(self):
        from strata_cli import cli

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            (home / "exe").write_bytes(b"")
            from strata_tokenizer import BYTE_TO_UNICODE
            tok_dir = home / "tok"
            tok_dir.mkdir()
            tokens = [BYTE_TO_UNICODE[b] for b in range(256)] + ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
            (tok_dir / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}), encoding="utf-8")
            (tok_dir / "merges.json").write_text("[]", encoding="utf-8")
            (tok_dir / "token_type.json").write_text(json.dumps([1] * 256 + [3, 3, 4, 4]), encoding="utf-8")
            (tok_dir / "tokenizer.json").write_text(
                json.dumps({"pre": "qwen35", "special_ids": {"tokenizer.ggml.eos_token_id": 257}}),
                encoding="utf-8",
            )
            cfg = home / "configs" / "strata-alpha.json"
            cfg.parent.mkdir()
            cfg.write_text(
                json.dumps({"exe": str(home / "exe"), "tokenizer": str(home / "tok"),
                            "args": ["--max-context", "32768"], "model_name": "alpha", "port": 8080}),
                encoding="utf-8",
            )
            with mock.patch.object(cli, "ROOT", home), \
                    mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(home)}), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                rc = cli.main(["serve", "--model", "alpha", "--plan", "--output", "json"])
            self.assertEqual(rc, 0)
            plan = json.loads(out.getvalue())["plan"]
            self.assertEqual(Path(plan["source_config"]).resolve(), cfg.resolve())

            with mock.patch.object(cli, "ROOT", home), \
                    mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(home)}), \
                    contextlib.redirect_stdout(io.StringIO()) as out2:
                rc2 = cli.main(["run", "alpha", "--plan", "--output", "json"])
            self.assertEqual(rc2, 0)
            plan2 = json.loads(out2.getvalue())["plan"]
            self.assertEqual(Path(plan2["source_config"]).resolve(), cfg.resolve())

            for argv in (
                ["run", "", "--plan", "--output", "json"],
                ["serve", "--model", "", "--plan", "--output", "json"],
            ):
                with mock.patch.object(cli, "ROOT", home), \
                        mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(home)}), \
                        contextlib.redirect_stdout(io.StringIO()) as err_out:
                    rc_empty = cli.main(argv)
                self.assertEqual(rc_empty, 5)
                payload = json.loads(err_out.getvalue())
                self.assertFalse(payload["ok"])

    def test_find_and_cli_resolve_config_tag_when_model_name_differs(self):
        from serve import models as registry
        from strata_cli import cli
        import polystrata

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            (home / "exe").write_bytes(b"")
            from strata_tokenizer import BYTE_TO_UNICODE
            tok_dir = home / "tok"
            tok_dir.mkdir()
            tokens = [BYTE_TO_UNICODE[b] for b in range(256)] + ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
            (tok_dir / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}), encoding="utf-8")
            (tok_dir / "merges.json").write_text("[]", encoding="utf-8")
            (tok_dir / "token_type.json").write_text(json.dumps([1] * 256 + [3, 3, 4, 4]), encoding="utf-8")
            (tok_dir / "tokenizer.json").write_text(
                json.dumps({"pre": "qwen35", "special_ids": {"tokenizer.ggml.eos_token_id": 257}}),
                encoding="utf-8",
            )
            cfg = home / "configs" / "strata-flint-q8_0.json"
            cfg.parent.mkdir()
            cfg.write_text(
                json.dumps({"exe": str(home / "exe"), "tokenizer": str(tok_dir),
                            "args": ["--max-context", "32768"], "model_name": "flint-40b-a2.5b-q8_0", "port": 8080}),
                encoding="utf-8",
            )

            discovered = registry.discover(home)
            self.assertEqual(registry.find(discovered, "flint-q8_0")["config"], str(cfg))
            self.assertEqual(registry.find(discovered, "FLINT-Q8_0")["id"], "flint-40b-a2.5b-q8_0")
            self.assertIsNone(registry.find(discovered, ""))

            # Ambiguous stripped tag and duplicate cfg.name/cfg.stem across two roots return None instead of guessing
            dup = [
                {"id": "model-a", "config": str(home / "r1" / "configs" / "strata-dup-q8_0.json")},
                {"id": "model-b", "config": str(home / "r2" / "configs" / "strata-dup-q8_0.json")},
            ]
            self.assertIsNone(registry.find(dup, "dup-q8_0"))
            self.assertIsNone(registry.find(dup, "strata-dup-q8_0"))
            self.assertIsNone(registry.find(dup, "strata-dup-q8_0.json"))
            self.assertEqual(registry.find(dup, "model-a")["id"], "model-a")
            self.assertEqual(registry.find(dup, "model-b")["id"], "model-b")

            with mock.patch.object(polystrata, "ROOT", home):
                self.assertEqual(Path(polystrata.entry_of("flint-q8_0")["config"]).resolve(), cfg.resolve())

            for argv in (
                ["run", "flint-q8_0", "--plan", "--output", "json"],
                ["serve", "--model", "flint-q8_0", "--plan", "--output", "json"],
            ):
                with mock.patch.object(cli, "ROOT", home), \
                        mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(home)}), \
                        contextlib.redirect_stdout(io.StringIO()) as out:
                    rc = cli.main(argv)
                self.assertEqual(rc, 0)
                self.assertEqual(Path(json.loads(out.getvalue())["plan"]["source_config"]).resolve(), cfg.resolve())


class TestF6McpAndAuxiliaryTools(unittest.TestCase):
    def test_mcp_configs_describe_removal_start_and_logs(self):
        import strata_mcp

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "install"
            src = base / "src"
            (src / "strata_control").mkdir(parents=True)
            (root / "configs").mkdir(parents=True)
            (root / "logs").mkdir(parents=True)
            (root / "scripts").mkdir(parents=True)
            (root / "data" / "packs" / "q2_0" / "tokenizer").mkdir(parents=True)
            (root / "data" / "packs" / "q2_0" / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
            exe = root / "engine" / "strata.exe"
            exe.parent.mkdir()
            exe.write_bytes(b"")
            gguf = root / "data" / "models" / "q2_0.gguf"
            gguf.parent.mkdir(parents=True)
            gguf.write_bytes(b"model")
            cfg_p = root / "configs" / "strata-q2_0.json"
            log_p = root / "logs" / "strata-q2_0.log"
            log_p.write_text("line1\nline2\n", encoding="utf-8")
            cfg_p.write_text(
                json.dumps({"exe": str(exe), "tokenizer": str(root / "data" / "packs" / "q2_0" / "tokenizer"),
                            "args": ["--native", str(gguf), "--pack", str(root / "data" / "packs" / "q2_0")],
                            "model_name": "q2_0", "log": str(log_p), "port": 8080}),
                encoding="utf-8",
            )
            (root / "configs" / "strata-q2_0.json.bak").write_text("{}", encoding="utf-8")
            (root / "configs" / "strata-q2_0.shared-settings.json").write_text("{}", encoding="utf-8")
            (root / "configs" / "strata-q2_0.warm-start").write_text("[]", encoding="utf-8")
            (root / "configs" / "strata-q2_0.tool-media").mkdir()
            (root / "configs" / "strata-q2_0.tool-media" / "a.png").write_bytes(b"png")
            script_ext = "bat" if strata_mcp.WIN else "sh"
            script_p = root / "scripts" / f"run-q2_0.{script_ext}"
            script_p.write_text("echo run\n", encoding="utf-8")

            s = strata_mcp.Strata(root, TERMINAL_ROOT)
            s.python_override = sys.executable
            self.assertEqual([p.resolve() for p in s.configs()], [cfg_p.resolve()])
            self.assertEqual(s.find_config("Q2_0").resolve(), cfg_p.resolve())
            with mock.patch.object(s, "disk_usage", return_value={"models": {"q2_0": {"gb": 0.1, "bytes": 100, "shared_bytes": 0}}}):
                desc = s.describe_config(cfg_p)
            self.assertEqual(desc["start_script"], script_p.name)
            self.assertTrue(desc["ready"])

            tools = strata_mcp.Tools(s)
            logs_res = tools.strata_logs(source="engine", model="q2_0", lines=10)
            self.assertEqual(logs_res["lines"], ["line1", "line2"])

            with mock.patch.object(strata_mcp, "port_free", return_value=True), \
                    mock.patch.object(strata_mcp, "spawn_detached", return_value=mock.MagicMock(pid=999)) as sp, \
                    mock.patch.object(strata_mcp, "proc_identity", return_value="id"), \
                    mock.patch.object(tools, "wait_ready", return_value={"summary": "ready"}):
                tools.strata_start(model="q2_0", port=19090, wait_seconds=1)
            cmd = sp.call_args.args[0]
            self.assertEqual(Path(cmd[1]).resolve(), (TERMINAL_ROOT / "serve" / "server.py").resolve())

            rm_res = tools.strata_remove(model="q2_0", confirm=True)
            self.assertEqual(rm_res["failed"], [])
            for gone in (cfg_p, log_p, script_p, root / "configs" / "strata-q2_0.json.bak",
                         root / "configs" / "strata-q2_0.shared-settings.json",
                         root / "configs" / "strata-q2_0.warm-start",
                         root / "configs" / "strata-q2_0.tool-media"):
                self.assertFalse(gone.exists(), f"{gone} should be deleted")

    def test_strata_bench_calibrate_and_entrypoint(self):
        import calibrate
        import strata_bench

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "configs").mkdir()
            c1 = root / "configs" / "strata-m1.json"
            c1.write_text(json.dumps({"exe": "e", "args": []}), encoding="utf-8")
            (root / "configs" / "strata-m1.shared-settings.json").write_text("{}", encoding="utf-8")
            self.assertEqual(strata_bench.find_configs(root), [c1])

            (root / "tok").mkdir()
            (root / "tok" / "vocab.json").write_text('{"a": 0, "b": 1, "ab": 2}', encoding="utf-8")
            (root / "tok" / "merges.txt").write_text("a b", encoding="utf-8")
            (root / "tok" / "token_type.json").write_text("[1, 1, 1]", encoding="utf-8")
            log_file = root / "logs" / "strata-cal.log"
            cal_cfg = {
                "exe": "e", "args": ["--max-context", "8192"], "tokenizer": str(root / "tok"),
                "log": str(log_file), "vision": {"exe": "v", "mmproj": "m", "model": "g", "gpu": False, "max_tokens": 300},
            }
            fake_vision = mock.MagicMock()
            fake_st = types.SimpleNamespace(Tokenizer=lambda *a: types.SimpleNamespace(encode=lambda *a, **k: [0]))
            with mock.patch.dict(sys.modules, {"strata_tokenizer": fake_st}), \
                    mock.patch("serve.server.StrataEngine", return_value=mock.MagicMock()), \
                    mock.patch("serve.server.Vision", return_value=fake_vision), \
                    mock.patch("serve.server.child_env", return_value={}), \
                    mock.patch("serve.server.vision_env", return_value={}), \
                    mock.patch.object(calibrate, "host_worker_extras", return_value=[]), \
                    mock.patch.object(calibrate, "measure", return_value={"settings": {}, "report": {}}):
                res = calibrate.run(cal_cfg)
            self.assertEqual(res, {"settings": {}, "report": {}})
            self.assertTrue(log_file.parent.is_dir())

        sh_text = (REPO_ROOT / "tools" / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn("/data/install/configs/strata-*.json /data/install/strata-*.json", sh_text)
        self.assertIn("*.shared-settings.json", sh_text)

    def test_mcp_standalone_copy_without_strata_control_and_precedence(self):
        import shutil
        import subprocess
        import strata_mcp

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            fake_src = base / "fake_src"
            fake_tools = fake_src / "tools"
            fake_tools.mkdir(parents=True)
            shutil.copy2(TERMINAL_ROOT / "tools" / "strata_mcp.py", fake_tools / "strata_mcp.py")

            fake_root = base / "fake_root"
            (fake_root / "configs").mkdir(parents=True)
            (fake_root / "logs").mkdir(parents=True)
            (fake_root / "scripts").mkdir(parents=True)
            (fake_root / "engine").mkdir(parents=True)
            (fake_root / "engine" / "strata.exe").write_bytes(b"")
            (fake_root / "tok").mkdir(parents=True)
            (fake_root / "tok" / "vocab.json").write_text("{}", encoding="utf-8")
            (fake_root / "model.gguf").write_bytes(b"g")
            cfg_p = fake_root / "configs" / "strata-q2_0.json"
            cfg_p.write_text(json.dumps({
                "exe": str(fake_root / "engine" / "strata.exe"),
                "tokenizer": str(fake_root / "tok"),
                "args": ["--native", str(fake_root / "model.gguf")],
                "model_name": "q2_0",
                "log": str(fake_root / "logs" / "strata-q2_0.log"),
            }), encoding="utf-8")
            script_ext = "bat" if strata_mcp.WIN else "sh"
            (fake_root / "scripts" / f"run-q2_0.{script_ext}").write_text("echo run\n", encoding="utf-8")

            env = {k: v for k, v in os.environ.items() if k.upper() != "PYTHONPATH"}
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            probe = (
                "import sys; "
                f"sys.path.insert(0, {str(fake_tools)!r}); "
                "import strata_mcp as M; "
                f"s = M.Strata({str(fake_root)!r}, {str(fake_src)!r}); "
                "cfgs = s.configs(); "
                "assert len(cfgs) == 1, cfgs; "
                "found = s.find_config('Q2_0'); "
                "assert found == cfgs[0], found; "
                "desc = s.describe_config(cfgs[0], usage={'models': {'q2_0': {'gb': 0.0, 'bytes': 0, 'shared_bytes': 0}}}); "
                "assert desc['model'] == 'q2_0' and desc['start_script'] is not None, desc; "
                "plan = s.removal_plan(cfgs[0]); "
                "assert len(plan['remove']) >= 2, plan; "
                "print('STANDALONE_OK')"
            )
            out_file = base / "stdout.txt"
            err_file = base / "stderr.txt"
            with open(out_file, "w", encoding="utf-8") as out_fh, open(err_file, "w", encoding="utf-8") as err_fh:
                res = subprocess.run([sys.executable, "-B", "-c", probe], cwd=str(base), env=env, stdout=out_fh, stderr=err_fh)
            stdout_text = out_file.read_text(encoding="utf-8")
            stderr_text = err_file.read_text(encoding="utf-8")
            self.assertEqual(res.returncode, 0, stderr_text + stdout_text)
            self.assertIn("STANDALONE_OK", stdout_text)

            # Precedence: when both self.src and DEFAULT_ROOT have strata_control/, self.src is inserted first
            src_a = base / "src_a"
            src_b = base / "src_b"
            (src_a / "strata_control").mkdir(parents=True)
            (src_b / "strata_control").mkdir(parents=True)
            s_inst = strata_mcp.Strata(fake_root, src_a)
            fake_path = ["existing_entry"]
            with mock.patch.object(strata_mcp, "DEFAULT_ROOT", src_b), \
                    mock.patch.object(strata_mcp.sys, "path", fake_path):
                s_inst._layout()
            self.assertEqual(fake_path[0], str(src_a))
            self.assertNotIn(str(src_b), fake_path)

    def test_mcp_older_strata_control_without_layout_and_standalone_install_plan(self):
        import shutil
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            fake_src = base / "fake_src"
            fake_tools = fake_src / "tools"
            fake_tools.mkdir(parents=True)
            shutil.copy2(TERMINAL_ROOT / "tools" / "strata_mcp.py", fake_tools / "strata_mcp.py")

            fake_root = base / "fake_root"
            (fake_root / "configs").mkdir(parents=True)
            cfg_p = fake_root / "configs" / "strata-q2_0.json"
            cfg_p.write_text(json.dumps({"exe": "e", "args": [], "model_name": "q2_0"}), encoding="utf-8")
            data_dir = base / "PolyStrata-data"
            data_dir.mkdir()

            env = {k: v for k, v in os.environ.items() if k.upper() != "PYTHONPATH"}
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["STRATA_SETTINGS_FILE"] = str(base / "settings.json")

            def _run_probe(tag: str, code: str) -> tuple[int, str, str]:
                out_f = base / f"{tag}_stdout.txt"
                err_f = base / f"{tag}_stderr.txt"
                with open(out_f, "w", encoding="utf-8") as ofh, open(err_f, "w", encoding="utf-8") as efh:
                    res = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(base), env=env, stdout=ofh, stderr=efh)
                return res.returncode, out_f.read_text(encoding="utf-8"), err_f.read_text(encoding="utf-8")

            # 1) Standalone copy without strata_control/: install_plan(..., ram=32) works via fit_label fallback
            probe_standalone = (
                "import sys; "
                f"sys.path.insert(0, {str(fake_tools)!r}); "
                "import strata_mcp as M; "
                f"s = M.Strata({str(fake_root)!r}, {str(fake_src)!r}); "
                "s.hardware = lambda: {'ram_gb': 32.0, 'gpus': [], 'mac': False}; "
                "s.recommend = lambda hw: {'family': 'qwen', 'model': 'Q2_0', 'context': 32768}; "
                f"plan, args = M.Tools(s).install_plan('qwen', 'Q2_0', 32768, 'no', None, None, None, None, None, None, {str(data_dir)!r}); "
                "assert plan['ram'] == 'may run slower', plan; "
                "assert '--model' in args and 'Q2_0' in args, args; "
                "print('STANDALONE_PLAN_OK')"
            )
            rc1, out1, err1 = _run_probe("standalone_plan", probe_standalone)
            self.assertEqual(rc1, 0, err1 + out1)
            self.assertIn("STANDALONE_PLAN_OK", out1)

            # 2) Older strata_control/__init__.py without layout.py / discovery.py / fit_label.py
            (fake_src / "strata_control").mkdir()
            (fake_src / "strata_control" / "__init__.py").write_text("# older package\n", encoding="utf-8")
            probe_older = (
                "import sys; "
                f"sys.path.insert(0, {str(fake_tools)!r}); "
                "import strata_mcp as M; "
                f"s = M.Strata({str(fake_root)!r}, {str(fake_src)!r}); "
                "cfgs = s.configs(); "
                "assert len(cfgs) == 1, cfgs; "
                f"assert str(s.settings_path()).endswith('settings.json'); "
                f"assert {str(fake_src)!r} not in sys.path, sys.path; "
                "assert 'strata_control' not in sys.modules, list(sys.modules); "
                "s.hardware = lambda: {'ram_gb': 64.0, 'gpus': [], 'mac': False}; "
                "s.recommend = lambda hw: {'family': 'qwen', 'model': 'Q2_0', 'context': 32768}; "
                f"plan, _ = M.Tools(s).install_plan('qwen', 'Q2_0', 32768, 'no', None, None, None, None, None, None, {str(data_dir)!r}); "
                "assert plan['ram'] == 'fits', plan; "
                "print('OLDER_FALLBACK_OK')"
            )
            rc2, out2, err2 = _run_probe("older_fallback", probe_older)
            self.assertEqual(rc2, 0, err2 + out2)
            self.assertIn("OLDER_FALLBACK_OK", out2)

            # 3) self.src has older strata_control without layout, while DEFAULT_ROOT has full strata_control/layout.py
            default_src = base / "default_src"
            (default_src / "strata_control").mkdir(parents=True)
            (default_src / "strata_control" / "__init__.py").write_text("", encoding="utf-8")
            (default_src / "strata_control" / "layout.py").write_text("MARKER = 'DEFAULT'\n", encoding="utf-8")
            probe_default = (
                "import sys; "
                f"sys.path.insert(0, {str(fake_tools)!r}); "
                "import strata_mcp as M; "
                f"from pathlib import Path; M.DEFAULT_ROOT = Path({str(default_src)!r}); "
                f"s = M.Strata({str(fake_root)!r}, {str(fake_src)!r}); "
                "lay = s._layout(); "
                "assert getattr(lay, 'MARKER', None) == 'DEFAULT', lay; "
                f"assert {str(fake_src)!r} not in sys.path, sys.path; "
                "print('DEFAULT_FALLTHROUGH_OK')"
            )
            rc3, out3, err3 = _run_probe("default_fallthrough", probe_default)
            self.assertEqual(rc3, 0, err3 + out3)
            self.assertIn("DEFAULT_FALLTHROUGH_OK", out3)


if __name__ == "__main__":
    unittest.main()
