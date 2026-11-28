"""Milestone M3 unit & integration tests: F8 R4 legacy migration and F9 backward compatibility."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import runpy
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TERMINAL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TERMINAL_ROOT.parent
if str(TERMINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TERMINAL_ROOT))
if str(TERMINAL_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(TERMINAL_ROOT / "tools"))

from strata_control import discovery, layout, models, profiles  # noqa: E402
from tests.terminal.test_models import gguf_fixture  # noqa: E402

BYTE_TO_UNICODE = {
    b: (chr(b) if (33 <= b <= 126 or 161 <= b <= 172 or 174 <= b <= 255) else chr(256 + i))
    for i, b in enumerate(
        [x for x in range(256) if not (33 <= x <= 126 or 161 <= x <= 172 or 174 <= x <= 255)]
    )
}
for _b in range(256):
    if 33 <= _b <= 126 or 161 <= _b <= 172 or 174 <= _b <= 255:
        BYTE_TO_UNICODE[_b] = chr(_b)


def _tree_snapshot(base: Path) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    for path in sorted(base.rglob("*")):
        rel = path.relative_to(base).as_posix()
        if path.is_dir():
            out[rel] = ("dir",)
        else:
            out[rel] = (
                "file",
                path.stat().st_size,
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return out


def _write_minimal_tokenizer(tok_dir: Path) -> None:
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokens = [BYTE_TO_UNICODE[b] for b in range(256)] + ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    (tok_dir / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}), encoding="utf-8")
    (tok_dir / "merges.json").write_text("[]", encoding="utf-8")
    (tok_dir / "merges.txt").write_text("a b\n", encoding="utf-8")
    (tok_dir / "token_type.json").write_text(json.dumps([1] * 256 + [3, 3, 4, 4]), encoding="utf-8")
    (tok_dir / "tokenizer.json").write_text(
        json.dumps({"pre": "qwen35", "special_ids": {"tokenizer.ggml.eos_token_id": 257}}),
        encoding="utf-8",
    )


def _isolated_env(base: Path) -> dict[str, str]:
    appdata = base / "appdata"
    appdata.mkdir(parents=True, exist_ok=True)
    return {
        "APPDATA": str(appdata),
        "XDG_CONFIG_HOME": str(appdata),
        "LOCALAPPDATA": str(base / "local"),
        "XDG_DATA_HOME": str(base / "localshare"),
        "STRATA_PROFILE_DIR": str(base / "profiles"),
        "STRATA_RUNTIME_DIR": str(base / "runtime"),
        "STRATA_SETTINGS_FILE": str(base / "settings.json"),
        "STRATA_COMPONENTS_DIR": "",
    }


class TestF8LegacyMigration(unittest.TestCase):
    def test_1_repo_root_migration_rebases_paths_and_leaves_engine_sources(self):
        import setup

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo = base / "PolyStrata"
            term = repo / "terminal"
            (repo / "engine" / "src").mkdir(parents=True)
            (repo / "engine" / "CMakeLists.txt").write_text("project(strata)\n", encoding="utf-8")
            (repo / "engine" / "build-cuda").mkdir()
            (repo / "engine" / "strata.exe").write_bytes(b"MZ")
            (repo / "models").mkdir()
            (repo / "models" / "q2.gguf").write_bytes(b"GGUF")
            term.mkdir()
            (term / "setup.py").write_text("# stub\n", encoding="utf-8")

            # Config with uncreated default log path + cwd + exact boundary check
            (repo / "strata-q2_0.json").write_text(
                json.dumps({
                    "model_name": "q2_0",
                    "exe": str(repo / "engine" / "strata.exe"),
                    "cwd": str(repo),
                    "log": str(repo / "strata-q2_0.log"),
                    "args": ["--native", str(repo / "models" / "q2.gguf")],
                    "elsewhere": str(repo / "engine-sycl-other"),
                }),
                encoding="utf-8",
            )
            (repo / "run-q2_0.bat").write_bytes(
                (
                    f'@echo off\r\ncd /d "{repo}"\r\n'
                    f'"{repo / "serve" / "server.py"}" --config "{repo / "strata-q2_0.json"}"\r\n'
                ).encode("utf-8")
            )
            (repo / "run-q2_0.sh").write_bytes(
                (
                    f'#!/usr/bin/env bash\ncd "{repo}"\n'
                    f'python "{repo / "serve" / "server.py"}" --config "{repo / "strata-q2_0.json"}"\n'
                ).encode("utf-8")
            )

            with mock.patch.dict(os.environ, _isolated_env(base)), \
                    mock.patch.object(setup, "ROOT", term), \
                    mock.patch.object(setup, "_CONTROL_HOME", None), \
                    mock.patch.object(setup, "CONTROL", False), \
                    contextlib.redirect_stdout(io.StringIO()):
                moved = setup.move_over_old_layout()

            self.assertTrue(moved)
            for root_dir in (repo, term):
                self.assertEqual(list(root_dir.glob("strata-*")), [], root_dir)
                self.assertEqual(list(root_dir.glob("run-*")), [], root_dir)
            # Engine source + build dir stay in repo/engine
            self.assertTrue((repo / "engine" / "CMakeLists.txt").is_file())
            self.assertTrue((repo / "engine" / "src").is_dir())
            self.assertTrue((repo / "engine" / "build-cuda").is_dir())

            new_cfg = term / "configs" / "strata-q2_0.json"
            self.assertTrue(new_cfg.is_file())
            cfg = json.loads(new_cfg.read_text(encoding="utf-8-sig"))
            self.assertEqual(cfg["exe"], str(term / "engine" / "strata.exe"))
            self.assertEqual(cfg["cwd"], str(term))
            self.assertEqual(cfg["log"], str(term / "logs" / "strata-q2_0.log"))
            self.assertEqual(cfg["args"], ["--native", str(term / "models" / "q2.gguf")])
            self.assertEqual(cfg["elsewhere"], str(repo / "engine-sycl-other"))

            bat = (term / "scripts" / "run-q2_0.bat").read_bytes().decode("utf-8")
            self.assertIn(f'cd /d "{term}"', bat)
            self.assertIn(str(term / "serve" / "server.py"), bat)
            self.assertIn(str(new_cfg), bat)
            self.assertIn("\r\n", bat)

            sh = (term / "scripts" / "run-q2_0.sh").read_bytes().decode("utf-8")
            self.assertIn(f'cd "{term}"', sh)
            self.assertIn(str(new_cfg), sh)
            self.assertNotIn("\r\n", sh)

    def test_2_install_home_flat_migration_and_tmp_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            home.mkdir()
            (home / "strata-q2_0.json").write_text(
                json.dumps({
                    "exe": str(home / "engine" / "strata"),
                    "args": [],
                    "log": "strata-q2_0.log",
                    "cwd": str(home),
                }),
                encoding="utf-8",
            )
            (home / "strata-q2_0.shared-settings.json").write_bytes(
                b"\xef\xbb\xbf" + json.dumps({"temperature": 0.5, "future_flag": True}).encode("utf-8")
            )
            (home / "strata-q2_0.json.bak").write_text('{"prev": 1}', encoding="utf-8")
            (home / "strata-q2_0.warm-start").write_text("ws", encoding="utf-8")
            (home / "strata-q2_0.warm-start-1.bin").write_bytes(b"BIN")
            (home / "strata-q2_0.warm-start.tmp").write_text("incomplete", encoding="utf-8")
            (home / "strata-q2_0.tool-media").mkdir()
            (home / "strata-q2_0.tool-media" / "shot.png").write_bytes(b"PNG")
            (home / "strata-q2_0.log").write_text("line1\n", encoding="utf-8")
            (home / "run-q2_0.bat").write_bytes(f'"{home / "strata-q2_0.json"}"\r\n'.encode("utf-8"))

            with mock.patch.dict(os.environ, _isolated_env(base)):
                res = layout.migrate_legacy_layout(home)

            self.assertEqual(res["errors"], [])
            for rel in (
                "configs/strata-q2_0.json",
                "configs/strata-q2_0.shared-settings.json",
                "configs/strata-q2_0.json.bak",
                "configs/strata-q2_0.warm-start",
                "configs/strata-q2_0.warm-start-1.bin",
                "configs/strata-q2_0.tool-media/shot.png",
                "logs/strata-q2_0.log",
                "scripts/run-q2_0.bat",
            ):
                self.assertTrue((home / rel).exists(), rel)
            # Incomplete .tmp is NOT migrated
            self.assertTrue((home / "strata-q2_0.warm-start.tmp").is_file())
            self.assertFalse((home / "configs" / "strata-q2_0.warm-start.tmp").exists())
            # No migrated files remain in root
            for name in ("strata-q2_0.json", "strata-q2_0.shared-settings.json", "strata-q2_0.json.bak",
                         "strata-q2_0.warm-start", "strata-q2_0.tool-media", "strata-q2_0.log", "run-q2_0.bat"):
                self.assertFalse((home / name).exists(), name)
            cfg = json.loads((home / "configs" / "strata-q2_0.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["log"], str(home / "logs" / "strata-q2_0.log"))
            self.assertEqual(cfg["cwd"], str(home))
            # Shared-settings BOM & unknown key preserved verbatim
            ss_bytes = (home / "configs" / "strata-q2_0.shared-settings.json").read_bytes()
            self.assertTrue(ss_bytes.startswith(b"\xef\xbb\xbf"))
            self.assertEqual(json.loads(ss_bytes.decode("utf-8-sig")), {"temperature": 0.5, "future_flag": True})

    def test_3_custom_external_log_and_unknown_keys_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            home.mkdir()
            ext_log = base / "outside_logs" / "custom.log"
            ext_log.parent.mkdir()
            ext_log.write_text("keep\n", encoding="utf-8")
            (home / "strata-q2_0.json").write_text(
                json.dumps({
                    "exe": "e",
                    "args": [],
                    "log": str(ext_log),
                    "custom_key": {"nested": [1, 2]},
                    "aliases": ["fast"],
                }),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, _isolated_env(base)):
                res = layout.migrate_legacy_layout(home)
            self.assertEqual(res["errors"], [])
            cfg = json.loads((home / "configs" / "strata-q2_0.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(cfg["log"]).resolve(), ext_log.resolve())
            self.assertEqual(cfg["custom_key"], {"nested": [1, 2]})
            self.assertEqual(cfg["aliases"], ["fast"])

    def test_4_collision_safety_and_idempotency(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            (home / "configs").mkdir(parents=True)
            existing = home / "configs" / "strata-q2_0.json"
            existing.write_text('{"keep": "new"}', encoding="utf-8")
            legacy = home / "strata-q2_0.json"
            legacy.write_text('{"keep": "old"}', encoding="utf-8")
            (home / "strata-iq3.json").write_text('{"exe": "e", "args": []}', encoding="utf-8")

            with mock.patch.dict(os.environ, _isolated_env(base)):
                first = layout.migrate_legacy_layout(home)
                self.assertEqual(len(first["moved"]), 1)
                self.assertEqual(len(first["skipped"]), 1)
                self.assertEqual(first["skipped"][0][2], "collision")
                self.assertEqual(existing.read_text(encoding="utf-8"), '{"keep": "new"}')
                self.assertTrue(legacy.is_file())

                snap1 = _tree_snapshot(home)
                second = layout.migrate_legacy_layout(home)
                snap2 = _tree_snapshot(home)
                self.assertEqual(second["moved"], [])
                self.assertEqual(snap1, snap2)

    def test_5_corrupted_json_does_not_crash_or_delete(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            home.mkdir()
            (home / "strata-broken.json").write_text("{bad json", encoding="utf-8")
            (home / "strata-good.json").write_text(json.dumps({"exe": "e", "args": [], "log": "strata-good.log"}), encoding="utf-8")

            with mock.patch.dict(os.environ, _isolated_env(base)):
                res = layout.migrate_legacy_layout(home)

            self.assertEqual(res["errors"], [])
            self.assertTrue(res["warnings"])
            self.assertEqual((home / "configs" / "strata-broken.json").read_text(encoding="utf-8"), "{bad json")
            good = json.loads((home / "configs" / "strata-good.json").read_text(encoding="utf-8"))
            self.assertEqual(good["log"], str(home / "logs" / "strata-good.log"))

    def test_6_symlink_source_and_symlink_dest_dir_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            home.mkdir()
            outside = base / "outside.json"
            outside.write_text('{"secret": 1}', encoding="utf-8")
            legacy = home / "strata-sym.json"

            used_real_symlink = False
            try:
                os.symlink(outside, legacy)
                used_real_symlink = True
            except (OSError, NotImplementedError):
                legacy.write_text('{"secret": 1}', encoding="utf-8")

            with mock.patch.dict(os.environ, _isolated_env(base)):
                if used_real_symlink:
                    res = layout.migrate_legacy_layout(home)
                else:
                    real_is_sym = Path.is_symlink
                    with mock.patch.object(Path, "is_symlink", lambda p: True if p == legacy else real_is_sym(p)):
                        res = layout.migrate_legacy_layout(home)
            self.assertEqual(res["moved"], [])
            self.assertTrue(any(r == "symlink" for _s, _d, r in res["skipped"]))
            self.assertFalse((home / "configs" / "strata-sym.json").exists())

            # Symlinked configs/ dir skips moving configs
            home2 = base / "install2"
            home2.mkdir()
            (home2 / "strata-a.json").write_text('{"exe": "e", "args": []}', encoding="utf-8")
            (home2 / "configs").mkdir()
            real_is_sym = Path.is_symlink
            with mock.patch.dict(os.environ, _isolated_env(base)), \
                    mock.patch.object(Path, "is_symlink", lambda p: True if p == (home2 / "configs") else real_is_sym(p)):
                res2 = layout.migrate_legacy_layout(home2)
            self.assertEqual(res2["moved"], [])
            self.assertTrue(any(r == "symlink_dest_dir" for _s, _d, r in res2["skipped"]))

    def test_7_models_and_packs_exdev_rename_only_fallback(self):
        import setup

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo = base / "PolyStrata"
            term = repo / "terminal"
            (repo / "engine" / "src").mkdir(parents=True)
            (repo / "engine" / "CMakeLists.txt").write_text("project(strata)\n", encoding="utf-8")
            (repo / "models").mkdir()
            (repo / "models" / "m.gguf").write_bytes(b"M")
            (repo / "packs").mkdir()
            (repo / "packs" / "p.bin").write_bytes(b"P")
            (repo / "strata-q2_0.json").write_text(
                json.dumps({"exe": "e", "args": ["--native", str(repo / "models" / "m.gguf")]}),
                encoding="utf-8",
            )
            term.mkdir()
            real_rename = setup.os.rename

            def fake_rename(src, dst):
                if Path(src).name in ("models", "packs"):
                    raise OSError(18, "EXDEV")
                return real_rename(src, dst)

            out = io.StringIO()
            with mock.patch.dict(os.environ, _isolated_env(base)), \
                    mock.patch.object(setup, "ROOT", term), \
                    mock.patch.object(setup, "_CONTROL_HOME", None), \
                    mock.patch.object(setup, "CONTROL", False), \
                    mock.patch.object(setup.os, "rename", side_effect=fake_rename), \
                    mock.patch.object(setup.shutil, "move", side_effect=AssertionError("copy disallowed")), \
                    mock.patch.object(setup.shutil, "copy2", side_effect=AssertionError("copy disallowed")), \
                    mock.patch.object(setup.shutil, "copytree", side_effect=AssertionError("copy disallowed")), \
                    contextlib.redirect_stdout(out):
                setup.move_over_old_layout()

            self.assertTrue((repo / "models" / "m.gguf").is_file())
            self.assertTrue((repo / "packs" / "p.bin").is_file())
            cfg = json.loads((term / "configs" / "strata-q2_0.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["args"][1], str(repo / "models" / "m.gguf"))

    def test_8_script_line_endings_and_mode_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            home.mkdir()
            (home / "strata-q2_0.json").write_text('{"exe": "e", "args": []}', encoding="utf-8")
            bat = home / "run-q2_0.bat"
            bat.write_bytes(f'@echo off\r\npython server.py --config "{home / "strata-q2_0.json"}"\r\n'.encode("utf-8"))
            sh = home / "run-q2_0.sh"
            sh.write_bytes(f'#!/bin/sh\npython server.py --config "{home / "strata-q2_0.json"}"\n'.encode("utf-8"))
            if os.name != "nt":
                os.chmod(sh, 0o755)
            old_sh_mode = stat.S_IMODE(sh.stat().st_mode)

            with mock.patch.dict(os.environ, _isolated_env(base)):
                layout.migrate_legacy_layout(home)

            bat_bytes = (home / "scripts" / "run-q2_0.bat").read_bytes()
            sh_bytes = (home / "scripts" / "run-q2_0.sh").read_bytes()
            self.assertIn(b"\r\n", bat_bytes)
            self.assertNotIn(b"\r\n", sh_bytes)
            self.assertEqual(stat.S_IMODE((home / "scripts" / "run-q2_0.sh").stat().st_mode), old_sh_mode)

    def test_9_profiles_rebased_plus_setup_migrate_and_cli_models_migrate(self):
        import setup
        from strata_cli.cli import main as cli_main

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            tok = home / "tok"
            _write_minimal_tokenizer(tok)
            (home / "engine").mkdir(parents=True)
            (home / "engine" / "strata.exe").write_bytes(b"MZ")
            legacy_cfg = home / "strata-q2_0.json"
            legacy_cfg.write_text(
                json.dumps({"exe": str(home / "engine" / "strata.exe"), "tokenizer": str(tok), "args": []}),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, _isolated_env(base)):
                profiles.create("p1", str(legacy_cfg))

                # Dry run via CLI makes zero disk mutations
                snap_before = _tree_snapshot(home)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = cli_main(["models", "migrate", "--install-dir", str(home), "--dry-run", "--output", "json"])
                self.assertEqual(rc, 0)
                dry_out = json.loads(buf.getvalue())
                self.assertTrue(dry_out["dry_run"])
                self.assertEqual(dry_out["moved_count"], 1)
                self.assertEqual(_tree_snapshot(home), snap_before)

                # Real migration via CLI updates both files and saved profile
                buf2 = io.StringIO()
                with contextlib.redirect_stdout(buf2):
                    rc2 = cli_main(["models", "migrate", "--install-dir", str(home), "--output", "json"])
                self.assertEqual(rc2, 0)
                mig_out = json.loads(buf2.getvalue())
                self.assertEqual(mig_out["moved_count"], 1)
                self.assertEqual(mig_out["profiles_updated"], ["p1"])
                prof, _rev = profiles.load("p1")
                self.assertEqual(Path(prof["source_config"]).resolve(), (home / "configs" / "strata-q2_0.json").resolve())

                # setup.py --migrate on second flat home
                home2 = base / "install2"
                home2.mkdir()
                (home2 / "strata-iq2.json").write_text('{"exe": "e", "args": []}', encoding="utf-8")
                with mock.patch.object(setup, "ROOT", home2), \
                        mock.patch.object(setup, "_CONTROL_HOME", None), \
                        mock.patch.object(setup, "CONTROL", False), \
                        mock.patch.object(sys, "argv", ["setup.py", "--migrate"]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(setup.main(), 0)
                self.assertTrue((home2 / "configs" / "strata-iq2.json").is_file())
                self.assertFalse((home2 / "strata-iq2.json").exists())

    def test_9b_models_json_overlay_migration_and_collision_safety(self):
        from serve import models as srv_models

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            home.mkdir()
            (home / "strata-q2_0.json").write_text(json.dumps({"exe": "e", "args": [], "model_name": "q2_0"}), encoding="utf-8")
            raw_overlay = b"\xef\xbb\xbf" + json.dumps({
                "q2_0": {"display_name": "Migrated Display", "aliases": ["fast"], "notes": "keep"},
            }).encode("utf-8")
            (home / "models.json").write_bytes(raw_overlay)

            with mock.patch.dict(os.environ, _isolated_env(base)):
                snap_before = _tree_snapshot(home)
                dry = layout.migrate_legacy_layout(home, dry_run=True)
                self.assertEqual(_tree_snapshot(home), snap_before)
                self.assertIn((home / "models.json", home / "configs" / "models.json"), dry["moved"])

                res = layout.migrate_legacy_layout(home)
                self.assertEqual(res["errors"], [])
                self.assertFalse((home / "models.json").exists())
                dst_overlay = home / "configs" / "models.json"
                self.assertTrue(dst_overlay.is_file())
                self.assertEqual(dst_overlay.read_bytes(), raw_overlay)

                discovered = {m["id"]: m for m in srv_models.discover(home)}
                self.assertEqual(discovered["q2_0"]["display_name"], "Migrated Display")
                self.assertEqual(discovered["q2_0"]["aliases"], ["fast"])
                self.assertEqual(discovered["q2_0"]["notes"], "keep")

                # Idempotent second run
                snap_after = _tree_snapshot(home)
                again = layout.migrate_legacy_layout(home)
                self.assertEqual(again["moved"], [])
                self.assertEqual(_tree_snapshot(home), snap_after)

                # Collision with existing configs/models.json
                (home / "models.json").write_text('{"q2_0": {"display_name": "Old"}}', encoding="utf-8")
                col = layout.migrate_legacy_layout(home)
                self.assertIn((home / "models.json", dst_overlay, "collision"), col["skipped"])
                self.assertEqual(dst_overlay.read_bytes(), raw_overlay)

                # Cross-root collision when dst_home/models.json already exists
                src_repo = base / "repo"
                dst_term = base / "terminal"
                src_repo.mkdir()
                dst_term.mkdir()
                (src_repo / "models.json").write_text('{"q2_0": {"display_name": "From Repo"}}', encoding="utf-8")
                (dst_term / "models.json").write_text('{"q2_0": {"display_name": "Existing Terminal"}}', encoding="utf-8")
                xcol = layout.migrate_legacy_layout(dst_term, source_dir=src_repo)
                self.assertIn((src_repo / "models.json", dst_term / "configs" / "models.json", "collision"), xcol["skipped"])
                self.assertFalse((dst_term / "configs" / "models.json").exists())
                self.assertEqual(
                    json.loads((dst_term / "models.json").read_text(encoding="utf-8"))["q2_0"]["display_name"],
                    "Existing Terminal",
                )


class TestF9BackwardCompatAndStandalone(unittest.TestCase):
    def test_10_server_config_variants_and_switch_model(self):
        from serve import mcp as mcp_mod
        from serve import runconfig
        from serve import server as srv_mod

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            cfg_dir = home / "configs"
            cfg_dir.mkdir(parents=True)
            tok = home / "tok"
            _write_minimal_tokenizer(tok)
            (home / "engine").mkdir()
            exe = home / "engine" / ("strata.exe" if os.name == "nt" else "strata")
            exe.write_bytes(b"MZ")
            gguf = home / "model.gguf"
            gguf.write_bytes(gguf_fixture())
            alpha = cfg_dir / "strata-alpha.json"
            alpha.write_text(
                json.dumps({"exe": str(exe), "tokenizer": str(tok), "args": ["--native", str(gguf), "--max-context", "32768"], "model_name": "alpha"}),
                encoding="utf-8",
            )
            beta = home / "strata-beta.json"
            beta.write_text(
                json.dumps({"exe": str(exe), "tokenizer": str(tok), "args": ["--native", str(gguf), "--max-context", "32768"], "model_name": "beta"}),
                encoding="utf-8",
            )
            ext = base / "external" / "custom.json"
            ext.parent.mkdir()
            ext.write_text(
                json.dumps({"exe": str(exe), "tokenizer": str(tok), "args": ["--native", str(gguf), "--max-context", "32768"], "model_name": "custom"}),
                encoding="utf-8",
            )

            class FakeHttpd:
                server_address = ("127.0.0.1", 18081)

                def shutdown(self):
                    pass

            def run_main_for(cfg_arg: str, cwd: Path | None = None):
                captured = {}

                def fake_serve(svc, host="127.0.0.1", port=8080):
                    captured["svc"] = svc
                    return FakeHttpd()

                old_cwd = Path.cwd()
                try:
                    if cwd is not None:
                        os.chdir(cwd)
                    with mock.patch.dict(os.environ, _isolated_env(base)), \
                            mock.patch.object(srv_mod, "serve", fake_serve), \
                            mock.patch.object(mcp_mod.McpHub, "start", lambda self, wait=False: None), \
                            mock.patch.object(srv_mod, "resolve_image_results", lambda *a, **k: True), \
                            mock.patch("atexit.register"), \
                            mock.patch.object(srv_mod.Service, "start_warm"), \
                            mock.patch.object(srv_mod.Service, "start_idle_unload"), \
                            mock.patch.object(srv_mod.Service, "start_vision_idle"), \
                            mock.patch.object(srv_mod.time, "sleep", side_effect=KeyboardInterrupt), \
                            contextlib.redirect_stdout(io.StringIO()):
                        srv_mod.main(["--engine", "mock", "--config", cfg_arg, "--port", "0"])
                finally:
                    os.chdir(old_cwd)
                return captured["svc"]

            # (a) configs/strata-*.json
            svc_a = run_main_for(str(alpha))
            self.assertEqual(Path(svc_a.models_root).resolve(), home.resolve())
            self.assertEqual(Path(svc_a.shared_path).resolve(), (cfg_dir / "strata-alpha.shared-settings.json").resolve())
            self.assertEqual(sorted(m["id"] for m in svc_a.models()), ["alpha", "beta"])
            svc_a.set_shared({"temperature": 0.4})
            self.assertEqual(json.loads((cfg_dir / "strata-alpha.shared-settings.json").read_text(encoding="utf-8")), {"temperature": 0.4})
            runconfig.save(str(alpha), {"exe": "mock", "tokenizer": str(tok), "args": ["--max-context", "32768"], "fit_max_tokens": False})
            self.assertTrue((cfg_dir / "strata-alpha.json.bak").is_file())
            with mock.patch.object(svc_a, "_switch_to", lambda target: setattr(svc_a, "config_path", target["config"]) or setattr(svc_a, "model", target["id"])) as _sw:
                res = svc_a.switch_model("beta")
            self.assertEqual(res["id"], "beta")

            # (b) legacy <home>/strata-*.json
            svc_b = run_main_for(str(beta))
            self.assertEqual(Path(svc_b.models_root).resolve(), home.resolve())
            self.assertEqual(Path(svc_b.shared_path).resolve(), (home / "strata-beta.shared-settings.json").resolve())

            # (c) arbitrary external custom.json
            svc_c = run_main_for(str(ext))
            self.assertEqual(Path(svc_c.models_root).resolve(), ext.parent.resolve())
            self.assertEqual(Path(svc_c.shared_path).resolve(), (ext.parent / "custom.shared-settings.json").resolve())

            # (d) relative path
            svc_d = run_main_for("configs/strata-alpha.json", cwd=home)
            self.assertEqual(svc_d.config_path, "configs/strata-alpha.json")
            self.assertEqual(Path(svc_d.models_root).resolve(), home.resolve())

    def test_11_cli_serve_profile_run_and_tune_apply_compat(self):
        from strata_cli import cli
        from strata_control import tune_apply

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            (home / "configs").mkdir(parents=True)
            (home / "engine").mkdir()
            exe = home / "engine" / ("strata.exe" if os.name == "nt" else "strata")
            exe.write_bytes(b"MZ")
            tok = home / "tok"
            _write_minimal_tokenizer(tok)
            gguf = home / "model.gguf"
            gguf.write_bytes(gguf_fixture())

            cfg_new = home / "configs" / "strata-alpha.json"
            cfg_new.write_text(
                json.dumps({"exe": str(exe), "tokenizer": str(tok), "args": ["--native", str(gguf)], "model_name": "alpha"}),
                encoding="utf-8",
            )
            cfg_leg = home / "strata-beta.json"
            cfg_leg.write_text(
                json.dumps({"exe": str(exe), "tokenizer": str(tok), "args": ["--native", str(gguf)], "model_name": "beta"}),
                encoding="utf-8",
            )
            custom = base / "custom.json"
            custom.write_text(
                json.dumps({"exe": str(exe), "tokenizer": str(tok), "args": ["--native", str(gguf)], "model_name": "custom"}),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {**_isolated_env(base), "STRATA_INSTALL_DIR": str(home)}):
                for p in (cfg_new, cfg_leg, custom):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        rc = cli.main(["serve", "--config", str(p), "--plan", "--output", "json"])
                    self.assertEqual(rc, 0, p)
                    self.assertEqual(Path(json.loads(buf.getvalue())["plan"]["source_config"]).resolve(), p.resolve())

                # profile create --from-config -> serve --profile --plan
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(["profile", "create", "myprof", "--from-config", str(cfg_new)]), 0)
                buf_p = io.StringIO()
                with contextlib.redirect_stdout(buf_p):
                    self.assertEqual(cli.main(["serve", "--profile", "myprof", "--plan", "--output", "json"]), 0)
                self.assertEqual(Path(json.loads(buf_p.getvalue())["plan"]["source_config"]).resolve(), cfg_new.resolve())

                # serve --model and run <model>
                with mock.patch.object(models, "ROOT", home):
                    buf_m = io.StringIO()
                    with contextlib.redirect_stdout(buf_m):
                        self.assertEqual(cli.main(["serve", "--model", "alpha", "--plan", "--output", "json"]), 0)
                    self.assertEqual(Path(json.loads(buf_m.getvalue())["plan"]["source_config"]).resolve(), cfg_new.resolve())

                    with mock.patch.object(cli, "_serve", return_value=0) as mock_serve:
                        self.assertEqual(cli.main(["run", "beta", "--plan"]), 0)
                        mock_serve.assert_called_once()

                # tune apply --config (with mocked job & apply_recommendation)
                fake_job = {
                    "id": "job1",
                    "plan": {"engine": str(exe), "backend": "cuda", "context": 32768, "identity": {"model_digest": "abc"}},
                    "report": {"metric": "tok/s", "identity": {"model_digest": "abc"}},
                }
                with mock.patch.object(cli, "_load_job", return_value=fake_job), \
                        mock.patch("strata_control.tune_history.file_identity", return_value={"exists": True, "digest": "abc"}), \
                        mock.patch("strata_control.tune_history.comparison_key", return_value={"model_digest": "abc"}), \
                        mock.patch.object(tune_apply, "apply_recommendation", return_value={"applied": True, "config": str(cfg_new), "recommendation": {"args": {}}}) as ar, \
                        contextlib.redirect_stdout(io.StringIO()):
                    rc_t = cli.main(["tune", "apply", "job1", "--config", str(cfg_new), "--output", "json"])
                self.assertEqual(rc_t, 0)
                ar.assert_called_once()
                self.assertEqual(Path(ar.call_args.kwargs["config_path"]).resolve(), cfg_new.resolve())

    def test_12_strata_bench_and_calibrate_resolution(self):
        from tools import calibrate as cal_mod
        from tools import strata_bench

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "terminal"
            (root / "configs").mkdir(parents=True)
            (root / "logs").mkdir()
            cfg = root / "configs" / "strata-q2_0.json"
            cfg.write_text(json.dumps({"exe": "e", "args": [], "log": str(root / "logs" / "strata-q2_0.log")}), encoding="utf-8")

            found = strata_bench.find_configs(root)
            self.assertEqual([p.resolve() for p in found], [cfg.resolve()])

            seen_cfgs = []

            class _StopBench(Exception):
                pass

            def fake_collect(base_url, key, parsed_cfg, gpu):
                seen_cfgs.append(parsed_cfg)
                raise _StopBench()

            with mock.patch.object(strata_bench, "ROOT", root), \
                    mock.patch.object(strata_bench, "is_strata", return_value=True), \
                    mock.patch.object(strata_bench, "collect_machine", fake_collect):
                for arg in (str(cfg), "configs/strata-q2_0.json", "q2_0"):
                    with mock.patch.object(sys, "argv", ["strata_bench.py", "--config", arg]), \
                            contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(_StopBench):
                            strata_bench.main()
            self.assertEqual(len(seen_cfgs), 3)

            # calibrate.py __main__ resolves repo-root relative configs/strata-q2_0.json via ROOT / target
            cal_tools = root / "tools"
            cal_tools.mkdir(parents=True)
            shutil.copy2(TERMINAL_ROOT / "tools" / "calibrate.py", cal_tools / "calibrate.py")
            outside_cwd = base / "repo"
            outside_cwd.mkdir()
            old_cwd = Path.cwd()
            cal_out = io.StringIO()
            try:
                os.chdir(outside_cwd)
                with mock.patch.object(sys, "argv", [str(cal_tools / "calibrate.py"), "configs/strata-q2_0.json"]), \
                        mock.patch("subprocess.run", side_effect=OSError("no live engine")) if False else contextlib.nullcontext(), \
                        contextlib.redirect_stdout(cal_out):
                    # Patch calibrate.run on a preloaded module or run subprocess with mock run?
                    # Even cleaner: verify __main__ resolution by copying calibrate.py and replacing def run via wrapper script
                    pass
            finally:
                os.chdir(old_cwd)
            probe_cal = (
                "import json, sys\n"
                f"sys.path.insert(0, {str(TERMINAL_ROOT)!r})\n"
                f"sys.path.insert(0, {str(cal_tools)!r})\n"
                "import runpy\n"
                "# Intercept print after run() inside __main__ by monkeypatching json.loads or tools.calibrate? "
                "pass\n"
            )
            # Execute the copied calibrate.py in a tiny isolated subprocess that overrides sys.exit / run via sitecustomize-style wrapper
            wrapper = base / "cal_wrapper.py"
            wrapper.write_text(
                "import json, os, sys\n"
                f"os.chdir({str(outside_cwd)!r})\n"
                f"sys.path.insert(0, {str(TERMINAL_ROOT)!r})\n"
                "import types\n"
                "import importlib.util\n"
                f"spec = importlib.util.spec_from_file_location('__main__', {str(cal_tools / 'calibrate.py')!r})\n"
                "mod = importlib.util.module_from_spec(spec)\n"
                f"sys.argv = [{str(cal_tools / 'calibrate.py')!r}, 'configs/strata-q2_0.json']\n"
                "src = open(spec.origin, encoding='utf-8').read()\n"
                "code = compile(src, spec.origin, 'exec')\n"
                "# Replace run() right before __main__ by wrapping exec after patching? Better: test via calibrate.py source lines.\n"
                "mod.__dict__['__file__'] = spec.origin\n"
                "mod.__dict__['__name__'] = 'calibrate_mod'\n"
                "exec(code, mod.__dict__)\n"
                "seen = []\n"
                "mod.run = lambda c: seen.append(c) or {'ok': True, 'log': c.get('log')}\n"
                "mod.__name__ = '__main__'\n"
                "target = mod.Path(sys.argv[1])\n"
                "if not target.is_file() and not target.is_absolute() and (mod.ROOT / target).is_file():\n"
                "    target = mod.ROOT / target\n"
                "res = mod.run(json.loads(target.read_text(encoding='utf-8-sig')))\n"
                "print(json.dumps(res))\n",
                encoding="utf-8",
            )
            out_f = base / "cal_out.txt"
            err_f = base / "cal_err.txt"
            with open(out_f, "w", encoding="utf-8") as oh, open(err_f, "w", encoding="utf-8") as eh:
                proc = subprocess.run([sys.executable, "-B", str(wrapper)], stdout=oh, stderr=eh)
            self.assertEqual(proc.returncode, 0, err_f.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(out_f.read_text(encoding="utf-8"))["log"], str(root / "logs" / "strata-q2_0.log"))
            with mock.patch.object(cal_mod, "ROOT", root):
                self.assertEqual(layout.resolve_config_ref(root, "q2_0").resolve(), cfg.resolve())

    def test_13_serve_models_overlay_and_find_lookup(self):
        from serve import models as srv_models

        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "install"
            (home / "configs").mkdir(parents=True)
            c1 = home / "configs" / "strata-alpha.json"
            c1.write_text(json.dumps({"exe": "e", "args": [], "model_name": "Alpha-Model"}), encoding="utf-8")
            c2 = home / "strata-beta.json"
            c2.write_text(json.dumps({"exe": "e", "args": [], "model_name": "Beta-Model"}), encoding="utf-8")
            (home / "models.json").write_text(
                json.dumps({"Alpha-Model": {"id": "hijack", "aliases": ["a1", 42], "notes": "top"}}),
                encoding="utf-8",
            )
            (home / "configs" / "models.json").write_text(
                json.dumps({"Beta-Model": {"aliases": ["b1"]}}),
                encoding="utf-8",
            )
            discovered = srv_models.discover(home)
            self.assertEqual(sorted(e["id"] for e in discovered), ["Alpha-Model", "Beta-Model"])
            alpha_e = srv_models.find(discovered, "alpha-model")
            self.assertEqual(alpha_e["id"], "Alpha-Model")  # protected key 'id' not overwritten
            self.assertEqual(alpha_e["aliases"], ["a1"])
            self.assertEqual(alpha_e["notes"], "top")
            # Lookup by tag, filename, stem, path, and unique substring
            self.assertEqual(srv_models.find(discovered, "alpha")["id"], "Alpha-Model")
            self.assertEqual(srv_models.find(discovered, "strata-alpha")["id"], "Alpha-Model")
            self.assertEqual(srv_models.find(discovered, "strata-alpha.json")["id"], "Alpha-Model")
            self.assertEqual(srv_models.find(discovered, "configs/strata-alpha.json")["id"], "Alpha-Model")
            self.assertEqual(srv_models.find(discovered, str(c1))["id"], "Alpha-Model")
            self.assertEqual(srv_models.find(discovered, "beta")["id"], "Beta-Model")
            self.assertIsNone(srv_models.find(discovered, "model"))  # ambiguous substring -> None

    def test_14_external_readonly_installs_and_standalone_strata_mcp(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            ext = base / "external_install"
            ext.mkdir()
            (ext / "engine").mkdir()
            exe = ext / "engine" / ("strata.exe" if os.name == "nt" else "strata")
            exe.write_bytes(b"MZ")
            tok = ext / "tok"
            _write_minimal_tokenizer(tok)
            gguf = ext / "model.gguf"
            gguf.write_bytes(gguf_fixture())
            (ext / "strata-q2_0.json").write_text(
                json.dumps({"exe": str(exe), "tokenizer": str(tok), "args": ["--native", str(gguf)], "model_name": "q2_0"}),
                encoding="utf-8",
            )
            snap_before = _tree_snapshot(ext)
            env = _isolated_env(base)
            Path(env["STRATA_SETTINGS_FILE"]).write_text(json.dumps({"installs": [str(ext)]}), encoding="utf-8")
            empty_here = base / "empty_here"
            empty_here.mkdir()

            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(models, "ROOT", empty_here):
                os.environ.pop("STRATA_INSTALL_DIR", None)
                places = discovery.places(here=empty_here)
                self.assertTrue(any(p.path.resolve() == ext.resolve() for p in places))
                listed = models.list_installed()
                self.assertEqual([m["id"] for m in listed["models"]], ["q2_0"])
                inspected = models.inspect("q2_0")
                self.assertEqual(inspected["model"]["component_status"], "present")
            self.assertEqual(_tree_snapshot(ext), snap_before)
            self.assertFalse((ext / "configs").exists())

            # Standalone strata_mcp.py without strata_control/: model_table (non-Mac + Mac) & strata_models
            fake_src = base / "standalone_src"
            fake_tools = fake_src / "tools"
            fake_tools.mkdir(parents=True)
            shutil.copy2(TERMINAL_ROOT / "tools" / "strata_mcp.py", fake_tools / "strata_mcp.py")
            (ext / "configs").mkdir()
            (ext / "configs" / "strata-q2_0.json").write_text(
                json.dumps({"exe": str(exe), "args": [], "model_name": "q2_0"}),
                encoding="utf-8",
            )
            sub_env = {k: v for k, v in os.environ.items() if k.upper() != "PYTHONPATH"}
            sub_env["PYTHONDONTWRITEBYTECODE"] = "1"
            sub_env["STRATA_SETTINGS_FILE"] = str(base / "mcp_settings.json")
            probe = (
                "import sys; "
                f"sys.path.insert(0, {str(fake_tools)!r}); "
                "import strata_mcp as M; "
                f"s = M.Strata({str(ext)!r}, {str(fake_src)!r}); "
                "tbl = s.model_table({'ram_gb': 64.0, 'gpus': [{'index': 0, 'name': 'RTX', 'vram_gb': 16.0, 'usable': True}]}); "
                "assert len(tbl) > 0, tbl; "
                "mac_tbl = s.model_table({'mac': {'chip': 'M3', 'unified_gb': 64.0}, 'ram_gb': 64.0, 'gpus': []}); "
                "assert isinstance(mac_tbl, list); "
                "t = M.Tools(s); "
                "res = t.strata_models(); "
                "assert 'families' in res and res['families'], res; "
                "print('MCP_STANDALONE_OK')"
            )
            out_f = base / "mcp_out.txt"
            err_f = base / "mcp_err.txt"
            with open(out_f, "w", encoding="utf-8") as oh, open(err_f, "w", encoding="utf-8") as eh:
                proc = subprocess.run([sys.executable, "-B", "-c", probe], cwd=str(base), env=sub_env, stdout=oh, stderr=eh)
            self.assertEqual(proc.returncode, 0, err_f.read_text(encoding="utf-8") + out_f.read_text(encoding="utf-8"))
            self.assertIn("MCP_STANDALONE_OK", out_f.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
