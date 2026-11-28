"""MAP-0215 Q7 (owner 11-16): a git clone made before 0.2.16 kept setup's state at its root (run configs, logs, run scripts, the model
and pack folders, the MCP state, and the engine setup put in engine/). After its pull to the 0.2.16 layout setup runs in terminal/,
and its first run carries that state over, once, with the paths inside the run configs and run scripts changed to the new places.
The engine's source entries in engine/ stay; nothing is replaced; nothing happens in the old layout or in an installed Terminal.

    python -m unittest tools.test_setup_layout_move_over
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class MoveOver(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        env_patch = mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(self.tmp / "profiles")})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        c = self.clone = self.tmp / "PolyStrata"
        (c / "engine" / "src").mkdir(parents=True)
        (c / "engine" / "CMakeLists.txt").write_text("project(strata VERSION 0.1.61)\n", encoding="utf-8")
        (c / "engine" / "build-sycl").mkdir()                       # a build folder: stays
        (c / "engine" / "strata.exe").write_bytes(b"MZ")            # the engine the old setup downloaded
        (c / "engine" / "BUILD.json").write_text("{}", encoding="utf-8")
        (c / "engine" / "lib").mkdir()
        (c / "engine" / "lib" / "cudart.dll").write_bytes(b"dll")
        (c / "models").mkdir()
        (c / "models" / "note.txt").write_text("x", encoding="utf-8")
        (c / ".strata-mcp").mkdir()
        self.exe = c / "engine" / "strata.exe"
        (c / "strata-q2_0.json").write_text(json.dumps({
            "model_name": "Qwen Q2_0", "exe": str(self.exe), "args": ["--model", str(c / "models" / "q2.gguf"), "--mtp", str(c / "mtp" / "rt")],
            "env": {"PATH": str(c / "engine" / "lib")}, "elsewhere": str(c / "engine-sycl-other")}), encoding="utf-8")
        (c / "run-q2_0.bat").write_text(f'"{c / ".venv" / "Scripts" / "python.exe"}" "{c / "strata.py"}" launch\r\n', encoding="utf-8")
        (c / "strata-setup.log").write_text("log", encoding="utf-8")
        (c / "terminal").mkdir()
        (c / "terminal" / "setup.py").write_text("# new\n", encoding="utf-8")
        self.root = c / "terminal"

    def run_it(self, root=None):
        out = io.StringIO()
        with mock.patch.object(setup, "ROOT", root or self.root), \
                mock.patch.object(setup, "_CONTROL_HOME", None), \
                mock.patch.object(setup, "CONTROL", False), \
                contextlib.redirect_stdout(out):
            moved = setup.move_over_old_layout()
        return moved, out.getvalue()

    def test_strata_polys_engine_moves_too(self):
        """poly.10's setup put strata-poly in the clone's engine-poly/: left there, its config named it outside terminal/
        and update_poly_engine() never replaced it (an update from poly.10 kept the 0.1.45 engine)."""
        c = self.clone
        (c / "engine-poly").mkdir()
        (c / "engine-poly" / "strata-poly.exe").write_bytes(b"MZ")
        (c / "engine-poly" / "BUILD.json").write_text('{"source": "release"}', encoding="utf-8")
        (c / "strata-qwen36-ud-q3_k_m.json").write_text(json.dumps({
            "model_name": "q", "exe": str(c / "engine-poly" / "strata-poly.exe"), "args": []}), encoding="utf-8")
        self.run_it()
        t = self.root
        self.assertTrue((t / "engine-poly" / "strata-poly.exe").exists())
        self.assertFalse((c / "engine-poly").exists())
        cfg = json.loads((t / "configs" / "strata-qwen36-ud-q3_k_m.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["exe"], str(t / "engine-poly" / "strata-poly.exe"))
        self.assertEqual(Path(cfg["exe"]).parent, t / setup.ENGINE_POLY_DIR)   # update_poly_engine()'s test for its own engine

    def test_the_old_state_moves_into_terminal_with_its_paths(self):
        moved, out = self.run_it()
        t, c = self.root, self.clone
        for rel in ("configs/strata-q2_0.json", "scripts/run-q2_0.bat", "logs/strata-setup.log", "models/note.txt", ".strata-mcp",
                    "engine/strata.exe", "engine/BUILD.json", "engine/lib/cudart.dll"):
            self.assertTrue((t / rel).exists(), rel)
        for rel in ("strata-q2_0.json", "run-q2_0.bat", "strata-setup.log"):
            self.assertFalse((t / rel).exists(), f"unexpected root file in terminal/: {rel}")
        for rel in ("strata-q2_0.json", "run-q2_0.bat", "strata-setup.log", "models", "engine/strata.exe", "engine/lib"):
            self.assertFalse((c / rel).exists(), rel)
        for rel in ("engine/CMakeLists.txt", "engine/src", "engine/build-sycl"):     # the engine's sources and a build folder stay
            self.assertTrue((c / rel).exists(), rel)
        cfg = json.loads((t / "configs" / "strata-q2_0.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["exe"], str(t / "engine" / "strata.exe"))
        self.assertEqual(cfg["args"][1], str(t / "models" / "q2.gguf"))
        self.assertEqual(cfg["env"]["PATH"], str(t / "engine" / "lib"))
        self.assertEqual(cfg["elsewhere"], str(c / "engine-sycl-other"), "only whole path parts are changed")
        bat = (t / "scripts" / "run-q2_0.bat").read_text(encoding="utf-8")
        self.assertIn(str(t / ".venv" / "Scripts" / "python.exe"), bat)
        self.assertIn(str(t / "strata.py"), bat)
        self.assertIn(str(c), out)
        self.assertEqual(len(moved), 8)

    def test_it_happens_once(self):
        self.run_it()
        (self.clone / "strata-late.json").write_text("{}", encoding="utf-8")
        moved, out = self.run_it()
        self.assertEqual((moved, out), ([], ""))
        self.assertTrue((self.clone / "strata-late.json").exists())

    def test_nothing_is_replaced(self):
        (self.root / "models").mkdir()
        (self.root / "models" / "mine.txt").write_text("keep", encoding="utf-8")
        self.run_it()
        self.assertTrue((self.clone / "models" / "note.txt").exists(), "an existing terminal/models is not replaced")
        self.assertEqual((self.root / "models" / "mine.txt").read_text(encoding="utf-8"), "keep")

    def test_models_and_packs_move_by_rename_only_and_stay_when_it_fails(self):
        """Coordinator 11-16: models/ and packs/ (hundreds of GB) are never copied. A rename that fails (another disk, files in use)
        leaves them where they are, with a warning, and the moved config still names them there."""
        (self.clone / "packs").mkdir()
        (self.clone / "packs" / "p.bin").write_bytes(b"pack")
        real = setup.os.rename

        def rename(src, dst):
            if Path(src).name in ("models", "packs"):
                raise OSError(18, "Invalid cross-device link")
            return real(src, dst)
        with mock.patch.object(setup.os, "rename", side_effect=rename), \
                mock.patch.object(setup.shutil, "move", side_effect=AssertionError("moved by copy")), \
                mock.patch.object(setup.shutil, "copytree", side_effect=AssertionError("copied")), \
                mock.patch.object(setup.shutil, "copy2", side_effect=AssertionError("copied")):
            moved, out = self.run_it()
        for name in ("models", "packs"):
            self.assertTrue((self.clone / name).is_dir(), name)
            self.assertFalse((self.root / name).exists(), name)
            self.assertIn(f"{self.clone / name} could not be moved", out)
        self.assertTrue((self.root / "configs" / "strata-q2_0.json").exists(), "the rest still moves")
        cfg = json.loads((self.root / "configs" / "strata-q2_0.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["args"][1], str(self.clone / "models" / "q2.gguf"), "the config keeps naming the models where they are")
        self.assertEqual(cfg["exe"], str(self.root / "engine" / "strata.exe"))

    def test_no_move_in_the_old_layout_or_an_installed_terminal(self):
        self.assertEqual(self.run_it(self.clone), ([], ""))
        installed = self.tmp / "share" / "strata" / "terminal"
        installed.mkdir(parents=True)
        self.assertEqual(self.run_it(installed), ([], ""))
        self.assertTrue((self.clone / "strata-q2_0.json").exists())

    def test_sidecars_move_to_configs_and_cwd_is_rebased(self):
        c, t = self.clone, self.root
        (c / "strata-q2_0.shared-settings.json").write_text('{"temperature": 0.7}', encoding="utf-8")
        (c / "strata-q2_0.json.bak").write_text('{"old": true}', encoding="utf-8")
        (c / "strata-q2_0.warm-start").write_text("ws", encoding="utf-8")
        (c / "strata-q2_0.tool-media").mkdir()
        (c / "strata-q2_0.tool-media" / "pic.png").write_bytes(b"PNG")
        (c / "strata-q2_0.json").write_text(json.dumps({
            "model_name": "Qwen Q2_0",
            "exe": str(self.exe),
            "cwd": str(c),
            "log": str(c / "strata-q2_0.log"),
            "args": ["--model", str(c / "models" / "q2.gguf")],
            "elsewhere": str(c / "engine-sycl-other"),
        }), encoding="utf-8")
        (c / "run-q2_0.bat").write_text(
            f'@echo off\r\ncd /d "{c}"\r\n"{c / ".venv" / "Scripts" / "python.exe"}" "{c / "serve" / "server.py"}" --config "{c / "strata-q2_0.json"}"\r\n',
            encoding="utf-8",
        )
        self.run_it()
        for rel in (
            "configs/strata-q2_0.json",
            "configs/strata-q2_0.shared-settings.json",
            "configs/strata-q2_0.json.bak",
            "configs/strata-q2_0.warm-start",
            "configs/strata-q2_0.tool-media/pic.png",
            "scripts/run-q2_0.bat",
            "logs/strata-setup.log",
        ):
            self.assertTrue((t / rel).exists(), rel)
        cfg = json.loads((t / "configs" / "strata-q2_0.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["cwd"], str(t))
        self.assertEqual(cfg["log"], str(t / "logs" / "strata-q2_0.log"))
        self.assertEqual(cfg["elsewhere"], str(c / "engine-sycl-other"))
        self.assertEqual(json.loads((t / "configs" / "strata-q2_0.shared-settings.json").read_text(encoding="utf-8")), {"temperature": 0.7})
        bat = (t / "scripts" / "run-q2_0.bat").read_bytes().decode("utf-8")
        self.assertIn(f'cd /d "{t}"', bat)
        self.assertIn(str(t / "configs" / "strata-q2_0.json"), bat)
        self.assertIn("\r\n", bat)

    def test_collision_and_symlink_safety(self):
        (self.root / "configs").mkdir(parents=True)
        existing = self.root / "configs" / "strata-q2_0.json"
        existing.write_text('{"keep": true}', encoding="utf-8")
        moved, _out = self.run_it()
        self.assertEqual(moved, [])
        self.assertEqual(existing.read_text(encoding="utf-8"), '{"keep": true}')
        self.assertFalse((self.root / "strata-q2_0.json").exists())

        # Direct migrate_legacy_layout symlink skip
        from strata_control import layout
        ext = self.tmp / "outside.json"
        ext.write_text('{"outside": 1}', encoding="utf-8")
        home = self.tmp / "sym_home"
        home.mkdir()
        legacy = home / "strata-sym.json"
        legacy.write_text('{"ok": 1}', encoding="utf-8")
        real_is_symlink = Path.is_symlink

        def fake_is_symlink(p):
            return True if p == legacy else real_is_symlink(p)

        with mock.patch.object(Path, "is_symlink", fake_is_symlink):
            res = layout.migrate_legacy_layout(home)
        self.assertEqual(res["moved"], [])
        self.assertTrue(any(reason == "symlink" for _s, _d, reason in res["skipped"]))
        self.assertFalse((home / "configs" / "strata-sym.json").exists())

    def test_models_json_overlay_moves_with_collision_and_symlink_safety(self):
        from serve import models as srv_models
        from strata_control import layout

        overlay = {"Qwen Q2_0": {"display_name": "Custom Root Name", "aliases": ["daily"], "notes": "keep"}}
        (self.clone / "models.json").write_text(json.dumps(overlay), encoding="utf-8")
        self.run_it()

        self.assertFalse((self.clone / "models.json").exists())
        migrated_overlay = self.root / "configs" / "models.json"
        self.assertTrue(migrated_overlay.is_file())
        self.assertEqual(json.loads(migrated_overlay.read_text(encoding="utf-8")), overlay)
        discovered = {m["id"]: m for m in srv_models.discover(self.root)}
        self.assertEqual(discovered["Qwen Q2_0"]["display_name"], "Custom Root Name")
        self.assertIn("daily", discovered["Qwen Q2_0"]["aliases"])
        self.assertEqual(discovered["Qwen Q2_0"]["notes"], "keep")

        # Collision with existing <terminal>/configs/models.json
        c2 = self.tmp / "PolyStrata_c2"
        t2 = c2 / "terminal"
        (c2 / "engine" / "src").mkdir(parents=True)
        (c2 / "engine" / "CMakeLists.txt").write_text("project(polystrata)\n", encoding="utf-8")
        (t2 / "configs").mkdir(parents=True)
        (t2 / "configs" / "models.json").write_text('{"q2_0": {"display_name": "Keep Configs"}}', encoding="utf-8")
        (c2 / "models.json").write_text('{"q2_0": {"display_name": "Old Root"}}', encoding="utf-8")
        self.run_it(root=t2)
        self.assertEqual(json.loads((t2 / "configs" / "models.json").read_text(encoding="utf-8"))["q2_0"]["display_name"], "Keep Configs")
        self.assertTrue((c2 / "models.json").is_file())

        # Cross-root collision with existing <terminal>/models.json
        c3 = self.tmp / "PolyStrata_c3"
        t3 = c3 / "terminal"
        (c3 / "engine" / "src").mkdir(parents=True)
        (c3 / "engine" / "CMakeLists.txt").write_text("project(polystrata)\n", encoding="utf-8")
        t3.mkdir(parents=True)
        (t3 / "models.json").write_text('{"q2_0": {"display_name": "Keep Terminal"}}', encoding="utf-8")
        (c3 / "models.json").write_text('{"q2_0": {"display_name": "Old Root"}}', encoding="utf-8")
        self.run_it(root=t3)
        self.assertEqual(json.loads((t3 / "models.json").read_text(encoding="utf-8"))["q2_0"]["display_name"], "Keep Terminal")
        self.assertTrue((c3 / "models.json").is_file())
        self.assertFalse((t3 / "configs" / "models.json").exists())

        # Symlink safety for models.json
        home = self.tmp / "sym_overlay_home"
        home.mkdir()
        sym_overlay = home / "models.json"
        sym_overlay.write_text('{"q2_0": {"display_name": "Sym"}}', encoding="utf-8")
        real_is_symlink = Path.is_symlink
        with mock.patch.object(Path, "is_symlink", lambda p: True if p == sym_overlay else real_is_symlink(p)):
            res = layout.migrate_legacy_layout(home)
        self.assertEqual(res["moved"], [])
        self.assertTrue(any(s == sym_overlay and reason == "symlink" for s, _d, reason in res["skipped"]))
        self.assertFalse((home / "configs" / "models.json").exists())


class MainMovesFirst(unittest.TestCase):
    def test_main_moves_the_old_state_before_it_reads_any_config(self):
        order = []
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(setup, "ROOT", Path(td)), \
                mock.patch.object(setup, "_CONTROL_HOME", None), \
                mock.patch.object(setup, "CONTROL", False), \
                mock.patch.object(sys, "argv", ["setup.py", "--update"]), \
                mock.patch.object(setup, "move_over_old_layout", side_effect=lambda: order.append("move")), \
                mock.patch.object(setup, "update_pull", side_effect=lambda argv: order.append("pull") or 0):
            self.assertEqual(setup.main(), 0)
        self.assertEqual(order, ["move", "pull"])


if __name__ == "__main__":
    unittest.main()
