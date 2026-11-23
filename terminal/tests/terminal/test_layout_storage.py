"""Unit tests for M1 F1 (strata_control.layout) and F2 (unified discovery & read-only legacy fallback).

Runs strictly in isolated temporary directories with zero external dependencies (unittest + stdlib).
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import models as serve_models
from strata_cli import ram_budget_move
from strata_cli.launch.catalog import Catalog
from strata_control import cfg_paths, discovery, layout, models, models_move, profiles, tune_apply
from strata_control.profiles import ProfileError


def _tree_snapshot(root: Path) -> tuple[set[str], dict[str, tuple[bytes, int]]]:
    dirs: set[str] = set()
    files: dict[str, tuple[bytes, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        rel_d = d.relative_to(root).as_posix()
        if rel_d != ".":
            dirs.add(rel_d)
        for name in filenames:
            p = d / name
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                files[rel] = (b"<symlink>", p.lstat().st_mtime_ns)
            else:
                files[rel] = (p.read_bytes(), p.stat().st_mtime_ns)
    return dirs, files


def _write_valid_run_config(
    path: Path,
    *,
    exe: Path,
    tok_dir: Path,
    model_name: str = "demo",
    args: list[str] | None = None,
    bom: bool = False,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exe": str(exe),
        "tokenizer": str(tok_dir),
        "model_name": model_name,
        "args": args if args is not None else ["--max-context", "8192"],
        **(extra or {}),
    }
    raw = json.dumps(payload, indent=1).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def _make_runtime_fixtures(base: Path) -> tuple[Path, Path]:
    exe = base / "engine.exe"
    exe.write_bytes(b"exe")
    tok = base / "tokenizer"
    tok.mkdir(parents=True, exist_ok=True)
    for name in ("vocab.json", "merges.txt", "token_type.json"):
        (tok / name).write_text("{}", encoding="utf-8")
    return exe, tok


class TestLayoutModule(unittest.TestCase):
    def test_pure_path_helpers_do_not_create_directories(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "install_home"
            home.mkdir()
            self.assertEqual(layout.configs_dir(home), home / "configs")
            self.assertEqual(layout.logs_dir(home), home / "logs")
            self.assertEqual(layout.scripts_dir(home), home / "scripts")
            cfg = layout.config_path(home, "Qwen3.6")
            self.assertEqual(cfg, home / "configs" / "strata-qwen3.6.json")
            self.assertEqual(
                layout.shared_settings_path(cfg),
                home / "configs" / "strata-qwen3.6.shared-settings.json",
            )
            self.assertEqual(layout.backup_path(cfg), home / "configs" / "strata-qwen3.6.json.bak")
            self.assertEqual(layout.log_path(home, "strata-Qwen3.6"), home / "logs" / "strata-qwen3.6.log")
            self.assertEqual(
                layout.script_path(home, "Qwen3.6", windows=True),
                home / "scripts" / "run-qwen3.6.bat",
            )
            self.assertEqual(
                layout.script_path(home, "Qwen3.6", windows=False),
                home / "scripts" / "run-qwen3.6.sh",
            )
            self.assertEqual(layout.install_home_for_config(cfg), home.resolve())
            self.assertEqual(layout.install_home_for_config(home / "strata-qwen3.6.json"), home.resolve())
            self.assertFalse((home / "configs").exists())
            self.assertFalse((home / "logs").exists())
            self.assertFalse((home / "scripts").exists())

    def test_normalize_tag_validation(self):
        self.assertEqual(layout.normalize_tag("  Strata-IQ3_S "), "iq3_s")
        self.assertEqual(layout.normalize_tag("тест-модель"), "тест-модель")
        self.assertEqual(layout.normalize_tag("STRATA-Тест"), "тест")
        for bad in (
            "",
            "   ",
            "strata-",
            "a/b",
            "a\\b",
            ".",
            "..",
            "..foo",
            "a..b",
            "foo.",
            "foo:bar",
            "foo\x00bar",
            "foo\nbar",
            "foo\x7fbar",
            "foo.shared-settings",
            "strata-foo.shared-settings",
            "strata-foo.json",
            "foo.json",
            None,
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                layout.normalize_tag(bad)  # type: ignore[arg-type]

    def test_iter_config_files_ignores_directories_and_hardens_ref_and_home_helpers(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            (home / "configs" / "strata-empty.json").mkdir(parents=True)
            self.assertEqual(layout.iter_config_files(home), [])
            self.assertFalse(discovery.has_configs(home))
            self.assertIsNone(layout.find_config(home, "empty"))
            self.assertIsNone(layout.resolve_config_ref(home, home / "configs" / "strata-empty.json"))

            # A directory in configs/ must not shadow a real legacy file of the same name
            legacy = home / "strata-empty.json"
            legacy.write_text("{}", encoding="utf-8")
            self.assertEqual(layout.iter_config_files(home), [legacy])

            # resolve_config_ref rejects *.shared-settings.json even by explicit path
            shared = home / "configs" / "strata-demo.shared-settings.json"
            shared.write_text("{}", encoding="utf-8")
            self.assertIsNone(layout.resolve_config_ref(home, shared))

            # install_home_for_config handles mixed-case Configs directory
            mixed_cfg = home / "Configs" / "strata-m1.json"
            mixed_cfg.parent.mkdir(exist_ok=True)
            mixed_cfg.write_text("{}", encoding="utf-8")
            self.assertEqual(layout.install_home_for_config(mixed_cfg), home.resolve())

    def test_ensure_layout_dirs_creates_all_three_subdirs(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            c, l, s = layout.ensure_layout_dirs(home)
            self.assertTrue(c.is_dir())
            self.assertTrue(l.is_dir())
            self.assertTrue(s.is_dir())

    def test_iter_config_files_precedence_and_exclusions(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            (home / "configs").mkdir()
            new_cfg = home / "configs" / "strata-q2_0.json"
            old_cfg = home / "strata-q2_0.json"
            legacy_only = home / "strata-iq3_s.json"
            shared_new = home / "configs" / "strata-q2_0.shared-settings.json"
            shared_old = home / "strata-q2_0.shared-settings.json"
            for p in (new_cfg, old_cfg, legacy_only, shared_new, shared_old):
                p.write_text("{}", encoding="utf-8")
            (home / "strata-.json").write_text("{}", encoding="utf-8")
            (home / "other.json").write_text("{}", encoding="utf-8")
            # Even if legacy has newer mtime, configs/ wins
            os.utime(new_cfg, (1000, 1000))
            os.utime(old_cfg, (9000, 9000))

            found = layout.iter_config_files(home)
            self.assertEqual(found, [new_cfg, legacy_only])
            self.assertEqual(layout.iter_config_files(home, include_legacy=False), [new_cfg])
            self.assertEqual(layout.find_config(home, "q2_0"), new_cfg)
            self.assertEqual(layout.find_config(home, "STRATA-Q2_0.JSON"), new_cfg)
            self.assertEqual(layout.find_config(home, "iq3_s"), legacy_only)
            self.assertIsNone(layout.find_config(home, "iq3_s", include_legacy=False))
            self.assertEqual(layout.resolve_config_ref(home, old_cfg), old_cfg)
            self.assertEqual(layout.resolve_config_ref(home, "q2_0"), new_cfg)
            self.assertIsNone(layout.resolve_config_ref(home, ""))

    def test_atomic_crud_helpers_with_bom_backup_and_delete(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            p = layout.write_model_config(home, "demo", {"exe": "e", "args": ["--a"]}, backup=True)
            self.assertEqual(p, home / "configs" / "strata-demo.json")
            self.assertFalse(layout.backup_path(p).exists())

            p2 = layout.write_model_config(home, "demo", {"exe": "e2", "args": ["--b"]}, backup=True)
            self.assertEqual(p, p2)
            self.assertEqual(json.loads(layout.backup_path(p).read_text(encoding="utf-8"))["exe"], "e")
            self.assertEqual(layout.read_json(p)["exe"], "e2")

            # BOM tolerance in read_json
            bom_file = home / "bom.json"
            bom_file.write_bytes(b"\xef\xbb\xbf" + b'{"ok": true}')
            self.assertEqual(layout.read_json(bom_file), {"ok": True})
            self.assertEqual(layout.read_json(home / "missing.json", default={}), {})

            # Sidecars + legacy deletion
            layout.shared_settings_path(p).write_text("{}", encoding="utf-8")
            legacy = home / "strata-demo.json"
            legacy.write_text("{}", encoding="utf-8")
            removed = layout.delete_model_config(home, "demo", include_legacy=True, remove_sidecars=True)
            self.assertEqual(
                set(removed),
                {p, layout.backup_path(p), layout.shared_settings_path(p), legacy},
            )
            self.assertFalse(p.exists())
            self.assertFalse(legacy.exists())


class TestDiscoveryAndModels(unittest.TestCase):
    def test_discovery_and_models_support_configs_dir_and_are_strictly_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            home.mkdir()
            exe, tok = _make_runtime_fixtures(base)
            new_cfg = layout.config_path(home, "q2_0")
            _write_valid_run_config(new_cfg, exe=exe, tok_dir=tok, model_name="q2_0", bom=True)
            layout.shared_settings_path(new_cfg).write_text("{}", encoding="utf-8")
            empty_dir = base / "empty_install"
            empty_dir.mkdir()

            before_home = _tree_snapshot(home)
            before_empty = _tree_snapshot(empty_dir)

            with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(home)}):
                pl = discovery.places()
                self.assertEqual(len(pl), 1)
                self.assertEqual(pl[0].path, home)
                self.assertEqual(pl[0].configs, 1)
                self.assertEqual(discovery.find_install()[0], home)

                listed = models.list_installed()
                self.assertEqual(listed["model_count"], 1)
                self.assertEqual(listed["models"][0]["id"], "q2_0")
                self.assertEqual(Path(listed["models"][0]["source_config"]), new_cfg)
                self.assertEqual(listed["models"][0]["component_status"], "present")

                inspected = models.inspect("q2_0")
                self.assertEqual(inspected["model"]["id"], "q2_0")
                self.assertTrue(inspected["components"]["ok"])

            with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(empty_dir)}):
                self.assertEqual(discovery.config_files(empty_dir), [])
                self.assertFalse(discovery.has_configs(empty_dir))
                self.assertEqual(models.list_installed()["model_count"], 0)

            self.assertEqual(_tree_snapshot(home), before_home)
            self.assertEqual(_tree_snapshot(empty_dir), before_empty)
            self.assertFalse((empty_dir / "configs").exists())
            self.assertFalse((home / ram_budget_move.MARK).exists())
            self.assertFalse((home / "configs" / ram_budget_move.MARK).exists())

    def test_duplicate_shadowing_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            exe, tok = _make_runtime_fixtures(base)
            new_cfg = layout.config_path(home, "dup")
            old_cfg = home / "strata-dup.json"
            legacy_cfg = home / "strata-legacy_only.json"
            _write_valid_run_config(new_cfg, exe=exe, tok_dir=tok, model_name="dup-new")
            _write_valid_run_config(old_cfg, exe=exe, tok_dir=tok, model_name="dup-old")
            _write_valid_run_config(legacy_cfg, exe=exe, tok_dir=tok, model_name="legacy_only")
            os.utime(new_cfg, (1000, 1000))
            os.utime(old_cfg, (9000, 9000))

            with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(home)}):
                pl = discovery.places()
                self.assertEqual(pl[0].configs, 2)
                listed = models.list_installed()
                by_id = {m["id"]: m for m in listed["models"]}
                self.assertEqual(set(by_id), {"dup", "legacy_only"})
                self.assertEqual(Path(by_id["dup"]["source_config"]), new_cfg)
                self.assertEqual(by_id["dup"]["name"], "dup-new")

    def test_max_configs_cap_across_combined_deduplicated_entries(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            (home / "configs").mkdir()
            for i in range(65):
                (home / "configs" / f"strata-m{i:03d}.json").write_text("{}", encoding="utf-8")
            # Overlapping names do NOT push over 128
            for i in range(65):
                (home / f"strata-m{i:03d}.json").write_text("{}", encoding="utf-8")
            self.assertEqual(len(models._paths(home)), 65)  # noqa: SLF001

            # Unique names across configs/ + legacy exceeding 128 raise ProfileError
            for i in range(65, 129):
                (home / f"strata-n{i:03d}.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ProfileError):
                models._paths(home)  # noqa: SLF001

    def test_oversized_and_symlink_configs_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            exe, tok = _make_runtime_fixtures(base)
            big = layout.config_path(home, "big")
            big.parent.mkdir(parents=True)
            big.write_bytes(b" " * (models.MAX_CONFIG_BYTES + 10))
            good = layout.config_path(home, "good")
            _write_valid_run_config(good, exe=exe, tok_dir=tok, model_name="good")

            with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(home)}):
                listed = models.list_installed()
                by_id = {m["id"]: m for m in listed["models"]}
                self.assertEqual(by_id["big"]["component_status"], "invalid")
                with self.assertRaises(ProfileError):
                    models.inspect("big")

                # Symlink rejection (real symlink when permitted, else mocked lstat)
                link = layout.config_path(home, "link")
                created_real_link = False
                try:
                    link.symlink_to(good)
                    created_real_link = True
                except (OSError, NotImplementedError):
                    pass

                if created_real_link:
                    listed2 = models.list_installed()
                    by_id2 = {m["id"]: m for m in listed2["models"]}
                    self.assertEqual(by_id2["link"]["component_status"], "invalid")
                    with self.assertRaises(ProfileError):
                        models.inspect("link")
                else:
                    fake_lstat = mock.Mock(st_mode=stat.S_IFLNK | 0o777)
                    with mock.patch.object(Path, "lstat", return_value=fake_lstat):
                        with self.assertRaises(ProfileError):
                            models._read(good)  # noqa: SLF001

    def test_models_paths_mixed_case_shadowing_and_broken_symlink_configs_dir(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            exe, tok = _make_runtime_fixtures(base)

            # Case 1: Invalid-ID mixed-case configs/strata-Foo.json must NOT shadow valid legacy <home>/strata-foo.json,
            # while valid configs/strata-bar.json shadows legacy <home>/strata-bar.json
            home_case = base / "install_case"
            (home_case / "configs").mkdir(parents=True)
            _write_valid_run_config(
                home_case / "configs" / "strata-Foo.json", exe=exe, tok_dir=tok, model_name="from-configs",
            )
            _write_valid_run_config(
                home_case / "strata-foo.json", exe=exe, tok_dir=tok, model_name="from-legacy",
            )
            _write_valid_run_config(
                home_case / "configs" / "strata-bar.json", exe=exe, tok_dir=tok, model_name="bar",
            )
            _write_valid_run_config(
                home_case / "strata-bar.json", exe=exe, tok_dir=tok, model_name="legacy-bar",
            )
            # Plain directory named strata-dir.json in configs/ must not shadow legacy strata-dir.json
            (home_case / "configs" / "strata-dir.json").mkdir()
            legacy_dir = home_case / "strata-dir.json"
            _write_valid_run_config(legacy_dir, exe=exe, tok_dir=tok, model_name="legacy-dir")

            with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(home_case)}):
                paths_ids = [mid for mid, _ in models._paths(home_case)]  # noqa: SLF001
                self.assertIn("foo", paths_ids)
                self.assertIn("bar", paths_ids)
                self.assertIn("dir", paths_ids)
                listed = models.list_installed()
                listed_ids = {m["id"]: m for m in listed["models"]}
                self.assertEqual(set(listed_ids), {"bar", "dir", "foo"})
                self.assertEqual(listed_ids["bar"]["name"], "bar")
                self.assertEqual(listed_ids["foo"]["name"], "from-legacy")

            # Case 2: Broken symlink configs/ does not crash _paths() or hide legacy configs
            home_sym = base / "install_sym"
            home_sym.mkdir()
            legacy_ok = home_sym / "strata-legacy.json"
            _write_valid_run_config(legacy_ok, exe=exe, tok_dir=tok, model_name="legacy")
            cfg_dir = layout.configs_dir(home_sym)
            try:
                cfg_dir.symlink_to(base / "does-not-exist", target_is_directory=True)
                created_broken = True
            except (OSError, NotImplementedError):
                created_broken = False

            with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(home_sym)}):
                if created_broken:
                    listed_sym = models.list_installed()
                    self.assertEqual([m["id"] for m in listed_sym["models"]], ["legacy"])
                else:
                    orig_is_dir = Path.is_dir
                    orig_is_symlink = Path.is_symlink

                    def fake_is_dir(p: Path) -> bool:
                        if p == cfg_dir:
                            return False
                        return orig_is_dir(p)

                    def fake_is_symlink(p: Path) -> bool:
                        if p == cfg_dir:
                            return True
                        return orig_is_symlink(p)

                    with mock.patch.object(Path, "is_dir", fake_is_dir), \
                            mock.patch.object(Path, "is_symlink", fake_is_symlink):
                        listed_sym = models.list_installed()
                        self.assertEqual([m["id"] for m in listed_sym["models"]], ["legacy"])


class TestServeModels(unittest.TestCase):
    def setUp(self):
        serve_models._seen.clear()  # noqa: SLF001

    def test_discover_configs_dir_shadowing_mtime_and_overlay(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            exe, tok = _make_runtime_fixtures(base)

            new_cfg = layout.config_path(home, "gemma")
            old_cfg = home / "strata-gemma.json"
            _write_valid_run_config(new_cfg, exe=exe, tok_dir=tok, model_name="gemma", extra={"display_name": "From Configs"})
            _write_valid_run_config(old_cfg, exe=exe, tok_dir=tok, model_name="gemma", extra={"display_name": "From Legacy"})
            os.utime(new_cfg, (1000, 1000))
            os.utime(old_cfg, (9000, 9000))

            # Separate filenames sharing model_name -> newer mtime wins
            alt_a = layout.config_path(home, "alt-a")
            alt_b = home / "strata-alt-b.json"
            _write_valid_run_config(alt_a, exe=exe, tok_dir=tok, model_name="shared-id", extra={"display_name": "Older Alt"})
            _write_valid_run_config(alt_b, exe=exe, tok_dir=tok, model_name="shared-id", extra={"display_name": "Newer Alt"})
            os.utime(alt_a, (1000, 1000))
            os.utime(alt_b, (9000, 9000))

            # Shared settings and oversized files ignored; BOM accepted
            layout.shared_settings_path(new_cfg).write_text('{"exe":"x","args":[]}', encoding="utf-8")
            (home / "configs" / "strata-bom.json").write_bytes(
                b"\xef\xbb\xbf" + json.dumps({"exe": str(exe), "tokenizer": str(tok), "args": [], "model_name": "bom"}).encode("utf-8")
            )
            (home / "configs" / "strata-huge.json").write_bytes(b"{" + b" " * (serve_models.MAX_CONFIG_BYTES + 10) + b"}")

            # models.json overlay: root + configs/models.json, protected keys untouched
            (home / "models.json").write_text(
                json.dumps({"gemma": {"notes": "root note", "display_name": "Root Overlay", "status": "hacked"}}),
                encoding="utf-8",
            )
            (home / "configs" / "models.json").write_text(
                json.dumps({"gemma": {"display_name": "Configs Overlay", "id": "evil"}}),
                encoding="utf-8",
            )

            before = _tree_snapshot(home)
            found = {e["id"]: e for e in serve_models.discover(home)}
            self.assertEqual(set(found), {"bom", "gemma", "shared-id"})
            self.assertEqual(Path(found["gemma"]["config"]), new_cfg)
            self.assertEqual(found["gemma"]["display_name"], "Configs Overlay")
            self.assertEqual(found["gemma"]["notes"], "root note")
            self.assertEqual(found["gemma"]["status"], "available")
            self.assertEqual(found["shared-id"]["display_name"], "Newer Alt")
            self.assertEqual(_tree_snapshot(home), before)

            # Pre-M2 server.py compatibility: discover(<home>/configs) also reads <home>/models.json
            found_sub = {e["id"]: e for e in serve_models.discover(home / "configs")}
            self.assertEqual(found_sub["gemma"]["notes"], "root note")
            self.assertEqual(found_sub["gemma"]["display_name"], "Configs Overlay")

    def test_describe_rejects_symlink_and_broken_symlink_does_not_crash_discover(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            exe, tok = _make_runtime_fixtures(base)
            good = layout.config_path(home, "good")
            _write_valid_run_config(good, exe=exe, tok_dir=tok, model_name="good")

            broken = layout.config_path(home, "broken")
            try:
                broken.symlink_to(home / "does-not-exist.json")
                created = True
            except (OSError, NotImplementedError):
                created = False

            if created:
                self.assertIsNone(serve_models.describe(broken))
                found = serve_models.discover(home)
                self.assertEqual([e["id"] for e in found], ["good"])
            else:
                fake_lstat = mock.Mock(st_mode=stat.S_IFLNK | 0o777, st_mtime=1.0)
                with mock.patch.object(Path, "lstat", return_value=fake_lstat):
                    self.assertIsNone(serve_models.describe(good))
                    self.assertEqual(serve_models.discover(home), [])

    def test_describe_and_discover_skip_non_string_fields_without_crashing(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            exe, tok = _make_runtime_fixtures(base)
            good = layout.config_path(home, "good")
            _write_valid_run_config(good, exe=exe, tok_dir=tok, model_name="good")

            bad_payloads = {
                "badexe-list": {"exe": ["bad"], "tokenizer": str(tok), "args": []},
                "badexe-int": {"exe": 123, "tokenizer": str(tok), "args": []},
                "badexe-empty": {"exe": "", "tokenizer": str(tok), "args": []},
                "badtok": {"exe": str(exe), "tokenizer": 123, "args": []},
                "badcwd": {"exe": str(exe), "tokenizer": str(tok), "cwd": 123, "args": []},
                "badcwd-bool": {"exe": str(exe), "tokenizer": str(tok), "cwd": False, "args": []},
                "badargs-int": {"exe": str(exe), "tokenizer": str(tok), "args": [123]},
                "badargs-dict": {"exe": str(exe), "tokenizer": str(tok), "args": [{"k": "v"}]},
            }
            for tag, payload in bad_payloads.items():
                p = layout.config_path(home, tag)
                p.write_text(json.dumps(payload), encoding="utf-8")
                self.assertIsNone(serve_models.describe(p), tag)

            found = serve_models.discover(home)
            self.assertEqual([e["id"] for e in found], ["good"])

    def test_overlay_normalizes_string_aliases_and_rejects_oversized_models_json(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            home = base / "install"
            exe, tok = _make_runtime_fixtures(base)
            cfg = layout.config_path(home, "m")
            _write_valid_run_config(cfg, exe=exe, tok_dir=tok, model_name="m")

            overlay = home / "configs" / "models.json"
            overlay.write_text(json.dumps({"m": {"aliases": "myalias, second-alias"}}), encoding="utf-8")
            found = serve_models.discover(home)
            self.assertEqual(found[0]["aliases"], ["myalias", "second-alias"])
            self.assertIs(serve_models.resolve(found, "myalias"), found[0])
            self.assertIs(serve_models.resolve(found, "SECOND-ALIAS"), found[0])
            self.assertIsNone(serve_models.resolve(found, "m_invalid_part"))
            self.assertIsNone(serve_models.resolve(found, "y"))

            # Oversized (> 1 MiB) root models.json is ignored cleanly
            big_overlay = home / "models.json"
            big_overlay.write_bytes(b'{"m": {"notes": "' + (b"x" * (serve_models.MAX_CONFIG_BYTES + 100)) + b'"}}')
            found2 = serve_models.discover(home)
            self.assertEqual(len(found2), 1)
            self.assertNotIn("notes", found2[0])

    def test_describe_non_ascii_context_mixed_case_configs_overlay_and_line_vram_guard(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            exe, tok = _make_runtime_fixtures(base)

            # 1. Non-ASCII context values (e.g. superscript '²' or Arabic digits) return None without ValueError
            home_ctx = base / "install_ctx"
            sup_cfg = layout.config_path(home_ctx, "sup")
            _write_valid_run_config(sup_cfg, exe=exe, tok_dir=tok, model_name="sup", args=["--max-context", "²"])
            ara_cfg = layout.config_path(home_ctx, "ara")
            _write_valid_run_config(ara_cfg, exe=exe, tok_dir=tok, model_name="ara", args=["--max-context", "١٢٣"])
            self.assertIsNone(serve_models.describe(sup_cfg)["context"])
            self.assertIsNone(serve_models.describe(ara_cfg)["context"])

            # 2. Mixed-case Configs/ dir passed to discover() reads parent <home>/models.json
            home_case = base / "install_case"
            configs_upper = home_case / "Configs"
            _write_valid_run_config(configs_upper / "strata-m.json", exe=exe, tok_dir=tok, model_name="m")
            (home_case / "models.json").write_text(json.dumps({"m": {"notes": "from-root"}}), encoding="utf-8")
            found_upper = serve_models.discover(configs_upper)
            self.assertEqual(len(found_upper), 1)
            self.assertEqual(found_upper[0].get("notes"), "from-root")

            # 3. line() handles non-dict / non-numeric / bool vram_gb overlays without TypeError, preserves valid format
            for bad_vram in ("bad", ["1", "2"], {"least": True, "all": 10}, {"least": "x", "all": 10}):
                (home_case / "models.json").write_text(json.dumps({"m": {"vram_gb": bad_vram}}), encoding="utf-8")
                entry = serve_models.discover(home_case)[0]
                rendered = serve_models.line(entry)
                self.assertIn("m", rendered)
                self.assertNotIn("VRAM", rendered)

            (home_case / "models.json").write_text(
                json.dumps({"m": {"vram_gb": {"least": 3.1, "all": 43.3}, "ram_gb": 42.3}}),
                encoding="utf-8",
            )
            valid_entry = serve_models.discover(home_case)[0]
            self.assertIn("VRAM 3.1-43.3 GB, file 42.3 GB", serve_models.line(valid_entry))


class TestDownstreamF2Modules(unittest.TestCase):
    def test_models_move_discovers_and_repoints_configs_subdir(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            old_data = base / "data_old"
            new_data = base / "data_new"
            (old_data / "models" / "demo").mkdir(parents=True)
            (old_data / "models" / "demo" / "m.gguf").write_bytes(b"gguf")

            install = base / "install"
            cfg = layout.config_path(install, "demo")
            cfg.parent.mkdir(parents=True)
            cfg.write_text(
                json.dumps({"exe": "e", "tokenizer": "t", "args": ["--model", str(old_data / "models" / "demo" / "m.gguf")]}),
                encoding="utf-8",
            )
            legacy_dup = install / "strata-demo.json"
            legacy_dup.write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")
            layout.shared_settings_path(cfg).write_text("{}", encoding="utf-8")

            cfgs = models_move.config_files(install)
            self.assertEqual(cfgs, [cfg])
            summary = models_move.move(old_data, new_data, configs=cfgs)
            self.assertTrue(summary["done"])
            updated = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertIn(str(new_data), updated["args"][1])

    def test_tune_apply_find_config_files_precedence_exclusions_and_bom(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            new_cfg = layout.config_path(root, "q2")
            old_cfg = root / "strata-q2.json"
            _write_valid_run_config(new_cfg, exe=root / "e", tok_dir=root / "t", model_name="q2", bom=True)
            _write_valid_run_config(old_cfg, exe=root / "e", tok_dir=root / "t", model_name="q2")
            layout.shared_settings_path(new_cfg).write_text('{"args": []}', encoding="utf-8")

            for skip in (".git", ".venv", ".rocm-win", "node_modules", "__pycache__"):
                d = root / skip
                d.mkdir()
                _write_valid_run_config(d / "strata-skip.json", exe=root / "e", tok_dir=root / "t")

            found = tune_apply.find_config_files(root)
            self.assertEqual(found, [new_cfg])

    def test_tune_apply_directory_in_configs_does_not_shadow_legacy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "configs" / "strata-empty.json").mkdir(parents=True)
            legacy = root / "strata-empty.json"
            _write_valid_run_config(legacy, exe=root / "e", tok_dir=root / "t", model_name="empty", args=[])
            self.assertEqual(tune_apply.find_config_files(root), [legacy])

    def test_ram_budget_move_migrates_configs_subdir_and_marks_inside_configs(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            cfg = layout.config_path(home, "unsloth-ud-q4_k_xl")
            cfg.parent.mkdir(parents=True)
            args = ["--native", "/models/Qwen.gguf", "--max-context", "8192", "--resident-budget-gib", "40"]
            cfg.write_text(
                json.dumps({"exe": "e", "args": args, "args_written": list(args), "model_name": "q4"}),
                encoding="utf-8",
            )
            layout.shared_settings_path(cfg).write_text('{"temperature": 0.7}', encoding="utf-8")

            import setup as setup_mod

            with mock.patch.object(setup_mod, "ram_gb", lambda: 63.7), \
                    mock.patch.object(setup_mod, "_unified_memory_pc", lambda: False):
                done = ram_budget_move.run([home], setup_module=setup_mod, say=lambda _t: None)
                self.assertEqual(done, {str(home): ["strata-unsloth-ud-q4_k_xl.json"]})
                self.assertFalse((home / ram_budget_move.MARK).exists())
                self.assertTrue((home / "configs" / ram_budget_move.MARK).is_file())
                self.assertTrue(layout.backup_path(cfg).is_file())
                updated = json.loads(cfg.read_text(encoding="utf-8"))
                self.assertEqual(updated[ram_budget_move.WORK_KEY], 40)
                # Second run is a no-op
                self.assertEqual(ram_budget_move.run([home], setup_module=setup_mod, say=lambda _t: None), {})

    def test_ram_budget_move_symlinked_configs_does_not_write_mark(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            target_dir = base / "real_configs"
            target_dir.mkdir()
            args = ["--native", "/models/Qwen.gguf", "--max-context", "8192", "--resident-budget-gib", "40"]
            cfg_in_sym = target_dir / "strata-unsloth-ud-q4_k_xl.json"
            cfg_in_sym.write_text(
                json.dumps({"exe": "e", "args": args, "args_written": list(args), "model_name": "q4"}),
                encoding="utf-8",
            )

            home = base / "install"
            home.mkdir()
            cfg_dir = layout.configs_dir(home)
            try:
                cfg_dir.symlink_to(target_dir, target_is_directory=True)
                created_link = True
            except (OSError, NotImplementedError):
                created_link = False
                cfg_dir.mkdir()
                (cfg_dir / cfg_in_sym.name).write_text(cfg_in_sym.read_text(encoding="utf-8"), encoding="utf-8")

            import setup as setup_mod

            orig_is_symlink = Path.is_symlink

            def fake_is_symlink(p: Path) -> bool:
                if p == cfg_dir:
                    return True
                return orig_is_symlink(p)

            ctx = mock.MagicMock() if created_link else mock.patch.object(Path, "is_symlink", fake_is_symlink)
            with ctx, \
                    mock.patch.object(setup_mod, "ram_gb", lambda: 63.7), \
                    mock.patch.object(setup_mod, "_unified_memory_pc", lambda: False):
                done = ram_budget_move.run([home], setup_module=setup_mod, say=lambda _t: None)
                self.assertEqual(done, {str(home): []})
                self.assertFalse((home / ram_budget_move.MARK).exists())
                self.assertFalse((cfg_dir / ram_budget_move.MARK).exists())

                # Even when a legacy file in home IS migrated, MARK must not be written while symlinked configs/ has unmigrated configs
                legacy_cfg = home / "strata-unsloth-ud-q4_k_xl-legacy.json"
                legacy_cfg.write_text(
                    json.dumps({"exe": "e", "args": args, "args_written": list(args), "model_name": "q4l"}),
                    encoding="utf-8",
                )
                done2 = ram_budget_move.run([home], setup_module=setup_mod, say=lambda _t: None)
                self.assertEqual(done2, {str(home): ["strata-unsloth-ud-q4_k_xl-legacy.json"]})
                self.assertFalse((home / ram_budget_move.MARK).exists())
                self.assertFalse((cfg_dir / ram_budget_move.MARK).exists())

    def test_launch_catalog_finds_configs_subdir_and_legacy_with_bom(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "install"
            other = base / "other"
            exe, tok = _make_runtime_fixtures(base)
            weights = base / "model.gguf"
            weights.write_bytes(b"x")

            new_cfg = layout.config_path(root, "iq3_s")
            old_cfg = root / "strata-iq3_s.json"
            _write_valid_run_config(
                new_cfg, exe=exe, tok_dir=tok, model_name="iq3_s",
                args=["--native", str(weights)], bom=True,
            )
            _write_valid_run_config(
                old_cfg, exe=exe, tok_dir=tok, model_name="iq3_s-legacy",
                args=["--native", str(weights)],
            )
            other_cfg = layout.config_path(other, "iq1_m")
            _write_valid_run_config(other_cfg, exe=exe, tok_dir=tok, model_name="iq1_m")

            with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(root)}):
                cat = Catalog(root=root, hardware={"ram_gb": 64.0, "gpus": []})
                self.assertEqual(cat.installed_config("IQ3_S"), new_cfg)
                self.assertTrue(cat.installed_with_files("iq3_s"))
                # Even if pre-M2 strata.configs() returns legacy old_cfg, configs/ shadows it
                fake_place = mock.Mock(path=other)
                with mock.patch.object(cat.strata, "configs", return_value=[old_cfg]), \
                        mock.patch.object(cat, "places", [fake_place]):
                    rows = cat.installed_configs()
                paths = [Path(r["path"]) for r in rows]
                self.assertIn(new_cfg, paths)
                self.assertNotIn(old_cfg, paths)
                self.assertIn(other_cfg, paths)

    def test_profiles_and_cfg_paths_work_with_configs_subdir(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            comp = base / "components"
            old_eng = comp / "engine" / "0.1.0-linux-cuda" / "strata-engine"
            new_eng_dir = comp / "engine" / "0.2.0-linux-cuda"
            new_eng_dir.mkdir(parents=True)
            new_eng = new_eng_dir / "strata-engine"
            new_eng.write_bytes(b"e")
            tok = base / "tok"
            tok.mkdir()
            (tok / "vocab.json").write_text("{}", encoding="utf-8")

            home = base / "install"
            cfg = layout.config_path(home, "demo")
            _write_valid_run_config(
                cfg,
                exe=old_eng,
                tok_dir=tok,
                model_name="demo",
                args=["--max-context", "8192"],
                extra={"log": str(layout.log_path(home, "demo"))},
            )
            with mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(base / "profiles"), "STRATA_COMPONENTS_DIR": str(comp)}):
                prof, rev = profiles.create("demo", str(cfg))
                self.assertEqual(Path(prof["source_config"]), cfg.resolve())
                sha = __import__("hashlib").sha256(cfg.read_bytes()).hexdigest()
                proposed, _ = profiles.propose_values("demo", ["port=8091"])
                saved, _, _ = profiles.set_values("demo", ["port=8091"])
                self.assertEqual(saved["overrides"]["port"], 8091)
                notes, problems = cfg_paths.repair([cfg])
                self.assertEqual(problems, [])
                self.assertEqual(len(notes), 1)
                repaired = json.loads(cfg.read_text(encoding="utf-8"))
                self.assertEqual(Path(repaired["exe"]), new_eng)
                self.assertEqual(repaired["log"], str(layout.log_path(home, "demo")))
                # New sha after repair
                new_sha = __import__("hashlib").sha256(cfg.read_bytes()).hexdigest()
                self.assertNotEqual(sha, new_sha)
                loaded, cur_rev = profiles.load("demo")
                proposed2 = {**loaded, "overrides": {"port": 8092}}
                profiles.save("demo", proposed2, expected_revision=cur_rev, expected_source_sha256=new_sha)


if __name__ == "__main__":
    unittest.main()
