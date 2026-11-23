"""Tier 5 E2E White-Box Adversarial Hardening Tests (18 tests).

Covers all 13 empirically confirmed bugs from Challenger 1 & 2 (BUG-1..BUG-13) plus
white-box adversarial edge cases across migration, standalone MCP, config lookup,
platform setup parity, unified launcher discovery, concurrent atomic saves, and
downstream layout-aware tools.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import importlib.util
import io
import json
import os
import stat
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

TERMINAL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TERMINAL_ROOT.parent
if str(TERMINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TERMINAL_ROOT))

from tests.e2e.harness import (  # noqa: E402
    IsolatedWorkspace,
    WorkspaceSnapshotGuard,
    dir_snapshot,
    make_fake_engine,
    make_gguf,
    make_tokenizer,
    root_clutter_entries,
)


class _Tier5Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._repo_guard = WorkspaceSnapshotGuard()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._repo_guard.assert_unchanged(cls.__name__)


class TestTier5BugRepro(_Tier5Base):
    """T5-01 .. T5-10: Regression tests for BUG-1 .. BUG-10."""

    def test_t5_01_migrate_rebase_relative_dot_and_windows_case(self):
        from strata_control import layout

        with IsolatedWorkspace() as ws:
            # 1A: Relative install_home=Path("inst") must not corrupt sibling "my_inst/..."
            inst = ws.base / "inst"
            inst.mkdir(parents=True)
            (inst / "strata-q2_0.json").write_text(
                json.dumps({
                    "exe": "e",
                    "args": [],
                    "other_ref": "my_inst/strata-q2_0.json",
                    "own_ref": "inst/strata-q2_0.json",
                }),
                encoding="utf-8",
            )
            old_cwd = Path.cwd()
            try:
                os.chdir(ws.base)
                res_rel = layout.migrate_legacy_layout(Path("inst"))
            finally:
                os.chdir(old_cwd)
            self.assertEqual(res_rel["errors"], [])
            cfg_rel = json.loads((inst / "configs" / "strata-q2_0.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg_rel["other_ref"], "my_inst/strata-q2_0.json")
            self.assertEqual(Path(cfg_rel["own_ref"]), Path("inst/configs/strata-q2_0.json"))

            # 1B: install_home=Path(".") must not double-prefix or corrupt "x/strata-q2_0.json"
            dot_inst = ws.base / "dot_inst"
            dot_inst.mkdir(parents=True)
            (dot_inst / "strata-q2_0.json").write_text(
                json.dumps({
                    "exe": "e",
                    "args": [],
                    "note": "x/strata-q2_0.json",
                    "self_rel": "strata-q2_0.json",
                    "self_abs": str(dot_inst / "strata-q2_0.json"),
                }),
                encoding="utf-8",
            )
            old_cwd = Path.cwd()
            try:
                os.chdir(dot_inst)
                res_dot = layout.migrate_legacy_layout(Path("."))
            finally:
                os.chdir(old_cwd)
            self.assertEqual(res_dot["errors"], [])
            cfg_dot = json.loads((dot_inst / "configs" / "strata-q2_0.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg_dot["note"], "x/strata-q2_0.json")
            self.assertEqual(Path(cfg_dot["self_rel"]), Path("configs/strata-q2_0.json"))
            self.assertEqual(Path(cfg_dot["self_abs"]).resolve(), (dot_inst / "configs" / "strata-q2_0.json").resolve())

            # 1C: On Windows, case-mismatched paths under install_home must rebase cleanly
            win_dir = ws.base / "WinInst"
            win_dir.mkdir(parents=True)
            raw_target = str(win_dir / "strata-q2_0.json")
            (win_dir / "strata-q2_0.json").write_text(
                json.dumps({
                    "exe": "e",
                    "args": [],
                    "self_lower": raw_target.lower(),
                }),
                encoding="utf-8",
            )
            res_win = layout.migrate_legacy_layout(win_dir)
            self.assertEqual(res_win["errors"], [])
            cfg_win = json.loads((win_dir / "configs" / "strata-q2_0.json").read_text(encoding="utf-8"))
            if os.name == "nt":
                self.assertEqual(
                    Path(cfg_win["self_lower"]).resolve(),
                    (win_dir / "configs" / "strata-q2_0.json").resolve(),
                )
            else:
                self.assertEqual(cfg_win["self_lower"], raw_target.lower())

    def test_t5_02_standalone_strata_mcp_intel_arc_recommend_and_install_plan(self):
        with IsolatedWorkspace() as ws:
            mcp_path = TERMINAL_ROOT / "tools" / "strata_mcp.py"
            saved_sys_path = list(sys.path)
            saved_mods = {
                k: sys.modules[k]
                for k in list(sys.modules)
                if k == "strata_control" or k.startswith("strata_control.") or k == "strata_mcp_standalone"
            }
            try:
                term_norm = os.path.normcase(str(TERMINAL_ROOT.resolve()))
                sys.path[:] = [
                    p for p in sys.path
                    if p and os.path.normcase(str(Path(p).resolve())) != term_norm
                ]
                for k in list(sys.modules):
                    if k == "strata_control" or k.startswith("strata_control."):
                        sys.modules.pop(k, None)
                sys.modules["strata_control"] = None  # type: ignore[assignment]

                spec = importlib.util.spec_from_file_location("strata_mcp_standalone", mcp_path)
                self.assertIsNotNone(spec)
                self.assertIsNotNone(spec.loader)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["strata_mcp_standalone"] = mod
                spec.loader.exec_module(mod)

                arc_hw = {
                    "ram_gb": 64.0,
                    "cpu": {"name": "Test CPU", "avx2": True, "avx512": False},
                    "gpus": [{
                        "index": 0,
                        "name": "Intel Arc B580",
                        "vendor": "intel",
                        "vram_gb": 12.0,
                        "usable": True,
                    }],
                    "engine_vendor": "intel",
                }
                st = mod.Strata(ws.install, ws.terminal_here)
                rec = st.recommend(arc_hw)
                self.assertEqual(rec["backend"], "sycl")
                self.assertTrue(rec.get("notes"), "expected Intel Arc guidance note in recommend()")

                tools = mod.Tools(st)
                with mock.patch.object(st, "hardware", return_value=arc_hw):
                    try:
                        plan = tools.install_plan(
                            "qwen", "Q2_0", 131072, None, None, None, None, "sycl", None, 8080, str(ws.data)
                        )
                        self.assertIsInstance(plan, dict)
                    except mod.ToolError as err:
                        self.assertTrue(str(err))
            finally:
                sys.path[:] = saved_sys_path
                for k in list(sys.modules):
                    if k == "strata_control" or k.startswith("strata_control.") or k == "strata_mcp_standalone":
                        sys.modules.pop(k, None)
                sys.modules.update(saved_mods)

    def test_t5_03_serve_models_find_unnormalized_dot_and_dotdot_from_outside(self):
        from serve import models as serve_models

        with IsolatedWorkspace() as ws:
            canon_home = ws.base / "canon_home"
            m_canon = ws.make_installed_model("q2_0", install_dir=canon_home, layout="configs")
            leg_home = ws.base / "leg_home"
            m_leg = ws.make_installed_model("q2_0", install_dir=leg_home, layout="legacy")
            outside = ws.base / "outside"
            outside.mkdir()

            canon_models = serve_models.discover(canon_home)
            leg_models = serve_models.discover(leg_home)
            combined_leg = serve_models.discover(canon_home) + serve_models.discover(leg_home)

            old_cwd = Path.cwd()
            try:
                os.chdir(outside)
                found_canon = serve_models.find(canon_models, "configs/../configs/strata-q2_0.json")
                self.assertIsNotNone(found_canon)
                self.assertEqual(Path(found_canon["config"]).resolve(), m_canon["cfg"].resolve())

                found_dot = serve_models.find(leg_models, "./strata-q2_0.json")
                self.assertIsNotNone(found_dot)
                self.assertEqual(Path(found_dot["config"]).resolve(), m_leg["cfg"].resolve())

                found_dotdot = serve_models.find(leg_models, "configs/../strata-q2_0.json")
                self.assertIsNotNone(found_dotdot)
                self.assertEqual(Path(found_dotdot["config"]).resolve(), m_leg["cfg"].resolve())

                # Traversal escaping above root must still be rejected
                self.assertIsNone(serve_models.find(canon_models, "../configs/strata-q2_0.json"))
                self.assertIsNone(serve_models.find(leg_models, "../strata-q2_0.json"))

                # Cross-root ambiguity must return None
                canon_home_2 = ws.base / "canon_home_2"
                ws.make_installed_model("q2_0", install_dir=canon_home_2, layout="configs")
                dup_canon = serve_models.discover(canon_home) + serve_models.discover(canon_home_2)
                self.assertIsNone(serve_models.find(dup_canon, "configs/../configs/strata-q2_0.json"))
                # ./strata-q2_0.json only matches legacy root, not configs/
                found_only_leg = serve_models.find(combined_leg, "./strata-q2_0.json")
                self.assertIsNotNone(found_only_leg)
                self.assertEqual(Path(found_only_leg["config"]).resolve(), m_leg["cfg"].resolve())
            finally:
                os.chdir(old_cwd)

    def test_t5_04_layout_resolve_config_ref_subpath_from_outside_cwd(self):
        from strata_control import layout

        with IsolatedWorkspace() as ws:
            cfg = ws.install / "configs" / "strata-1.5b.json"
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(json.dumps({"exe": "e", "args": []}), encoding="utf-8")
            outside = ws.base / "deep" / "outside"
            outside.mkdir(parents=True)
            (ws.base / "escape.json").write_text("{}", encoding="utf-8")

            old_cwd = Path.cwd()
            try:
                os.chdir(outside)
                got_fwd = layout.resolve_config_ref(ws.install, "configs/strata-1.5b.json")
                got_bck = layout.resolve_config_ref(ws.install, r"configs\strata-1.5b.json")
                self.assertIsNotNone(got_fwd)
                self.assertIsNotNone(got_bck)
                self.assertEqual(got_fwd.resolve(), cfg.resolve())
                self.assertEqual(got_bck.resolve(), cfg.resolve())
                self.assertIsNone(layout.resolve_config_ref(ws.install, "../escape.json"))
            finally:
                os.chdir(old_cwd)

    def test_t5_05_setup_intel_start_sycl_legacy_script_in_home(self):
        import setup as S

        sycl_dir = REPO_ROOT / "engine" / "sycl"
        saved_sys_path = list(sys.path)
        try:
            if str(sycl_dir) not in sys.path:
                sys.path.insert(0, str(sycl_dir))
            import setup_intel

            with IsolatedWorkspace() as ws:
                fake_term = ws.base / "fake_term"
                (fake_term / "configs").mkdir(parents=True)
                cfg_p = fake_term / "configs" / "strata-q2_0.json"
                cfg_p.write_text(json.dumps({"backend": "sycl", "model_name": "q2_0", "port": 8080}), encoding="utf-8")
                legacy_sh = fake_term / "run-q2_0.sh"
                legacy_sh.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
                fake_engine_root = ws.base / "fake_engine"
                fake_engine_root.mkdir()
                exec_calls: list[tuple[str, list[str]]] = []

                attrs_to_restore = {
                    name: getattr(S, name)
                    for name in (
                        "WIN", "home", "gpus", "amd_gpus", "amd_problem", "hip_vision",
                        "build_engine_hip", "hipblaslt_table", "ram_gb", "mem_have",
                        "bench_tips", "write_setup_config", "write_run_script", "start", "say", "main",
                    )
                }
                orig_argv = list(sys.argv)
                try:
                    def fake_execv(path, argv):
                        exec_calls.append((str(path), [str(x) for x in argv]))
                        raise SystemExit(0)

                    def fake_main():
                        return S.start(cfg_p)

                    with mock.patch.object(setup_intel, "ROOT", fake_engine_root), \
                            mock.patch.object(setup_intel, "READER_DIR", fake_engine_root / "build-vision"), \
                            mock.patch.object(setup_intel, "intel_devices", return_value=([{"index": 0, "name": "Arc B580", "vram_gb": 12.0, "driver": "xe", "devid": "e20b"}], [])), \
                            mock.patch.object(setup_intel, "sycl_engine", return_value=(fake_engine_root / "build-sycl-aot" / "strata", None)), \
                            mock.patch.object(setup_intel.os, "execv", side_effect=fake_execv), \
                            mock.patch.object(S, "WIN", False), \
                            mock.patch.object(S, "home", lambda: fake_term), \
                            mock.patch.object(S, "ram_gb", lambda: 64.0), \
                            mock.patch.object(S, "main", fake_main), \
                            contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit) as cm:
                            setup_intel.install(["--yes"])
                        self.assertEqual(cm.exception.code, 0)
                finally:
                    for k, v in attrs_to_restore.items():
                        setattr(S, k, v)
                    sys.argv[:] = orig_argv

                self.assertEqual(len(exec_calls), 1)
                self.assertEqual(exec_calls[0][0], "/bin/sh")
                self.assertEqual(Path(exec_calls[0][1][1]).resolve(), legacy_sh.resolve())
        finally:
            sys.path[:] = saved_sys_path

    def test_t5_06_setup_mac_setup_poly_reads_legacy_previous_config(self):
        import setup as S

        metal_dir = REPO_ROOT / "engine" / "metal"
        saved_sys_path = list(sys.path)
        try:
            if str(metal_dir) not in sys.path:
                sys.path.insert(0, str(metal_dir))
            import setup_mac

            with IsolatedWorkspace() as ws:
                fake_home = ws.base / "mac_home"
                fake_home.mkdir()
                legacy_cfg = fake_home / "strata-glm4.json"
                legacy_cfg.write_text(
                    json.dumps({"fit_max_tokens": 16384, "args": ["--mtp"], "parallel": 2}),
                    encoding="utf-8",
                )
                fam_id = next(f for f in setup_mac.MAC_POLY_FAMILIES if "glm" in f or S.FAMILIES[f].get("tag") == "glm")
                fam = S.FAMILIES[fam_id]
                model = next(iter(S.family_models(fam_id)))
                tag = fam["tag"] + model
                # Write legacy config matching exact computed tag
                tag_legacy = fake_home / f"strata-{tag.lower()}.json"
                tag_legacy.write_text(
                    json.dumps({"fit_max_tokens": 16384, "args": ["--mtp"], "parallel": 2}),
                    encoding="utf-8",
                )

                gguf_dir = ws.base / "mac_ggufs"
                gguf_dir.mkdir()
                for i in range(1, S.model_shards(fam, model) + 1):
                    make_gguf(gguf_dir / S.model_file(fam, model, i), architecture="glm5-next")
                pack_dir = ws.data / "packs" / tag.lower()
                make_tokenizer(pack_dir / "tokenizer")
                if fam.get("draft"):
                    draft_file = ws.data / fam["draft"]
                    draft_file.parent.mkdir(parents=True, exist_ok=True)
                    draft_file.write_bytes(b"draft")
                if d := S.family_models(fam_id)[model]:
                    if d.get("draft_model"):
                        (gguf_dir / Path(d["draft_model"]).name).write_bytes(b"draft")

                seen_previous: list[dict] = []
                orig_apply = setup_mac.P.apply_options

                def spy_apply(args, a_ns, previous):
                    seen_previous.append(dict(previous))
                    return orig_apply(args, a_ns, previous)

                ns = types.SimpleNamespace(
                    family=fam_id,
                    model=model,
                    context=8192,
                    vision="no",
                    vision_tokens=None,
                    models_dir=None,
                    gguf_dir=str(gguf_dir),
                    update=False,
                    build=False,
                    no_download_models=True,
                    yes=True,
                    conversation_cache_mib=None,
                    vram_reserve_mib=None,
                    port=8080,
                    host=None,
                    api_key=None,
                    browser=False,
                    parallel=None,
                    poly_kv=None,
                    no_scripts=True,
                    no_start=True,
                )
                fake_exe = make_fake_engine(ws.base / "mac_build", poly=True)
                fake_header = types.SimpleNamespace(metadata={"general.architecture": "glm5-next"})
                fake_est = {"ok": True, "total_gb": 10.0, "budget_gb": 32.0}

                with mock.patch.object(S, "home", lambda: fake_home), \
                        mock.patch.object(S, "pip_install", lambda *a, **k: None), \
                        mock.patch.object(S, "llama_override", lambda: None), \
                        mock.patch.object(S, "verify_sha256", lambda *a, **k: None), \
                        mock.patch.object(S, "check_shards", lambda *a, **k: None), \
                        mock.patch.object(S, "run", lambda *a, **k: types.SimpleNamespace(returncode=0)), \
                        mock.patch.object(S, "poly_kv_whole", lambda *a, **k: None), \
                        mock.patch.object(S, "write_setup_config", lambda *a, **k: None), \
                        mock.patch.object(S, "say", lambda *a, **k: None), \
                        mock.patch.object(S, "warn", lambda *a, **k: None), \
                        mock.patch.object(S, "ok", lambda *a, **k: None), \
                        mock.patch.object(setup_mac, "build_engine", lambda *a, **k: fake_exe), \
                        mock.patch.object(setup_mac.P, "read_headers", return_value=([fake_header], None)), \
                        mock.patch.object(setup_mac.P, "estimate", return_value=fake_est), \
                        mock.patch.object(setup_mac.P, "report", lambda *a, **k: None), \
                        mock.patch.object(setup_mac.P, "apply_options", side_effect=spy_apply):
                    rc = setup_mac.setup_poly(ns, {"ram_gb": 64.0, "perf_cores": 8}, ws.data, [])
                self.assertEqual(rc, 0)
                self.assertEqual(len(seen_previous), 1)
                self.assertEqual(seen_previous[0].get("fit_max_tokens"), 16384)
                self.assertEqual(seen_previous[0].get("args"), ["--mtp"])
        finally:
            sys.path[:] = saved_sys_path

    def test_t5_07_polystrata_respects_strata_install_dir(self):
        import polystrata

        with IsolatedWorkspace() as ws:
            ext_dir = ws.base / "external_install"
            ws.make_installed_model("q2_0", install_dir=ext_dir, layout="configs")
            empty_root = ws.base / "empty_terminal"
            empty_root.mkdir()

            with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(ext_dir)}), \
                    mock.patch.object(polystrata, "ROOT", empty_root), \
                    mock.patch.object(polystrata.terminal, "api", return_value=(0, {})), \
                    mock.patch.object(sys, "argv", ["polystrata.py", "--list-models"]):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = polystrata.main()
            self.assertEqual(rc, 0, buf.getvalue())
            self.assertIn("q2_0", buf.getvalue())

    def test_t5_08_mixed_case_is_config_file_and_models_shadow_parity(self):
        from strata_control import layout, models as ctrl_models

        with IsolatedWorkspace() as ws:
            self.assertEqual(layout.is_config_file("Strata-q2_0.json"), layout.is_log_file("Strata-q2_0.log"))
            self.assertTrue(layout.is_config_file("Strata-q2_0.json"))
            self.assertFalse(layout.is_config_file("Strata-q2_0.SHARED-SETTINGS.JSON"))

            # Invalid-ID mixed-case configs/strata-Q2_0.json must not shadow valid legacy strata-q2_0.json into 0 models
            root = ws.base / "mixed_case_home"
            (root / "configs").mkdir(parents=True)
            ws.make_installed_model("q2_0", install_dir=root, layout="configs", filename_override="strata-Q2_0.json")
            ws.make_installed_model("q2_0", install_dir=root, layout="legacy", filename_override="strata-q2_0.json")

            listed = ctrl_models.list_installed(override=str(root))
            self.assertEqual([m["id"] for m in listed["models"]], ["q2_0"])

    def test_t5_09_concurrent_runconfig_save_atomic(self):
        from serve import runconfig

        with IsolatedWorkspace() as ws:
            cfg_path = ws.install / "configs" / "strata-conc.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps({"exe": "e", "args": [], "counter": 0}), encoding="utf-8")

            errors: list[BaseException] = []

            def worker(tid: int) -> None:
                try:
                    for it in range(20):
                        runconfig.save(cfg_path, {"exe": "e", "args": [], "tid": tid, "it": it})
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futs = [pool.submit(worker, tid) for tid in range(4)]
                for f in futs:
                    f.result()

            self.assertEqual(errors, [], f"concurrent runconfig.save failed: {errors}")
            parsed = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertIn("tid", parsed)
            self.assertTrue((cfg_path.parent / "strata-conc.json.bak").is_file())
            self.assertEqual(root_clutter_entries(ws.install), [])

    def test_t5_10_migrate_relative_warm_start_seed_file(self):
        from strata_control import layout

        with IsolatedWorkspace() as ws:
            home = ws.base / "warm_home"
            home.mkdir()
            seed = home / "seed.json"
            seed.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}), encoding="utf-8")
            (home / "strata-q2_0.json").write_text(
                json.dumps({"exe": "e", "args": [], "warm_start": "seed.json"}),
                encoding="utf-8",
            )

            res = layout.migrate_legacy_layout(home)
            self.assertEqual(res["errors"], [])
            # Non-strata-* user file stays at home/seed.json
            self.assertTrue(seed.is_file())
            migrated_cfg = home / "configs" / "strata-q2_0.json"
            self.assertTrue(migrated_cfg.is_file())
            data = json.loads(migrated_cfg.read_text(encoding="utf-8"))
            resolved_seed = (migrated_cfg.parent / data["warm_start"]).resolve()
            self.assertEqual(resolved_seed, seed.resolve())
            self.assertTrue(resolved_seed.is_file())


class TestTier5WhiteBoxCoverage(_Tier5Base):
    """T5-11 .. T5-15: Additional white-box adversarial coverage."""

    def test_t5_11_setup_migrate_cli_and_collision_idempotency(self):
        import setup

        with IsolatedWorkspace(control_mode=True) as ws:
            home = ws.install
            (home / "strata-q2_0.json").write_text(
                json.dumps({"exe": str(home / "engine" / "strata"), "args": [], "log": "strata-q2_0.log"}),
                encoding="utf-8",
            )
            (home / "strata-q2_0.shared-settings.json").write_text(json.dumps({"temperature": 0.6}), encoding="utf-8")
            (home / "strata-q2_0.json.bak").write_text("{}", encoding="utf-8")
            (home / "strata-q2_0.warm-start").write_text("[1,2,3]", encoding="utf-8")
            (home / "strata-q2_0.warm-start-0.bin").write_bytes(b"WS")
            (home / "strata-q2_0.tool-media").mkdir()
            (home / "strata-q2_0.tool-media" / "img.png").write_bytes(b"PNG")
            (home / "strata-q2_0.log").write_text("started\n", encoding="utf-8")
            (home / "run-q2_0.bat").write_bytes(f'"{home / "strata-q2_0.json"}"\r\n'.encode("utf-8"))

            # Colliding config already exists in configs/
            (home / "configs").mkdir(parents=True)
            colliding = home / "configs" / "strata-iq3.json"
            colliding.write_text('{"keep": "canonical"}', encoding="utf-8")
            (home / "strata-iq3.json").write_text('{"keep": "legacy"}', encoding="utf-8")

            with mock.patch.object(setup, "move_over_old_layout", lambda: []), \
                    mock.patch.object(sys, "argv", ["setup.py", "--migrate"]):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = setup.main()
            self.assertEqual(rc, 0)
            for rel in (
                "configs/strata-q2_0.json",
                "configs/strata-q2_0.shared-settings.json",
                "configs/strata-q2_0.json.bak",
                "configs/strata-q2_0.warm-start",
                "configs/strata-q2_0.warm-start-0.bin",
                "configs/strata-q2_0.tool-media/img.png",
                "logs/strata-q2_0.log",
                "scripts/run-q2_0.bat",
            ):
                self.assertTrue((home / rel).exists(), rel)
            self.assertEqual(json.loads(colliding.read_text(encoding="utf-8")), {"keep": "canonical"})
            self.assertTrue((home / "strata-iq3.json").is_file())

            # Remove colliding legacy file and verify repeat --migrate is idempotent
            (home / "strata-iq3.json").unlink()
            snap_before = dir_snapshot(home)
            with mock.patch.object(setup, "move_over_old_layout", lambda: []), \
                    mock.patch.object(sys, "argv", ["setup.py", "--migrate"]):
                buf2 = io.StringIO()
                with contextlib.redirect_stdout(buf2):
                    rc2 = setup.main()
            self.assertEqual(rc2, 0)
            self.assertEqual(dir_snapshot(home), snap_before)

    def test_t5_12_cli_models_migrate_dry_run_and_apply_with_install_dir(self):
        from strata_control import profiles

        with IsolatedWorkspace() as ws:
            m = ws.make_installed_model("q2_0", layout="legacy", create_log=True, create_script=True)
            profiles.create("prof1", str(m["cfg"]))

            snap_before = dir_snapshot(ws.install)
            rc_dry, out_dry, err_dry = ws.run_cli(["models", "migrate", "--dry-run", "--output", "json"])
            self.assertEqual(rc_dry, 0, err_dry)
            dry_json = json.loads(out_dry)
            self.assertTrue(dry_json["dry_run"])
            self.assertGreaterEqual(dry_json["moved_count"], 1)
            self.assertEqual(dir_snapshot(ws.install), snap_before)

            rc_apply, out_apply, err_apply = ws.run_cli(["models", "migrate", "--output", "json"])
            self.assertEqual(rc_apply, 0, err_apply)
            apply_json = json.loads(out_apply)
            self.assertFalse(apply_json["dry_run"])
            self.assertEqual(apply_json["profiles_updated"], ["prof1"])
            self.assertEqual(root_clutter_entries(ws.install), [])
            prof, _rev = profiles.load("prof1")
            self.assertEqual(
                Path(prof["source_config"]).resolve(),
                (ws.install / "configs" / "strata-q2_0.json").resolve(),
            )

    def test_t5_13_non_dict_json_and_bom_cross_reader_resilience(self):
        from serve import models as serve_models
        from serve.server_parts import http_api
        from strata_control import discovery, layout, models as ctrl_models
        from tools import strata_bench
        from tools.strata_mcp import Strata

        with IsolatedWorkspace() as ws:
            cfg_dir = ws.install / "configs"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            m = ws.make_installed_model("good", layout="configs")
            raw_good = m["cfg"].read_bytes()
            m["cfg"].write_bytes(b"\xef\xbb\xbf" + raw_good)

            (cfg_dir / "strata-list.json").write_text("[]", encoding="utf-8")
            (cfg_dir / "strata-null.json").write_text("null", encoding="utf-8")
            (cfg_dir / "strata-trunc.json").write_text("{bad", encoding="utf-8")

            ss_bad = cfg_dir / "strata-good.shared-settings.json"
            ss_bad.write_text("[]", encoding="utf-8")

            srv_res = serve_models.discover(ws.install)
            self.assertEqual([x["id"] for x in srv_res], ["good"])

            ctrl_res = ctrl_models.list_installed(override=str(ws.install))
            by_id = {x["id"]: x for x in ctrl_res["models"]}
            self.assertEqual(by_id["good"]["component_status"], "present")
            for bad_id in ("list", "null", "trunc"):
                self.assertEqual(by_id[bad_id]["component_status"], "invalid")

            self.assertEqual(len(discovery.config_files(ws.install)), 4)
            self.assertEqual([p.resolve() for p in strata_bench.find_configs(ws.install)], [m["cfg"].resolve()])
            st = Strata(ws.install, ws.terminal_here)
            installed_res = st.installed()
            self.assertEqual([x["model"] for x in installed_res["models"] if x["ready"]], ["good"])
            self.assertEqual({x["model"] for x in installed_res["models"]}, {"good", "list", "null", "trunc"})
            self.assertEqual(http_api.load_shared_settings(ss_bad), {})

            # Legacy non-dict / truncated configs survive migration without crashing
            leg_home = ws.base / "leg_nondict"
            leg_home.mkdir()
            (leg_home / "strata-arr.json").write_text("[]", encoding="utf-8")
            (leg_home / "strata-bad.json").write_text("{", encoding="utf-8")
            mig = layout.migrate_legacy_layout(leg_home)
            self.assertEqual(mig["errors"], [])
            self.assertEqual((leg_home / "configs" / "strata-arr.json").read_text(encoding="utf-8"), "[]")
            self.assertEqual((leg_home / "configs" / "strata-bad.json").read_text(encoding="utf-8"), "{")

    def test_t5_14_setup_no_scripts_and_calibrate_bench_tag_lookup(self):
        from strata_control import layout
        from tools import calibrate as cal_mod, strata_bench

        with IsolatedWorkspace(control_mode=True) as ws:
            code, out, err = ws.run_setup(
                ["--family", "qwen", "--model", "Q2_0", "--yes", "--no-start", "--no-scripts", "--vision", "no"]
            )
            self.assertEqual(code, 0, f"setup --no-scripts failed: {out}\n{err}")
            cfg_p = ws.install / "configs" / "strata-q2_0.json"
            self.assertTrue(cfg_p.is_file())
            self.assertEqual(root_clutter_entries(ws.install), [])
            scripts_dir = ws.install / "scripts"
            self.assertEqual(list(scripts_dir.glob("run-*.*")) if scripts_dir.is_dir() else [], [])

            found = strata_bench.find_configs(ws.install)
            self.assertEqual([p.resolve() for p in found], [cfg_p.resolve()])

            seen_cfgs: list[dict] = []

            class _StopBench(Exception):
                pass

            def fake_collect(base_url, key, parsed_cfg, gpu):
                seen_cfgs.append(parsed_cfg)
                raise _StopBench()

            with mock.patch.object(strata_bench, "ROOT", ws.install), \
                    mock.patch.object(strata_bench, "is_strata", return_value=True), \
                    mock.patch.object(strata_bench, "collect_machine", fake_collect):
                for arg in ("q2_0", "configs/strata-q2_0.json"):
                    with mock.patch.object(sys, "argv", ["strata_bench.py", "--config", arg]), \
                            contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(_StopBench):
                            strata_bench.main()
            self.assertEqual(len(seen_cfgs), 2)

            with mock.patch.object(cal_mod, "ROOT", ws.install):
                self.assertEqual(layout.resolve_config_ref(ws.install, "q2_0").resolve(), cfg_p.resolve())
                self.assertEqual(
                    layout.resolve_config_ref(ws.install, "configs/strata-q2_0.json").resolve(),
                    cfg_p.resolve(),
                )

    def test_t5_15_models_move_ram_budget_move_tune_apply_and_adopt_in_configs(self):
        from strata_cli import ram_budget_move
        from strata_cli.launch import catalog as launch_catalog, runner as launch_runner
        from strata_control import models_move, tune_apply, tune_history

        with IsolatedWorkspace() as ws:
            # 1) models_move & ram_budget_move inside configs/
            m = ws.make_installed_model(
                "q2_0",
                layout="configs",
                extra_cfg={
                    "args": [
                        "--native", str(ws.data / "models" / "Q2_0" / "Qwen3.8-Flash-Next-q2_0-00001-of-00001.gguf"),
                        "--pack", str(ws.data / "packs" / "q2_0"),
                        "--resident-budget-gib", "40",
                        "--max-context", "32768",
                    ]
                },
            )
            self.assertEqual([p.resolve() for p in models_move.config_files(ws.install)], [m["cfg"].resolve()])
            new_data = ws.base / "new_data"
            new_data.mkdir()
            rc_mv, out_mv, err_mv = ws.run_cli(
                ["models", "move", str(new_data), "--from", str(ws.data), "--install-dir", str(ws.install), "--output", "json"]
            )
            self.assertEqual(rc_mv, 0, f"{out_mv}\n{err_mv}")
            self.assertIn(str(new_data), json.loads(m["cfg"].read_text(encoding="utf-8"))["args"][1])

            fake_setup = mock.MagicMock()
            fake_setup.migrate_run_configs.return_value = [m["cfg"].name]
            moved_rb = ram_budget_move.run(folders=[ws.install], setup_module=fake_setup, say=lambda _t: None)
            self.assertTrue(moved_rb.get(str(ws.install)))
            self.assertTrue((ws.install / "configs" / ram_budget_move.MARK).is_file())
            self.assertFalse((ws.install / ram_budget_move.MARK).exists())

            # 2) tune_apply + rollback in configs/
            before_bytes = m["cfg"].read_bytes()
            ident = tune_history.file_identity(m["cfg"])
            job = {
                "id": "job-t5-15",
                "state": "completed",
                "report": {
                    "recommendation_available": True,
                    "recommendation": {"args": {"--pcie-frac": "0.50"}},
                    "identity": ident,
                },
            }
            state_dir = ws.base / "tune_state"
            applied = tune_apply.apply_recommendation(
                job=job,
                config_path=m["cfg"],
                current_identity=ident,
                state_dir=state_dir,
                verify=lambda _cand: 4,
            )
            self.assertIn("--pcie-frac", json.loads(m["cfg"].read_text(encoding="utf-8"))["args"])
            tune_apply.rollback(application_id=applied["application"]["application_id"], state_dir=state_dir)
            self.assertEqual(m["cfg"].read_bytes(), before_bytes)

            # 3) runner.adopt_prepared_files writes config to configs/ and log to logs/
            adopt_home = ws.base / "adopt_home"
            adopt_home.mkdir()
            make_fake_engine(ws.components / "engine" / "0.1.66-linux-x86_64-sycl")
            (ws.components / "engine" / "active.json").write_text(
                json.dumps({"folder": "0.1.66-linux-x86_64-sycl"}),
                encoding="utf-8",
            )
            adopt_data = ws.base / "adopt_data"
            gguf_dir = adopt_data / "models" / "Q2_0"
            make_gguf(gguf_dir / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00001.gguf")
            pack_dir = adopt_data / "packs" / "q2_0"
            make_tokenizer(pack_dir / "tokenizer")
            (pack_dir / "experts.bin").write_bytes(b"experts")
            (pack_dir / "native_experts.txt").write_text("0\n", encoding="utf-8")

            settings = types.SimpleNamespace(
                family="qwen",
                model="Q2_0",
                model_path=str(gguf_dir),
                data_dir=str(adopt_data),
                context=32768,
                kv=None,
            )
            comp = types.SimpleNamespace(main_vram_gb=12.0)
            cat = mock.MagicMock(spec=launch_catalog.Catalog)
            cat.root = adopt_home
            cat.src = ws.terminal_here
            cat.tables.return_value = ({}, {"qwen": {"name": "qwen3.8-flash-next", "tag": ""}}, {}, "builtin")
            cat.tag_of.return_value = "Q2_0"
            cat.data_dir.return_value = adopt_data
            cat.computer.return_value = comp

            runner_obj = object.__new__(launch_runner.Runner)
            runner_obj.s = settings
            runner_obj._catalog = cat
            runner_obj.notices = []
            runner_obj.emit = lambda *a, **k: None

            with mock.patch("strata_control.engine_component.detect_backend", return_value=("sycl", "test")), \
                    mock.patch("strata_cli.launch.adopt.plan") as mock_plan, \
                    mock.patch("strata_cli.launch.adopt.write") as mock_write:
                mock_plan.return_value = ({"exe": "component:strata-engine", "args": [], "log": str(adopt_home / "logs" / "strata-q2_0.log")}, [])
                def _fake_write(path, cfg):
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_text(json.dumps(cfg), encoding="utf-8")
                mock_write.side_effect = _fake_write
                adopted_path = runner_obj.adopt_prepared_files("no NVIDIA GPU found", chosen=True)

            self.assertEqual(adopted_path.resolve(), (adopt_home / "configs" / "strata-q2_0.json").resolve())
            self.assertTrue(adopted_path.is_file())
            self.assertEqual(
                Path(mock_plan.call_args.kwargs["log"]).resolve(),
                (adopt_home / "logs" / "strata-q2_0.log").resolve(),
            )
            self.assertEqual(root_clutter_entries(adopt_home), [])

    def test_t5_16_setup_migrate_run_configs_canonical_precedence(self):
        import setup
        from strata_control import user_args

        with IsolatedWorkspace() as ws:
            folder = ws.base / "mig_folder"
            cfg_dir = folder / "configs"
            cfg_dir.mkdir(parents=True)
            canon_cfg = {
                "exe": "e",
                "args": ["--native", "m.gguf", "--resident-budget-gib", "40"],
            }
            user_args.stamp(canon_cfg)
            canon_path = cfg_dir / "strata-q2_0.json"
            canon_path.write_text(json.dumps(canon_cfg), encoding="utf-8")

            legacy_shadow = {
                "exe": "e",
                "args": ["--native", "m.gguf"],
            }
            shadow_path = folder / "strata-q2_0.json"
            shadow_path.write_text(json.dumps(legacy_shadow), encoding="utf-8")

            other_legacy = {
                "exe": "e",
                "args": ["--native", "m2.gguf", "--resident-budget-gib", "40"],
            }
            user_args.stamp(other_legacy)
            other_path = folder / "strata-iq3_s.json"
            other_path.write_text(json.dumps(other_legacy), encoding="utf-8")

            with mock.patch.object(setup, "ram_gb", return_value=64.0), \
                    mock.patch.object(setup, "_unified_memory_pc", return_value=False):
                moved = setup.migrate_run_configs(folder)

            self.assertEqual(sorted(moved), ["strata-iq3_s.json", "strata-q2_0.json"])
            updated_canon = json.loads(canon_path.read_text(encoding="utf-8"))
            self.assertEqual(setup.flag_value(updated_canon["args"], "--resident-budget-gib"), "47")
            self.assertTrue((cfg_dir / "strata-q2_0.json.bak").is_file())
            self.assertEqual(json.loads(shadow_path.read_text(encoding="utf-8")), legacy_shadow)
            updated_other = json.loads(other_path.read_text(encoding="utf-8"))
            self.assertEqual(setup.flag_value(updated_other["args"], "--resident-budget-gib"), "47")

    def test_t5_17_strata_mcp_broken_symlink_describe_config_resilience(self):
        from tools.strata_mcp import Strata, Tools

        with IsolatedWorkspace() as ws:
            ws.make_installed_model("good", layout="configs")
            cfg_dir = ws.install / "configs"
            broken_cfg = cfg_dir / "strata-brokenlink.json"
            broken_cfg.write_text("{}", encoding="utf-8")
            valid_sym_cfg = cfg_dir / "strata-validlink.json"
            valid_sym_cfg.write_text((cfg_dir / "strata-good.json").read_text(encoding="utf-8"), encoding="utf-8")

            broken_key = os.path.normcase(os.path.abspath(str(broken_cfg)))
            sym_keys = {
                broken_key,
                os.path.normcase(os.path.abspath(str(valid_sym_cfg))),
            }
            orig_is_symlink = Path.is_symlink
            orig_stat = Path.stat
            orig_lstat = Path.lstat

            def fake_is_symlink(p: Path) -> bool:
                if os.path.normcase(os.path.abspath(str(p))) in sym_keys:
                    return True
                return orig_is_symlink(p)

            def fake_stat(p: Path, *args, **kwargs):
                follow = kwargs.get("follow_symlinks", True)
                if follow and os.path.normcase(os.path.abspath(str(p))) == broken_key:
                    raise FileNotFoundError(2, "No such file or directory", str(p))
                return orig_stat(p, *args, **kwargs)

            st = Strata(ws.install, ws.terminal_here)
            tools = Tools(st)
            with mock.patch.object(Path, "is_symlink", fake_is_symlink), \
                    mock.patch.object(Path, "stat", fake_stat):
                inst = st.installed()
                by_model = {m["model"]: m for m in inst["models"]}
                self.assertEqual(set(by_model), {"good", "brokenlink", "validlink"})
                self.assertTrue(by_model["good"]["ready"])
                self.assertFalse(by_model["brokenlink"]["ready"])
                self.assertIsNotNone(by_model["brokenlink"]["last_used"])
                self.assertFalse(by_model["validlink"]["ready"])
                status = tools.strata_status(hardware=False)
                self.assertIn("installed", status)

                def fake_lstat_err(p: Path, *args, **kwargs):
                    if os.path.normcase(os.path.abspath(str(p))) == broken_key:
                        raise OSError("stat failed")
                    return orig_lstat(p, *args, **kwargs)

                with mock.patch.object(Path, "lstat", fake_lstat_err):
                    desc = st.describe_config(broken_cfg)
                    self.assertFalse(desc["ready"])
                    self.assertIsNone(desc["last_used"])

    def test_t5_18_migrate_rebase_mixed_windows_separators(self):
        from strata_control import layout

        with IsolatedWorkspace() as ws:
            p1 = ws.base / "p1_mixed"
            p1.mkdir(parents=True)
            p1_str = str(p1)
            p1_fwd = p1_str.replace("\\", "/")
            sib = ws.base / "p1_mixed_other" / "strata-q2_0.json"

            (p1 / "strata-q2_0.json").write_text(
                json.dumps({
                    "exe": "e",
                    "args": [],
                    "all_native": str(p1 / "strata-q2_0.json"),
                    "all_fwd": f"{p1_fwd}/strata-q2_0.json",
                    "mixed1": f"{p1_fwd}\\strata-q2_0.json",
                    "mixed2": f"{p1_str}/strata-q2_0.json",
                    "mixed_log": f"{p1_fwd}\\strata-q2_0.log",
                    "sibling": str(sib),
                }),
                encoding="utf-8",
            )
            (p1 / "strata-q2_0.log").write_text("ok\n", encoding="utf-8")
            (p1 / "run-q2_0.bat").write_bytes(
                f'"{p1_fwd}\\strata-q2_0.json"\r\n'.encode("utf-8")
            )

            res = layout.migrate_legacy_layout(p1)
            self.assertEqual(res["errors"], [])

            migrated_cfg = p1 / "configs" / "strata-q2_0.json"
            migrated_log = p1 / "logs" / "strata-q2_0.log"
            migrated_bat = p1 / "scripts" / "run-q2_0.bat"
            self.assertTrue(migrated_cfg.is_file())
            self.assertTrue(migrated_log.is_file())
            self.assertTrue(migrated_bat.is_file())

            data = json.loads(migrated_cfg.read_text(encoding="utf-8"))
            self.assertEqual(Path(data["all_native"]).resolve(), migrated_cfg.resolve())
            self.assertEqual(Path(data["all_fwd"]).resolve(), migrated_cfg.resolve())
            self.assertIn("/", data["all_fwd"])
            self.assertEqual(data["sibling"], str(sib))
            if os.name == "nt":
                self.assertEqual(Path(data["mixed1"]).resolve(), migrated_cfg.resolve())
                self.assertEqual(Path(data["mixed2"]).resolve(), migrated_cfg.resolve())
                self.assertEqual(Path(data["mixed_log"]).resolve(), migrated_log.resolve())
                bat_text = migrated_bat.read_bytes().decode("utf-8")
                self.assertIn("configs\\strata-q2_0.json", bat_text)
                self.assertTrue(bat_text.endswith("\r\n"))


if __name__ == "__main__":
    unittest.main()
