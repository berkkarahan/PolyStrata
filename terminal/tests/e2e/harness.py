"""Stdlib-only E2E test harness for PolyStrata model config & runtime storage tests.

Provides:
- `WorkspaceSnapshotGuard`: verifies the repository root and `terminal/`
  have zero added, deleted, or modified files during test execution.
- `IsolatedWorkspace`: temp-dir workspace with full environment isolation,
  module-global patching, synthetic GGUF/tokenizer/engine builders, and
  sandbox-safe HTTP (`make_handler`), CLI, setup, and file-redirected subprocess
  invokers.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from email.parser import BytesParser
from pathlib import Path
from unittest import mock

TERMINAL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TERMINAL_ROOT.parent
if str(TERMINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TERMINAL_ROOT))

CLUTTER_EXACT = set()
CLUTTER_SUFFIXES = (".log", ".json.bak")
CLUTTER_RE = re.compile(
    r"^(?:strata-.*\.json(?:\.bak|\.tmp)?|strata-.*\.shared-settings\.json|"
    r"strata-.*\.log|run-.*\.(?:bat|sh)|.*\.warm-start(?:\..*)?|.*\.tool-media)$",
    re.I,
)


def is_per_model_clutter_name(name: str) -> bool:
    """Return True if a top-level filename/dirname is a per-model generated artifact."""
    return bool(CLUTTER_RE.fullmatch(name))


def root_clutter_entries(directory: Path) -> list[str]:
    """Return sorted top-level entry names in `directory` that are per-model artifacts."""
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir() if is_per_model_clutter_name(p.name))


def dir_snapshot(base: Path, *, exclude_dirs: tuple[str, ...] = ("__pycache__", ".git", ".agents")) -> dict[str, tuple]:
    """Capture recursive deterministic state (type, size, mtime_ns) of `base`."""
    out: dict[str, tuple] = {}
    if not base.exists():
        return out
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in exclude_dirs)
        rpath = Path(root)
        for d in dirs:
            p = rpath / d
            rel = p.relative_to(base).as_posix()
            out[rel] = ("dir",)
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            p = rpath / f
            rel = p.relative_to(base).as_posix()
            try:
                st = p.lstat()
                out[rel] = ("file", st.st_size, st.st_mtime_ns)
            except OSError:
                out[rel] = ("missing",)
    return out


def assert_dir_unchanged(
    testcase: unittest.TestCase | None,
    base: Path,
    before: dict[str, tuple],
    *,
    ignore_prefixes: tuple[str, ...] = (),
    label: str = "",
) -> None:
    """Assert `base` snapshot has zero added, removed, or modified entries."""
    after = dir_snapshot(base)
    if ignore_prefixes:
        before = {k: v for k, v in before.items() if not k.startswith(ignore_prefixes)}
        after = {k: v for k, v in after.items() if not k.startswith(ignore_prefixes)}
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(k for k in before if k in after and after[k] != before[k])
    if added or removed or modified:
        msg = (
            f"Directory mutation detected{' (' + label + ')' if label else ''} at {base}: "
            f"added={added[:10]}, removed={removed[:10]}, modified={modified[:10]}"
        )
        if testcase is not None:
            testcase.fail(msg)
        raise AssertionError(msg)


def _repo_top_snapshot(repo_root: Path = REPO_ROOT, terminal_root: Path = TERMINAL_ROOT) -> dict[str, tuple]:
    """Snapshot top-level entries of repo_root & terminal_root plus layout subdirs."""
    out: dict[str, tuple] = {}
    skip = {".git", ".agents", "__pycache__"}

    def record_dir_top(prefix: str, folder: Path) -> None:
        if not folder.is_dir():
            return
        for entry in sorted(folder.iterdir(), key=lambda p: p.name):
            if entry.name in skip or entry.name.endswith(".pyc"):
                continue
            key = f"{prefix}/{entry.name}"
            try:
                st = entry.lstat()
                if entry.is_dir():
                    out[key] = ("dir", st.st_mtime_ns)
                else:
                    out[key] = ("file", st.st_size, st.st_mtime_ns)
            except OSError:
                out[key] = ("err",)

    record_dir_top("repo", repo_root)
    record_dir_top("terminal", terminal_root)
    for sub in ("configs", "logs", "scripts"):
        sub_path = terminal_root / sub
        if sub_path.is_dir():
            for rel, val in dir_snapshot(sub_path).items():
                out[f"terminal/{sub}/{rel}"] = val
    return out


class WorkspaceSnapshotGuard:
    """Guards the repository root and `terminal/` against accidental mutations."""

    def __init__(self, repo_root: Path = REPO_ROOT, terminal_root: Path = TERMINAL_ROOT):
        self.repo_root = Path(repo_root)
        self.terminal_root = Path(terminal_root)
        self.before = _repo_top_snapshot(self.repo_root, self.terminal_root)

    def assert_unchanged(self, label: str = "") -> None:
        after = _repo_top_snapshot(self.repo_root, self.terminal_root)
        added = sorted(set(after) - set(self.before))
        removed = sorted(set(self.before) - set(after))
        modified = sorted(k for k in self.before if k in after and after[k] != self.before[k])
        if added or removed or modified:
            raise AssertionError(
                f"Workspace mutation detected{' (' + label + ')' if label else ''}: "
                f"added={added}, removed={removed}, modified={modified}"
            )

    def __enter__(self) -> "WorkspaceSnapshotGuard":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.assert_unchanged()


def make_gguf(path: Path, *, architecture: str = "qwen4exp", expert_type: int = 42, include_ple: bool = True) -> Path:
    """Write a minimal valid GGUF v3 file readable by tools/gguf_reader.py."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def string(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    pairs = [
        ("general.architecture", 8, string(architecture)),
        ("general.name", 8, string(f"Test {architecture}")),
        ("general.size_label", 8, string("35B-A3B")),
        (f"{architecture}.context_length", 4, struct.pack("<I", 262144)),
        ("tokenizer.ggml.model", 8, string("gpt2")),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1) + string("a")),
        ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 1) + string("a b")),
    ]
    tensors = [
        ("blk.0.ffn_gate_exps.weight", 64 if expert_type == 42 else 256, expert_type, 0),
    ]
    if include_ple:
        tensors.append(("per_layer_token_embd.weight", 64, 0, 0))
    header = (
        b"GGUF"
        + struct.pack("<IQQ", 3, len(tensors), len(pairs))
        + b"".join(string(k) + struct.pack("<I", kind) + val for k, kind, val in pairs)
    )
    offset = 0
    for name, elements, ttype, _ in tensors:
        header += string(name) + struct.pack("<IQIQ", 1, elements, ttype, offset)
        offset += 256
    pad = b"\x00" * (-len(header) % 32)
    data = header + pad + b"\x00" * max(offset, 256)
    path.write_bytes(data)
    return path


def make_tokenizer(path: Path) -> Path:
    """Write a minimal valid tokenizer directory loadable by strata_tokenizer.Tokenizer.from_dir."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    from strata_tokenizer import BYTE_TO_UNICODE

    tokens = [BYTE_TO_UNICODE[b] for b in range(256)] + [
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
    ]
    vocab = {tok: idx for idx, tok in enumerate(tokens)}
    token_types = [1] * 256 + [3, 3, 4, 4]
    (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (path / "merges.json").write_text("[]", encoding="utf-8")
    (path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (path / "token_type.json").write_text(json.dumps(token_types), encoding="utf-8")
    (path / "tokenizer.json").write_text(
        json.dumps({"pre": "qwen35", "special_ids": {"tokenizer.ggml.eos_token_id": 257}}),
        encoding="utf-8",
    )
    return path


def make_fake_engine(eng_dir: Path, *, exe_name: str | None = None, poly: bool = False, version: str = "0.1.66") -> Path:
    """Write a dummy engine executable and BUILD.json inside `eng_dir`."""
    eng_dir = Path(eng_dir)
    eng_dir.mkdir(parents=True, exist_ok=True)
    if exe_name is None:
        base = "strata-poly" if poly else "strata"
        exe_name = f"{base}.exe" if os.name == "nt" else base
    exe = eng_dir / exe_name
    exe.write_bytes(b"fake-engine-binary")
    try:
        exe.chmod(0o755)
    except OSError:
        pass
    (eng_dir / "BUILD.json").write_text(
        json.dumps({"version": version, "source": "prebuilt", "cuda_dirs": [], "lib_dirs": []}),
        encoding="utf-8",
    )
    return exe


def make_fake_engine_component(
    components_dir: Path,
    *,
    backend: str | None = None,
    version: str = "0.1.66",
) -> tuple[Path, str]:
    """Install a synthetic schema-1 engine component in `components_dir` and return `(folder, backend)`."""
    from strata_control import engine_component as ec

    plat = ec.host_platform()
    if backend is not None:
        chosen_backend = backend
    elif plat == "macos-arm64":
        chosen_backend = "metal"
    elif plat.startswith("macos"):
        chosen_backend = "cpu"
    else:
        chosen_backend = "cuda"

    folder = Path(components_dir) / "engine" / f"{version}-{plat}-{chosen_backend}"
    folder.mkdir(parents=True, exist_ok=True)
    entry = "strata.exe" if os.name == "nt" else "strata"
    entry_path = folder / entry
    raw = b"fake-engine-binary"
    entry_path.write_bytes(raw)
    if os.name != "nt":
        try:
            entry_path.chmod(0o755)
        except OSError:
            pass

    manifest = {
        "schema_version": 1,
        "id": "strata-engine",
        "name": "strata-engine",
        "version": version,
        "platform": plat,
        "backend": chosen_backend,
        "entrypoint": entry,
        "files": [
            {
                "path": entry,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        ],
    }
    (folder / ec.MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return folder, chosen_backend


class _FakeSock:
    """Minimal socket stand-in for in-process `BaseHTTPRequestHandler` execution."""

    def __init__(self, peer=("127.0.0.1", 55000), local=("127.0.0.1", 18080)):
        self._peer = peer
        self._local = local
        self._timeout = 5.0

    def getsockname(self):
        return self._local

    def getpeername(self):
        return self._peer

    def gettimeout(self):
        return self._timeout

    def settimeout(self, t):
        self._timeout = t

    def sendall(self, data):
        return len(data)

    def recv(self, n, flags=0):
        return b""

    def fileno(self):
        # Negative fd triggers OSError/ValueError inside _watch_client select(), making it return cleanly.
        return -1

    def close(self):
        pass


def call_http(svc, method: str, path: str, body=None, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes, object]:
    """Invoke `serve.server.make_handler(svc)` in-process without opening TCP sockets."""
    from serve import server as srv_mod

    H = srv_mod.make_handler(svc)
    raw_body = b""
    hdrs = {"Host": "127.0.0.1:18080"}
    if headers:
        hdrs.update(headers)
    if body is not None:
        if isinstance(body, (dict, list)):
            raw_body = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            raw_body = body.encode("utf-8")
        else:
            raw_body = bytes(body)
        hdrs.setdefault("Content-Length", str(len(raw_body)))

    req_lines = [f"{method.upper()} {path} HTTP/1.0"]
    for k, v in hdrs.items():
        req_lines.append(f"{k}: {v}")
    raw_req = ("\r\n".join(req_lines) + "\r\n\r\n").encode("utf-8") + raw_body

    h = H.__new__(H)
    h.rfile = io.BytesIO(raw_req)
    h.wfile = io.BytesIO()
    h.connection = _FakeSock()
    h.request = h.connection
    h.client_address = ("127.0.0.1", 55000)
    h.server = types.SimpleNamespace(server_name="127.0.0.1", server_port=18080)
    h.close_connection = True
    h.handle_one_request()

    resp_bytes = h.wfile.getvalue()
    head, _, resp_body = resp_bytes.partition(b"\r\n\r\n")
    head_lines = head.split(b"\r\n")
    status_line = head_lines[0].decode("iso-8859-1", "replace") if head_lines else "HTTP/1.0 500"
    parts = status_line.split(" ", 2)
    status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 500
    raw_hdrs = b"\r\n".join(head_lines[1:])
    parsed_hdrs = {k.lower(): v for k, v in BytesParser().parsebytes(raw_hdrs).items()}
    parsed_json = None
    if resp_body:
        try:
            parsed_json = json.loads(resp_body.decode("utf-8"))
        except ValueError:
            parsed_json = None
    return status, parsed_hdrs, resp_body, parsed_json


class _CapturedService(Exception):
    def __init__(self, svc):
        self.svc = svc


class IsolatedWorkspace:
    """Self-contained temporary workspace isolating env vars, settings, profiles, and module ROOTs."""

    def __init__(
        self,
        *,
        suffix_label: str = "e2e",
        repo_layout: bool = False,
        control_mode: bool = False,
        install_as_here: bool = False,
    ):
        self.suffix_label = suffix_label
        self.repo_layout = repo_layout
        self.control_mode = control_mode
        self.install_as_here = install_as_here
        self._stack = contextlib.ExitStack()
        self.tmp: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> "IsolatedWorkspace":
        self.tmp = tempfile.TemporaryDirectory(prefix=f"polystrata-{self.suffix_label}-")
        self._stack.callback(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

        self.user_home = self.base / "home"
        self.roaming = self.base / "roaming"
        self.local = self.base / "local"
        self.xdg_config = self.base / "xdg-config"
        self.xdg_state = self.base / "xdg-state"
        self.xdg_data = self.base / "xdg-data"
        self.xdg_cache = self.base / "xdg-cache"
        self.documents = self.base / "documents"
        self.data = self.base / "PolyStrata-data"
        self.components = self.base / "components"
        self.profiles = self.base / "config" / "profiles"

        if self.repo_layout:
            self.repo = self.base / "repo"
            self.terminal_here = self.repo / "terminal"
            self.install = self.terminal_here
            (self.repo / "engine").mkdir(parents=True)
            (self.repo / "engine" / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
        else:
            self.repo = self.base / "repo"
            self.terminal_here = self.base / "terminal_here"
            self.install = self.terminal_here if self.install_as_here else (self.base / "install")

        for d in (
            self.user_home,
            self.roaming,
            self.local,
            self.xdg_config,
            self.xdg_state,
            self.xdg_data,
            self.xdg_cache,
            self.documents,
            self.data,
            self.components,
            self.profiles,
            self.repo,
            self.terminal_here,
            self.install,
        ):
            d.mkdir(parents=True, exist_ok=True)

        src_data = TERMINAL_ROOT / "data"
        dst_data = self.terminal_here / "data"
        if src_data.is_dir() and not dst_data.exists():
            shutil.copytree(src_data, dst_data)

        self.env = {
            "HOME": str(self.user_home),
            "USERPROFILE": str(self.user_home),
            "APPDATA": str(self.roaming),
            "LOCALAPPDATA": str(self.local),
            "XDG_CONFIG_HOME": str(self.xdg_config),
            "XDG_STATE_HOME": str(self.xdg_state),
            "XDG_DATA_HOME": str(self.xdg_data),
            "XDG_CACHE_HOME": str(self.xdg_cache),
            "XDG_DOCUMENTS_DIR": str(self.documents),
            "STRATA_DOCUMENTS_DIR": str(self.documents),
            "STRATA_INSTALL_DIR": str(self.install),
            "STRATA_DATA_DIR": str(self.data),
            "STRATA_COMPONENTS_DIR": str(self.components),
            "STRATA_PROFILE_DIR": str(self.profiles),
            "STRATA_PRESET_DIR": str(self.base / "config" / "presets"),
            "STRATA_LAUNCH_PRESET_DIR": str(self.base / "config" / "launch-presets"),
            "STRATA_RUNTIME_DIR": str(self.base / "state" / "runtime"),
            "STRATA_TUNE_DIR": str(self.base / "state" / "tuning"),
            "STRATA_TIMESKIP_DIR": str(self.base / "state" / "time-skip"),
            "STRATA_DRAFT_COSTS_DIR": str(self.base / "state" / "draft-costs"),
            "STRATA_LASTRUN": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.env_clear = (
            "STRATA_SETTINGS_FILE",
            "STRATA_CONTROL_INSTALL",
            "STRATA_CONTROL_DIR",
            "STRATA_CONNECTION_DIR",
            "STRATA_ROOT",
            "STRATA_ENGINE_BACKEND",
            "STRATA_REQUIRE_SIGNED_COMPONENTS",
            "STRATA_NEED_ENGINE",
            "STRATA_ALLOW_NO_ENGINE",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "STRATA_API_KEY",
            "STRATA_TELEMETRY_KEY",
        )
        self._stack.enter_context(mock.patch.dict(os.environ, self.env))
        for k in self.env_clear:
            os.environ.pop(k, None)

        self._patch_module_globals()
        return self

    def _patch_module_globals(self) -> None:
        import setup
        import polystrata
        from serve import models as serve_models
        from strata_control import models as ctrl_models
        from strata_control import private_files
        from strata_cli import cli as cli_mod
        from strata_cli.launch import catalog as launch_catalog

        orig_sys_path = list(sys.path)

        def _restore_sys_path() -> None:
            sys.path[:] = orig_sys_path

        self._stack.callback(_restore_sys_path)

        setup_root = self.terminal_here
        control_home = self.install if self.control_mode else None
        serve_models._seen.clear()
        self._stack.callback(serve_models._seen.clear)

        self._stack.enter_context(mock.patch.object(private_files, "adopt_user_defaults", lambda: False))

        self._stack.enter_context(mock.patch.object(setup, "ROOT", setup_root))
        self._stack.enter_context(mock.patch.object(setup, "CONTROL", bool(self.control_mode)))
        self._stack.enter_context(mock.patch.object(setup, "_CONTROL_HOME", control_home))
        self._stack.enter_context(mock.patch.object(setup, "_RUN_CONFIGS", []))
        self._stack.enter_context(mock.patch.object(serve_models, "ROOT", self.install))
        self._stack.enter_context(mock.patch.object(ctrl_models, "ROOT", self.terminal_here))
        self._stack.enter_context(mock.patch.object(polystrata, "ROOT", self.install))
        self._stack.enter_context(mock.patch.object(cli_mod, "ROOT", self.install))
        self._stack.enter_context(mock.patch.object(launch_catalog, "installer_folder", lambda: self.install))

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stack.close()

    @property
    def home_dir(self) -> Path:
        return self.install if self.control_mode else self.terminal_here

    def child_env(self, extra: dict[str, str] | None = None, *, clear: tuple[str, ...] = ()) -> dict[str, str]:
        env = dict(os.environ)
        for k in self.env_clear:
            env.pop(k, None)
        env.update(self.env)
        for k in clear:
            env.pop(k, None)
        if extra:
            env.update(extra)
        return env

    def make_installed_model(
        self,
        tag: str,
        *,
        install_dir: Path | None = None,
        data_dir: Path | None = None,
        layout: str = "configs",
        context: int = 32768,
        backend: str = "cuda",
        architecture: str = "qwen4exp",
        model_name: str | None = None,
        extra_cfg: dict | None = None,
        shared_settings: dict | None = None,
        create_log: bool = False,
        create_script: bool = False,
        write_bak: bool = False,
        filename_override: str | None = None,
    ) -> dict[str, Path]:
        """Create a complete synthetic model install (gguf, pack, tokenizer, engine, config)."""
        home = Path(install_dir or self.install)
        data = Path(data_dir or self.data)
        tag_norm = tag.lower()
        eng_dir = home / "engine"
        exe = make_fake_engine(eng_dir)
        tok_dir = data / "packs" / tag_norm / "tokenizer"
        make_tokenizer(tok_dir)
        pack_dir = data / "packs" / tag_norm
        (pack_dir / "experts.bin").write_bytes(b"experts")
        (pack_dir / "native_experts.txt").write_text("0\n", encoding="utf-8")
        mtp_dir = data / "mtp" / tag_norm
        (mtp_dir / "rt").mkdir(parents=True, exist_ok=True)
        (mtp_dir / "rt" / "experts.bin").write_bytes(b"mtp")
        gguf_dir = data / "models" / tag.upper()
        gguf_path = make_gguf(
            gguf_dir / f"Qwen3.8-Flash-Next-{tag}-00001-of-00001.gguf",
            architecture=architecture,
        )

        if layout == "configs":
            cfg_dir = home / "configs"
            log_file = home / "logs" / f"strata-{tag_norm}.log"
            script_file = home / "scripts" / f"run-{tag_norm}.{'bat' if os.name == 'nt' else 'sh'}"
        elif layout == "legacy":
            cfg_dir = home
            log_file = home / f"strata-{tag_norm}.log"
            script_file = home / f"run-{tag_norm}.{'bat' if os.name == 'nt' else 'sh'}"
        else:
            cfg_dir = Path(layout)
            log_file = cfg_dir / f"{tag_norm}.log"
            script_file = cfg_dir / f"run-{tag_norm}.{'bat' if os.name == 'nt' else 'sh'}"
        cfg_dir.mkdir(parents=True, exist_ok=True)

        cfg_name = filename_override or f"strata-{tag_norm}.json"
        cfg_path = cfg_dir / cfg_name
        cfg = {
            "exe": str(exe),
            "tokenizer": str(tok_dir),
            "cwd": str(home),
            "args": [
                "--native", str(gguf_path),
                "--pack", str(pack_dir),
                "--mtp", str(mtp_dir),
                "--max-context", str(context),
            ],
            "backend": backend,
            "model_name": model_name or tag_norm,
            "port": 8080,
            "log": str(log_file),
        }
        if extra_cfg:
            cfg.update(extra_cfg)
        cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")

        if shared_settings is not None:
            ss_path = Path(str(cfg_path.with_suffix("")) + ".shared-settings.json")
            ss_path.write_text(json.dumps(shared_settings, indent=1), encoding="utf-8")
        if write_bak:
            bak_path = cfg_path.with_name(cfg_path.name + ".bak")
            bak_path.write_bytes(cfg_path.read_bytes())
        if create_log:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text(f"[strata] log for {tag_norm}\n", encoding="utf-8")
        if create_script:
            script_file.parent.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                script_file.write_bytes(
                    f"@echo off\r\ncd /d \"{home}\"\r\n\"{sys.executable}\" \"{home / 'serve' / 'server.py'}\" "
                    f"--engine strata --config \"{cfg_path}\" --port 8080\r\n".encode("utf-8")
                )
            else:
                script_file.write_text(
                    f"#!/bin/sh\ncd \"{home}\"\nexec \"{sys.executable}\" \"{home / 'serve' / 'server.py'}\" "
                    f"--engine strata --config \"{cfg_path}\" --port 8080\n",
                    encoding="utf-8",
                )
                script_file.chmod(0o755)
        return {
            "home": home,
            "data": data,
            "cfg": cfg_path,
            "log": log_file,
            "script": script_file,
            "exe": exe,
            "tok": tok_dir,
            "pack": pack_dir,
            "mtp": mtp_dir,
            "gguf": gguf_path,
        }

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        from strata_cli.cli import main

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as e:
                code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
        return int(code or 0), out.getvalue(), err.getvalue()

    def run_polystrata(self, argv: list[str]) -> tuple[int, str, str]:
        import polystrata

        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["polystrata.py", *argv]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = polystrata.main()
            except SystemExit as e:
                if isinstance(e.code, int):
                    code = e.code
                elif e.code is None:
                    code = 0
                else:
                    err.write(str(e.code) + "\n")
                    code = 1
        return int(code or 0), out.getvalue(), err.getvalue()

    def run_setup(
        self,
        argv: list[str],
        *,
        ram_gb: float = 128.0,
        gpus: list[dict] | None = None,
        mock_move_over: bool = True,
        extra_patches: tuple = (),
    ) -> tuple[int, str, str]:
        """Run `setup.main()` in-process with hardware/network/downloads mocked (never mocking config/script writers)."""
        import setup
        from strata_control import private_files

        t = self.terminal_here
        (t / "data" / "mtp" / "rt").mkdir(parents=True, exist_ok=True)
        if not (t / "data" / "mtp" / "rt" / "experts.bin").exists():
            (t / "data" / "mtp" / "rt" / "experts.bin").write_bytes(b"mtp")
        eng = t / "engine"
        make_fake_engine(eng)
        eng_poly = t / "engine-poly"
        make_fake_engine(eng_poly, poly=True)

        if gpus is None:
            gpus = [{"index": 0, "name": "NVIDIA GeForce RTX 5090", "vram_gb": 31.8, "arch": "120", "driver": "580.97"}]

        class _FakeGGUF:
            def __init__(self, path):
                self.metadata = {
                    "general.architecture": "qwen4exp",
                    "general.size_label": "35B-A3B",
                }
                self.tensors = [
                    types.SimpleNamespace(name="per_layer_token_embd.weight"),
                    types.SimpleNamespace(name="blk.0.ffn_gate_exps.weight"),
                ]

        def _infer_gguf_arch(name: str) -> str:
            low = name.lower()
            if low.startswith("mmproj"):
                return "qwen4exp"
            if low.startswith("flint-"):
                return "flint"
            if low.startswith("gemma"):
                return "gemma4"
            if low.startswith("glm"):
                return "glm5-next"
            if low.startswith("lfm"):
                return "lfm2moe"
            if "inkling" in low:
                return "inkling2_loop"
            if "clover" in low:
                return "clover_moe"
            if (
                low.startswith("qwen3.6")
                or low.startswith("groot")
                or low.startswith("ornith")
                or low.startswith("tiel")
                or low.startswith("minicpm")
            ):
                return "qwen35moe"
            if low.startswith("kolibri"):
                return "kolibri1"
            if low.startswith("deepseek"):
                return "deepseek4"
            if "laguna" in low:
                return "laguna"
            return "qwen4exp"

        def fake_download(url, dst, what=None):
            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.suffix == ".gguf":
                make_gguf(dst, architecture=_infer_gguf_arch(dst.name))
            else:
                dst.write_bytes(b"stub")
            setup.mark(dst)

        def fake_run(cmd, *a, **kw):
            cmd = [str(x) for x in cmd]
            if "--out" in cmd:
                out_dir = Path(cmd[cmd.index("--out") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                make_tokenizer(out_dir / "tokenizer")
                (out_dir / "native_experts.txt").write_text("0\n", encoding="utf-8")
                (out_dir / "index.txt").write_text("0\n", encoding="utf-8")
                (out_dir / "experts.bin").write_bytes(b"experts")
            return types.SimpleNamespace(returncode=0)

        patches = [
            mock.patch.object(setup.sys, "platform", "win32" if setup.WIN else "linux"),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d, *a, **k: (self.data, [])),
            mock.patch.object(setup, "gpus", lambda: gpus),
            mock.patch.object(setup, "card_speeds", lambda: {}),
            mock.patch.object(setup, "amd_gpus", lambda: []),
            mock.patch.object(setup, "ram_gb", lambda: ram_gb),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, False)),
            mock.patch.object(setup, "cpu_cores", lambda: None),
            mock.patch.object(setup, "vision_threads", lambda logical=None: 6),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "is_wsl", lambda: False),
            mock.patch.object(setup, "rotational_disk", lambda p: None),
            mock.patch.object(setup, "ple_ram_platform_ok", lambda: False),
            mock.patch.object(setup, "free_gb", lambda p: 900.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", lambda *a, **k: eng),
            mock.patch.object(setup, "get_poly_prebuilt", lambda *a, **k: eng_poly),
            mock.patch.object(setup, "build_engine_poly", lambda *a, **k: eng_poly),
            mock.patch.object(setup, "build_engine_poly_hip", lambda *a, **k: eng_poly),
            mock.patch.object(setup, "update_installed_engine", lambda *a, **k: None),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda *a, **k: None),
            mock.patch.object(setup, "verify_poly_download", lambda *a, **k: None),
            mock.patch.object(setup, "verify_poly_own", lambda *a, **k: None),
            mock.patch.object(setup, "run", fake_run),
            mock.patch.object(setup, "mtp_corrupt", lambda *a, **k: False),
            mock.patch.object(setup, "refresh_draft_vocab", lambda *a, **k: None),
            mock.patch.object(setup, "saved_calibration", lambda cfg: None),
            mock.patch.object(setup, "calibrate_config", mock.Mock(side_effect=AssertionError("calibrated"))),
            mock.patch.object(private_files, "adopt_user_defaults", lambda: None),
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=_FakeGGUF)}),
            mock.patch.object(sys, "argv", ["setup.py", "--models-dir", str(self.data / "models"), *argv]),
        ]
        if mock_move_over:
            patches.append(mock.patch.object(setup, "move_over_old_layout", lambda: []))
        patches.extend(extra_patches)

        out, err = io.StringIO(), io.StringIO()
        code = 0
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    code = setup.main()
                except SystemExit as e:
                    code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
        return int(code or 0), out.getvalue(), err.getvalue()

    @contextlib.contextmanager
    def start_mock_service(
        self,
        cfg_path: Path | None,
        *,
        extra_args: tuple[str, ...] = (),
        start_mcp: bool = False,
    ):
        """Initialize a real `serve.server.Service` via `server.main(['--engine', 'mock', ...])`."""
        import atexit
        from serve import appendlog, lifeline, mcp as mcp_mod, server as srv_mod

        def fake_serve(svc, host="127.0.0.1", port=8080):
            raise _CapturedService(svc)

        def fake_start_model(cfg, tok, lazy=False):
            if cfg.get("log"):
                fh = appendlog.open_append(cfg["log"])
                fh.close()
            return srv_mod.MockEngine(tok, ["Switched mock reply."], delay_s=0.0), None, {}, None, []

        argv = ["--engine", "mock", "--port", "0"]
        if cfg_path is not None:
            argv += ["--config", str(cfg_path)]
        argv.extend(extra_args)

        svc = None
        out, err = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(srv_mod, "start_model", fake_start_model))
            with contextlib.ExitStack() as call_stack:
                call_stack.enter_context(mock.patch.object(srv_mod, "serve", fake_serve))
                call_stack.enter_context(mock.patch.object(lifeline, "install_stop_handlers", lambda *a, **k: None))
                call_stack.enter_context(mock.patch.object(atexit, "register", lambda *a, **k: None))
                if not start_mcp:
                    call_stack.enter_context(
                        mock.patch.object(mcp_mod.McpHub, "start", lambda self_hub, wait=False: None)
                    )
                call_stack.enter_context(contextlib.redirect_stdout(out))
                call_stack.enter_context(contextlib.redirect_stderr(err))
                try:
                    srv_mod.main(argv)
                except _CapturedService as cap:
                    svc = cap.svc
            if svc is None:
                raise AssertionError(f"server.main({argv!r}) did not reach serve(); stdout={out.getvalue()!r} stderr={err.getvalue()!r}")
            try:
                yield svc
            finally:
                if getattr(svc, "idle_stop", None):
                    svc.idle_stop.set()
                if getattr(svc, "warm_stop", None):
                    svc.warm_stop.set()
                if getattr(svc, "vision_idle_stop", None):
                    svc.vision_idle_stop.set()
                if getattr(svc, "mcp", None) is not None:
                    try:
                        svc.mcp.wait(5.0)
                    except Exception:
                        pass
                    try:
                        svc.mcp.close()
                    except Exception:
                        pass
                if getattr(svc, "keep_awake", None) is not None and hasattr(svc.keep_awake, "close"):
                    try:
                        svc.keep_awake.close()
                    except Exception:
                        pass
                idle_t = getattr(svc, "idle_thread", None)
                if idle_t is not None and idle_t.is_alive():
                    idle_t.join(timeout=2.0)
                if getattr(svc, "engine", None) and hasattr(svc.engine, "close"):
                    try:
                        svc.engine.close()
                    except Exception:
                        pass

    def run_subprocess_to_file(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, str]:
        """Run a subprocess redirecting stdout/stderr to a temp file (avoids CreatePipe WinError 5)."""
        log_path = self.base / f"subproc-{os.getpid()}.out"
        with open(log_path, "wb") as fh:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd or self.base),
                env=env or self.child_env(),
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        output = log_path.read_text(encoding="utf-8", errors="replace")
        log_path.unlink(missing_ok=True)
        return proc.returncode, output
