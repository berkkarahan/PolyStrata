"""Canonical storage layout, discovery, lookup, and atomic CRUD helpers for PolyStrata model configs.

Organizes per-model runtime artifacts under an install home into dedicated subfolders:
- ``<home>/configs/``: ``strata-<tag>.json`` and co-located sidecars (``.shared-settings.json``, ``.bak``, etc.)
- ``<home>/logs/``: ``strata-<tag>.log``
- ``<home>/scripts/``: ``run-<tag>.bat`` / ``run-<tag>.sh``

All discovery and lookup helpers are strictly read-only and never create directories or write files.
Directory creation happens only via ``ensure_layout_dirs()`` or explicit write helpers.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from strata_control import atomic

CONFIGS_DIR = "configs"
LOGS_DIR = "logs"
SCRIPTS_DIR = "scripts"
SHARED_SETTINGS_SUFFIX = ".shared-settings.json"


# ---- Canonical Directory & Path Helpers (pure; NEVER create directories) ----

def configs_dir(home: Path | str) -> Path:
    """Canonical directory for per-model configs and co-located sidecars: ``<home>/configs``."""
    return Path(home) / CONFIGS_DIR


def logs_dir(home: Path | str) -> Path:
    """Canonical directory for per-model and server logs: ``<home>/logs``."""
    return Path(home) / LOGS_DIR


def scripts_dir(home: Path | str) -> Path:
    """Canonical directory for generated startup scripts: ``<home>/scripts``."""
    return Path(home) / SCRIPTS_DIR


def ensure_layout_dirs(home: Path | str) -> tuple[Path, Path, Path]:
    """Create ``configs/``, ``logs/``, and ``scripts/`` under ``home``.

    Only call on explicit write paths — NEVER during read-only discovery.
    """
    c, l, s = configs_dir(home), logs_dir(home), scripts_dir(home)
    for d in (c, l, s):
        d.mkdir(parents=True, exist_ok=True)
    return c, l, s


def normalize_tag(tag: str) -> str:
    """Normalize a model tag (strip leading ``strata-`` if passed, lowercase)."""
    if not isinstance(tag, str):
        raise ValueError(f"invalid model tag: {tag!r}")
    t = tag.strip()
    if t.lower().startswith("strata-"):
        t = t[len("strata-"):]
    t = t.lower()
    if (
        not t
        or "/" in t
        or "\\" in t
        or ":" in t
        or any(ord(c) < 32 or ord(c) == 127 for c in t)
        or t in (".", "..")
        or ".." in t
        or t.endswith(".")
        or t.endswith(".json")
        or t.endswith(".shared-settings")
    ):
        raise ValueError(f"invalid model tag: {tag!r}")
    return t


def config_path(home: Path | str, tag: str) -> Path:
    """Canonical config path: ``<home>/configs/strata-<tag.lower()>.json``."""
    return configs_dir(home) / f"strata-{normalize_tag(tag)}.json"


def shared_settings_path(cfg_path: Path | str) -> Path:
    """Co-located shared settings path: ``<stem>.shared-settings.json`` beside ``cfg_path``."""
    p = Path(cfg_path)
    return p.with_name(p.stem + SHARED_SETTINGS_SUFFIX)


def backup_path(cfg_path: Path | str) -> Path:
    """Co-located backup path: ``<filename>.bak`` beside ``cfg_path``."""
    p = Path(cfg_path)
    return p.with_name(p.name + ".bak")


def log_path(home: Path | str, tag: str) -> Path:
    """Canonical log path: ``<home>/logs/strata-<tag.lower()>.log``."""
    return logs_dir(home) / f"strata-{normalize_tag(tag)}.log"


def script_path(home: Path | str, tag: str, *, windows: bool | None = None) -> Path:
    """Canonical startup script path: ``<home>/scripts/run-<tag.lower()>.{bat|sh}``."""
    is_win = (os.name == "nt") if windows is None else windows
    ext = "bat" if is_win else "sh"
    return scripts_dir(home) / f"run-{normalize_tag(tag)}.{ext}"


def install_home_for_config(cfg_path: Path | str) -> Path:
    """Resolve install home from ``<home>/configs/strata-*.json`` or legacy ``<home>/strata-*.json``."""
    parent = Path(cfg_path).resolve().parent
    return parent.parent if parent.name.lower() == CONFIGS_DIR else parent


# ---- Read-Only Config Identification, Discovery & Lookup ----

def is_config_file(path: Path | str) -> bool:
    """True if filename matches ``strata-<tag>.json`` (case-insensitive) and is not ``*.shared-settings.json``.

    Does not stat or follow symlinks.
    """
    n = Path(path).name.lower()
    return (
        n.startswith("strata-")
        and n.endswith(".json")
        and not n.endswith(SHARED_SETTINGS_SUFFIX)
        and len(n) > len("strata-.json")
    )


def tag_from_config(path: Path | str) -> str | None:
    """Extract ``<tag>`` from ``strata-<tag>.json``, or None if not a config filename."""
    p = Path(path)
    if not is_config_file(p):
        return None
    return p.name[len("strata-"):-len(".json")]


def _is_non_dir_entry(path: Path) -> bool:
    try:
        return path.is_symlink() or not path.is_dir()
    except OSError:
        return False


def iter_config_files(home: Path | str, *, include_legacy: bool = True) -> list[Path]:
    """Read-only discovery of ``strata-*.json`` under ``configs_dir(home)``, then optional legacy ``<home>/``.

    Uses ``iterdir()`` so symlinks are preserved for downstream validation (e.g. ``models._read()``).
    Plain directories named ``strata-*.json`` are ignored.
    Deduplicates by case-insensitive filename, preferring ``configs_dir(home)`` over legacy ``<home>/``.
    NEVER creates directories or writes files.
    """
    base = Path(home)
    search_dirs = [configs_dir(base)]
    if include_legacy:
        search_dirs.append(base)
    out: list[Path] = []
    seen: set[str] = set()
    for d in search_dirs:
        try:
            if not d.is_dir():
                continue
            entries = sorted(
                (p for p in d.iterdir() if is_config_file(p) and _is_non_dir_entry(p)),
                key=lambda p: p.name.lower(),
            )
        except OSError:
            continue
        for p in entries:
            key = p.name.lower()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def find_config(home: Path | str, name_or_tag: str, *, include_legacy: bool = True) -> Path | None:
    """Find a config in ``home`` by tag, stem (``strata-<tag>``), or filename (``strata-<tag>.json``), case-insensitive.

    Prefers ``configs_dir(home)`` over legacy ``<home>/``. Read-only.
    """
    raw = str(name_or_tag or "").strip()
    if not raw:
        return None
    low = raw.lower()
    want_tag = low[len("strata-"):] if low.startswith("strata-") else low
    if want_tag.endswith(".json"):
        want_tag = want_tag[:-len(".json")]
    for p in iter_config_files(home, include_legacy=include_legacy):
        t = (tag_from_config(p) or "").lower()
        if low in (t, p.stem.lower(), p.name.lower()) or (want_tag and want_tag == t):
            return p
    return None


def resolve_config_ref(home: Path | str, ref: str | Path, *, include_legacy: bool = True) -> Path | None:
    """Resolve an explicit file path OR a model tag/name under ``home`` (read-only)."""
    if not ref:
        return None
    p = Path(ref)
    try:
        if p.is_file() and not p.name.lower().endswith(SHARED_SETTINGS_SUFFIX):
            return p
    except OSError:
        pass
    ref_str = str(ref)
    if not p.is_absolute() and not p.anchor and ("/" in ref_str or "\\" in ref_str):
        home_p = Path(home)
        rel_candidates = [p]
        if "\\" in ref_str:
            rel_candidates.append(Path(ref_str.replace("\\", "/")))
        try:
            home_resolved = home_p.resolve()
        except OSError:
            home_resolved = None
        for rel in rel_candidates:
            cand = home_p / rel
            try:
                if cand.is_file() and not cand.name.lower().endswith(SHARED_SETTINGS_SUFFIX):
                    if home_resolved is not None:
                        cand_resolved = cand.resolve()
                        if cand_resolved != home_resolved and home_resolved not in cand_resolved.parents:
                            continue
                        if not include_legacy and cand_resolved.parent == home_resolved and is_config_file(cand_resolved):
                            continue
                    return cand
            except OSError:
                pass
    return find_config(home, ref_str, include_legacy=include_legacy)


# ---- Atomic JSON & Config CRUD Helpers ----

def read_json(path: Path | str, *, default: Any = None) -> Any:
    """Read UTF-8 (with optional BOM) JSON cleanly; return ``default`` on OSError/UnicodeError/ValueError."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError):
        return default


def atomic_write_bytes(path: Path | str, payload: bytes, *, backup: bool = False) -> Path | None:
    """Atomically write ``payload`` to ``path`` using mkstemp + fsync + ``atomic.replace``.

    Returns the backup ``Path`` if created, else ``None``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    bak: Path | None = None
    if backup and target.is_file():
        bak = backup_path(target)
        shutil.copyfile(target, bak)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        atomic.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return bak


def atomic_write_json(
    path: Path | str,
    data: dict[str, Any],
    *,
    indent: int = 1,
    ensure_ascii: bool = False,
    trailing_newline: bool = False,
    backup: bool = False,
) -> Path | None:
    """Serialize ``data`` as JSON and write atomically."""
    text = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii) + ("\n" if trailing_newline else "")
    return atomic_write_bytes(path, text.encode("utf-8"), backup=backup)


def write_model_config(home: Path | str, tag: str, cfg: dict[str, Any], *, backup: bool = True) -> Path:
    """Write model config to canonical ``<home>/configs/strata-<tag>.json`` atomically."""
    target = config_path(home, tag)
    atomic_write_json(target, cfg, indent=1, backup=backup)
    return target


def delete_model_config(
    home: Path | str,
    tag: str,
    *,
    include_legacy: bool = False,
    remove_sidecars: bool = False,
) -> list[Path]:
    """Delete matching config (and optionally ``.bak`` / ``.shared-settings.json``) under ``home``.

    Returns list of removed paths.
    """
    canon = normalize_tag(tag)
    removed: list[Path] = []
    targets = [config_path(home, canon)]
    if include_legacy:
        targets.append(Path(home) / f"strata-{canon}.json")
    for cfg in targets:
        candidates = [cfg]
        if remove_sidecars:
            candidates.extend([backup_path(cfg), shared_settings_path(cfg)])
        for p in candidates:
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                    removed.append(p)
            except OSError:
                pass
    return removed


# ---- Legacy Layout Artifact Classifiers & Idempotent Migration ----

def is_shared_settings_file(path: Path | str) -> bool:
    """True if filename matches ``<stem>.shared-settings.json`` (non-empty stem)."""
    n = Path(path).name.lower()
    return n.endswith(SHARED_SETTINGS_SUFFIX) and len(n) > len(SHARED_SETTINGS_SUFFIX)


def is_backup_file(path: Path | str) -> bool:
    """True if filename matches ``strata-<tag>.json.bak``."""
    n = Path(path).name.lower()
    return n.startswith("strata-") and n.endswith(".json.bak") and len(n) > len("strata-.json.bak")


def is_warm_start_file(path: Path | str) -> bool:
    """True if filename matches ``strata-<tag>.warm-start*`` (excluding incomplete ``.tmp``)."""
    n = Path(path).name.lower()
    return (
        n.startswith("strata-")
        and (n.endswith(".warm-start") or ".warm-start-" in n)
        and not n.endswith(".tmp")
    )


def is_tool_media_dir(path: Path | str) -> bool:
    """True if name matches ``strata-<tag>.tool-media``."""
    n = Path(path).name.lower()
    return n.startswith("strata-") and n.endswith(".tool-media") and len(n) > len("strata-.tool-media")


def is_models_overlay_file(path: Path | str) -> bool:
    """True if filename is ``models.json``."""
    return Path(path).name == "models.json"


def is_log_file(path: Path | str) -> bool:
    """True if filename matches ``strata-<tag>.log``."""
    n = Path(path).name.lower()
    return n.startswith("strata-") and n.endswith(".log") and len(n) > len("strata-.log")


def is_run_script_file(path: Path | str) -> bool:
    """True if filename matches ``run-<tag>.bat`` or ``run-<tag>.sh``."""
    n = Path(path).name.lower()
    return (
        n.startswith("run-")
        and (n.endswith(".bat") or n.endswith(".sh"))
        and len(n) > len("run-.sh")
    )


def legacy_artifact_target(install_home: Path | str, entry: Path | str) -> Path | None:
    """Return canonical destination under ``install_home`` for a legacy flat artifact, or ``None``."""
    p = Path(entry)
    if (
        is_config_file(p)
        or is_shared_settings_file(p)
        or is_backup_file(p)
        or is_warm_start_file(p)
        or is_tool_media_dir(p)
        or is_models_overlay_file(p)
    ):
        return configs_dir(install_home) / p.name
    if is_log_file(p):
        return logs_dir(install_home) / p.name
    if is_run_script_file(p):
        return scripts_dir(install_home) / p.name
    return None


def _rebase_profiles(moved_cfg_map: dict[str, Path], *, warn: Callable[[str], Any] | None = None) -> list[str]:
    """Lazily update saved profiles whose ``source_config`` points at a migrated config."""
    if not moved_cfg_map:
        return []
    try:
        from strata_control import profiles
        pdir = profiles.profile_dir()
        if not pdir.is_dir():
            return []
        entries = sorted(pdir.iterdir(), key=lambda p: p.name.lower())
    except (ImportError, OSError):
        return []
    updated: list[str] = []
    for entry in entries:
        if entry.suffix != ".json" or entry.name.startswith(".") or entry.is_symlink():
            continue
        name = entry.stem
        try:
            prof, _rev = profiles.load(name)
        except (profiles.ProfileError, OSError):
            continue
        src_cfg = prof.get("source_config")
        if not isinstance(src_cfg, str) or not src_cfg:
            continue
        keys = [os.path.normcase(os.path.abspath(src_cfg))]
        try:
            keys.append(os.path.normcase(str(Path(src_cfg).resolve())))
        except OSError:
            pass
        new_dst = next((moved_cfg_map[k] for k in keys if k in moved_cfg_map), None)
        if new_dst is None or not new_dst.is_file():
            continue
        try:
            proposed, rev = profiles.propose_source(name, str(new_dst))
            profiles.save(name, proposed, expected_revision=rev)
            updated.append(name)
        except (profiles.ProfileError, OSError) as exc:
            if warn is not None:
                warn(f"profile {name}: could not update source_config: {exc}")
    return updated


def migrate_legacy_layout(
    install_home: Path | str,
    *,
    source_dir: Path | str | None = None,
    extra_swaps: Sequence[tuple[Path | str, Path | str]] = (),
    exact_swaps: Sequence[tuple[Path | str, Path | str]] = (),
    dry_run: bool = False,
    warn: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Idempotently migrate legacy per-model files from ``source_dir`` (or ``install_home``) into ``configs/``, ``logs/``, ``scripts/``.

    - Moves only via ``os.rename()`` (never copies or clobbers collisions).
    - Skips symlinked sources, symlinked destination subdirs, and any existing destination.
    - Rebases moved valid ``strata-*.json`` configs (preserving corrupted JSON and unknown keys),
      default ``log`` paths (even if the ``.log`` file was not yet created), moved ``run-*.{bat,sh}``
      scripts (preserving ``st_mode`` and exact line endings), and saved ``profiles`` ``source_config`` refs.
    """
    dst_home = Path(install_home)
    src_root = Path(source_dir) if source_dir is not None else dst_home
    moved: list[tuple[Path, Path]] = []
    skipped: list[tuple[Path, Path, str]] = []
    errors: list[str] = []
    warnings: list[str] = []

    def _warn(msg: str) -> None:
        warnings.append(msg)
        if warn is not None:
            warn(msg)

    if dst_home.is_symlink():
        msg = f"{dst_home}: symlinked install home skipped"
        _warn(msg)
        errors.append(msg)
        return {"moved": moved, "skipped": skipped, "errors": errors, "warnings": warnings, "profiles_updated": [], "dry_run": dry_run}
    if src_root.is_symlink():
        msg = f"{src_root}: symlinked source directory skipped"
        _warn(msg)
        errors.append(msg)
        return {"moved": moved, "skipped": skipped, "errors": errors, "warnings": warnings, "profiles_updated": [], "dry_run": dry_run}
    if not src_root.is_dir():
        return {"moved": moved, "skipped": skipped, "errors": errors, "warnings": warnings, "profiles_updated": [], "dry_run": dry_run}

    try:
        entries = sorted(src_root.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        msg = f"{src_root}: could not scan directory: {exc}"
        _warn(msg)
        errors.append(msg)
        return {"moved": moved, "skipped": skipped, "errors": errors, "warnings": warnings, "profiles_updated": [], "dry_run": dry_run}

    planned: list[tuple[Path, Path]] = []
    for p in entries:
        dst = legacy_artifact_target(dst_home, p)
        if dst is None:
            continue
        if p.is_symlink():
            skipped.append((p, dst, "symlink"))
            _warn(f"{p} skipped: symlink")
            continue
        try:
            if is_tool_media_dir(p):
                if not p.is_dir():
                    continue
            elif not p.is_file():
                continue
        except OSError:
            continue
        if dst.parent.is_symlink():
            skipped.append((p, dst, "symlink_dest_dir"))
            _warn(f"{dst.parent} is a symlink; skipping {p}")
            continue
        if dst.exists() or dst.is_symlink():
            skipped.append((p, dst, "collision"))
            continue
        if (
            is_models_overlay_file(p)
            and os.path.normcase(os.path.abspath(src_root)) != os.path.normcase(os.path.abspath(dst_home))
            and ((dst_home / p.name).exists() or (dst_home / p.name).is_symlink())
        ):
            skipped.append((p, dst, "collision"))
            continue
        if os.path.abspath(p) == os.path.abspath(dst):
            continue
        planned.append((p, dst))

    if dry_run:
        return {
            "moved": planned,
            "skipped": skipped,
            "errors": errors,
            "warnings": warnings,
            "profiles_updated": [],
            "dry_run": True,
        }

    for src, dst in planned:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
            moved.append((src, dst))
        except OSError as exc:
            msg = f"{src} could not be moved to {dst}: {exc}"
            errors.append(msg)
            _warn(msg)

    if not moved:
        return {
            "moved": moved,
            "skipped": skipped,
            "errors": errors,
            "warnings": warnings,
            "profiles_updated": [],
            "dry_run": False,
        }

    def _expand_swaps(items: Sequence[tuple[Path | str, Path | str]]) -> list[tuple[str, str]]:
        raw: list[tuple[str, str]] = []
        for a, b in items:
            sa, sb = str(a), str(b)
            raw.append((sa, sb))
            if not os.path.isabs(sa):
                raw.append((os.path.abspath(sa), os.path.abspath(sb)))
        return raw

    prefix_raw = _expand_swaps([*moved, *extra_swaps])
    exact_raw = _expand_swaps(exact_swaps)

    def _norm_key(s: str) -> str:
        return s.replace("\\", "/").lower() if os.name == "nt" else s

    def _escape_path(s: str) -> str:
        if os.name == "nt":
            return "".join(r"[\\/]" if ch in ("\\", "/") else re.escape(ch) for ch in s)
        return re.escape(s)

    patterns: list[tuple[str, str, bool]] = []
    seen_pat: set[tuple[str, bool]] = set()
    for is_prefix, pairs in ((True, prefix_raw), (False, exact_raw)):
        for a, b in pairs:
            variants = ((a, b),) if os.name == "nt" else ((a, b), (a.replace("\\", "/"), b.replace("\\", "/")))
            for sa, sb in variants:
                if not sa:
                    continue
                k = (_norm_key(sa), is_prefix)
                if k not in seen_pat:
                    seen_pat.add(k)
                    patterns.append((sa, sb, is_prefix))
    patterns.sort(key=lambda item: (-len(item[0]), 0 if item[2] else 1))

    repl_map: dict[str, tuple[str, str]] = {}
    regex_branches: list[str] = []
    for sa, sb, is_prefix in patterns:
        nk = _norm_key(sa)
        repl_map.setdefault(nk, (sb, sb.replace("\\", "/")))
        trail = r"[\\/\"'\s;,]" if is_prefix else r"[\"\'\s;,]"
        regex_branches.append(f"{_escape_path(sa)}(?=$|{trail})")
    rebase_re = (
        re.compile(
            r"(?<![^\s\"'=;,])(?:" + "|".join(regex_branches) + r")",
            re.IGNORECASE if os.name == "nt" else 0,
        )
        if regex_branches
        else None
    )

    def _pick_repl(m: re.Match[str]) -> str:
        matched = m.group(0)
        sb_native, sb_fwd = repl_map[_norm_key(matched)]
        if os.name == "nt" and "/" in matched and "\\" not in matched:
            return sb_fwd
        return sb_native

    def rebase(text: str) -> str:
        if rebase_re is None:
            return text
        return rebase_re.sub(_pick_repl, text)

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return rebase(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    src_root_norm = os.path.normcase(os.path.abspath(src_root))
    dst_home_norm = os.path.normcase(os.path.abspath(dst_home))
    moved_cfg_map: dict[str, Path] = {}

    for src, dst in moved:
        if is_config_file(dst):
            try:
                moved_cfg_map[os.path.normcase(os.path.abspath(src))] = dst.resolve()
                moved_cfg_map[os.path.normcase(str(src.resolve()))] = dst.resolve()
            except OSError:
                moved_cfg_map[os.path.normcase(str(src))] = dst
            try:
                data = json.loads(dst.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError, ValueError) as exc:
                _warn(f"{dst}: the paths in it could not be updated: {exc}")
                continue
            if not isinstance(data, dict):
                continue
            new_data = walk(data)
            raw_log = new_data.get("log")
            if isinstance(raw_log, str) and raw_log.strip():
                log_p = Path(raw_log)
                is_bare = (not log_p.is_absolute()) and log_p.parent == Path(".")
                same_root = False
                if not is_bare:
                    try:
                        log_parent_norm = os.path.normcase(os.path.abspath(log_p.parent))
                        same_root = log_parent_norm in (src_root_norm, dst_home_norm)
                    except OSError:
                        pass
                if is_bare or same_root:
                    new_data["log"] = str(logs_dir(dst_home) / log_p.name)
            raw_ws = new_data.get("warm_start")
            if isinstance(raw_ws, str) and raw_ws.strip():
                ws_p = Path(raw_ws)
                if not ws_p.is_absolute() and not ws_p.anchor:
                    try:
                        dst_cand = dst.parent / ws_p
                        if not dst_cand.exists():
                            src_resolved = src_root.resolve()
                            old_seed = (src.parent / ws_p).resolve()
                            if old_seed.is_file() and (old_seed == src_resolved or src_resolved in old_seed.parents):
                                new_data["warm_start"] = Path(
                                    os.path.relpath(old_seed, dst.parent.resolve())
                                ).as_posix()
                    except OSError:
                        pass
            if new_data != data:
                try:
                    atomic_write_json(dst, new_data, indent=2, trailing_newline=True, backup=False)
                except OSError as exc:
                    msg = f"{dst}: the paths in it could not be updated: {exc}"
                    errors.append(msg)
                    _warn(msg)
        elif is_run_script_file(dst):
            try:
                mode = dst.stat().st_mode
                old_text = dst.read_bytes().decode("utf-8", "surrogateescape")
                new_text = rebase(old_text)
                if new_text != old_text:
                    atomic_write_bytes(dst, new_text.encode("utf-8", "surrogateescape"), backup=False)
                    try:
                        os.chmod(dst, stat.S_IMODE(mode))
                    except OSError:
                        pass
            except OSError as exc:
                msg = f"{dst}: the paths in it could not be updated: {exc}"
                errors.append(msg)
                _warn(msg)

    profiles_updated = _rebase_profiles(moved_cfg_map, warn=_warn)
    return {
        "moved": moved,
        "skipped": skipped,
        "errors": errors,
        "warnings": warnings,
        "profiles_updated": profiles_updated,
        "dry_run": False,
    }
