"""serve/models.py - the models this installation has, and what each one is.

A model is a run config: a strata-<name>.json that setup wrote, with the engine, its arguments and the tokenizer.
The registry lists them.  What a model is comes from its own file where it has one (a GGUF says its architecture,
its size, its context and whether it carries a draft block), and models.json in the same folder may add to that
or correct it: an object of model names, each with any of the keys of an entry.

    {"qwen3.6-35b-a3b": {"display_name": "Qwen3.6 35B-A3B", "notes": "the daily one"}}

Nothing here starts or stops an engine: that is the server's (Service.switch_model).
"""
from __future__ import annotations
import sys
if sys.platform == "win32":
    # CPython 3.12's WMI helper (_wmi, behind platform.system() / uname()) takes an access violation in a worker thread when WMI refuses the
    # account (0x80041003) and the caller has already gone (DIAG-WINHOST-0216). Without _wmi, platform uses the registry and the environment.
    sys.modules["_wmi"] = None

import json
import os
import posixpath
import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from strata_control import layout  # noqa: E402

MAX_CONFIG_BYTES = 1024 * 1024

# what an architecture is run by, and what the registry calls it
BACKENDS = {
    "glm5-next": "glm5-next", "glm5next": "glm5-next",
    "qwen35moe": "qwen35moe",
    "clover_moe": "clover_moe",
    "gemma4": "gemma4",
    "deepseek4": "deepseek4",
    "lfm2moe": "lfm2moe",
    "flint": "flint",
    "kolibri1": "kolibri1",
    "laguna": "laguna",
    "inkling2_loop": "inkling2_loop",
    "inkling2": "inkling2_loop",   # Unsloth's top-level files: the same engine (inkling2.cpp)
}
# a routed expert's tensors: they stay in the file and are held in VRAM as far as it has room
EXPERT = re.compile(r"\.ffn_(gate|up|down|gate_up)_exps\.weight$")
_seen: dict = {}   # (path, size, mtime) -> what the file says


def _arg(args: list, flag: str):
    """The value after `flag` in an engine's argument list, or None."""
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def _gguf(path: Path) -> dict:
    """What a GGUF says of itself: read once per file as it is now."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in _seen:
        return _seen[key]
    from gguf_reader import GGUFFile
    g = GGUFFile(path)
    m = g.metadata
    arch = str(m.get("general.architecture") or "")
    dense = experts = 0
    for t in g.tensors:
        n = t.expected_bytes() or 0
        if EXPERT.search(t.name):
            experts += n
        else:
            dense += n
    # a split file's other parts hold most of the experts: the first part's sizes are then a lower bound
    parts = int(m.get("split.count") or 1)
    out = {
        "backend": BACKENDS.get(arch, arch or "unknown"),
        "runnable": arch in BACKENDS,
        "display_name": m.get("general.name") or m.get("general.basename"),
        "size_label": m.get("general.size_label"),
        "context_trained": m.get(arch + ".context_length"),
        "layers": m.get(arch + ".block_count"),
        "experts": m.get(arch + ".expert_count"),
        "experts_used": m.get(arch + ".expert_used_count"),
        "mtp": bool(m.get(arch + ".nextn_predict_layers")),
        "dense_bytes": dense, "expert_bytes": experts, "parts": parts,
        "types": g.by_type(),
    }
    _seen[key] = out
    return out


def _params(label) -> tuple[str | None, str | None]:
    """("35B", "3B") from a size label like "35B-A3B": all the parameters, and the ones a token uses."""
    m = re.match(r"^(\d+(?:\.\d+)?[BMK])(?:-A(\d+(?:\.\d+)?[BMK]))?", str(label or ""), re.I)
    return (m.group(1), m.group(2)) if m else (None, None)


def _quant(name: str, types: dict) -> str | None:
    """The quantization a file's name states (UD-Q3_K_M, 3.0bit, ...), or else its most common tensor type."""
    m = re.search(r"((?:UD-)?I?Q\d[A-Z0-9_]*|\d(?:\.\d)?bit[\w-]*|APEX[\w-]*|BF16|F16)", name, re.I)
    if m:
        return m.group(1)
    quantized = [t for t in types if t not in ("F32", "F16", "BF16")]
    return quantized[0] if quantized else None


def _aliases(value) -> list[str]:
    """The config's other names for the model (#297: a list, or one comma-separated string); [] for anything else."""
    if isinstance(value, str):
        value = value.split(",")
    return [x.strip() for x in value if isinstance(x, str) and x.strip()] if isinstance(value, list) else []


def _read_bounded_json(path: Path):
    """Read a regular, non-symlink UTF-8 (or BOM) JSON file up to MAX_CONFIG_BYTES, or None on error."""
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            details = os.fstat(stream.fileno())
            if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_CONFIG_BYTES:
                return None
            raw = stream.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            return None
        return json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, ValueError):
        return None


def describe(config: Path) -> dict | None:
    """The registry's entry for a run config, or None when the file is not one."""
    cfg = _read_bounded_json(Path(config))
    if not isinstance(cfg, dict):
        return None
    exe_val = cfg.get("exe")
    args_val = cfg.get("args")
    cwd_val = cfg.get("cwd")
    tok_val = cfg.get("tokenizer")
    if (
        not isinstance(exe_val, str)
        or not exe_val
        or not isinstance(args_val, list)
        or not all(isinstance(a, str) for a in args_val)
        or (cwd_val is not None and not isinstance(cwd_val, str))
        or (tok_val is not None and not isinstance(tok_val, str))
    ):
        return None
    args = list(args_val)
    cwd = Path(cwd_val or ROOT)
    model = _arg(args, "--model")
    path = (cwd / model) if model and not Path(model).is_absolute() else Path(model) if model else None
    exe = Path(exe_val) if Path(exe_val).is_absolute() else cwd / exe_val
    ctx = _arg(args, "--max-context") or _arg(args, "--ctx")
    model_name = cfg.get("model_name")
    entry_id = model_name if isinstance(model_name, str) and model_name else Path(config).stem.replace("strata-", "", 1)
    e = {
        "id": entry_id,
        "config": str(config),
        "display_name": None, "backend": "qwen4exp", "path": str(path) if path else None,
        "quantization": None, "params_total": None, "params_active": None,
        "context": int(ctx) if isinstance(ctx, str) and 1 <= len(ctx) <= 8 and ctx.isascii() and ctx.isdecimal() else None,
        "multimodal": bool(cfg.get("vision")),
        "aliases": _aliases(cfg.get("aliases")),
        "mtp": bool(_arg(args, "--draft-model")) and "--no-draft" not in args,
        "ram_gb": None, "vram_gb": None,
        "status": "available", "problem": None,
    }
    tok = Path(tok_val or "")
    try:
        if not exe.exists():
            e["status"], e["problem"] = "unavailable", f"the engine is missing ({exe})"
        elif not (tok / "vocab.json").exists():
            e["status"], e["problem"] = "unavailable", f"the tokenizer is missing ({tok})"
    except (OSError, ValueError) as err:
        e["status"], e["problem"] = "unavailable", f"the config path is invalid ({err})"
    if path is not None and path.suffix == ".gguf":
        try:
            path_exists = path.exists()
        except (OSError, ValueError) as err:
            e["status"], e["problem"] = "unavailable", f"the model file is invalid ({err})"
            path_exists = False
        if not path_exists:
            if e["status"] == "available":
                e["status"], e["problem"] = "unavailable", f"the model file is missing ({path})"
        else:
            try:
                g = _gguf(path)
            except (OSError, ValueError, KeyError) as err:
                e["status"], e["problem"] = "unavailable", f"the model file cannot be read ({err})"
            else:
                e["backend"] = g["backend"]
                e["display_name"] = g["display_name"]
                e["params_total"], e["params_active"] = _params(g["size_label"])
                e["quantization"] = _quant(path.name, g["types"])
                e["mtp"] = (e["mtp"] or g["mtp"]) and "--no-draft" not in args
                if g["parts"] == 1:
                    # the file is read in place: all of it is file cache at most, and VRAM holds the dense
                    # weights whole and the experts as far as the card has room
                    e["ram_gb"] = round((g["dense_bytes"] + g["expert_bytes"]) / 1e9, 1)
                    e["vram_gb"] = {"least": round(g["dense_bytes"] / 1e9 + 1.0, 1),
                                    "all": round((g["dense_bytes"] + g["expert_bytes"]) / 1e9 + 1.0, 1)}
                if not g["runnable"]:
                    e["status"], e["problem"] = "unavailable", f"no engine here runs a '{g['backend']}' model"
    # the name setup gave it, else the file's own (a file's can be a converter's leftover), else the id
    disp = cfg.get("display_name")
    e["display_name"] = (disp if isinstance(disp, str) and disp else None) or e["display_name"] or e["id"]
    return e


def _mtime(path: Path) -> float:
    try:
        return path.lstat().st_mtime
    except OSError:
        return 0.0


def _read_models_overlay(path: Path) -> dict:
    data = _read_bounded_json(path)
    return data if isinstance(data, dict) else {}


def discover(root: Path = ROOT) -> list[dict]:
    """Every model of the installation, by name.  Two configs of one name: the newer file is the model."""
    root = Path(root)
    overlay_paths: list[Path] = []
    if root.name.lower() == layout.CONFIGS_DIR:
        overlay_paths.append(root.parent / "models.json")
    overlay_paths.append(root / "models.json")
    if root.name.lower() != layout.CONFIGS_DIR:
        overlay_paths.append(layout.configs_dir(root) / "models.json")
    extra: dict = {}
    for overlay_path in overlay_paths:
        layer = _read_models_overlay(overlay_path)
        for k, v in layer.items():
            if isinstance(v, dict):
                cur = extra.get(k)
                extra[k] = {**cur, **v} if isinstance(cur, dict) else dict(v)
    out: dict[str, dict] = {}
    candidates = sorted(
        layout.iter_config_files(root),
        key=lambda p: (_mtime(p), 1 if p.parent.name.lower() == layout.CONFIGS_DIR else 0),
        reverse=True,
    )
    for p in candidates:
        e = describe(p)
        if e is None or e["id"] in out:
            continue
        more = extra.get(e["id"])
        if isinstance(more, dict):
            e.update({k: v for k, v in more.items() if k not in ("id", "config", "status", "problem")})
            if "aliases" in more:
                e["aliases"] = _aliases(more["aliases"])
        out[e["id"]] = e
    return sorted(out.values(), key=lambda e: e["id"])


def find(models: list[dict], name: str) -> dict | None:
    """A model by its name, by a config file's name/tag/path, or by the one name that contains what was typed."""
    name = name.strip()
    low = name.lower()
    if not low:
        return None
    if "/" in name or "\\" in name:
        path_matches = []
        p_in = Path(name)
        try:
            resolved_in = str(p_in.resolve()).lower()
        except OSError:
            resolved_in = os.path.abspath(name).lower()
        is_rel = not p_in.is_absolute() and not p_in.anchor
        norm_rel = posixpath.normpath(name.replace("\\", "/")) if is_rel else ""
        escapes_root = is_rel and (norm_rel == ".." or norm_rel.startswith("../"))
        rel_parts = (
            tuple(part.lower() for part in norm_rel.split("/") if part not in ("", "."))
            if is_rel and not escapes_root
            else ()
        )
        for e in models:
            cfg_p = Path(e["config"])
            try:
                resolved_cfg = str(cfg_p.resolve()).lower()
            except OSError:
                resolved_cfg = os.path.abspath(e["config"]).lower()
            if resolved_cfg == resolved_in:
                path_matches.append(e)
            elif len(rel_parts) >= 2:
                cfg_parts = tuple(part.lower() for part in cfg_p.parts)
                if len(cfg_parts) >= len(rel_parts) and cfg_parts[-len(rel_parts):] == rel_parts:
                    path_matches.append(e)
            elif len(rel_parts) == 1:
                if cfg_p.parent.name.lower() != layout.CONFIGS_DIR and cfg_p.name.lower() == rel_parts[0]:
                    path_matches.append(e)
        if len(path_matches) == 1:
            return path_matches[0]
        if len(path_matches) > 1:
            return None
    id_matches = [e for e in models if e["id"].lower() == low]
    if len(id_matches) == 1:
        return id_matches[0]
    if len(id_matches) > 1:
        return None
    cfg_matches = [
        e for e in models
        if low in (Path(e["config"]).name.lower(), Path(e["config"]).stem.lower())
    ]
    if len(cfg_matches) == 1:
        return cfg_matches[0]
    if len(cfg_matches) > 1:
        return None
    tag_matches = [
        e for e in models
        if Path(e["config"]).stem.lower().startswith("strata-")
        and len(Path(e["config"]).stem) > len("strata-")
        and Path(e["config"]).stem.lower().removeprefix("strata-") == low
    ]
    if len(tag_matches) == 1:
        return tag_matches[0]
    if len(tag_matches) > 1:
        return None
    close = [e for e in models if low in e["id"].lower()]
    return close[0] if len(close) == 1 else None


def resolve(models: list[dict], name) -> dict | None:
    """#38: the model a request's `model` names: its name, one of its aliases or its config file's name, exactly (case
    aside).  Never a part of a name, as find() takes: a client must not start a switch by a name that only looks
    like one of them."""
    if not isinstance(name, str) or not name.strip():
        return None
    low = name.strip().lower()
    for e in models:
        config = Path(e["config"])
        if low in (e["id"].lower(), config.name.lower(), config.stem.lower(), *(a.lower() for a in e.get("aliases", ()) if isinstance(a, str))):
            return e
    return None


def line(e: dict, current: str | None = None) -> str:
    """One line of a listing."""
    size = "-".join(x for x in (e["params_total"], "A" + e["params_active"] if e["params_active"] else None) if x)
    vram = e.get("vram_gb")
    ram = e.get("ram_gb")
    if (
        isinstance(vram, dict)
        and isinstance(vram.get("least"), (int, float)) and not isinstance(vram.get("least"), bool)
        and isinstance(vram.get("all"), (int, float)) and not isinstance(vram.get("all"), bool)
    ):
        need = f"VRAM {vram['least']:g}-{vram['all']:g} GB"
        if isinstance(ram, (int, float)) and not isinstance(ram, bool):
            need += f", file {ram:g} GB"
    else:
        need = ""
    mark = "*" if e["id"] == current else " "
    what = ", ".join(x for x in (size, e["quantization"], e["backend"], f"context {e['context']}" if e["context"] else "",
                                 "images" if e["multimodal"] else "", "MTP" if e["mtp"] else "", need) if x)
    tail = "" if e["status"] != "unavailable" else f"  [{e['problem']}]"
    return f"{mark} {e['id']:28s} {what}{tail}"


def main() -> int:
    for e in discover():
        print(line(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
