# -*- coding: utf-8 -*-
"""
plugins/heretic/backend/api.py

Heretic plugin backend endpoints (content-agnostic infrastructure) with UI-supporting APIs.

Features
--------
1) Library layout detection + helpers
   - GET  /{prefix}/library/info
   - POST /{prefix}/library/open_folder
   - POST /{prefix}/library/flatten_layout   (move src/heretic -> heretic, confirmation handled by UI)

2) Debug Hooks (StreamInterceptor stats capture)
   - GET  /{prefix}/hooks/modules            (list named_modules)
   - POST /{prefix}/hooks/enable             (attach)
   - POST /{prefix}/hooks/disable            (detach)
   - GET  /{prefix}/hooks/stats              (latest capture)

3) SVD Weight Analysis (model diff + low-rank export)
   - GET  /{prefix}/weightdiff/candidates    (list checkpoint files under models/text)
   - POST /{prefix}/weightdiff/calc          (load A,B -> diff in memory)
   - POST /{prefix}/weightdiff/export_lora   (SVD low-rank -> safetensors)

Compatibility
-------------
Host versions differ in how plugin routers are mounted. We expose routes on BOTH:
  - /heretic/*        (router)
  - /api/heretic/*    (router_api)

No semantic policies are encoded here; it is purely tensor / file / plumbing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.engines.text import text_engine
from .stream_interceptor import StreamInterceptor
from .weight_diff_utils import calculate_weight_diff, convert_diff_to_lora, save_diff_as_lora_safetensors

try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None


router = APIRouter(prefix="/heretic", tags=["heretic"])
router_api = APIRouter(prefix="/api/heretic", tags=["heretic"])

# Singleton interceptor bound to current inference model
_interceptor: Optional[StreamInterceptor] = None
_last_diff: Optional[Dict[str, torch.Tensor]] = None
_last_diff_meta: Dict[str, Any] = {}


# -----------------------------
# Helpers: paths / layout
# -----------------------------

def _plugin_root() -> Path:
    # .../plugins/heretic/backend/api.py -> .../plugins/heretic
    return Path(__file__).resolve().parents[1]

def _tool_root() -> Path:
    # .../plugins/heretic -> .../(tool root)
    return _plugin_root().parents[1]

def _paths(plugin_root: Path) -> Dict[str, Path]:
    expected_root = plugin_root / "heretic_master"
    alt_root = plugin_root / "heretic-master"

    return {
        "plugin_root": plugin_root,
        "expected_root": expected_root,
        "alt_root": alt_root,
        "expected_direct_init": expected_root / "heretic" / "__init__.py",
        "expected_src_init": expected_root / "src" / "heretic" / "__init__.py",
        "alt_direct_init": alt_root / "heretic" / "__init__.py",
        "alt_src_init": alt_root / "src" / "heretic" / "__init__.py",
        "expected_direct_dir": expected_root / "heretic",
        "expected_src_dir": expected_root / "src" / "heretic",
    }

def _detect_layout(plugin_root: Path) -> Dict[str, Any]:
    p = _paths(plugin_root)

    expected_root = p["expected_root"]
    alt_root = p["alt_root"]

    expected_exists = expected_root.exists()
    alt_exists = alt_root.exists()

    expected_direct_ok = p["expected_direct_init"].exists()
    expected_src_ok = p["expected_src_init"].exists()
    alt_direct_ok = p["alt_direct_init"].exists()
    alt_src_ok = p["alt_src_init"].exists()

    layout_raw = "missing"
    status = "MISSING"
    layout = "missing"

    if expected_direct_ok:
        layout = "flat"
        layout_raw = "direct"
        status = "OK"
    elif expected_src_ok:
        layout = "src"
        layout_raw = "src_layout"
        status = "WARN"
    elif alt_direct_ok:
        layout = "hyphen"
        layout_raw = "hyphen_direct"
        status = "ERROR"
    elif alt_src_ok:
        layout = "hyphen_src"
        layout_raw = "hyphen_src_layout"
        status = "ERROR"

    rename_required = bool(alt_exists) and not expected_exists
    can_auto_rename = False  # conservative: UI may offer rename in future
    can_auto_flatten = bool(expected_src_ok) and (not p["expected_direct_dir"].exists())

    notes: List[str] = [
        "推奨配置（基本）: plugins/heretic/heretic_master/heretic/...",
        "zip展開直後のフォルダ名が 'heretic-master' の場合、Python import の都合で 'heretic_master' にリネームしてください（- → _）。",
        "Heretic本体は upstream としてそのまま置き、ツール側の接着層（backend/frontend）は plugins/heretic/ 配下で管理します。",
    ]

    manual_steps: List[str] = []
    if layout == "src":
        manual_steps = [
            "検出: heretic_master/src/heretic 配下に Heretic を確認しました。",
            "推奨: plugins/heretic/heretic_master/heretic 直下に配置してください（src を挟まない）。",
            "手動で次を実行してください（非破壊・移動のみ）:",
            "  1) 'plugins/heretic/heretic_master/src/heretic' を 'plugins/heretic/heretic_master/heretic' に移動",
            "  2) 移動後、空の 'src' フォルダは残っていてもOK（不要なら削除可）",
            r"Windows コマンド例: move ""plugins\heretic\heretic_master\src\heretic"" ""plugins\heretic\heretic_master\heretic""",
        ]
    elif layout in ("hyphen", "hyphen_src"):
        manual_steps = [
            "検出: フォルダ名が 'heretic-master'（ハイフン）です。",
            "推奨: フォルダ名を 'heretic_master'（アンダーバー）へリネームしてください（- → _）。",
            "手動リネーム例:",
            r"  rename ""plugins\heretic\heretic-master"" ""heretic_master""",
        ]
        if layout == "hyphen_src":
            manual_steps.append("追加注意: 内部レイアウトが src/heretic です。リネーム後に src レイアウトの手順も確認してください。")

    info: Dict[str, Any] = {
        "plugin_root": str(p["plugin_root"]),
        "expected_root": str(expected_root),
        "expected_exists": expected_exists,
        "expected_direct_init": str(p["expected_direct_init"]),
        "expected_src_init": str(p["expected_src_init"]),
        "expected_direct_ok": expected_direct_ok,
        "expected_src_ok": expected_src_ok,

        "alt_root": str(alt_root),
        "alt_exists": alt_exists,
        "alt_direct_init": str(p["alt_direct_init"]),
        "alt_src_init": str(p["alt_src_init"]),
        "alt_direct_ok": alt_direct_ok,
        "alt_src_ok": alt_src_ok,

        "layout": layout,
        "layout_raw": layout_raw,
        "status": status,

        "rename_required": rename_required,
        "can_auto_rename": can_auto_rename,
        "can_auto_flatten": can_auto_flatten,

        "github_url": "https://github.com/p-e-w/heretic",

        "paths": {
            "expected_root": str(expected_root),
            "expected_flat_dir": str(p["expected_direct_dir"]),
            "expected_src_dir": str(p["expected_src_dir"]),
            "found_root": str(expected_root) if expected_root.exists() else (str(alt_root) if alt_root.exists() else ""),
            "src_root": str(expected_root / "src"),
            "hyphen_root": str(alt_root),
        },
        "notes": notes,
        "manual_steps": manual_steps,
    }
    return info


def _library_info() -> JSONResponse:
    plugin_root = _plugin_root()
    return JSONResponse(_detect_layout(plugin_root))


def _open_library_folder() -> JSONResponse:
    plugin_root = _plugin_root()
    target = plugin_root / "heretic_master"
    if not target.exists():
        target = plugin_root

    try:
        import os
        import subprocess
        import sys

        path_str = str(target)

        if sys.platform.startswith("win"):
            os.startfile(path_str)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path_str])
        else:
            subprocess.Popen(["xdg-open", path_str])

        return JSONResponse({"ok": True, "opened": path_str})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)


def _flatten_layout() -> JSONResponse:
    """
    Safe, non-destructive helper for common upstream zip layout:
      plugins/heretic/heretic_master/src/heretic  ->  plugins/heretic/heretic_master/heretic
    """
    plugin_root = _plugin_root()
    p = _paths(plugin_root)
    src_dir = p["expected_src_dir"]
    dst_dir = p["expected_direct_dir"]

    if not src_dir.exists():
        return JSONResponse({"ok": False, "performed": False, "reason": "src_layout_not_found",
                             "src": str(src_dir), "dst": str(dst_dir),
                             "message": "移動元 (src/heretic) が見つからないため実行できません。"}, status_code=400)
    if dst_dir.exists():
        return JSONResponse({"ok": False, "performed": False, "reason": "destination_exists",
                             "src": str(src_dir), "dst": str(dst_dir),
                             "message": "移動先 (heretic) が既に存在するため実行できません。手動で内容を確認してください。"}, status_code=409)

    try:
        import shutil
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        moved_to = shutil.move(str(src_dir), str(dst_dir))
        return JSONResponse({"ok": True, "performed": True, "src": str(src_dir), "dst": str(dst_dir),
                             "moved_to": str(moved_to),
                             "message": "移動が完了しました。必要ならブラウザを更新して判定結果を確認してください。"})
    except Exception as e:
        return JSONResponse({"ok": False, "performed": False, "reason": "exception",
                             "src": str(src_dir), "dst": str(dst_dir),
                             "error": f"{type(e).__name__}: {e}",
                             "message": "移動処理で例外が発生しました。権限/ファイルロック等を確認してください。"}, status_code=500)


# -----------------------------
# Debug Hooks UI APIs
# -----------------------------

class HookEnableReq(BaseModel):
    module_name: str
    hook_type: str = "forward_hook"  # or forward_pre_hook


class HookDisableReq(BaseModel):
    pass


@router.get("/hooks/modules")
def hooks_modules(q: str = "", limit: int = 200):
    model = getattr(text_engine, "inference_model", None)
    if model is None:
        return {"ok": False, "message": "推論モデルがロードされていません（先にText推論でモデルをロードしてください）", "count": 0, "modules": []}
    ql = (q or "").lower().strip()
    names = []
    for name, _ in model.named_modules():
        if not name:
            continue
        if ql and ql not in name.lower():
            continue
        names.append(name)
        if len(names) >= limit:
            break
    return {"ok": True, "count": len(names), "modules": names}


@router.post("/hooks/enable")
def hooks_enable(req: HookEnableReq):
    global _interceptor
    model = getattr(text_engine, "inference_model", None)
    if model is None:
        return {"ok": False, "message": "推論モデルがロードされていません（先にText推論でモデルをロードしてください）", "count": 0, "modules": []}
    if _interceptor is None or _interceptor.model is not model:
        _interceptor = StreamInterceptor(model)
    try:
        _interceptor.attach_by_name(req.module_name, req.hook_type)
    except Exception as e:
        raise HTTPException(400, f"Hook有効化失敗: {e}")
    return {"ok": True, "module_name": req.module_name, "hook_type": req.hook_type}


@router.post("/hooks/disable")
def hooks_disable(_: HookDisableReq = HookDisableReq()):
    global _interceptor
    if _interceptor is None:
        return {"ok": True, "disabled": False}
    _interceptor.detach_all()
    return {"ok": True, "disabled": True}


@router.get("/hooks/stats")
def hooks_stats():
    if _interceptor is None:
        return {"ok": True, "stats": None}
    return {"ok": True, "stats": _interceptor.get_latest_stats()}


# -----------------------------
# SVD Weight Analysis UI APIs
# -----------------------------

def _models_dir_text() -> Path:
    return _tool_root() / "models" / "text"

def _lora_out_dir_text() -> Path:
    return _tool_root() / "lora_adapters" / "text" / "heretic"

def _is_checkpoint(p: Path) -> bool:
    suf = p.suffix.lower()
    if suf in (".safetensors", ".bin", ".pt", ".pth"):
        return True
    return False

@router.get("/weightdiff/candidates")
def weightdiff_candidates(limit: int = 2000):
    root = _models_dir_text()
    files: List[Dict[str, Any]] = []
    if root.exists():
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if _is_checkpoint(p):
                files.append({"path": str(p), "name": p.name, "bytes": p.stat().st_size})
                if len(files) >= limit:
                    break
    files.sort(key=lambda x: x["name"])
    return {"ok": True, "root": str(root), "count": len(files), "files": files}


class DiffCalcReq(BaseModel):
    modelA_path: str
    modelB_path: str


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".safetensors":
        if safetensors_load_file is None:
            raise RuntimeError("safetensors が利用できません")
        return safetensors_load_file(str(p))
    obj = torch.load(str(p), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        # common HF format: {"model": state_dict} etc. try best-effort
        for k in ("model", "module", "net", "weights"):
            v = obj.get(k)
            if isinstance(v, dict):
                return v
        return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    raise RuntimeError("Unsupported checkpoint format")


@router.post("/weightdiff/calc")
def weightdiff_calc(req: DiffCalcReq):
    global _last_diff, _last_diff_meta
    try:
        sdA = _load_state_dict(req.modelA_path)
        sdB = _load_state_dict(req.modelB_path)
    except Exception as e:
        raise HTTPException(400, f"チェックポイント読込失敗: {e}")

    diff = calculate_weight_diff(sdA, sdB)

    norms: List[Tuple[str, float, Tuple[int, ...]]] = []
    for k, v in diff.items():
        if isinstance(v, torch.Tensor) and v.ndim in (1, 2):
            norms.append((k, float(v.float().norm().item()), tuple(v.shape)))
    norms.sort(key=lambda x: x[1], reverse=True)

    _last_diff = diff
    _last_diff_meta = {
        "modelA_path": req.modelA_path,
        "modelB_path": req.modelB_path,
        "n_keys": len(diff),
        "top": norms[:50],
        "ts": time.time(),
    }
    return {"ok": True, "meta": _last_diff_meta}


class ExportLoraReq(BaseModel):
    out_path: str = ""     # if empty, auto path
    rank: int = 8
    key_regex: str = ""    # optional filter for parameter keys
    max_keys: int = 64     # safety limit


@router.post("/weightdiff/export_lora")
def weightdiff_export_lora(req: ExportLoraReq):
    global _last_diff, _last_diff_meta
    if _last_diff is None:
        raise HTTPException(400, "先に diff を計算してください（/weightdiff/calc）")

    rank = int(req.rank)
    max_keys = int(req.max_keys)
    if rank < 1 or rank > 256:
        raise HTTPException(400, "rank は 1〜256 で指定してください")
    if max_keys < 1 or max_keys > 512:
        raise HTTPException(400, "max_keys は 1〜512 で指定してください")

    import re
    rx = None
    if req.key_regex:
        try:
            rx = re.compile(req.key_regex)
        except Exception as e:
            raise HTTPException(400, f"key_regex が不正です: {e}")

    # Auto output path
    out_path = req.out_path.strip()
    if not out_path:
        out_dir = _lora_out_dir_text()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = str(out_dir / f"diff_svd_r{rank}_{ts}.safetensors")
    else:
        out_path = str(Path(out_path))

    # Build LoRA tensors
    lora_tensors: Dict[str, torch.Tensor] = {}
    count = 0
    for k, v in _last_diff.items():
        if count >= max_keys:
            break
        if rx and not rx.search(k):
            continue
        if not isinstance(v, torch.Tensor):
            continue

        # Only 2D matrices (linear weights) are directly LoRA-able
        if v.ndim == 2:
            A, B = convert_diff_to_lora(v, rank=rank)
            lora_tensors[f"{k}.lora_A"] = A.contiguous()
            lora_tensors[f"{k}.lora_B"] = B.contiguous()
            count += 1
        elif v.ndim == 1:
            # Bias: store as is (analysis convenience)
            lora_tensors[f"{k}.bias_diff"] = v.contiguous()
            count += 1

    if count == 0:
        raise HTTPException(400, "対象となるテンソルがありません（2D weight / 1D bias を含むキーにマッチするか確認してください）")

    try:
        save_diff_as_lora_safetensors(
            out_path=out_path,
            lora_tensors=lora_tensors,
            metadata={
                "plugin": "heretic",
                "type": "diff_svd_lora",
                "rank": str(rank),
                "modelA_path": str(_last_diff_meta.get("modelA_path", "")),
                "modelB_path": str(_last_diff_meta.get("modelB_path", "")),
                "n_keys_used": str(count),
                "key_regex": req.key_regex or "",
            },
        )
    except Exception as e:
        raise HTTPException(400, f"LoRA書き出し失敗: {e}")

    return {"ok": True, "out_path": out_path, "rank": rank, "n_keys_used": count}


# Register routes on BOTH routers (same handlers)
for r in (router, router_api):
    r.add_api_route("/library/info", _library_info, methods=["GET"])
    r.add_api_route("/library/open_folder", _open_library_folder, methods=["POST"])
    r.add_api_route("/library/flatten_layout", _flatten_layout, methods=["POST"])

    r.add_api_route("/hooks/modules", hooks_modules, methods=["GET"])
    r.add_api_route("/hooks/enable", hooks_enable, methods=["POST"])
    r.add_api_route("/hooks/disable", hooks_disable, methods=["POST"])
    r.add_api_route("/hooks/stats", hooks_stats, methods=["GET"])

    r.add_api_route("/weightdiff/candidates", weightdiff_candidates, methods=["GET"])
    r.add_api_route("/weightdiff/calc", weightdiff_calc, methods=["POST"])
    r.add_api_route("/weightdiff/export_lora", weightdiff_export_lora, methods=["POST"])

@router.post("/register_artifact")
def register_artifact(payload: dict):
    """
    Heretic outputs を既存のモデル管理（ディレクトリスキャン）に登録するためのコピー。

    payload:
      {
        "job_id": "...",
        "kind": "models" | "lora",
        "src_rel": "models" | "lora/xxx.safetensors" | "models/MyModel",
        "dst_name": "任意: 保存先名"
      }
    """
    job_id = (payload.get("job_id") or "").strip()
    kind = (payload.get("kind") or "").strip()
    src_rel = (payload.get("src_rel") or "").strip()
    dst_name = (payload.get("dst_name") or "").strip()

    if not job_id or not kind or not src_rel:
        raise HTTPException(400, "job_id/kind/src_rel が必要です")

    st = _job_runner.public_dict(job_id)
    if not st:
        raise HTTPException(404, "job_id が見つかりません")

    out = Path(st["outputs_dir"])
    src = (out / src_rel).resolve()
    if not str(src).startswith(str(out.resolve())):
        raise HTTPException(400, "不正なパスです")
    if not src.exists():
        raise HTTPException(404, f"src not found: {src_rel}")

    # destination
    if kind == "models":
        dst_root = Path(settings.dirs["text"]["models"]).resolve()
    elif kind == "lora":
        dst_root = Path(settings.dirs["text"]["output"]).resolve()
    else:
        raise HTTPException(400, f"unknown kind: {kind}")

    dst_root.mkdir(parents=True, exist_ok=True)

    if not dst_name:
        dst_name = src.name

    dst = (dst_root / dst_name).resolve()
    if not str(dst).startswith(str(dst_root)):
        raise HTTPException(400, "不正な dst_name です")
    if dst.exists():
        raise HTTPException(400, f"既に存在します: {dst_name}")

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    return {"ok": True, "dst": str(dst), "dst_name": dst_name}
