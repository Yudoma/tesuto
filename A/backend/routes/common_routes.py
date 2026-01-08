# -*- coding: utf-8 -*-
"""backend/routes/common_routes.py

設計Aの「共通ルート」相当。

- /api/jobs/* : ジョブ状態の確認
- /api/artifacts/* : 成果物取得

BK33 既存の同期推論/学習APIは維持しつつ、
非同期ジョブ（SQLite Queue）で実運用に寄せるための土台を追加します。
"""

from __future__ import annotations

import os
import platform
import subprocess
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.sqlite_queue import sqlite_queue
from backend.core.env_snapshot import snapshot_env
from backend.core.dataset_report import build_dataset_report
from backend.core.artifact_store import artifact_store
from lora_config import settings


router = APIRouter()


class OpenPathRequest(BaseModel):
    """OSのファイルマネージャでフォルダを開く。

    セキュリティより利便性を優先するローカルツール向け。
    - key を指定すると、サーバ側で既知のディレクトリに解決
    - path を指定すると、相対パスはアプリの base_dir 配下に解決
    """

    key: str | None = None
    path: str | None = None


_KNOWN_DIR_KEYS = {
    # models
    "models_text": settings.dirs["text"]["models"],
    "models_image": settings.dirs["image"]["models"],
    "models_audio": settings.dirs["audio"]["models"],
    # datasets
    "datasets_text": settings.dirs["text"]["datasets"],
    "datasets_image": settings.dirs["image"]["datasets"],
    "datasets_audio": settings.dirs["audio"]["datasets"],
    # outputs
    "lora_adapters_root": settings.output_root,
    "lora_adapters_text": settings.dirs["text"]["output"],
    "lora_adapters_image": settings.dirs["image"]["output"],
    "lora_adapters_audio": settings.dirs["audio"]["output"],
}


def _resolve_open_target(req: OpenPathRequest) -> Path:
    if req.key:
        p = _KNOWN_DIR_KEYS.get(req.key)
        if not p:
            raise HTTPException(400, f"Unknown key: {req.key}")
        return Path(p)

    if not req.path:
        raise HTTPException(400, "path or key is required")

    raw = str(req.path).strip().strip('"')
    if not raw:
        raise HTTPException(400, "path is empty")

    p = Path(raw)
    # 相対パスはアプリの base_dir 配下に解決
    if not p.is_absolute():
        p = (settings.base_dir / p).resolve()
    return p


def _open_in_file_manager(target: Path) -> None:
    # ファイルなら親フォルダを開く
    folder = target
    if target.exists() and target.is_file():
        folder = target.parent

    if not folder.exists():
        raise HTTPException(404, f"Path not found: {folder}")

    system = platform.system().lower()
    try:
        if system.startswith("win"):
            # explorer でフォルダを開く
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif system == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except Exception as e:
        raise HTTPException(500, f"Failed to open folder: {e}")


@router.post("/utils/open_path")
def open_path(req: OpenPathRequest):
    """指定パスのあるフォルダをOSで開く。

    フロントエンドの「フォルダを開く」ボタン向け。
    """
    target = _resolve_open_target(req)
    _open_in_file_manager(target)
    return {"status": "ok"}



@router.get("/utils/read_text_file")
def read_text_file(path: str, max_lines: int = 600):
    """指定テキストファイルを読み、末尾 max_lines 行を返す。

    ローカルツール前提。安全のため、base_dir 配下に解決できるパスのみ許可。
    """
    try:
        p = Path(path)
        if not p.is_absolute():
            p = (settings.base_dir / p).resolve()
        else:
            p = p.resolve()

        base = Path(settings.base_dir).resolve()
        if base not in p.parents and p != base:
            raise HTTPException(400, "Invalid path (outside base_dir)")

        if not p.exists() or not p.is_file():
            raise HTTPException(404, "File not found")

        # 末尾 max_lines 行
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        tail = lines[-max_lines:] if max_lines > 0 else lines
        return {"path": str(p), "lines": tail}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to read file: {e}")


@router.get("/utils/paths")
def get_paths():
    """UI/運用向け: 主要フォルダの絶対パス一覧を返す。"""
    def norm(x: Path) -> str:
        try:
            return str(Path(x).resolve())
        except Exception:
            return str(x)

    paths = {
        "base_dir": norm(settings.base_dir),
        "models_text": norm(settings.dirs["text"]["models"]),
        "models_image": norm(settings.dirs["image"]["models"]),
        "models_audio": norm(settings.dirs["audio"]["models"]),
        "datasets_text": norm(settings.dirs["text"]["datasets"]),
        "datasets_image": norm(settings.dirs["image"]["datasets"]),
        "datasets_audio": norm(settings.dirs["audio"]["datasets"]),
        "lora_adapters_root": norm(settings.output_root),
        "lora_adapters_text": norm(settings.dirs["text"]["output"]),
        "lora_adapters_image": norm(settings.dirs["image"]["output"]),
        "lora_adapters_audio": norm(settings.dirs["audio"]["output"]),
        "logs_dir": norm(settings.logs_dir),
        "static_dir": norm(settings.base_dir / "static"),
    }
    return {"paths": paths}




@router.get("/capabilities")
def get_capabilities():
    """UI が参照する対応機能一覧（例外を出さない）。"""
    try:
        tp = third_party_status(settings.base_dir)
    except Exception as e:
        tp = {
            "gpt_sovits_repo_ok": False,
            "gpt_sovits_repo_path": str((settings.base_dir / "third_party" / "GPT-SoVITS").resolve()),
            "gpt_sovits_repo_url": "https://github.com/RVC-Boss/GPT-SoVITS",
            "xtts_repo_ok": False,
            "xtts_repo_path": str((settings.base_dir / "third_party" / "XTTS").resolve()),
            "xtts_repo_url": "https://github.com/coqui-ai/TTS",
            "error": str(e),
        }

    return {
        "capabilities": {"text": True, "image": True, "audio": True},
        "audio_lora": {
            "vc_gpt_sovits": bool(tp.get("gpt_sovits_repo_ok")),
            "tts_xtts": bool(tp.get("xtts_repo_ok")),
        },
        "third_party": tp,
    }

@router.get("/api/capabilities")
def get_capabilities_api():
    return get_capabilities()

@router.get("/utils/capabilities")
def get_capabilities_utils():
    return get_capabilities()


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = sqlite_queue.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"job": job}


@router.get("/artifacts/{modality}/{artifact_id}/meta")
def get_artifact_meta(modality: str, artifact_id: str):
    meta = artifact_store.get_meta(modality, artifact_id)
    if not meta:
        raise HTTPException(404, "Artifact meta not found")
    return {"meta": meta}



@router.get("/artifacts/compare")
def compare_artifacts(a_id: str, b_id: str):
    """2つの artifact の meta を比較（A/B比較・回帰比較向け）"""
    try:
        diff = artifact_store.compare_meta(a_id, b_id)
        return {"status": "ok", "diff": diff}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/artifacts/{modality}/{artifact_id}/file")
def download_artifact_file(modality: str, artifact_id: str):
    p = artifact_store.get_output_path(modality, artifact_id)
    if p is None or (not p.exists()):
        raise HTTPException(404, "Artifact file not found")

    # FileResponse は Windows パスでも動作する
    return FileResponse(
        path=str(p),
        filename=p.name,
        media_type="application/octet-stream",
    )

@router.post("/datasets/validate")
def validate_dataset(payload: dict):
    """データセット検査。致命傷があれば ok=False を返す。"""
    mode = payload.get("mode")  # text/image/audio
    dataset = payload.get("dataset")  # name or path
    kind = payload.get("kind")  # optional: tts/vc
    msgs = []
    ok = True

    def fail(m):
        nonlocal ok
        ok = False
        msgs.append(m)

    # Resolve path
    p = None
    try:
        if mode == "text":
            p = (settings.dirs["text"]["datasets"] / str(dataset)).resolve()
        elif mode == "image":
            p = (settings.dirs["image"]["datasets"] / str(dataset)).resolve()
        elif mode == "audio":
            p = (settings.dirs["audio"]["datasets"] / str(dataset)).resolve()
        else:
            p = Path(str(dataset)).resolve()
    except Exception:
        p = Path(str(dataset)).resolve()

    if not p.exists():
        fail(f"データセットが見つかりません: {p}")
        return {"ok": ok, "messages": msgs, "path": str(p)}

    if p.is_file():
        # single file datasets (text)
        if p.stat().st_size == 0:
            fail("データセットファイルが空です。")
        return {"ok": ok, "messages": msgs, "path": str(p)}

    # folder checks
    files = [x for x in p.rglob("*") if x.is_file()]
    if len(files) == 0:
        fail("データセットフォルダにファイルがありません。")
        return {"ok": ok, "messages": msgs, "path": str(p)}

    if mode == "image":
        exts = {".png",".jpg",".jpeg",".webp",".bmp"}
        imgs = [x for x in files if x.suffix.lower() in exts]
        if len(imgs) == 0:
            fail("画像ファイルが見つかりません（png/jpg/webp など）。")
    if mode == "audio":
        if kind == "tts":
            meta = p / "metadata.csv"
            wavs = p / "wavs"
            if not meta.exists():
                fail("metadata.csv が見つかりません（LJSpeech形式）。")
            if not wavs.exists():
                fail("wavs/ が見つかりません（LJSpeech形式）。")
            else:
                w = [x for x in wavs.glob("*.wav")]
                if len(w) == 0:
                    fail("wavs/ に wav がありません。")
        else:
            # vc or generic
            wavs = [x for x in files if x.suffix.lower()==".wav"]
            if len(wavs) == 0:
                fail("wav が見つかりません。")

    # soft warnings
    if ok and len(files) < 5:
        msgs.append("警告: データ数が少ないため、学習品質が不安定になる可能性があります。")

    return {"ok": ok, "messages": msgs, "path": str(p)}

@router.post("/utils/refresh_models")
def refresh_models():
    """即時反映用: registry を返す（フロントから呼ぶだけでOK）"""
    from backend.registry.index import load_registry
    return load_registry(settings.base_dir)


# =============================================================================
# D/E: Dataset reports & Environment/Setup logs
# =============================================================================

@router.get("/utils/list_setup_logs")
def list_setup_logs():
    """setup系ログを一覧化（UI閲覧用）。"""
    logs_dir = settings.logs_dir
    items = []
    for p in logs_dir.glob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if "setup" not in name:
            continue
        if p.suffix.lower() not in {".log", ".txt"}:
            continue
        try:
            st = p.stat()
            items.append({"name": p.name, "path": str(p), "size": st.st_size, "mtime": int(st.st_mtime)})
        except Exception:
            items.append({"name": p.name, "path": str(p), "size": None, "mtime": None})
    items.sort(key=lambda x: (x.get("mtime") or 0), reverse=True)
    return {"items": items}

@router.get("/runs/dataset_report")
def get_dataset_report(job_id: str):
    """runs/<job_id>/dataset_report.json を返す。"""
    run_dir = settings.runs_root / str(job_id)
    p = run_dir / "dataset_report.json"
    if not p.exists():
        raise HTTPException(404, "dataset_report.json not found")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to read dataset_report.json: {e}")

@router.get("/runs/env")
def get_env_snapshot(job_id: str):
    """runs/<job_id>/env.json を返す。"""
    run_dir = settings.runs_root / str(job_id)
    p = run_dir / "env.json"
    if not p.exists():
        raise HTTPException(404, "env.json not found")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to read env.json: {e}")

@router.get("/runs/env_diff")
def get_env_diff(job_id: str):
    """指定ジョブの env.json と現在環境の差分を返す。"""
    before = get_env_snapshot(job_id)
    after = snapshot_env()
    bpk = before.get("packages") or {}
    apk = after.get("packages") or {}
    added = {}
    removed = {}
    changed = {}
    for k, v in apk.items():
        if k not in bpk:
            added[k] = v
        elif str(bpk.get(k)) != str(v):
            changed[k] = {"before": bpk.get(k), "after": v}
    for k, v in bpk.items():
        if k not in apk:
            removed[k] = v
    return {
        "job_id": job_id,
        "added": dict(sorted(added.items())),
        "removed": dict(sorted(removed.items())),
        "changed": dict(sorted(changed.items())),
        "before_meta": {k: before.get(k) for k in ["python_version","platform","torch_version","cuda_available","cuda_version","gpu_name"] if before.get(k) is not None},
        "after_meta": {k: after.get(k) for k in ["python_version","platform","torch_version","cuda_available","cuda_version","gpu_name"] if after.get(k) is not None},
    }

@router.get("/datasets/history")
def dataset_history(mode: str, dataset: str):
    """指定データセットが使われたジョブ履歴を返す（runs/logs走査）。"""
    mode = (mode or "").lower()
    hist_file = None
    if mode == "text":
        hist_file = settings.logs_dir / "history.json"
    elif mode == "image":
        hist_file = settings.logs_dir / "history_image.json"
    elif mode == "audio":
        hist_file = settings.logs_dir / "history_audio.json"
    else:
        raise HTTPException(400, "mode must be text/image/audio")

    if not hist_file.exists():
        return {"items": []}

    try:
        data = json.loads(hist_file.read_text(encoding="utf-8"))
        items = data.get("history") or []
    except Exception as e:
        raise HTTPException(500, f"Failed to read history: {e}")

    # dataset は name かフルパスが来る。末尾一致も許容。
    needle = str(dataset)
    out = []
    for it in items:
        ds = str(it.get("dataset") or "")
        if ds == needle or ds.endswith(needle) or needle.endswith(ds):
            out.append(it)
    return {"items": out}
