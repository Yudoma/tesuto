# -*- coding: utf-8 -*-
"""backend/routes/audio_routes.py

Audio (XTTS中心 + VC(RVC/GPT-SoVITS)) モダリティ専用API。
main_router.py 側で prefix="/audio" を付与する想定。

最低限の要件:
- POST /audio/train/start
- POST /audio/inference/load
- POST /audio/inference/generate

実運用に必要なため、/datasets /models /train/status 等も同梱します。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from lora_config import settings
from backend.core.job_manager import job_manager
from backend.engines.audio import audio_engine
from backend.core.job_spec import JobSpec, ModelRef
from backend.core.sqlite_queue import sqlite_queue


def get_audio_presets():
    return [
        {"id": "natural", "label": "自然（おすすめ）", "description": "正規化/韻律/音量正規化を標準で適用"},
        {"id": "clear", "label": "ハキハキ", "description": "句読点をやや増やして聞き取りやすく"},
    ]


router = APIRouter()


class DownloadModelRequest(BaseModel):
    repo_id: str = Field(..., description="HuggingFace repo id")




@router.get("/models")
def list_audio_models():
    models_dir = settings.dirs["audio"]["models"]
    res: List[Dict[str, Any]] = []
    if models_dir.exists():
        for p in models_dir.iterdir():
            if p.is_dir():
                res.append({"name": p.name, "type": "folder", "path": str(p)})
            elif p.is_file():
                res.append({"name": p.name, "type": "file", "path": str(p)})
    return {"models": res}


@router.get("/models/{model_name}/meta")
def model_meta(model_name: str):
    """モデルのメタ情報（UI表示用）"""
    model_dir = settings.dirs["audio"]["models"] / model_name
    if not model_dir.exists():
        raise HTTPException(404, "Model not found")
    # directory size
    total = 0
    files = []
    for p in model_dir.rglob("*"):
        if p.is_file():
            try:
                sz = p.stat().st_size
                total += sz
                files.append({"path": str(p.relative_to(model_dir)), "size": sz})
            except Exception:
                pass
    try:
        mtime = model_dir.stat().st_mtime
    except Exception:
        mtime = None
    return {
        "name": model_name,
        "path": str(model_dir.resolve()),
        "total_bytes": total,
        "file_count": len(files),
        "mtime": mtime,
        "files_top": sorted(files, key=lambda x: x.get("size", 0), reverse=True)[:20],
        "modality": "audio"
    }

@router.get("/models/{model_name}/predelete_check")
def model_predelete_check(model_name: str):
    """モデル削除前の参照チェック（軽量）"""
    refs = []
    # history files（存在する範囲）
    candidates = [
        settings.logs_dir / "history.json",
        settings.logs_dir / "history_audio.json",
        settings.logs_dir / "history_audio.json"
    ]
    for hf in candidates:
        try:
            if hf.exists():
                data = json.loads(hf.read_text(encoding="utf-8"))
                for item in data:
                    if str(item.get("model","")) == model_name:
                        refs.append({"where": str(hf.name), "job_id": item.get("id"), "timestamp": item.get("timestamp"), "status": item.get("status")})
        except Exception:
            pass

@router.delete("/models/{model_name}")
def delete_audio_model(model_name: str):
    models_dir = settings.dirs["audio"]["models"]
    path = (models_dir / model_name).resolve()
    # 安全確認: models_dir配下のみ許可
    if models_dir.resolve() not in path.parents and path != models_dir.resolve():
        raise HTTPException(400, "Invalid model path")
    if not path.exists():
        raise HTTPException(404, "Model not found")
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return {"status": "deleted", "name": model_name}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete: {e}")

@router.post("/models/download")
def download_audio_model(req: DownloadModelRequest, background_tasks: BackgroundTasks):
    """HuggingFaceから音声モデルをダウンロード (バックグラウンド)"""
    from huggingface_hub import snapshot_download

    def _bg_download(repo_id: str):
        try:
            local_dir = settings.dirs["audio"]["models"] / repo_id.split("/")[-1]
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Audio Download] Starting: {repo_id} -> {local_dir}")
            snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
            print(f"[Audio Download] Completed: {repo_id}")
        except Exception as e:
            print(f"[Audio Download] Failed: {e}")

    background_tasks.add_task(_bg_download, req.repo_id)
    return {"status": "started", "repo_id": req.repo_id}



    active = False
    try:
        st = audio_engine.get_training_status()
        if (st or {}).get("job_id") and (st or {}).get("status") == "running":
            if ((st or {}).get("params") or {}).get("base_model") == model_name or (st or {}).get("base_model") == model_name:
                active = True
    except Exception:
        pass

    return {
        "model": model_name,
        "reference_count": len(refs),
        "references": refs,
        "active_job_using": active
    }


@router.get("/inference/presets")
def api_audio_inference_presets() -> List[Dict[str, str]]:
    """音声生成の用途別プリセット一覧（UI用）"""
    return get_audio_presets()


def _normalize_rel_posix(path: str) -> str:
    p = (path or '').replace('\\', '/').lstrip('/')
    p = re.sub(r'^[A-Za-z]:', '', p).lstrip('/')
    p = re.sub(r'/+', '/', p)
    return p


def _safe_under(base: Path, rel_path: str) -> Path:
    rel = _normalize_rel_posix(rel_path)
    if rel in ('', '.'):
        raise HTTPException(400, 'Invalid path.')
    base = base.resolve()
    cand = (base / rel).resolve()
    if not str(cand).startswith(str(base)):
        raise HTTPException(400, 'Path traversal detected.')
    return cand


class AudioTrainParams(BaseModel):
    epochs: int = 1
    train_batch_size: int = 1

    whisper_model: str = Field(default='small', description='faster-whisper model name')
    language: str = Field(default='ja', description='ASR language (ja/en/auto等)')

    slice_min_sec: float = 3.0
    slice_max_sec: float = 10.0
    slice_target_sec: float = 8.0
    slice_hop_sec: float = 6.0

    # GPT-SoVITS 連携（外部委譲）
    gpt_sovits_repo: Optional[str] = None
    custom_train_cmd: Optional[str] = None


class AudioTrainStartRequest(BaseModel):
    base_model: str = ''
    dataset: str
    params: AudioTrainParams


class AudioInferenceLoadRequest(BaseModel):
    # models/audio 配下のフォルダ名 or 任意パス
    model_dir: str
    gpt_sovits_repo: Optional[str] = None
    custom_infer_cmd: Optional[str] = None

    # v2: TTS/VC の選択（省略可）
    tts_backend: Optional[str] = Field(default=None, description="TTSバックエンド: xtts / gpt_sovits")
    vc_backend: Optional[str] = Field(default=None, description="VCバックエンド: none / rvc / gpt_sovits_vc")

    # XTTS
    xtts_model_id: Optional[str] = Field(default=None, description="XTTS model id (例: tts_models/.../xtts_v2) またはローカルモデル")
    xtts_language: str = Field(default="ja", description="XTTS language code (ja/en/...)")

    # RVC
    rvc_repo: Optional[str] = Field(default=None, description="RVC repo パス（任意）")
    rvc_custom_cmd: Optional[str] = Field(default=None, description="RVC VC カスタムコマンド（任意）: {repo} {in_wav} {ref} {out}")

    # GPT-SoVITS VC
    gpt_sovits_vc_repo: Optional[str] = Field(default=None, description="GPT-SoVITS(VC) repo パス（任意）")
    gpt_sovits_vc_custom_cmd: Optional[str] = Field(default=None, description="GPT-SoVITS(VC) カスタムコマンド: {repo} {in_wav} {ref} {out}")



class AudioGenerateRequest(BaseModel):
    text: str
    reference_audio: str
    output_format: str = 'wav'
    preset_id: Optional[str] = Field(default=None, description="用途別プリセット: natural / mimic")
    postprocess: bool = Field(default=True, description="後処理（音量正規化）を行う")
    target_lufs: Optional[float] = Field(default=None, description="LUFS正規化ターゲット（例: -16.0）。未指定時はRMS正規化")
    custom_infer_cmd: Optional[str] = None

    # v2: TTS/VC
    tts_backend: Optional[str] = Field(default=None, description="TTSバックエンド: xtts / gpt_sovits")
    vc_backend: Optional[str] = Field(default=None, description="VCバックエンド: none / rvc / gpt_sovits_vc")
    xtts_model_id: Optional[str] = Field(default=None, description="XTTS model id")
    xtts_language: str = Field(default="ja", description="XTTS language")
    rvc_repo: Optional[str] = Field(default=None, description="RVC repo")
    rvc_custom_cmd: Optional[str] = Field(default=None, description="RVC custom VC cmd")
    gpt_sovits_vc_repo: Optional[str] = Field(default=None, description="GPT-SoVITS VC repo")
    gpt_sovits_vc_custom_cmd: Optional[str] = Field(default=None, description="GPT-SoVITS VC custom cmd")



class AudioGenerateJobRequest(BaseModel):
    """設計A: 非同期ジョブ投入用（Workerで音声生成）。"""

    model_dir: str = Field(..., description="models/audio 配下のフォルダ名 or 任意パス")
    gpt_sovits_repo: Optional[str] = Field(default=None, description="GPT-SoVITS repo パス（任意）")
    custom_infer_cmd: Optional[str] = Field(default=None, description="推論用カスタムコマンド（任意）")

    text: str
    reference_audio: str
    output_format: str = 'wav'
    preset_id: Optional[str] = Field(default=None, description="用途別プリセット: natural / mimic")
    postprocess: bool = Field(default=True, description="後処理（音量正規化）を行う")
    target_lufs: Optional[float] = Field(default=None, description="LUFS正規化ターゲット（例: -16.0）。未指定時はRMS正規化")

    # v2: TTS/VC
    tts_backend: Optional[str] = Field(default=None, description="TTSバックエンド: xtts / gpt_sovits")
    vc_backend: Optional[str] = Field(default=None, description="VCバックエンド: none / rvc / gpt_sovits_vc")
    xtts_model_id: Optional[str] = Field(default=None, description="XTTS model id")
    xtts_language: str = Field(default="ja", description="XTTS language")
    rvc_repo: Optional[str] = Field(default=None, description="RVC repo")
    rvc_custom_cmd: Optional[str] = Field(default=None, description="RVC custom VC cmd")
    gpt_sovits_vc_repo: Optional[str] = Field(default=None, description="GPT-SoVITS VC repo")
    gpt_sovits_vc_custom_cmd: Optional[str] = Field(default=None, description="GPT-SoVITS VC custom cmd")



class AudioGenerateStreamRequest(BaseModel):
    text: str
    reference_audio_path: str
    gpt_sovits_repo: Optional[str] = None
    custom_infer_cmd: Optional[str] = None
    output_format: str = "wav"
    preset_id: Optional[str] = Field(default=None, description="用途別プリセット: natural / mimic")
    max_chunk_len: int = 120

    # v2: TTS/VC
    tts_backend: Optional[str] = Field(default=None, description="TTSバックエンド: xtts / gpt_sovits")
    vc_backend: Optional[str] = Field(default=None, description="VCバックエンド: none / rvc / gpt_sovits_vc")
    xtts_model_id: Optional[str] = Field(default=None, description="XTTS model id")
    xtts_language: str = Field(default="ja", description="XTTS language")
    rvc_repo: Optional[str] = Field(default=None, description="RVC repo")
    rvc_custom_cmd: Optional[str] = Field(default=None, description="RVC custom VC cmd")
    gpt_sovits_vc_repo: Optional[str] = Field(default=None, description="GPT-SoVITS VC repo")
    gpt_sovits_vc_custom_cmd: Optional[str] = Field(default=None, description="GPT-SoVITS VC custom cmd")



@router.get('/datasets')
def list_audio_datasets():
    ds_dir = settings.dirs['audio']['datasets']
    res: List[Dict[str, Any]] = []
    if ds_dir.exists():
        for d in ds_dir.iterdir():
            if d.is_dir():
                res.append({'name': d.name, 'path': str(d)})
    return {'datasets': res}


@router.get('/models')
def list_audio_models():
    md = settings.dirs['audio']['models']
    res: List[Dict[str, Any]] = []
    if md.exists():
        for p in md.iterdir():
            if p.is_dir():
                res.append({'name': p.name, 'path': str(p)})
            elif p.is_file():
                res.append({'name': p.name, 'path': str(p)})
    return {'models': res}


@router.post('/train/start')
def train_start(req: AudioTrainStartRequest):
    try:
        return audio_engine.start_training(req.base_model, req.dataset, req.params.model_dump())
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post('/train/stop')
def train_stop():
    return audio_engine.stop_training()


@router.get('/train/status')
def train_status():
    return audio_engine.get_training_status()


@router.get('/train/history')
def train_history():
    return audio_engine.get_training_history()


@router.post('/inference/load')
def inference_load(req: AudioInferenceLoadRequest):
    # まずモデルディレクトリをロード
    res = audio_engine.load_inference_model(req.model_dir, adapter_path=None)

    # repo/cmd をエンジンに記憶（generate で毎回渡しても良いが、UI体験を安定させる）
    try:
        with audio_engine._lock:  # type: ignore[attr-defined]
            if req.gpt_sovits_repo:
                audio_engine._infer_repo = Path(req.gpt_sovits_repo)  # type: ignore[attr-defined]
            if req.custom_infer_cmd:
                audio_engine._infer_custom_cmd = req.custom_infer_cmd  # type: ignore[attr-defined]
            if req.tts_backend:
                audio_engine._tts_backend = str(req.tts_backend)  # type: ignore[attr-defined]
            if req.vc_backend:
                audio_engine._vc_backend = str(req.vc_backend)  # type: ignore[attr-defined]
            if req.xtts_model_id:
                audio_engine._xtts_model_id = str(req.xtts_model_id)  # type: ignore[attr-defined]
            if req.rvc_repo:
                audio_engine._rvc_repo = Path(str(req.rvc_repo))  # type: ignore[attr-defined]
            if req.rvc_custom_cmd:
                audio_engine._rvc_custom_cmd = str(req.rvc_custom_cmd)  # type: ignore[attr-defined]
            if req.gpt_sovits_vc_repo:
                audio_engine._gpt_sovits_vc_repo = Path(str(req.gpt_sovits_vc_repo))  # type: ignore[attr-defined]
            if req.gpt_sovits_vc_custom_cmd:
                audio_engine._gpt_sovits_vc_custom_cmd = str(req.gpt_sovits_vc_custom_cmd)  # type: ignore[attr-defined]
    except Exception:
        pass

    return res


@router.post('/inference/generate')
def inference_generate(req: AudioGenerateRequest):
    try:
        return audio_engine.generate_audio(
            text=req.text,
            preset_id=req.preset_id,
            reference_audio_path=req.reference_audio,
            output_format=req.output_format,
            custom_infer_cmd=req.custom_infer_cmd,
            tts_backend=req.tts_backend,
            vc_backend=req.vc_backend,
            xtts_model_id=req.xtts_model_id,
            xtts_language=req.xtts_language,
            rvc_repo=req.rvc_repo,
            rvc_custom_cmd=req.rvc_custom_cmd,
            gpt_sovits_vc_repo=req.gpt_sovits_vc_repo,
            gpt_sovits_vc_custom_cmd=req.gpt_sovits_vc_custom_cmd,
            postprocess=req.postprocess,
            target_lufs=req.target_lufs,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post('/inference/enqueue_generate')
def inference_enqueue_generate(req: AudioGenerateJobRequest):
    """設計A: 音声生成ジョブをキューへ投入。

    注意:
    - Worker を起動していない場合、ジョブは queued のままになります。
    - 同期生成が必要な場合は既存の /inference/generate を使用してください。
    """

    spec = JobSpec(
        job_type="audio_generate",
        prompt_source={"text": req.text},
        compiled_prompt={"text": req.text},
        model_ref=ModelRef(model_id=req.model_dir, backend=str(req.tts_backend or "gpt_sovits")),
        generation_params={
            "reference_audio": req.reference_audio,
            "output_format": req.output_format,
            "gpt_sovits_repo": req.gpt_sovits_repo,
            "custom_infer_cmd": req.custom_infer_cmd,
            "tts_backend": req.tts_backend,
            "vc_backend": req.vc_backend,
            "xtts_model_id": req.xtts_model_id,
            "xtts_language": req.xtts_language,
            "rvc_repo": req.rvc_repo,
            "rvc_custom_cmd": req.rvc_custom_cmd,
            "gpt_sovits_vc_repo": req.gpt_sovits_vc_repo,
            "gpt_sovits_vc_custom_cmd": req.gpt_sovits_vc_custom_cmd,
        },
        seed=None,
    )
    spec.ensure_request_id()
    spec_hash = spec.hash()
    job_id = sqlite_queue.enqueue(spec.job_type, spec.to_dict(), spec_hash)
    return {"status": "queued", "job_id": job_id, "spec_hash": spec_hash}


@router.get("/train/status/{job_id}")
def get_training_status_by_id(job_id: str):
    """学習ステータス取得（job_id指定）"""
    return job_manager.get_status(job_id)

@router.post("/train/cancel/{job_id}")
def cancel_training(job_id: str):
    """学習キャンセル（job_id指定）"""
    job_manager.stop_job(job_id)
    return {"status": "cancel_requested", "job_id": job_id}
@router.post("/train/rerun/{job_id}")
def train_rerun(job_id: str):
    """過去ジョブの設定で再実行（履歴から再現）"""
    return audio_engine.rerun_training(job_id)


