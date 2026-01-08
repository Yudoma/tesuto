# -*- coding: utf-8 -*-
""" 
backend/routes/image_routes.py

Image (Diffusers) モダリティ専用API。

prefix は main_router.py 側で /image を付与しているため、
このファイル内のパスは ""（空）基準です。

例:
- GET  /api/image/models
- POST /api/image/train/start
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from lora_config import settings
from backend.core.job_manager import job_manager
from backend.engines.image import image_engine
from backend.engines.image_presets import get_image_presets
from backend.core.job_spec import JobSpec, ModelRef, AdapterRef
from backend.core.sqlite_queue import sqlite_queue


router = APIRouter()


class DownloadModelRequest(BaseModel):
    repo_id: str = Field(..., description="HuggingFace repo id")


# ===========================================================
# Helpers
# ===========================================================


def _normalize_rel_posix(path: str) -> str:
    p = (path or "").replace("\\", "/").lstrip("/")
    p = re.sub(r"^[A-Za-z]:", "", p).lstrip("/")
    p = re.sub(r"/+", "/", p)
    return p


def _safe_under(base: Path, rel_path: str) -> Path:
    rel = _normalize_rel_posix(rel_path)
    if rel in ("", "."):
        raise HTTPException(400, "Invalid path.")
    base = base.resolve()
    candidate = (base / rel).resolve()
    if not str(candidate).startswith(str(base)):
        raise HTTPException(400, "Path traversal detected.")
    return candidate


# ===========================================================
# Request/Response Models
# ===========================================================


class ImageTrainParams(BaseModel):
    # モデル系
    model_type: str = Field(default="sdxl", description="sdxl or sd15")
    resolution: int = Field(default=1024, description="base resolution (sdxl=1024, sd15=512)")

    # 学習系
    epochs: int = 1
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: Optional[str] = None  # JSON文字列で渡せる

    # VRAM節約
    mixed_precision: str = "fp16"  # no / fp16 / bf16
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    use_xformers: bool = False
    allow_tf32: bool = True

    # データ
    caption_ext: str = ".txt"
    shuffle_tags: bool = True
    caption_dropout: float = 0.0

    # その他
    max_train_steps: int = 0
    save_every_n_steps: int = 200
    sample_prompt: Optional[str] = None  # (任意) 学習中サンプル生成用プロンプト
    seed: int = 42
    output_name: Optional[str] = None


class ImageTrainStartRequest(BaseModel):
    base_model: str
    dataset: str
    params: ImageTrainParams


class ImageInferenceLoadRequest(BaseModel):
    base_model: str
    adapter_path: Optional[str] = None


class ImageGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance_scale: float = 5.0
    seed: int = 0
    adapter_path: Optional[str] = None
    lora_scale: float = 1.0


class ImageGenerateAdvancedRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg: float = 7.0
    seed: Optional[int] = None
    adapter_path: Optional[str] = None
    lora_scale: float = 1.0

    # advanced
    preset_id: Optional[str] = None
    scheduler: str = ""
    hires_scale: float = 1.5
    hires_steps: int = 15
    hires_denoise: float = 0.35
    use_refiner: bool = False
    refiner_model: Optional[str] = None

    # ControlNet（任意）
    controlnet_type: Optional[str] = None
    controlnet_model: Optional[str] = None
    control_image_base64: Optional[str] = None

    # Inpaint / Img2Img（任意・後方互換）
    init_image_base64: Optional[str] = None
    mask_image_base64: Optional[str] = None
    inpaint_mode: Optional[str] = Field(default=None, description="inpaint / img2img / outpaint（未指定なら自動）")


class ImageGenerateJobRequest(ImageGenerateAdvancedRequest):
    """設計A: 非同期ジョブ投入用（Workerで生成）。

    既存 /inference/generate_advanced は同期生成（互換維持）。
    実運用では GPU 負荷を HTTP プロセスから切り離したいので、
    この API は sqlite_queue へジョブを enqueue します。
    """

    base_model: str = Field(..., description="diffusers base model name (models/image 配下)")
    adapter_path: Optional[str] = Field(default=None, description="LoRA adapter path (optional)")
    lora_scale: float = 1.0



# ===========================================================
# Validation helpers (Inpaint / ControlNet)
# ===========================================================

def _validate_inpaint_controlnet(req: ImageGenerateAdvancedRequest) -> None:
    """Inpaint / Img2Img / Outpaint / ControlNet の入力整合性を検証します（仕様固定）。

    仕様（設計A・明文化）:
    - init_image_base64 がある場合: img2img または inpaint/outpaint を行う。
    - mask_image_base64 がある場合: init_image_base64 も必須。
    - inpaint_mode が "inpaint" / "outpaint" の場合: init + mask が必須。
    - inpaint_mode が "img2img" の場合: init は必須、mask は禁止（mask があるなら inpaint を使う）。
    - ControlNet は「単体」利用を原則とし、Inpaint(=mask利用) と同時指定は不可（相互排他）。
      ※将来的に併用を許可する場合は、Engine 側で明示対応しこの検証条件を更新すること。
    """
    try:
        init_b64 = (req.init_image_base64 or "").strip()
        mask_b64 = (req.mask_image_base64 or "").strip()
        mode = (req.inpaint_mode or "").strip().lower()
        cn = (req.controlnet_type or "").strip()

        if mask_b64 and (not init_b64):
            raise HTTPException(400, "mask_image_base64 を指定する場合は init_image_base64 も必須です。")

        if mode in ("inpaint", "outpaint"):
            if not init_b64 or not mask_b64:
                raise HTTPException(400, f"inpaint_mode={mode} の場合は init_image_base64 と mask_image_base64 の両方が必須です。")

        if mode == "img2img":
            if not init_b64:
                raise HTTPException(400, "inpaint_mode=img2img の場合は init_image_base64 が必須です。")
            if mask_b64:
                raise HTTPException(400, "inpaint_mode=img2img の場合、mask_image_base64 は指定できません（inpaint を使用してください）。")

        # mode 未指定なら自動だが、mask ありなら inpaint、initのみなら img2img とみなす
        # ここでは入力の一貫性だけ確認する

        if cn and mask_b64:
            raise HTTPException(400, "ControlNet と Inpaint（mask指定）の併用は現在サポートしていません。どちらか一方を使用してください。")

        # controlnet を使う場合、control_image_base64 が必要（UI/運用ミス防止）
        if cn:
            if not (req.control_image_base64 or "").strip():
                raise HTTPException(400, "ControlNet を使用する場合は control_image_base64 が必須です。")
    except HTTPException:
        raise
    except Exception:
        # fail-soft: ここで例外を出さない（ただし整合性が崩れると Engine 側で落ちる）
        return
# ===========================================================
# Routes: Models / Datasets
# ===========================================================


@router.get("/models")
def list_image_models():
    models_dir = settings.dirs["image"]["models"]
    res: List[Dict[str, Any]] = []
    if models_dir.exists():
        for p in models_dir.iterdir():
            if p.is_dir():
                # diffusers形式かを簡易判定
                if (p / "model_index.json").exists() or (p / "unet").exists():
                    res.append({"name": p.name, "type": "diffusers", "path": str(p)})
                else:
                    res.append({"name": p.name, "type": "folder", "path": str(p)})
            elif p.is_file():
                suf = p.suffix.lower()
                if suf in [".safetensors", ".ckpt"]:
                    res.append({"name": p.name, "type": "single_file", "path": str(p)})
                else:
                    res.append({"name": p.name, "type": "file", "path": str(p)})
    return {"models": res}

@router.post("/models/download")
def download_image_model(req: DownloadModelRequest, background_tasks: BackgroundTasks):
    """HuggingFaceから画像モデルをダウンロード (バックグラウンド)"""
    from huggingface_hub import snapshot_download

    def _bg_download(repo_id: str):
        try:
            # 保存先は models/image/<repoの末尾>
            local_dir = settings.dirs["image"]["models"] / repo_id.split("/")[-1]
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Image Download] Starting: {repo_id} -> {local_dir}")
            snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
            print(f"[Image Download] Completed: {repo_id}")
        except Exception as e:
            print(f"[Image Download] Failed: {e}")

    background_tasks.add_task(_bg_download, req.repo_id)
    return {"status": "started", "repo_id": req.repo_id}




@router.get("/models/{model_name}/meta")
def model_meta(model_name: str):
    """モデルのメタ情報（UI表示用）"""
    model_dir = settings.dirs["image"]["models"] / model_name
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
        "modality": "image"
    }

@router.get("/models/{model_name}/predelete_check")
def model_predelete_check(model_name: str):
    """モデル削除前の参照チェック（軽量）"""
    refs = []
    # history files（存在する範囲）
    candidates = [
        settings.logs_dir / "history.json",
        settings.logs_dir / "history_image.json",
        settings.logs_dir / "history_image.json"
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

    active = False
    try:
        st = image_engine.get_training_status()
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


@router.delete("/models/{model_name}")
def delete_image_model(model_name: str):
    path = _safe_under(settings.dirs["image"]["models"], model_name)
    if not path.exists():
        raise HTTPException(404, "Model not found")
    try:
        shutil.rmtree(path)
        return {"status": "deleted", "name": model_name}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete: {e}")


@router.get("/datasets")
def list_image_datasets():
    ds_dir = settings.dirs["image"]["datasets"]
    res: List[Dict[str, Any]] = []
    if ds_dir.exists():
        for d in ds_dir.iterdir():
            if d.is_dir():
                # 画像ファイル数（目安）を付与して UI に情報を返す
                ex = None
                cnt = 0
                try:
                    for p in d.iterdir():
                        if not p.is_file():
                            continue
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                            cnt += 1
                            if ex is None:
                                ex = p.name
                except Exception:
                    # 権限や破損ディレクトリ等は無視（ローカル運用の利便性優先）
                    cnt = cnt
                res.append({"name": d.name, "path": str(d), "count": cnt, "example": ex})
            elif d.is_file():
                # 単一zip等は Phase 1 では想定しないが、念のため
                res.append({"name": d.name, "path": str(d), "count": 0, "example": None})
    return {"datasets": res}


# ===========================================================
# Routes: Training
# ===========================================================


@router.post("/train/start")
def train_start(req: ImageTrainStartRequest):
    try:
        return image_engine.start_training(req.base_model, req.dataset, req.params.model_dump())
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/train/stop")
def train_stop():
    return image_engine.stop_training()


@router.get("/train/status")
def train_status():
    return image_engine.get_training_status()

@router.get("/train/latest_sample")
def train_latest_sample():
    """学習中に生成された最新サンプル画像（sample_latest.png または sample_*.png）を返す。"""
    try:
        st = image_engine.get_training_status()
        params = (st or {}).get("params") or {}
        out_dir = params.get("output_dir")
        if not out_dir:
            return {"exists": False}

        out_path = Path(str(out_dir))
        samples_dir = out_path / "samples"
        if not samples_dir.exists():
            return {"exists": False}

        # 優先: sample_latest.png
        cand = samples_dir / "sample_latest.png"
        if cand.exists():
            data = cand.read_bytes()
        else:
            # mtime が新しい順で sample_*.png を探索
            pngs = sorted(samples_dir.glob("sample_*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not pngs:
                return {"exists": False}
            cand = pngs[0]
            data = cand.read_bytes()

        import base64
        b64 = base64.b64encode(data).decode("utf-8")
        return {"exists": True, "filename": cand.name, "image_base64": "data:image/png;base64," + b64}
    except Exception as e:
        raise HTTPException(500, str(e))



@router.post("/train/rerun/{job_id}")
def train_rerun(job_id: str):
    """過去ジョブの設定で再実行（履歴から再現）"""
    return image_engine.rerun_training(job_id)

@router.get("/train/history")
def train_history():
    return image_engine.get_training_history()


# ===========================================================
# Routes: Inference
# ===========================================================


@router.post("/inference/load")
def inference_load(req: ImageInferenceLoadRequest):
    try:
        return image_engine.load_inference_model(req.base_model, req.adapter_path)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/inference/unload")
def inference_unload():
    return image_engine.unload_inference_model()


@router.get("/inference/is_loaded")
def inference_is_loaded():
    return {"loaded": bool(image_engine.is_inference_model_loaded())}


@router.get("/inference/presets")
def inference_presets():
    return {"presets": get_image_presets()}


@router.post("/inference/enqueue_generate_advanced")
def inference_enqueue_generate_advanced(req: ImageGenerateJobRequest):
    """設計A: 生成ジョブをキューに投入し、job_id を返す。

    注意:
    - Worker を起動していない場合、ジョブは queued のままになります。
    - 同期生成が必要な場合は既存の /inference/generate_advanced を使用してください。
    """

    # JobSpec の構築（再現性のためseedは確定させる）
    spec = JobSpec(
        job_type="image_generate_advanced",
        prompt_source={"prompt": req.prompt, "negative_prompt": req.negative_prompt},
        compiled_prompt={"prompt": req.prompt, "negative_prompt": req.negative_prompt},
        model_ref=ModelRef(model_id=req.base_model, backend="diffusers"),
        adapter_refs=[AdapterRef(adapter_id=req.adapter_path, weight=float(req.lora_scale))] if req.adapter_path else [],
        generation_params={
            "width": int(req.width),
            "height": int(req.height),
            "steps": int(req.steps),
            "cfg": float(req.cfg),
            "scheduler": str(req.scheduler or ""),
            "preset_id": req.preset_id,
            "hires_scale": float(req.hires_scale),
            "hires_steps": int(req.hires_steps),
            "hires_denoise": float(req.hires_denoise),
            "use_refiner": bool(req.use_refiner),
            "refiner_model": req.refiner_model,
            "controlnet_type": req.controlnet_type,
            "controlnet_model": req.controlnet_model,
            "control_image_base64": req.control_image_base64,
            "init_image_base64": req.init_image_base64,
            "mask_image_base64": req.mask_image_base64,
            "inpaint_mode": req.inpaint_mode,
        },
        seed=req.seed,
        runtime_hints={"device": "cuda" if settings else ""},
    )
    spec.ensure_request_id()
    spec.ensure_seed()
    spec_hash = spec.hash()
    job_id = sqlite_queue.enqueue(spec.job_type, spec.to_dict(), spec_hash)
    return {"status": "queued", "job_id": job_id, "spec_hash": spec_hash, "seed": spec.seed}


@router.post("/inference/generate_advanced")
def inference_generate_advanced(req: ImageGenerateAdvancedRequest):
    _validate_inpaint_controlnet(req)
    if not image_engine.is_inference_model_loaded():
        raise HTTPException(400, "Inference model is not loaded.")

    res = image_engine.generate_image_advanced(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        steps=req.steps,
        cfg=req.cfg,
        seed=req.seed,
        adapter_path=req.adapter_path,
        lora_scale=req.lora_scale,
        scheduler=req.scheduler,
        preset_id=req.preset_id,
        hires_scale=req.hires_scale,
        hires_steps=req.hires_steps,
        hires_denoise=req.hires_denoise,
        use_refiner=req.use_refiner,
        refiner_model=req.refiner_model,
        controlnet_type=req.controlnet_type,
        controlnet_model=req.controlnet_model,
        control_image_base64=req.control_image_base64,
        init_image_base64=req.init_image_base64,
        mask_image_base64=req.mask_image_base64,
        inpaint_mode=req.inpaint_mode,
    )
    if res.get("status") != "ok":
        raise HTTPException(500, res.get("message", "generation failed"))
    return res


@router.post("/inference/enqueue_generate_advanced")
def enqueue_generate_advanced(req: ImageGenerateJobRequest):
    _validate_inpaint_controlnet(req)
    """画像生成（advanced）をジョブキューへ投入。

    返す job_id を /api/jobs/{job_id} でポーリングしてください。
    Worker が完了すると result に artifact_id が入ります。
    """

    # JobSpec を構築（最小）
    spec = JobSpec(
        job_type="image_generate_advanced",
        prompt_source={"prompt": req.prompt, "negative_prompt": req.negative_prompt},
        compiled_prompt={"prompt": req.prompt, "negative_prompt": req.negative_prompt},
        model_ref=ModelRef(model_id=req.base_model, backend="diffusers"),
        adapter_refs=[AdapterRef(adapter_id=req.adapter_path, weight=float(req.lora_scale))] if req.adapter_path else [],
        generation_params={
            "width": int(req.width),
            "height": int(req.height),
            "steps": int(req.steps),
            "cfg": float(req.cfg),
            "seed": req.seed,
            "scheduler": req.scheduler,
            "preset_id": req.preset_id,
            "hires_scale": float(req.hires_scale),
            "hires_steps": int(req.hires_steps),
            "hires_denoise": float(req.hires_denoise),
            "use_refiner": bool(req.use_refiner),
            "refiner_model": req.refiner_model,
            "controlnet_type": req.controlnet_type,
            "controlnet_model": req.controlnet_model,
            "control_image_base64": req.control_image_base64,
            "lora_scale": float(req.lora_scale),
        },
        seed=req.seed,
        app_version="BK33",
        pipeline_version="designA",
    )
    spec.ensure_request_id().ensure_seed()
    spec_hash = spec.hash()
    job_id = sqlite_queue.enqueue("image_generate_advanced", spec.to_dict(), spec_hash)
    return {"status": "queued", "job_id": job_id, "spec_hash": spec_hash, "seed": spec.seed}


@router.post("/inference/enqueue_generate_advanced")
def inference_enqueue_generate_advanced(req: ImageGenerateJobRequest):
    """画像生成ジョブをキューへ投入し、job_id を返す。

    注意:
    - Worker（backend/workers/run_worker.py）を別プロセスで起動している前提。
    - UIは段階的移行が可能なように、同期APIは残している。
    """
    # JobSpec（再現性の核）
    spec = JobSpec(
        job_type="image_generate_advanced",
        prompt_source={"prompt": req.prompt, "negative_prompt": req.negative_prompt},
        compiled_prompt={
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "preset_id": req.preset_id,
        },
        model_ref=ModelRef(model_id=req.base_model, backend="diffusers"),
        adapter_refs=[AdapterRef(adapter_id=req.adapter_path, weight=float(req.lora_scale))] if req.adapter_path else [],
        generation_params={
            "width": int(req.width),
            "height": int(req.height),
            "steps": int(req.steps),
            "cfg": float(req.cfg),
            "scheduler": req.scheduler,
            "hires_scale": float(req.hires_scale),
            "hires_steps": int(req.hires_steps),
            "hires_denoise": float(req.hires_denoise),
            "use_refiner": bool(req.use_refiner),
            "refiner_model": req.refiner_model,
            "controlnet_type": req.controlnet_type,
            "controlnet_model": req.controlnet_model,
            "control_image_base64": req.control_image_base64,
            "preset_id": req.preset_id,
        },
        seed=req.seed,
        runtime_hints={"device": "cuda" if settings else ""},
    )
    spec.ensure_request_id().ensure_seed()
    spec_hash = spec.hash()
    job_id = sqlite_queue.enqueue(job_type=spec.job_type, spec=spec.to_dict(), spec_hash=spec_hash)
    return {"status": "queued", "job_id": job_id, "spec_hash": spec_hash, "seed": spec.seed}



@router.post("/inference/generate")
def inference_generate(req: ImageGenerateRequest):
    if not image_engine.is_inference_model_loaded():
        raise HTTPException(400, "Inference model is not loaded.")

    res = image_engine.generate_image(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        adapter_path=req.adapter_path,
        lora_scale=req.lora_scale,
    )
    if res.get("status") != "ok":
        raise HTTPException(500, res.get("message", "generation failed"))
    return res


@router.get("/train/status/{job_id}")
def get_training_status_by_id(job_id: str):
    """学習ステータス取得（job_id指定）"""
    return job_manager.get_status(job_id)

@router.post("/train/cancel/{job_id}")
def cancel_training(job_id: str):
    """学習キャンセル（job_id指定）"""
    job_manager.stop_job(job_id)
    return {"status": "cancel_requested", "job_id": job_id}
