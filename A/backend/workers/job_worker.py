# -*- coding: utf-8 -*-
"""backend/workers/job_worker.py

設計A: ジョブキュー（SQLite）をポーリングして実行する単機ワーカー。

使い方（Windows例）:

  (venv) > python -m backend.workers.job_worker --worker_id w1 --types image_generate_advanced,audio_generate

ポイント:
- このワーカーは FastAPI サーバーとは別プロセスで起動します。
- BK33 既存の同期 API はそのまま使えますが、
  実運用では enqueue API + ワーカーを推奨します。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

from lora_config import settings
from backend.core.job_spec import JobSpec
from backend.core.sqlite_queue import sqlite_queue
from backend.engines.image import image_engine
from backend.engines.audio import audio_engine


def _ensure_cuda_env() -> None:
    # 文字化け防止
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _handle_image_generate_advanced(spec_d: Dict[str, Any]) -> Dict[str, Any]:
    spec = JobSpec.from_dict(spec_d)
    mr = spec.model_ref.model_id if spec.model_ref else ""
    adapter = None
    if spec.adapter_refs:
        adapter = spec.adapter_refs[0].adapter_id

    # モデルを必要に応じてロード
    if not image_engine.is_inference_model_loaded():
        image_engine.load_inference_model(mr, adapter)
    else:
        # base_model が違う場合はリロード
        # image_engine は内部で base_model を保持しているため、簡易に unload->load
        try:
            if getattr(image_engine, "_pipe_base_model", None) != mr:
                image_engine.unload_inference_model()
                image_engine.load_inference_model(mr, adapter)
        except Exception:
            pass

    gp = spec.generation_params or {}
    res = image_engine.generate_image_advanced(
        prompt=(spec.compiled_prompt or {}).get("prompt", ""),
        negative_prompt=(spec.compiled_prompt or {}).get("negative_prompt", ""),
        width=int(gp.get("width", 1024)),
        height=int(gp.get("height", 1024)),
        steps=int(gp.get("steps", 30)),
        cfg=float(gp.get("cfg", 7.0)),
        seed=spec.seed,
        adapter_path=adapter,
        lora_scale=float(gp.get("lora_scale", 1.0)) if gp.get("lora_scale") is not None else 1.0,
        scheduler=str(gp.get("scheduler", "")),
        preset_id=gp.get("preset_id"),
        hires_scale=float(gp.get("hires_scale", 1.5)),
        hires_steps=int(gp.get("hires_steps", 15)),
        hires_denoise=float(gp.get("hires_denoise", 0.35)),
        use_refiner=bool(gp.get("use_refiner", False)),
        refiner_model=gp.get("refiner_model"),
        controlnet_type=gp.get("controlnet_type"),
        controlnet_model=gp.get("controlnet_model"),
        control_image_base64=gp.get("control_image_base64"),
        init_image_base64=gp.get("init_image_base64"),
        mask_image_base64=gp.get("mask_image_base64"),
        inpaint_mode=gp.get("inpaint_mode"),
    )
    return res


def _handle_audio_generate(spec_d: Dict[str, Any]) -> Dict[str, Any]:
    spec = JobSpec.from_dict(spec_d)
    mr = spec.model_ref.model_id if spec.model_ref else ""
    gp = spec.generation_params or {}

    # モデルロード（model_dir）
    if not audio_engine.is_inference_model_loaded():
        audio_engine.load_inference_model(mr, adapter_path=None)
    else:
        try:
            if getattr(audio_engine, "_infer_model_dir", None) is None:
                audio_engine.load_inference_model(mr, adapter_path=None)
        except Exception:
            pass

    # repo/cmd は spec の優先度で反映
    try:
        with audio_engine._lock:  # type: ignore[attr-defined]
            if gp.get("gpt_sovits_repo"):
                from pathlib import Path

                audio_engine._infer_repo = Path(str(gp.get("gpt_sovits_repo")))  # type: ignore[attr-defined]
            if gp.get("custom_infer_cmd"):
                audio_engine._infer_custom_cmd = str(gp.get("custom_infer_cmd"))  # type: ignore[attr-defined]
    except Exception:
        pass

    res = audio_engine.generate_audio(
        text=(spec.compiled_prompt or {}).get("text", ""),
        reference_audio_path=str(gp.get("reference_audio", "")),
        gpt_sovits_repo=gp.get("gpt_sovits_repo"),
        custom_infer_cmd=gp.get("custom_infer_cmd"),
        output_format=str(gp.get("output_format", "wav")),
        tts_backend=gp.get("tts_backend"),
        vc_backend=gp.get("vc_backend"),
        xtts_model_id=gp.get("xtts_model_id"),
        xtts_language=str(gp.get("xtts_language", "ja")),
        rvc_repo=gp.get("rvc_repo"),
        rvc_custom_cmd=gp.get("rvc_custom_cmd"),
        gpt_sovits_vc_repo=gp.get("gpt_sovits_vc_repo"),
        gpt_sovits_vc_custom_cmd=gp.get("gpt_sovits_vc_custom_cmd"),
    )
    return res


HANDLERS = {
    "image_generate_advanced": _handle_image_generate_advanced,
    "audio_generate": _handle_audio_generate,
}


def main(argv=None) -> int:
    _ensure_cuda_env()
    p = argparse.ArgumentParser()
    p.add_argument("--worker_id", default="worker1", help="ワーカー識別子（ログ/DB用）")
    p.add_argument(
        "--types",
        default="image_generate_advanced,audio_generate",
        help="処理する job_type のCSV（例: image_generate_advanced,audio_generate）",
    )
    p.add_argument("--poll_interval", type=float, default=0.5, help="ポーリング間隔（秒）")
    args = p.parse_args(argv)

    types = [t.strip() for t in str(args.types).split(",") if t.strip()]
    if not types:
        types = ["image_generate_advanced", "audio_generate"]

    print("=" * 60)
    print("BK33 Job Worker (DesignA)")
    print(f"worker_id: {args.worker_id}")
    print(f"job_types : {types}")
    print(f"db_path   : {sqlite_queue.db_path}")
    print("=" * 60)

    while True:
        did = False
        for jt in types:
            job = sqlite_queue.dequeue(args.worker_id, jt, timeout_sec=0)
            if not job:
                continue
            did = True
            job_id = job["job_id"]
            spec_d = job["spec"]

            print(f"[worker] start job_id={job_id} type={jt}")
            try:
                handler = HANDLERS.get(jt)
                if handler is None:
                    raise RuntimeError(f"No handler for job_type={jt}")
                sqlite_queue.heartbeat(job_id, 0.01)
                res = handler(spec_d)
                if isinstance(res, dict) and res.get("status") == "ok":
                    sqlite_queue.ack(job_id, "succeeded", result=res, error="")
                    print(f"[worker] succeeded job_id={job_id}")
                else:
                    sqlite_queue.ack(job_id, "failed", result=res if isinstance(res, dict) else None, error=str(res))
                    print(f"[worker] failed job_id={job_id}: {res}")
            except Exception as e:
                sqlite_queue.ack(job_id, "failed", result=None, error=str(e))
                print(f"[worker] exception job_id={job_id}: {e}")

        if not did:
            time.sleep(max(0.1, float(args.poll_interval)))


if __name__ == "__main__":
    raise SystemExit(main())
