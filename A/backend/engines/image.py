# -*- coding: utf-8 -*-
""" 
backend/engines/image.py

Image (Diffusers) モダリティ用エンジン。

- JobManager を使って train_image.py をサブプロセス起動し、ログを監視
- 推論パイプライン（SDXL/SD1.5）をロードし、LoRA を適用して画像生成（Base64返却）

設計方針:
- TextEngine と同様のインターフェースを維持（BaseEngine）
- Windows + Consumer GPU を前提に、FP16/Offload などの保険を用意
"""

from __future__ import annotations

import base64
import gc
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from lora_config import settings
from backend.core.dataset_report import build_dataset_report
from backend.core.job_manager import job_manager
from backend.core.artifact_store import artifact_store
def _sha256_data_uri(data_uri: str) -> str:
    """data:image/...;base64,.... の中身(bytes)のsha256。失敗時は空。"""
    try:
        import base64
        s = (data_uri or "").strip()
        if not s:
            return ""
        if "," in s:
            s = s.split(",", 1)[1]
        b = base64.b64decode(s, validate=False)
        return _sha256_bytes(b)
    except Exception:
        return ""

from backend.engines.image_presets import get_image_presets, find_preset
from backend.engines.image_pipelines import apply_scheduler, prepare_pipe_common, hires_img2img, refine, decode_base64_image_to_pil, decode_and_preprocess_control_image, get_inpaint_pipeline
from backend.engines.image_metadata import model_identity
from backend.engines.base import BaseEngine


class ImageEngine(BaseEngine):
    def __init__(self):
        self.worker_script = settings.base_dir / "backend" / "workers" / "train_image.py"

        self._pipe = None
        self._pipe_base_model = None
        self._pipe_adapter_path = None
        self._pipe_is_sdxl = False
        self._img2img_pipe = None
        self._refiner_pipe = None
        self._refiner_model = None
        # ControlNet / Reference 系（設計A: 破綻補正の標準化）
        self._control_pipes = {}  # (base_model, controlnet_id) -> pipe
        self._controlnet_last = None
        self._pipe_lock = threading.Lock()

        # 画像学習の履歴はテキストと分ける
        self._history_file = settings.logs_dir / "history_image.json"

    # =========================================================================
    # Training Job Management (BaseEngine Implementation)
    # =========================================================================

    def start_training(self, base_model: str, dataset: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # -------------------------------------------------
        # Image LoRA Training (diffusers/accelerate)
        # train_image.py を JobManager 経由でサブプロセス起動する。
        # -------------------------------------------------
        base_model_path = settings.dirs["image"]["models"] / base_model
        if not base_model_path.exists():
            base_model_path = Path(base_model)

        dataset_path = settings.dirs["image"]["datasets"] / dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = settings.dirs["image"]["output"] / f"{dataset}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        worker_script = Path(__file__).resolve().parents[1] / "workers" / "train_image.py"
        if not worker_script.exists():
            # フォールバック: 実行ディレクトリ相対
            worker_script = Path("backend/workers/train_image.py")

        cmd = [
            sys.executable,
            str(worker_script),
            "--dataset_dir", str(dataset_path),
            "--output_dir", str(out_dir),
            "--base_model_path", str(base_model_path),
        ]

        # params は UI 側でキー名がブレることがあるため正規化する
        key_map = {
            "use_xformers": "xformers",
        }

        allowed_value_keys = {
            "resolution",
            "train_batch_size",
            "gradient_accumulation_steps",
            "epochs",
            "learning_rate",
            "max_train_steps",
            "lr_scheduler",
            "warmup_steps",
            "warmup_ratio",
            "seed",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "mixed_precision",
            "tag_delim",
            "caption_dropout",
            "max_token_length",
            "snr_gamma",
            "save_every_n_steps",
            "sample_prompt",
        }

        allowed_flag_keys = {
            "gradient_checkpointing",
            "use_8bit_adam",
            "xformers",
            "shuffle_tags",
        }

        for k, v in (params or {}).items():
            if k is None:
                continue
            k = key_map.get(str(k), str(k))

            if v is None:
                continue

            if isinstance(v, bool):
                if k in allowed_flag_keys and v:
                    cmd.append(f"--{k}")
                continue

            if k not in allowed_value_keys:
                continue

            cmd.extend([f"--{k}", str(v)])

        ui_params = dict(params or {})
        ui_params.update({
            "base_model": base_model,
            "dataset": dataset,
            "output_dir": str(out_dir),
        })

        job_info = self.job_manager.start_job(
            cmd=cmd,
            params=ui_params,
            cwd=Path(__file__).resolve().parents[2],
            env=os.environ.copy(),
            log_prefix="train_image",
        )

        # データセット検査結果を runs/<job_id>/dataset_report.json に保存（ベストエフォート）
        try:
            job_id = job_info.get("job_id") or job_info.get("id") or job_info.get("jobId")
            if job_id:
                run_dir = settings.runs_root / str(job_id)
                run_dir.mkdir(parents=True, exist_ok=True)
                rep = build_dataset_report("image", dataset_path, params=ui_params)
                (run_dir / "dataset_report.json").write_text(
                    json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except Exception:
            pass


        try:
            if isinstance(params, dict):
                params["log_file"] = job_info.get("log_file")
        except Exception:
            pass
        try:
            self._append_history(job_info.get("job_id"), base_model, dataset, params, "running")
        except Exception:
            pass
        return job_info
    def stop_training(self) -> Dict[str, str]:
        job_manager.stop_job()
        return {"status": "stopped"}

    def get_training_status(self) -> Dict[str, Any]:
        status = job_manager.get_status()
        if status.get("status") in ["completed", "failed", "stopped"]:
            self._update_history_status(status.get("job_id"), status.get("status"))
        return status

    
    def rerun_training(self, job_id: str) -> Dict[str, Any]:
        """履歴から同一条件で再学習を開始する"""
        hist = self.get_training_history().get("history") or []
        target = None
        for item in hist:
            if str(item.get("id")) == str(job_id):
                target = item
                break
        if not target:
            raise RuntimeError("履歴が見つかりませんでした。")
        model = target.get("model")
        dataset = target.get("dataset")
        params = (target.get("params") or {})
        return self.start_training(model, dataset, params)

    def get_training_history(self) -> Dict[str, Any]:
        if not self._history_file.exists():
            return {"history": []}
        try:
            hist = json.loads(self._history_file.read_text(encoding="utf-8"))
            hist.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return {"history": hist}
        except Exception:
            return {"history": []}

    def _append_history(self, job_id: str, model: str, dataset: str, params: Dict[str, Any], status: str):
        hist = []
        if self._history_file.exists():
            try:
                hist = json.loads(self._history_file.read_text(encoding="utf-8"))
            except Exception:
                hist = []
        entry = {
            "id": job_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "dataset": dataset,
            "status": status,
            "params": params,
            "final_loss": None,
        }
        hist.append(entry)
        self._history_file.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")

    def _update_history_status(self, job_id: Optional[str], status: str):
        if not job_id or not self._history_file.exists():
            return
        try:
            hist = json.loads(self._history_file.read_text(encoding="utf-8"))
            updated = False
            for item in hist:
                if item.get("id") != job_id:
                    continue
                if item.get("status") not in ["running", "idle"]:
                    continue
                item["status"] = status

                # ログ末尾から loss を拾う（train_image.py は {"loss": ...} を出す）
                loss = None
                try:
                    job_logs = job_manager.current_job.get("logs", [])
                    for line in reversed(job_logs):
                        if '"loss"' in line:
                            try:
                                data = json.loads(line)
                                if isinstance(data, dict) and "loss" in data:
                                    loss = data.get("loss")
                                    break
                            except Exception:
                                continue
                except Exception:
                    pass
                item["final_loss"] = loss
                updated = True
                break
            if updated:
                self._history_file.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    # =========================================================================
    # Inference / Verification (BaseEngine Implementation)
    # =========================================================================

    def load_inference_model(self, base_model: str, adapter_path: Optional[str] = None) -> Dict[str, Any]:
        with self._pipe_lock:
            self.unload_inference_model()

            base_model_path = settings.dirs["image"]["models"] / base_model
            if not base_model_path.exists():
                base_model_path = Path(base_model)

            # adapter は output/image or models/image にある想定
            adapter_full = None
            if adapter_path:
                p1 = settings.dirs["image"]["output"] / adapter_path
                p2 = settings.dirs["image"]["models"] / adapter_path
                if p1.exists():
                    adapter_full = p1
                elif p2.exists():
                    adapter_full = p2

            try:
                from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

                torch_dtype = torch.float16
                # Windowsの一部環境では bf16 が不安定な場合があるため、推論は fp16 を優先

                                # SDXL / SD1.5 自動判定:
                # - Diffusersフォルダ: model_index.json の _class_name を参照（無ければ try-load で判定）
                # - 単一モデルファイル（.safetensors/.ckpt）: SDXL を先に from_single_file で試し、失敗時に SD1.5 へフォールバック
                is_sdxl = False
                pipe = None
                load_err = None

                is_single_file = base_model_path.is_file() and base_model_path.suffix.lower() in [".safetensors", ".ckpt"]

                def _try_load_sdxl():
                    if is_single_file:
                        return StableDiffusionXLPipeline.from_single_file(
                            str(base_model_path),
                            torch_dtype=torch_dtype,
                        )
                    # diffusersフォルダ
                    variant = "fp16" if (base_model_path / "unet").exists() else None
                    kwargs = {
                        "torch_dtype": torch_dtype,
                        "use_safetensors": True,
                    }
                    if variant is not None:
                        kwargs["variant"] = variant
                    return StableDiffusionXLPipeline.from_pretrained(str(base_model_path), **kwargs)

                def _try_load_sd15():
                    if is_single_file:
                        return StableDiffusionPipeline.from_single_file(
                            str(base_model_path),
                            torch_dtype=torch_dtype,
                            safety_checker=None,
                        )
                    return StableDiffusionPipeline.from_pretrained(
                        str(base_model_path),
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        safety_checker=None,
                    )

                if not is_single_file:
                    # model_index.json があればそれを優先
                    try:
                        mi = base_model_path / "model_index.json"
                        if mi.exists():
                            data = json.loads(mi.read_text(encoding="utf-8"))
                            cls = str(data.get("_class_name", ""))
                            if "StableDiffusionXLPipeline" in cls:
                                pipe = _try_load_sdxl()
                                is_sdxl = True
                    except Exception:
                        pipe = None

                if pipe is None:
                    # try-load で判定
                    try:
                        pipe = _try_load_sdxl()
                        is_sdxl = True
                    except Exception as e:
                        load_err = e

                if pipe is None:
                    try:
                        pipe = _try_load_sd15()
                        is_sdxl = False
                    except Exception as e:
                        raise RuntimeError(f"Failed to load base model: {base_model_path} (sdxl_err={load_err}, sd_err={e})")
# メモリ節約：安全側でオフロードを使えるようにする
                # ただし offload は速度が落ちるので、まず cuda を試す
                if torch.cuda.is_available():
                    pipe = pipe.to("cuda")
                else:
                    pipe = pipe.to("cpu")

                pipe.set_progress_bar_config(disable=True)

                if adapter_full is not None:
                    pipe.load_lora_weights(str(adapter_full))
                    # adapter の強度は generate_image で cross_attention_kwargs で調整

                self._pipe = pipe
                self._pipe_base_model = str(base_model)
                self._pipe_adapter_path = str(adapter_path) if adapter_path else None
                self._pipe_is_sdxl = bool(is_sdxl)
                self._img2img_pipe = None
                self._refiner_pipe = None
                self._refiner_model = None

                return {"status": "loaded", "base_model": base_model, "adapter_path": adapter_path}

            except Exception as e:
                return {"status": "error", "message": str(e)}

    def unload_inference_model(self) -> Dict[str, str]:
        with self._pipe_lock:
            if self._pipe is not None:
                try:
                    # LoRAが載っている場合は外す（存在しない場合もあるので安全に）
                    if hasattr(self._pipe, "unload_lora_weights"):
                        try:
                            self._pipe.unload_lora_weights()
                        except Exception:
                            pass
                except Exception:
                    pass

                self._pipe = None
                self._pipe_base_model = None
                self._pipe_adapter_path = None
                self._pipe_is_sdxl = False
                self._img2img_pipe = None
                self._refiner_pipe = None
                self._refiner_model = None
                self._last_pipe_use = None
                torch.cuda.empty_cache()
            return {"status": "unloaded"}

    def is_inference_model_loaded(self) -> bool:
        return self._pipe is not None

    # =========================================================================
    # Image-specific API
    # =========================================================================

    




    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
        adapter_path: Optional[str] = None,
        lora_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """ロード済みのパイプラインで画像生成し、Base64 PNG を返す。

        adapter_path が指定された場合:
          - 既にロード済みの LoRA と異なるなら、可能であれば LoRA を差し替える
          - diffusers のバージョン差異により unload_lora_weights が無い場合があるため、
            その場合は安全側で「新しい LoRA を追加でロード」し、同名 adapter なら set_adapters を試す
        """
        with self._pipe_lock:
            if self._pipe is None:
                return {"status": "error", "message": "Inference model is not loaded."}

            pipe = self._pipe
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # adapter の解決（相対指定を許可）
            adapter_full: Optional[Path] = None
            if adapter_path:
                p1 = settings.dirs["image"]["output"] / adapter_path
                p2 = settings.dirs["image"]["models"] / adapter_path
                if p1.exists():
                    adapter_full = p1
                elif p2.exists():
                    adapter_full = p2
                else:
                    # 絶対/任意パスも許可（存在検証のみ）
                    cand = Path(adapter_path)
                    if cand.exists():
                        adapter_full = cand

            # LoRA 差し替え（可能なら）
            try:
                if adapter_full is not None:
                    cur = str(self._pipe_adapter_path) if self._pipe_adapter_path else None
                    newp = str(adapter_path)
                    if cur != newp:
                        # 既存LoRAを外せるAPIがあれば外す
                        if hasattr(pipe, "unload_lora_weights"):
                            try:
                                pipe.unload_lora_weights()
                            except Exception:
                                pass
                        # 新しいLoRAをロード
                        pipe.load_lora_weights(str(adapter_full))
                        self._pipe_adapter_path = newp
            except Exception as e:
                return {"status": "error", "message": f"Failed to load LoRA adapter: {e}"}

            if seed is None or int(seed) == 0:
                seed = int(time.time() * 1000) % (2**31 - 1)

            generator = torch.Generator(device=device).manual_seed(int(seed))

            # LoRA強度: diffusers の一部版では cross_attention_kwargs={"scale": ...} が有効
            cross_attention_kwargs = {"scale": float(lora_scale)} if lora_scale is not None else None

            try:
                out = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    width=int(width),
                    height=int(height),
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
                image = out.images[0]

                buf = io.BytesIO()
                image.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                b64 = base64.b64encode(png_bytes).decode("utf-8")

                meta = {
                    "seed": int(seed),
                    "request_params": {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "width": int(width),
                        "height": int(height),
                        "steps": int(num_inference_steps),
                        "guidance_scale": float(guidance_scale),
                        "adapter_path": adapter_path,
                        "lora_scale": float(lora_scale),
                    },
                    "warnings": warnings,
                    "resolved_models": model_identity(self._pipe_base_model or "", adapter_path),
                }

                saved = artifact_store.save("image", png_bytes, "png", meta)
                return {
                    "status": "ok",
                    "seed": int(seed),
                    "artifact_id": saved.get("artifact_id"),
                    "image_base64": "data:image/png;base64," + b64,
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}



    def generate_image_advanced(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        adapter_path: Optional[str] = None,
        lora_scale: float = 1.0,
        scheduler: str = "",
        preset_id: Optional[str] = None,
        hires_scale: float = 1.5,
        hires_steps: int = 15,
        hires_denoise: float = 0.35,
        use_refiner: bool = False,
        refiner_model: Optional[str] = None,
        # ControlNet（Canny/Depth/OpenPose等）: 破綻抑制用（任意）
        controlnet_type: Optional[str] = None,
        controlnet_model: Optional[str] = None,
        control_image_base64: Optional[str] = None,
        init_image_base64: Optional[str] = None,
        mask_image_base64: Optional[str] = None,
        inpaint_mode: Optional[str] = None,
        # 将来拡張用（未実装でもI/Fだけ固定しておく）
        ip_adapter_path: Optional[str] = None,
        ip_adapter_scale: float = 1.0,
        reference_image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """設計A向けの標準工程（Hi-Res/Refiner）を通す生成。
        既存 generate_image は互換維持のため残し、本APIは追加で提供する。
        """
        with self._pipe_lock:
            if self._pipe is None:
                return {"status": "error", "message": "Inference model is not loaded."}

            # プリセットのデフォルト適用（指定がある場合）
            if preset_id:
                p = find_preset(preset_id)
                if p:
                    d = p.get("defaults") or {}
                    steps = int(d.get("steps", steps))
                    cfg = float(d.get("cfg", cfg))
                    scheduler = str(d.get("scheduler", scheduler or ""))
                    hires_scale = float(d.get("hires_scale", hires_scale))
                    hires_steps = int(d.get("hires_steps", hires_steps))
                    hires_denoise = float(d.get("hires_denoise", hires_denoise))
                    use_refiner = bool(d.get("use_refiner", use_refiner))
                    if not negative_prompt and p.get("negative_prompt"):
                        negative_prompt = str(p.get("negative_prompt"))

            import random
            if seed is None:
                seed = random.randint(0, 2**31 - 1)

            pipe = self._pipe

            # ControlNet（任意）: 破綻抑制/構図固定
            # - controlnet_type: "canny" / "depth" / "openpose" / "custom"
            # - controlnet_model: settings.dirs["image"]["controlnet"] 配下のフォルダ名 or 絶対パス
            # - control_image_base64: data:image/...;base64,....
            if controlnet_type and control_image_base64:
                try:
                    from PIL import Image
                    from diffusers import ControlNetModel
                    # SDXL / SD1.5 でPipelineクラスが異なる
                    try:
                        from diffusers import StableDiffusionXLControlNetPipeline as _XLCP
                    except Exception:
                        _XLCP = None
                    try:
                        from diffusers import StableDiffusionControlNetPipeline as _CP
                    except Exception:
                        _CP = None

                    # data URI / base64 -> PIL (+ controlnet_type 前処理)
                    control_img = decode_and_preprocess_control_image(
                        control_image_base64,
                        controlnet_type=controlnet_type,
                        device=('cuda' if torch.cuda.is_available() else 'cpu'),
                    )

                    # controlnet model path
                    cn_id = (controlnet_model or "").strip()
                    if not cn_id:
                        # type別のデフォルト候補（ユーザーが models/image/controlnet に配置する想定）
                        # 例: canny_sdxl / depth_sdxl / openpose_sdxl 等
                        cn_id = f"{controlnet_type}_sdxl" if self._pipe_is_sdxl else f"{controlnet_type}"
                    cn_path = settings.dirs["image"]["controlnet"] / cn_id
                    if not cn_path.exists():
                        cn_path = Path(cn_id)

                    if not cn_path.exists():
                        return {
                            "status": "error",
                            "message": f"ControlNetモデルが見つかりません: {cn_id}\n"
                                       f"配置先: {settings.dirs['image']['controlnet']}\n"
                                       "例）models/image/controlnet/canny_sdxl/（diffusers形式）",
                        }

                    key = (str(self._pipe_base_model or ""), str(cn_path))
                    cn_pipe = self._control_pipes.get(key)

                    if cn_pipe is None:
                        controlnet = ControlNetModel.from_pretrained(
                            str(cn_path),
                            torch_dtype=torch.float16,
                        )
                        if self._pipe_is_sdxl and _XLCP is not None:
                            cn_pipe = _XLCP.from_pretrained(
                                str(self._pipe_base_model or ""),
                                controlnet=controlnet,
                                torch_dtype=torch.float16,
                                variant="fp16",
                            )
                        elif (not self._pipe_is_sdxl) and _CP is not None:
                            cn_pipe = _CP.from_pretrained(
                                str(self._pipe_base_model or ""),
                                controlnet=controlnet,
                                torch_dtype=torch.float16,
                            )
                        else:
                            return {"status": "error", "message": "この環境のdiffusersではControlNetパイプラインが利用できません（diffusersの更新が必要です）。"}

                        cn_pipe = cn_pipe.to(device)
                        # 既存最適化を適用
                        prepare_pipe_common(cn_pipe, enable_xformers=True, vae_slicing=True, vae_tiling=False)
                        self._control_pipes[key] = cn_pipe

                    pipe = cn_pipe
                    # control image は後段で pipe(...) に渡すため保持
                    _control_img = control_img
                except Exception as e:
                    return {"status": "error", "message": f"ControlNetの初期化に失敗しました: {e}"}
            else:
                _control_img = None

            # 共通最適化
            prepare_pipe_common(pipe, enable_xformers=True, vae_slicing=True, vae_tiling=False)
            apply_scheduler(pipe, scheduler)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # LoRA の適用は既存 generate_image と同じロジックを再利用するため、
            # 一旦 generate_image 相当の adapter 切替だけ実施
            #（unload_lora_weights が無い場合の差異吸収も既存実装に寄せる）
            # NOTE: 既存 generate_image のコードを二重化しないため、簡易に set_adapters だけ試す。
            try:
                if adapter_path:
                    adapter_full = None
                    p1 = settings.dirs["image"]["output"] / adapter_path
                    p2 = settings.dirs["image"]["models"] / adapter_path
                    if p1.exists():
                        adapter_full = p1
                    elif p2.exists():
                        adapter_full = p2
                    else:
                        cand = Path(adapter_path)
                        if cand.exists():
                            adapter_full = cand
                    if adapter_full is not None:
                        cur = str(self._pipe_adapter_path) if self._pipe_adapter_path else None
                        if cur != str(adapter_path):
                            if hasattr(pipe, "unload_lora_weights"):
                                try:
                                    pipe.unload_lora_weights()
                                except Exception:
                                    pass
                            pipe.load_lora_weights(str(adapter_full))
                            self._pipe_adapter_path = str(adapter_path)
            except Exception:
                pass

            # base生成（Text2Img / ControlNet / Img2Img / Inpaint）
            try:
                generator = torch.Generator(device=device).manual_seed(int(seed))

                init_b64 = (init_image_base64 or "").strip()
                mask_b64 = (mask_image_base64 or "").strip()
                mode = (inpaint_mode or "").strip().lower()

                # 自動判定（互換）
                if not mode:
                    if init_b64 and mask_b64:
                        mode = "inpaint"
                    elif init_b64:
                        mode = "img2img"
                    else:
                        mode = "text2img"

                warnings: List[str] = []

                # 将来拡張用ガード: BK43では Inpaint + ControlNet の同時利用は未対応（破綻回避）
                if mode in ("inpaint", "outpaint") and controlnet_type and control_image_base64:
                    return {
                        "status": "error",
                        "message": "BK43では Inpaint/Outpaint と ControlNet の同時利用は未対応です（将来拡張用ガード）。プリセット運用ではどちらか一方を選択してください。",
                    }

                # 将来拡張I/F（未実装）: 指定された場合は保存のみ（動作は変えない）
                if (ip_adapter_path or "").strip():
                    warnings.append("ip_adapter はBK43では未実装です（I/Fのみ予約）。指定値は meta に保存されます。")
                if (reference_image_base64 or "").strip():
                    warnings.append("reference_image はBK43では未実装です（I/Fのみ予約）。指定値は meta に保存されます。")

                # 1) Inpaint / Outpaint
                if mode in ("inpaint", "outpaint") and init_b64 and mask_b64:
                    inpaint_pipe = get_inpaint_pipeline(str(self._pipe_base_model or self._pipe_model_id or ""), device=device, dtype=self._torch_dtype)
                    if inpaint_pipe is None:
                        return {"status": "error", "message": "InpaintPipeline の初期化に失敗しました。SDXL Inpaintモデル（diffusers形式）が必要です。"}
                    apply_scheduler(inpaint_pipe, scheduler)

                    init_img = decode_base64_image_to_pil(init_b64).convert("RGB")
                    mask_img = decode_base64_image_to_pil(mask_b64).convert("L")

                    # サイズ合わせ（マスクも同サイズに）
                    init_img = init_img.resize((int(width), int(height)))
                    mask_img = mask_img.resize((int(width), int(height)))

                    # outpaint: マスクが「外側」想定の場合、運用により白黒が逆になりやすい。
                    # ここでは自動反転はしない（誤反転で破壊するため）。
                    out = inpaint_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=init_img,
                        mask_image=mask_img,
                        num_inference_steps=int(steps),
                        guidance_scale=float(cfg),
                        generator=generator,
                    )
                    image = out.images[0]

                # 2) Img2Img（入力画像からの変換）
                elif mode == "img2img" and init_b64:
                    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
                    init_img = decode_base64_image_to_pil(init_b64).convert("RGB")
                    init_img = init_img.resize((int(width), int(height)))

                    # pipe.components から組み立て（同一componentsなので ControlNet ではなく base pipe を前提）
                    if self._img2img_pipe is None:
                        if self._pipe_is_sdxl:
                            self._img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
                        else:
                            self._img2img_pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
                        self._img2img_pipe = self._img2img_pipe.to(device)
                        prepare_pipe_common(self._img2img_pipe, enable_xformers=True, vae_slicing=True, vae_tiling=False)

                    apply_scheduler(self._img2img_pipe, scheduler)

                    # strength は UI/API からまだ出していないので、破綻しにくい値を固定
                    strength = 0.55
                    out = self._img2img_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=init_img,
                        strength=float(strength),
                        num_inference_steps=int(steps),
                        guidance_scale=float(cfg),
                        generator=generator,
                        cross_attention_kwargs={"scale": float(lora_scale)} if adapter_path else None,
                    )
                    image = out.images[0]

                # 3) Text2Img / ControlNet
                else:
                    kwargs = dict(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=int(width),
                        height=int(height),
                        num_inference_steps=int(steps),
                        guidance_scale=float(cfg),
                        generator=generator,
                        cross_attention_kwargs={"scale": float(lora_scale)} if adapter_path else None,
                    )
                    # ControlNet Pipeline の場合のみ image を渡す
                    if _control_img is not None:
                        kwargs["image"] = _control_img
                    out = pipe(**kwargs)
                    image = out.images[0]

            except Exception as e:
                return {"status": "error", "message": str(e)}

            # Hi-Res（二段img2img）
            hires_used = False
            if float(hires_scale) and float(hires_scale) > 1.01:
                try:
                    from PIL import Image
                    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline

                    # pipe.components から組み立て
                    if self._img2img_pipe is None:
                        if self._pipe_is_sdxl:
                            self._img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
                        else:
                            self._img2img_pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
                        self._img2img_pipe = self._img2img_pipe.to(device)
                        prepare_pipe_common(self._img2img_pipe, enable_xformers=True, vae_slicing=True, vae_tiling=False)
                        apply_scheduler(self._img2img_pipe, scheduler)

                    new_w = int(int(width) * float(hires_scale))
                    new_h = int(int(height) * float(hires_scale))
                    image_up = image.resize((new_w, new_h), resample=Image.LANCZOS)
                    image = hires_img2img(
                        self._img2img_pipe,
                        image_up,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=int(seed),
                        steps=int(hires_steps),
                        cfg=float(cfg),
                        strength=float(hires_denoise),
                    )
                    hires_used = True
                except Exception:
                    hires_used = False

            # Refiner（二段: SDXLのみ）
            refiner_used = False
            if use_refiner and self._pipe_is_sdxl:
                try:
                    from diffusers import StableDiffusionXLImg2ImgPipeline
                    # refiner model の解決
                    resolved = None
                    if refiner_model:
                        rp = settings.dirs["image"].get("refiner")
                        # 直接パス/名前を許容
                        cand = Path(refiner_model)
                        if not cand.exists() and rp is not None:
                            p = Path(rp) / refiner_model
                            if p.exists():
                                cand = p
                        if cand.exists():
                            resolved = cand
                    else:
                        rp = settings.dirs["image"].get("refiner")
                        if rp is not None and Path(rp).exists():
                            # 最初のサブディレクトリを採用
                            subs = [p for p in Path(rp).iterdir() if p.is_dir()]
                            if subs:
                                resolved = subs[0]

                    if resolved is not None:
                        if self._refiner_pipe is None or self._refiner_model != str(resolved):
                            self._refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                                str(resolved),
                                torch_dtype=torch.float16,
                            ).to(device)
                            prepare_pipe_common(self._refiner_pipe, enable_xformers=True, vae_slicing=True, vae_tiling=False)
                            self._refiner_model = str(resolved)

                        image = refine(
                            self._refiner_pipe,
                            image,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            seed=int(seed),
                            steps=max(10, int(hires_steps)),
                            cfg=float(cfg),
                        )
                        refiner_used = True
                except Exception:
                    refiner_used = False

            # PNG
            try:
                from PIL import Image
                import io as _io
                buf = _io.BytesIO()
                image.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                b64 = base64.b64encode(png_bytes).decode("utf-8")

                meta = {
                    "seed": int(seed),
                    "request_params": {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "width": int(width),
                        "height": int(height),
                        "steps": int(steps),
                        "cfg": float(cfg),
                        "scheduler": scheduler,
                        "hires": {
                            "scale": float(hires_scale),
                            "steps": int(hires_steps),
                            "denoise": float(hires_denoise),
                            "used": bool(hires_used),
                        },
                        "refiner": {
                            "used": bool(refiner_used),
                            "model": self._refiner_model,
                        },
                        "lora_scale": float(lora_scale),
                        "preset_id": preset_id,
                        "controlnet": {
                            "type": controlnet_type or "",
                            "model": controlnet_model or "",
                            "used": bool(_control_img is not None),
                            "control_image_sha256": _sha256_data_uri(control_image_base64 or ""),
                        },
                        "inpaint": {
                            "mode": (inpaint_mode or "").strip().lower(),
                            "init_provided": bool((init_image_base64 or "").strip()),
                            "mask_provided": bool((mask_image_base64 or "").strip()),
                            "init_image_sha256": _sha256_data_uri(init_image_base64 or ""),
                            "mask_image_sha256": _sha256_data_uri(mask_image_base64 or ""),
                        },
                        "ip_adapter": {
                            "path": (ip_adapter_path or ""),
                            "scale": float(ip_adapter_scale),
                            "used": bool((ip_adapter_path or "").strip()),
                        },
                        "reference": {
                            "used": bool((reference_image_base64 or "").strip()),
                            "image_sha256": _sha256_data_uri(reference_image_base64 or ""),
                        },
                    },
                    "resolved_models": model_identity(self._pipe_base_model or "", adapter_path, {"refiner": self._refiner_model}),
                }
                saved = artifact_store.save("image", png_bytes, "png", meta)
                return {
                    "status": "ok",
                    "seed": int(seed),
                    "artifact_id": saved.get("artifact_id"),
                    "image_base64": "data:image/png;base64," + b64,
                    "hires_used": bool(hires_used),
                    "refiner_used": bool(refiner_used),
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}

# シングルトン
image_engine = ImageEngine()
