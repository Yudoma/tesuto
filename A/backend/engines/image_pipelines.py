# -*- coding: utf-8 -*-
"""backend/engines/image_pipelines.py

Diffusers のパイプライン補助関数群。

- 後方互換を優先しつつ、パイプラインの最適化/スケジューラ切替/Hi-Res(img2img)/Refiner を提供します。
- ControlNet は「任意機能」として、指定された場合のみ ControlNetModel を組み込みます。

注意:
- 本ファイルは Engine から呼ばれるため、例外で落ちない（fail-soft）ことを優先します。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def _try(fn):
    try:
        return fn()
    except Exception:
        return None


def prepare_pipe_common(
    pipe,
    *,
    enable_xformers: bool = True,
    vae_slicing: bool = True,
    vae_tiling: bool = False,
) -> None:
    """一般的なVRAM節約オプションを可能な範囲で有効化します（存在する場合のみ）。"""
    if enable_xformers:
        _try(lambda: getattr(pipe, "enable_xformers_memory_efficient_attention")())
    if vae_slicing:
        _try(lambda: getattr(pipe, "enable_vae_slicing")())
    if vae_tiling:
        _try(lambda: getattr(pipe, "enable_vae_tiling")())


def apply_scheduler(pipe, scheduler_name: str) -> None:
    """scheduler 名を指定して差し替えます。未指定/失敗時は何もしません。"""
    if not scheduler_name:
        return
    name = str(scheduler_name).lower().strip()
    if not name:
        return
    try:
        from diffusers import (
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            DDIMScheduler,
            PNDMScheduler,
            UniPCMultistepScheduler,
            LCMScheduler,
        )
    except Exception:
        return

    try:
        if name in ("euler", "euler_discrete"):
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif name in ("euler_a", "euler_ancestral", "euler_ancestral_discrete"):
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif name in ("dpmpp_2m", "dpmpp", "dpm_solver_multistep", "dpm++"):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif name in ("ddim",):
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif name in ("pndm",):
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        elif name in ("unipc", "unipc_multistep"):
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        elif name in ("lcm", "lcm_scheduler"):
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        else:
            return
    except Exception:
        return


def hires_img2img(
    img2img_pipe,
    image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    strength: float,
):
    """img2img で簡易 Hi-Res（二段目）を行います。"""
    gen = torch.Generator(device=getattr(img2img_pipe, "device", "cpu")).manual_seed(int(seed))
    out = img2img_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        generator=gen,
    )
    return out.images[0]


def refine(
    refiner_pipe,
    image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
):
    """SDXL Refiner 等を適用します。"""
    gen = torch.Generator(device=getattr(refiner_pipe, "device", "cpu")).manual_seed(int(seed))
    out = refiner_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        generator=gen,
    )
    return out.images[0]


def build_controlnet_pipeline(
    *,
    base_pipe,
    controlnet_model_path: str,
    is_xl: bool,
    torch_dtype: Any = torch.float16,
):
    """既存の base_pipe を元に ControlNet 対応パイプラインを構築します。

    返り値:
      ControlNet対応のパイプライン

    方針:
    - base_pipe のコンポーネント（UNet/VAE/TextEncoder/Tokenizer/Scheduler 等）を再利用
    - diffusers の ControlNet パイプラインが利用できない環境では例外
    """
    from diffusers import ControlNetModel

    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch_dtype)

    if is_xl:
        try:
            from diffusers import StableDiffusionXLControlNetPipeline
        except Exception as e:
            raise RuntimeError("StableDiffusionXLControlNetPipeline が利用できません。diffusers を更新してください。") from e

        cn_pipe = StableDiffusionXLControlNetPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            text_encoder_2=getattr(base_pipe, "text_encoder_2", None),
            tokenizer=base_pipe.tokenizer,
            tokenizer_2=getattr(base_pipe, "tokenizer_2", None),
            unet=base_pipe.unet,
            scheduler=base_pipe.scheduler,
            controlnet=controlnet,
        )
    else:
        try:
            from diffusers import StableDiffusionControlNetPipeline
        except Exception as e:
            raise RuntimeError("StableDiffusionControlNetPipeline が利用できません。diffusers を更新してください。") from e

        cn_pipe = StableDiffusionControlNetPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            unet=base_pipe.unet,
            scheduler=base_pipe.scheduler,
            controlnet=controlnet,
            safety_checker=getattr(base_pipe, "safety_checker", None),
            feature_extractor=getattr(base_pipe, "feature_extractor", None),
        )

    return cn_pipe


def decode_base64_image_to_pil(image_base64: str):
    """data URI / base64 文字列を PIL.Image に変換します。"""
    try:
        from PIL import Image
        import base64, io

        b64s = str(image_base64)
        if "," in b64s:
            b64s = b64s.split(",", 1)[1]
        raw = base64.b64decode(b64s)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"control image decode failed: {e}") from e



# ============================================================
# ControlNet 前処理（設計A：命中率と再現性のため）
# ============================================================

def preprocess_control_image(
    controlnet_type: str,
    control_image_pil,
    *,
    device: str = "cuda",
):
    """ControlNet 用の conditioning 画像を controlnet_type ごとに生成します。

    - canny: エッジ画像（白背景に黒線）を生成
    - depth: 深度推定による depth map を生成
    - openpose: 姿勢推定による pose map を生成
    - custom/unknown/None: 入力画像をそのまま返す（後方互換）

    重要:
    - 依存ライブラリが無い場合でも落とさない（fail-soft）
    - 生成できない場合は入力画像を返す（後方互換）
    """
    try:
        ctype = (controlnet_type or "").lower().strip()
        if not ctype or ctype in ("none", "off", "disable", "disabled"):
            return control_image_pil

        # 1) canny: OpenCV があればそれを優先、無ければ PIL で簡易エッジ
        if ctype in ("canny", "edge", "edges"):
            try:
                import numpy as np
                import cv2  # type: ignore
                img = np.array(control_image_pil.convert("RGB"))
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                from PIL import Image
                return Image.fromarray(edges_rgb)
            except Exception:
                # PIL fallback
                try:
                    from PIL import Image, ImageFilter, ImageOps
                    g = control_image_pil.convert("L")
                    g = g.filter(ImageFilter.FIND_EDGES)
                    g = ImageOps.autocontrast(g)
                    return g.convert("RGB")
                except Exception:
                    return control_image_pil

        # 2) depth / openpose: controlnet_aux を優先（Windows環境で導入されがち）
        #    無い場合は入力画像を返す
        if ctype in ("depth", "midas", "zoedepth"):
            try:
                from controlnet_aux import MidasDetector  # type: ignore
                det = MidasDetector.from_pretrained("lllyasviel/Annotators")
                # controlnet_aux は numpy/PIL 両対応
                depth = det(control_image_pil)
                return depth.convert("RGB") if hasattr(depth, "convert") else control_image_pil
            except Exception:
                return control_image_pil

        if ctype in ("openpose", "pose", "poses"):
            try:
                from controlnet_aux import OpenposeDetector  # type: ignore
                det = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
                pose = det(control_image_pil)
                return pose.convert("RGB") if hasattr(pose, "convert") else control_image_pil
            except Exception:
                return control_image_pil

        # unknown/custom
        return control_image_pil
    except Exception:
        return control_image_pil


def decode_and_preprocess_control_image(
    control_image_base64: str,
    *,
    controlnet_type: Optional[str] = None,
    device: str = "cuda",
):
    """base64 画像を decode し、controlnet_type が指定されていれば前処理して返します。"""
    pil = decode_base64_image_to_pil(control_image_base64)
    if controlnet_type:
        pil = preprocess_control_image(controlnet_type, pil, device=device)
    return pil


# ============================================================
# Inpaint / Img2Img Pipeline（設計A: 将来拡張の足場）
# ============================================================

_INPAINT_PIPE_CACHE: Dict[str, Any] = {}

def get_inpaint_pipeline(model_id: str, *, dtype: torch.dtype, device: str):
    """SDXL Inpaint Pipeline を取得（キャッシュ）。無ければ None。"""
    try:
        key = f"{model_id}::{str(dtype)}::{device}"
        if key in _INPAINT_PIPE_CACHE:
            return _INPAINT_PIPE_CACHE[key]
        try:
            from diffusers import StableDiffusionXLInpaintPipeline  # type: ignore
        except Exception:
            return None
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype in (torch.float16, torch.bfloat16) else None,
        )
        pipe = prepare_pipe_common(pipe)
        pipe.to(device)
        _INPAINT_PIPE_CACHE[key] = pipe
        return pipe
    except Exception:
        return None
