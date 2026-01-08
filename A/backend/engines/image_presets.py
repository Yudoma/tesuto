# -*- coding: utf-8 -*-
"""backend/engines/image_presets.py
画像生成の用途別プリセット（日本人向け「勝ちパターン」）。

方針:
- UI / API から preset_id を指定した場合に、Engine 側で defaults を上書き適用する。
- 既存挙動（preset 未指定）は一切変更しない。
- プリセットは「ControlNet / Refiner / Scheduler / VAE」まで含めて再現性を担保する。
  ※実際の model_id / vae_id は環境差が大きいので、既定では空文字（未指定）とし、
    運用時に models/image/* 配下の構成に合わせて埋める。
"""
from __future__ import annotations

from typing import Any, Dict, List

# 互換: 既存のネガティブプロンプトを維持しつつ、日本語運用で破綻しやすい要素も足す
DEFAULT_NEGATIVE = (
    "lowres, blurry, bad anatomy, bad hands, extra fingers, worst quality, low quality, "
    "text, watermark, logo, signature, jpeg artifacts, out of frame"
)

# SDXL 想定の「勝ちやすい」既定（環境差があるため過度に固定しない）
DEFAULTS_BASE = {
    "steps": 28,
    "cfg": 5.5,
    "scheduler": "dpmpp_2m",
    "hires_scale": 1.6,
    "hires_steps": 15,
    "hires_denoise": 0.35,
    "use_refiner": True,
    # 環境依存: 未指定なら既存挙動（Engine 側のロード済みモデル）を使う
    "refiner_model": "",
    "vae": "",
    # ControlNet（任意）: UI で control image を渡した時に効かせるための推奨
    "controlnet_type": "",
    "controlnet_model": "",
}


# BK43: 勝ち設定（プリセット）を“凍結”として扱うためのメタ情報。
# - UI側で危険/破綻しやすい項目を隠し、プリセットの再現性を担保する目的。
# - Engine側の既存挙動は維持（preset未指定時の自由度は残す）。
FROZEN_IMAGE_PRESET_IDS = {
    "photo_person",
    "anime_illus",
    "product_packshot",
    "lineart_to_color",
}

def _with_frozen_meta(p: Dict[str, Any]) -> Dict[str, Any]:
    try:
        pid = str(p.get("id") or "")
        if pid in FROZEN_IMAGE_PRESET_IDS:
            p = dict(p)
            p["frozen"] = True
            p["frozen_reason"] = "BK43勝ち設定を維持し、再現性を固定するため"
        return p
    except Exception:
        return p

def get_image_presets() -> List[Dict[str, Any]]:
    """用途別プリセット一覧を返します。

    BK43運用方針:
    - ここに定義されたプリセットは「凍結プリセット」として扱い、再現性を最優先します。
    - UI側は frozen=true のプリセット選択時、危険/破綻しやすい調整項目を隠す想定です。
      （Engine側の既存挙動は維持し、preset未指定時は従来通り自由に調整可能）
    """
    presets = [
                {
                    "id": "photo_person",
                    "label": "写真風・人物（安定）",
                    "description": "人物の破綻（手/顔）を抑え、自然な質感を狙う。Refiner推奨。",
                    "defaults": {
                        **DEFAULTS_BASE,
                        "cfg": 5.0,
                        "hires_scale": 1.7,
                        "hires_denoise": 0.33,
                        "use_refiner": True,
                        # 人物は Canny より Depth が安定することが多い（control image がある場合のみ）
                        "controlnet_type": "depth",
                        "controlnet_model": "depth_sdxl",
                    },
                    "negative_prompt": DEFAULT_NEGATIVE,
                },
                {
                    "id": "anime_person",
                    "label": "アニメ・人物（輪郭安定）",
                    "description": "輪郭と目を安定させ、アニメ絵の破綻を抑える。Refinerは任意。",
                    "defaults": {
                        **DEFAULTS_BASE,
                        "steps": 30,
                        "cfg": 6.5,
                        "scheduler": "euler_a",
                        "hires_scale": 1.6,
                        "hires_denoise": 0.38,
                        "use_refiner": False,
                        "controlnet_type": "canny",
                        "controlnet_model": "canny_sdxl",
                    },
                    "negative_prompt": DEFAULT_NEGATIVE + ", realistic, photo, skin pores",
                },
                {
                    "id": "background",
                    "label": "背景・風景（見栄え）",
                    "description": "構図の破綻を抑え、細部の情報量を増やす。Refiner推奨。",
                    "defaults": {
                        **DEFAULTS_BASE,
                        "steps": 32,
                        "cfg": 5.5,
                        "hires_scale": 1.8,
                        "hires_denoise": 0.32,
                        "use_refiner": True,
                        "controlnet_type": "canny",
                        "controlnet_model": "canny_sdxl",
                    },
                    "negative_prompt": DEFAULT_NEGATIVE,
                },
                {
                    "id": "product",
                    "label": "物撮り・商品（シャープ）",
                    "description": "シャープさ・ライティング・歪みを抑える。Refiner推奨。",
                    "defaults": {
                        **DEFAULTS_BASE,
                        "steps": 26,
                        "cfg": 5.0,
                        "scheduler": "dpmpp_2m",
                        "hires_scale": 1.5,
                        "hires_denoise": 0.28,
                        "use_refiner": True,
                        "controlnet_type": "canny",
                        "controlnet_model": "canny_sdxl",
                    },
                    "negative_prompt": DEFAULT_NEGATIVE,
                },
                {
                    "id": "illustration",
                    "label": "イラスト（汎用）",
                    "description": "アニメ/写実の中間。破綻を抑えつつ表現幅を確保。",
                    "defaults": {
                        **DEFAULTS_BASE,
                        "steps": 28,
                        "cfg": 6.0,
                        "scheduler": "euler",
                        "hires_scale": 1.6,
                        "hires_denoise": 0.36,
                        "use_refiner": False,
                    },
                    "negative_prompt": DEFAULT_NEGATIVE,
                },
                {
                    "id": "icon",
                    "label": "アイコン（小サイズ安定）",
                    "description": "小さな絵での破綻を減らし、輪郭をはっきりさせる。",
                    "defaults": {
                        **DEFAULTS_BASE,
                        "steps": 24,
                        "cfg": 6.5,
                        "hires_scale": 1.25,
                        "hires_steps": 10,
                        "hires_denoise": 0.35,
                        "use_refiner": False,
                    },
                    "negative_prompt": DEFAULT_NEGATIVE,
                },

                {
                    "id": "photo_person_refiner",
                    "label": "人物・実写（Refiner強め）",
                    "description": "人物の肌/顔の破綻を抑え、質感を上げる（Refiner を使う）。",
                    "defaults": {
                        **DEFAULTS_BASE,
                        "steps": 32,
                        "cfg": 6.8,
                        "hires_scale": 1.5,
                        "hires_steps": 18,
                        "hires_denoise": 0.35,
                        "use_refiner": True,
                    },
                    "negative_prompt": DEFAULT_NEGATIVE,
                },
                {
                    "id": "inpaint_fix",
                    "label": "インペイント（修復）",
                    "description": "欠損/ノイズ/手指などの部分修復向け（init+mask 必須）。",
                    "defaults": {
                        **DEFAULTS_BASE,
                        "steps": 28,
                        "cfg": 6.5,
                        "hires_scale": 1.0,
                        "use_refiner": False,
                        "inpaint_mode": "inpaint",
                    },
                    "negative_prompt": DEFAULT_NEGATIVE,
                },
                {
                    "id": "canny_controlnet",
                    "label": "ControlNet（Canny：構図固定）",
                    "description": "エッジ画像（Canny）で構図を固定して生成する。control_image が必須。",
                    "defaults": {
                        **DEFAULTS_BASE,
                        "steps": 26,
                        "cfg": 6.5,
                        "hires_scale": 1.5,
                        "hires_steps": 14,
                        "hires_denoise": 0.35,
                        "use_refiner": False,
                        "controlnet_type": "canny",
                    },
                    "negative_prompt": DEFAULT_NEGATIVE,
                },
    ]
    return [_with_frozen_meta(p) for p in presets]

def find_preset(preset_id: str) -> Dict[str, Any]:
    """preset_id からプリセットを返します。見つからなければ空dict。"""
    if not preset_id:
        return {}
    for p in get_image_presets():
        if p.get("id") == preset_id:
            return p
    return {}