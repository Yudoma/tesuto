# -*- coding: utf-8 -*-
"""
backend/routes/main_router.py
アプリケーション全体のAPIルーター定義。

LoRA Factory v14（Modular Architecture）では、各モダリティ（Text / Image / Audio）のルートを
このファイルで集約し、FastAPI アプリ側 (lora_app.py) から /api プレフィックスでマウントします。

重要:
- 既存のフロントエンド互換性のため、Text(LLM) のルートは prefix なしで include します。
- Image/Audio は衝突回避のため prefix を付けます（/image, /audio）
"""

from fastapi import APIRouter, HTTPException

from backend.core.system_info import get_system_info, get_image_system_info, get_audio_system_info

from backend.routes.text_routes import router as text_router
from backend.routes.image_routes import router as image_router
from backend.routes.audio_routes import router as audio_router
from backend.routes.common_routes import router as common_router

router = APIRouter()

# ===========================================================
# Common / System
# ===========================================================

@router.get("/system_info", tags=["System"])
def api_system_info():
    """システム情報（CPU/RAM/GPU/FlashAttention2確認など）"""
    try:
        return get_system_info()
    except Exception as e:
        raise HTTPException(500, f"system_info取得失敗: {e}")


@router.get("/system_info/image", tags=["System"])
def api_system_info_image():
    """画像（Diffusers）生成/学習に関係するシステム情報"""
    try:
        return get_image_system_info()
    except Exception as e:
        raise HTTPException(500, f"system_info(image)取得失敗: {e}")


@router.get("/system_info/audio", tags=["System"])
def api_system_info_audio():
    """音声（Voice）生成/学習に関係するシステム情報"""
    try:
        return get_audio_system_info()
    except Exception as e:
        raise HTTPException(500, f"system_info(audio)取得失敗: {e}")

# ===========================================================
# Modality Routers
# ===========================================================

# Text (LLM) Routes
# 既存のフロントエンド(app.js / text_ui.js)との互換性を維持するため、prefixなしでマウントします。
# これにより /api/models, /api/train/start などがそのまま利用可能です。
router.include_router(text_router, tags=["Text (LLM)"])

# Common (Jobs / Artifacts)
router.include_router(common_router, tags=["Common"])

# Image Generation Routes
router.include_router(image_router, prefix="/image", tags=["Image Generation"])

# Audio/Voice Routes
router.include_router(audio_router, prefix="/audio", tags=["Audio/Voice"])
