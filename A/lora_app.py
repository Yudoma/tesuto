# -*- coding: utf-8 -*-
"""
lora_app.py
LoRA FactoryのFastAPIアプリケーション定義。
lora_server.py から読み込まれます。
(v14: Modular Architecture Support)
"""
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# 設定のインポート
from lora_config import settings

# 新しいメインルーターのインポート (バックエンド統合ルーター)
from backend.routes.main_router import router as main_router
from backend.routes.plugin_system_api import router as plugin_system_api
from backend.plugin_system import load_backend_routers


app = FastAPI(title="LoRA Factory")

# ============================================================
# CORS設定 (ローカル開発用)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ルーティング (API)
# ============================================================
# 全モダリティ（Text/Image/Audio）のAPIを集約した main_router を /api プレフィックスでマウント
app.include_router(main_router, prefix="/api")

# Plugin discovery API
app.include_router(plugin_system_api, prefix="/api")

# Backend plugin routers (optional)
for _r in load_backend_routers():
    try:
        app.include_router(_r, prefix="/api")
    except Exception:
        pass


# ============================================================
# 静的ファイル (Frontend)
# ============================================================
# リファクタリング後の構成では 'static' フォルダを使用します
static_dir = settings.base_dir / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Plugins (optional frontend modules)
plugins_dir = settings.base_dir / "plugins"
if plugins_dir.exists():
    app.mount("/plugins", StaticFiles(directory=str(plugins_dir)), name="plugins")


# ============================================================
# ルート (UI配信)
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    """
    メインのダッシュボード (index.html) を返す
    """
    index_path = static_dir / "index.html"
    
    # ファイルが存在しない場合のフォールバック（開発初期用）
    if not index_path.exists():
        return HTMLResponse(
            """
            <html>
                <head><title>LoRA Factory</title></head>
                <body style="font-family: sans-serif; padding: 2rem; background-color: #0d1117; color: #c9d1d9;">
                    <h1>LoRA Factory Backend is Running</h1>
                    <p>UI file (static/index.html) was not found.</p>
                    <p>Please ensure frontend files are placed in the 'static' directory.</p>
                </body>
            </html>
            """
        )
    
    return FileResponse(str(index_path))

@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse("", status_code=204)

# ============================================================
# 起動時処理
# ============================================================
@app.on_event("startup")
async def startup_event():
    print("--- LoRA Factory Initialized (Modular Backend) ---")
    print(f"API Host: {settings.host}:{settings.port}")
    print(f"Static Dir: {static_dir}")