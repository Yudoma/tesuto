# -*- coding: utf-8 -*-
"""
lora_server.py
LoRA Factory (学習ツール) の起動エントリーポイント。
run_lora.bat から呼び出されます。
"""
import os
import uvicorn
from lora_config import settings

def main():
    """
    Uvicornサーバーを起動する。
    ポート設定などは lora_config.py (または環境変数) に準拠。
    """
    # ホストとポートの設定
    host = settings.host
    port = settings.port
    
    print(f"Starting LoRA Factory Server on http://{host}:{port}")
    print(f"Models Dir: {settings.models_dir}")
    print(f"Output Dir: {settings.output_dir}")

    # アプリケーションの起動
    # reload=False: 学習中の不用意な再起動を防ぐため
    uvicorn.run(
        "lora_app:app",
        host=host,
        port=port,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    main()