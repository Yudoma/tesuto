@echo off
:: バッチファイルがあるフォルダを作業ディレクトリにする（重要）
cd /d %~dp0

:: 仮想環境を有効化
call venv_lora\Scripts\activate

:: 終了しても画面を閉じないようにする
cmd /k