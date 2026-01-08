@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set VENV_DIR=venv_lora
set PORT=8081

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [Error] Virtual environment not found.
    echo Please run "python setup_lora_env.py" first.
    pause
    exit /b 1
)

echo ======================================================
echo  LoRA Factory (Train ^& Verify)
echo  URL: http://localhost:%PORT%
echo  Venv: %VENV_DIR%
echo ======================================================

"%VENV_DIR%\Scripts\python.exe" lora_server.py
pause
