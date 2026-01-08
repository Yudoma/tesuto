# -*- coding: utf-8 -*-
"""
LoRA Factory: environment setup script (Windows-friendly)

目的
- LoRA 学習/検証に必要な Python 依存関係を「確実に」導入する
- Windows で入らない/壊れやすい高速化系（xformers / triton 等）は既定で無理に入れない
- ただし、明示フラグでインストール可能（失敗してもセットアップ自体は継続）
- 常にログを出す（コンソール + logs/ にタイムスタンプ付きログ）

重要（今回の不具合の原因）
- setup_lora_env.py を "venv の Python" ではなく "システム Python" で実行すると、
  依存がグローバルに入り、torch/xformers が意図せず入れ替わって CUDA が無効化することがあります。
  -> 本スクリプトは「venv を作成して、その venv の python でインストール」を既定動作にします。

使い方（推奨）
  1) ルートフォルダで:
     py -3.11 setup_lora_env.py
     -> .\\venv_lora を作成し、その中に一括インストールします

  2) 既存 venv に入れる:
     py -3.11 setup_lora_env.py --venv .\\venv_lora

  3) 既存の python を指定（上級者）:
     py -3.11 setup_lora_env.py --python .\\venv_lora\\Scripts\\python.exe

オプション（高速化/省メモリ系）
  - --install-optional : xformers / bitsandbytes を試行（Windows では torch の入れ替わりを防ぐため --no-deps で入れます）

注意
- triton は Windows では通常 pip から入りません（SKIP）
- CUDA 版 torch は PyTorch 公式 index-url を使います（既定: cu121）
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import platform
import shutil
import subprocess
import sys

# --- Windows console encoding safety (avoid cp932 UnicodeEncodeError) ---
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
# ----------------------------------------------------------------------
from pathlib import Path
from typing import List, Optional, Tuple


# -----------------------------
# Logging
# -----------------------------
def _setup_logger(log_dir: Path, verbose: bool = True) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"setup_lora_env_{ts}.log"

    logger = logging.getLogger("setup_lora_env")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File: always DEBUG
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Log file: %s", str(log_file))
    return logger, log_file


# -----------------------------
# Subprocess helpers
# -----------------------------
def _run(cmd: List[str], logger: logging.Logger, check: bool = True) -> subprocess.CompletedProcess:
    logger.info("RUN: %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="backslashreplace")
    if p.stdout:
        logger.debug("STDOUT:\n%s", p.stdout.rstrip())
    if p.stderr:
        logger.debug("STDERR:\n%s", p.stderr.rstrip())
    if check and p.returncode != 0:
        logger.error("[COMMAND FAILED] rc=%s cmd=%s", p.returncode, " ".join(cmd))
        if p.stdout:
            logger.error("stdout:\n%s", p.stdout)
        if p.stderr:
            logger.error("stderr:\n%s", p.stderr)
        raise RuntimeError(f"Command failed (rc={p.returncode}): {' '.join(cmd)}")
    return p


def _detect_windows() -> bool:
    return platform.system().lower().startswith("win")


def _has_nvidia_gpu(logger: logging.Logger) -> bool:
    try:
        if shutil.which("nvidia-smi"):
            p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, encoding="utf-8", errors="backslashreplace")
            if p.returncode == 0 and p.stdout.strip():
                logger.info("nvidia-smi detected:\n%s", p.stdout.strip())
                return True
        logger.info("nvidia-smi not found or no GPUs detected (best-effort).")
        return False
    except Exception as e:
        logger.info("nvidia-smi check skipped: %s", e)
        return False


def _venv_python(venv_dir: Path) -> Path:
    if _detect_windows():
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv(venv_dir: Path, base_python: str, logger: logging.Logger) -> Path:
    py = _venv_python(venv_dir)
    if py.exists():
        logger.info("[OK] venv exists: %s", str(venv_dir))
        return py
    logger.info("[Setup] Creating venv: %s", str(venv_dir))
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    _run([base_python, "-m", "venv", str(venv_dir)], logger, check=True)
    if not py.exists():
        raise RuntimeError(f"venv python not found after create: {py}")
    logger.info("[OK] venv created: %s", str(venv_dir))
    return py


def _pip_install(python_exe: str, pkgs: List[str], logger: logging.Logger, *,
                 upgrade: bool = True,
                 extra_args: Optional[List[str]] = None,
                 allow_fail: bool = False) -> bool:
    if not pkgs:
        return True
    cmd = [python_exe, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd += pkgs
    if extra_args:
        cmd += extra_args
    try:
        _run(cmd, logger, check=True)
        return True
    except Exception as e:
        if allow_fail:
            logger.warning("[OPTIONAL INSTALL FAILED] %s | pkgs=%s", e, pkgs)
            return False
        raise


def _pip_check(python_exe: str, logger: logging.Logger) -> bool:
    try:
        _run([python_exe, "-m", "pip", "check"], logger, check=True)
        return True
    except Exception:
        logger.warning("pip check reported issues (see above).")
        return False


def _torch_probe(python_exe: str, logger: logging.Logger) -> dict:
    code = r"""
import json
try:
    import torch
    info = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
    print(json.dumps(info, ensure_ascii=False))
except Exception as e:
    print(json.dumps({"error": str(e)}, ensure_ascii=False))
"""
    p = subprocess.run([python_exe, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="backslashreplace")
    out = (p.stdout or "").strip().splitlines()
    if not out:
        return {"error": "torch probe produced no output"}
    try:
        info = json.loads(out[-1])
    except Exception:
        logger.info("Torch probe raw output:\n%s", p.stdout)
        return {"error": "torch probe json parse failed"}
    return info


# -----------------------------
# Package plan
# -----------------------------
def _required_packages() -> List[str]:
    """BK47 実装で必要になる依存を Windows で入る範囲でまとめた必須セット。

    ここは **学習/検証/推論/データ処理** の土台。
    高速化系（xformers/flash-attn/triton等）は別枠（optional / 手動）に分離。
    """
    return [
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.6",
        "python-multipart>=0.0.9",
        "requests>=2.31",
        "tqdm>=4.66",
        "rich>=13.7",
        "pyyaml>=6.0.1",
        "psutil>=5.9",
        "numpy>=1.26,<2.0",   # 修正: numpy 2.0は古いPandasと非互換のため制限
        "pandas>=1.4,<2.0",  # TTS(>=0.22) が pandas<2.0 を要求するため固定
        "scipy>=1.11",
        "scikit-learn>=1.4",
        "sentence-transformers>=3.0",
        "faiss-cpu>=1.8",
        "transformers>=4.41",
        "accelerate>=0.32",
        "datasets>=2.20",
        "peft>=0.11",
        "trl>=0.9",
        "safetensors>=0.4",
        "einops>=0.7",
        "omegaconf>=2.3",
        "packaging>=24.0",
        "soundfile>=0.12",
        "librosa>=0.10",
        "pydub>=0.25",
        "ffmpeg-python>=0.2",
        "pyloudnorm>=0.1",
        "langdetect>=1.0.9",
        "faster-whisper>=1.0.3",
        "diffusers>=0.30",
        "compel>=2.0",
        "opencv-python-headless>=4.9",
        "pillow<12",
        "huggingface_hub>=0.23",
    ]


def _optional_packages(install_optional: bool, logger: logging.Logger) -> List[Tuple[str, List[str], List[str]]]:
    """
    returns list of (group_name, pkgs, extra_pip_args)
    """
    groups: List[Tuple[str, List[str], List[str]]] = []

    # Windows: triton is generally unavailable on pip
    if not _detect_windows() and install_optional:
        groups.append(("triton (kernel compiler)", ["triton>=2.1"], []))
    else:
        logger.info("Windows detected (or optional disabled): triton will be SKIP.")

    if install_optional:
        # 重要: xformers/bitsandbytes が torch を勝手に入れ替えないように --no-deps を付与
        # （必要であればユーザー側で互換版を指定してインストールする方が安全）
        groups.append(("xformers (optional, no-deps)", ["xformers"], ["--no-deps"]))
        groups.append(("bitsandbytes (optional, no-deps)", ["bitsandbytes"], ["--no-deps"]))

        # BK47 追加パッケージ群
        groups.append(("bk47_quality", ['matplotlib', 'tensorboard', 'wandb', 'mlflow', 'prometheus-client', 'py-spy'], []))
        groups.append(("bk47_controlnet", ['controlnet-aux>=0.0.9'], []))
        groups.append(("bk47_tts", ['TTS>=0.22'], []))
        groups.append(("bk47_openai", ['openai>=1.0'], []))
        groups.append(("bk47_unsloth", ['unsloth'], []))

    return groups


# -----------------------------
# Torch install (CUDA-aware)
# -----------------------------
def _install_torch(python_exe: str, logger: logging.Logger, prefer_cuda: bool = True) -> None:
    """
    - NVIDIA GPU がある場合: 既定で cu121 を入れる（Windows 前提）
    - ない場合: CPU 版
    """
    has_gpu = _has_nvidia_gpu(logger)
    variant = os.environ.get("TORCH_VARIANT", "").strip().lower()
    if not variant:
        if prefer_cuda and has_gpu:
            variant = "cu121"
        else:
            variant = "cpu"

    # User can override with TORCH_VARIANT=cu118 / cpu など
    logger.info("[Torch] Selected TORCH_VARIANT=%s", variant)

    if variant == "cpu":
        # CPU builds from PyPI are fine
        _pip_install(python_exe, ["torch", "torchvision", "torchaudio"], logger, upgrade=True, allow_fail=False)
        return

    # CUDA builds: use official PyTorch index-url
    index_url = f"https://download.pytorch.org/whl/{variant}"
    # IMPORTANT: install as a matched set from the same index-url to avoid version mismatch
    _pip_install(
        python_exe,
        ["torch", "torchvision", "torchaudio"],
        logger,
        upgrade=True,
        extra_args=["--index-url", index_url],
        allow_fail=False,
    )
    logger.info("[Torch] Installed from %s", index_url)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default="venv_lora", help="venv directory to create/use (default: .\\venv_lora)")
    ap.add_argument("--python", default=None, help="Target python executable (if set, venv is not created/used).")
    ap.add_argument("--log-dir", default="logs", help="Directory to store log files (default: .\\logs).")
    ap.add_argument("--no-upgrade-pip", action="store_true", help="Do not upgrade pip/setuptools/wheel.")
    ap.add_argument("--quiet", action="store_true", help="Less console output (still logs everything to file).")
    ap.add_argument("--install-optional", action="store_true", help="Try to install optional perf packages (xformers/bitsandbytes).")
    ap.add_argument("--no-cuda", action="store_true", help="Force CPU torch install (ignore GPU).")
    args = ap.parse_args()

    logger, log_file = _setup_logger(Path(args.log_dir).resolve(), verbose=not args.quiet)

    base_python = sys.executable  # the interpreter running this script
    if args.python:
        target_python = Path(args.python).resolve()
        logger.info("Mode: target python specified (no venv create/use)")
        logger.info("Python: %s", str(target_python))
    else:
        venv_dir = Path(args.venv).resolve()
        venv_py = _ensure_venv(venv_dir, base_python, logger)
        target_python = venv_py
        logger.info("Mode: venv managed")
        logger.info("Venv : %s", str(venv_dir))
        logger.info("Python: %s", str(target_python))

    logger.info("Platform: %s | %s", platform.platform(), platform.machine())

    # pip baseline
    if not args.no_upgrade_pip:
        logger.info("Upgrading pip/setuptools/wheel ...")
        _pip_install(str(target_python), ["pip", "setuptools", "wheel"], logger, upgrade=True, allow_fail=False)

    # Torch first (so later packages don't accidentally downgrade/replace it)
    logger.info("Installing torch/torchvision/torchaudio ...")
    _install_torch(str(target_python), logger, prefer_cuda=(not args.no_cuda))

    # Required packages
    req = _required_packages()
    logger.info("Installing REQUIRED packages (%d) ...", len(req))
    _pip_install(str(target_python), req, logger, upgrade=True, allow_fail=False)

    # Optional packages
    opt_groups = _optional_packages(args.install_optional, logger)
    failed = []
    for name, pkgs, extra in opt_groups:
        logger.info("Installing OPTIONAL group: %s | %s", name, ", ".join(pkgs))
        ok = _pip_install(str(target_python), pkgs, logger, upgrade=True, extra_args=extra, allow_fail=True)
        if not ok:
            failed.append(name)

    # Sanity check
    logger.info("Running pip check ...")
    ok_check = _pip_check(str(target_python), logger)

    # Torch probe
    logger.info("Probing torch/cuda ...")
    info = _torch_probe(str(target_python), logger)
    logger.info("Torch probe: %s", info)

    # Summary (human-friendly)
    logger.info("==============================================")
    logger.info("SETUP SUMMARY")
    logger.info("- Target Python : %s", str(target_python))
    logger.info("- Log File      : %s", str(log_file))
    logger.info("- pip check      : %s", "OK" if ok_check else "NG (see above)")
    if failed:
        logger.info("- Optional       : FAILED (%s)", ", ".join(failed))
    else:
        logger.info("- Optional       : OK")
    logger.info("==============================================")

    # If CUDA expected but not available, provide concrete guidance
    if isinstance(info, dict) and not info.get("cuda_available", False) and not args.no_cuda:
        logger.warning("CUDA が無効です。torch が CPU 版になっている可能性があります。")
        logger.warning("対処: venv を削除して再実行 -> rmdir /s /q venv_lora && py -3.11 setup_lora_env.py")
        logger.warning("または TORCH_VARIANT を明示: set TORCH_VARIANT=cu121 && py -3.11 setup_lora_env.py")

    logger.info("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())