#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup_lora_env_optional_next_venv_lora.py  (SAFE TORCH + NO XFORMERS)

ユーザー方針:
- xformers は諦める（インストールしない / 試行もしない）

要件:
- **torch を絶対に壊さない（勝手にアップグレード/ダウングレード/再解決させない）**
- すべての pip install は **必ず venv_lora の python.exe** で実行（activate 依存なし）
- 実行ログを常時出力

安全設計:
- venv_lora 内の torch バージョンを取得し、constraints で **torch==検出値** を固定
- `--upgrade-strategy only-if-needed` で無駄な巻き込みを抑制
- 主要ステップ後に torch の import とバージョン一致を検証し、変化があれば **即停止**

使い方:
  py setup_lora_env_optional_next_venv_lora.py
  py setup_lora_env_optional_next_venv_lora.py --only monitoring
  py setup_lora_env_optional_next_venv_lora.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


# ---------------------------
# Optional package groups
# ---------------------------
# NOTE:
# - xformers は完全に除外（インストール対象に含めない）
OPTIONAL_GROUPS = {
    # 高速化/省メモリ（入らなくても動く）
    "speed_vram": [
        "bitsandbytes",    # Windowsで失敗しやすい（失敗しても続行）
        "torchao",         # torch関連（torch固定）
        "optimum",         # transformers周辺（torch固定）
    ],
    # 監視/可視化/補助（運用に便利）
    "monitoring": [
        "tensorboard",
        "wandb",
        "mlflow",
        "prometheus-client",
        "py-spy",
    ],
    # 便利系
    "quality_of_life": [
        "rich",
        "tqdm",
        "psutil",
        "matplotlib",
    ],
    # 任意
    "onnx": [
        "onnxruntime-gpu",
    ],
    # BK47: コア依存（このツールの主要機能を網羅）
    "voice": [
        # 音声(TTS)のみを先に入れる（失敗切り分け用）
        "TTS==0.22.0",
        "ffmpeg-python>=0.2",
        "pydub>=0.25",
        "soundfile>=0.12",
        "librosa>=0.10",
    ],
    # BK47: コア依存（このツールの主要機能を網羅）
    "bk47_core": [
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.6",
        "python-multipart>=0.0.9",
        "requests>=2.31",
        "tqdm>=4.66",
        "rich>=13.7",
        "pyyaml>=6.0.1",
        "psutil>=5.9",
        "numpy>=1.26",
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
    ],
    # BK47: optional (bk47_quality)
    "bk47_quality": [
        "matplotlib",
        "tensorboard",
        "wandb",
        "mlflow",
        "prometheus-client",
        "py-spy",
    ],
    # BK47: optional (bk47_controlnet)
    "bk47_controlnet": [
        "controlnet-aux>=0.0.9",
    ],
    # BK47: optional (bk47_tts)
    "bk47_tts": [
        "TTS>=0.22",
    ],
    # BK47: optional (bk47_openai)
    "bk47_openai": [
        "openai>=1.0",
    ],
    # BK47: optional (bk47_unsloth)
    "bk47_unsloth": [
        "unsloth",
    ],
}

DEFAULT_EXCLUDE = {
    # Windowsでは unsloth が依存として xformers/flash-attn 系を引き込みやすく、
    # 長いパス問題等で pip 全体が失敗して「TTSなど本命も入らない」原因になります。
    # そのため既定(all)からは除外し、必要な人だけ --only bk47_unsloth で入れてください。
    "unsloth",
}

ALL_TARGETS = sorted({p for pkgs in OPTIONAL_GROUPS.values() for p in pkgs} - DEFAULT_EXCLUDE)


# ---------------------------
# Utilities
# ---------------------------
def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        ensure_dir(log_file.parent)
        self.fp = open(log_file, "w", encoding="utf-8", newline="\n")

    def close(self):
        try:
            self.fp.close()
        except Exception:
            pass

    def _write(self, s: str):
        self.fp.write(s + "\n")
        self.fp.flush()

    def info(self, s: str):
        print(s)
        self._write(s)

    def section(self, title: str):
        bar = "=" * 78
        self.info(bar)
        self.info(title)
        self.info(bar)


def run_cmd(cmd: List[str], logger: Logger, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    logger.info(f"$ {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, encoding="utf-8", errors="backslashreplace")
    out = (p.stdout or "").rstrip()
    err = (p.stderr or "").rstrip()
    if out:
        logger.info(out)
    if err:
        logger.info(err)
    return p.returncode, out, err


def detect_project_root() -> Path:
    return Path(__file__).resolve().parent


def venv_python_path(root: Path) -> Path:
    return root / "venv_lora" / "Scripts" / "python.exe"


def ensure_venv_lora(root: Path, logger: Logger) -> Path:
    pyexe = venv_python_path(root)
    if pyexe.exists():
        logger.info(f"[venv_lora] {pyexe}")
        return pyexe
    raise RuntimeError(f"venv_lora not found: {pyexe} (先に setup_lora_env.py を実行してください)")


def get_torch_version(pyexe: Path, logger: Logger) -> str:
    rc, out, _ = run_cmd([str(pyexe), "-c", "import torch; print(torch.__version__)"], logger)
    if rc != 0 or not out.strip():
        raise RuntimeError("torch が venv_lora に見つかりません。先に setup_lora_env.py を実行してください。")
    return out.strip()


def assert_torch_unchanged(pyexe: Path, logger: Logger, expected: str, when: str) -> None:
    got = get_torch_version(pyexe, logger)
    if got != expected:
        logger.section("FATAL: torch version changed (STOP)")
        logger.info(f"[when] {when}")
        logger.info(f"[expected] {expected}")
        logger.info(f"[got] {got}")
        raise RuntimeError("torch が変化しました。安全のため処理を停止します。venv_lora を作り直すか torch を元のバージョンに戻してください。")
    logger.info(f"[torch OK] {got} (unchanged) @ {when}")


def write_constraints(root: Path, logger: Logger, torch_ver: str) -> Path:
    logs_dir = root / "logs"
    ensure_dir(logs_dir)
    cpath = logs_dir / f"constraints_torch_{torch_ver.replace('+','_')}_{now_stamp()}.txt"
    cpath.write_text(f"torch=={torch_ver}\n", encoding="utf-8")
    logger.info(f"[constraints] {cpath}")
    logger.info(f"torch=={torch_ver}")
    return cpath


def pip_install(pyexe: Path, pkgs: List[str], logger: Logger, constraints: Path, *,
                no_deps: bool = False) -> int:
    if not pkgs:
        return 0
    cmd = [str(pyexe), "-m", "pip", "install",
           "--upgrade", "--upgrade-strategy", "only-if-needed",
           "--constraint", str(constraints)]
    if no_deps:
        cmd.append("--no-deps")
    cmd += pkgs
    rc, _, _ = run_cmd(cmd, logger)
    return rc


def pip_check(pyexe: Path, logger: Logger) -> int:
    rc, _, _ = run_cmd([str(pyexe), "-m", "pip", "check"], logger)
    return rc


# ---------------------------
# Main
# ---------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(OPTIONAL_GROUPS.keys()) + ["all"], default="all",
                    help="インストールするグループを限定（デフォルト all）。all は Windows 安定性のため unsloth を除外。unsloth が必要なら --only bk47_unsloth を指定。")
    ap.add_argument("--dry-run", action="store_true", help="実行内容を出すだけ（インストールしない）")
    args = ap.parse_args()

    root = detect_project_root()
    logs_dir = root / "logs"
    ensure_dir(logs_dir)
    log_file = logs_dir / f"setup_lora_env_optional_next_SAFE_TORCH_NO_XFORMERS_{now_stamp()}.log"

    logger = Logger(log_file)
    try:
        logger.section("setup_lora_env_optional_next_venv_lora.py (SAFE TORCH / NO XFORMERS) start")
        logger.info(f"[log] {log_file}")
        logger.info(f"[root] {root}")

        pyexe = ensure_venv_lora(root, logger)

        torch_ver = get_torch_version(pyexe, logger)
        logger.section("Detect torch (pin it by constraints)")
        logger.info(f"[torch] {torch_ver}")
        constraints = write_constraints(root, logger, torch_ver)

        if args.only == "all":
            targets = ALL_TARGETS
        else:
            targets = OPTIONAL_GROUPS.get(args.only, [])

        logger.section("Plan")
        logger.info(f"[target python] {pyexe}")
        logger.info(f"[group] {args.only}")
        for p in targets:
            logger.info(f"  - {p}")

        if args.dry_run:
            logger.info("[DRY-RUN] No installation performed.")
            return 0

        # bitsandbytes は torch を巻き込みやすいことがあるので --no-deps で best-effort
        safe_first = [p for p in targets if p != "bitsandbytes"]
        risky = [p for p in targets if p == "bitsandbytes"]

        if safe_first:
            logger.section("Step 1: install optional packages (torch pinned)")
            rc = pip_install(pyexe, safe_first, logger, constraints, no_deps=False)
            assert_torch_unchanged(pyexe, logger, torch_ver, "after step1")
            if rc != 0:
                logger.info("[WARN] Some packages failed. Continue to next step.")

        if risky:
            logger.section("Step 2: best-effort bitsandbytes with --no-deps (torch pinned)")
            rc = pip_install(pyexe, risky, logger, constraints, no_deps=True)
            assert_torch_unchanged(pyexe, logger, torch_ver, "after bitsandbytes")
            if rc != 0:
                logger.info("[SKIP] bitsandbytes install failed (kept torch safe).")

        logger.section("pip check")
        pip_check(pyexe, logger)
        assert_torch_unchanged(pyexe, logger, torch_ver, "final")

        logger.section("Done")
        return 0
    except Exception as e:
        try:
            logger.section("ERROR")
            logger.info(str(e))
        except Exception:
            pass
        return 1
    finally:
        try:
            logger.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
