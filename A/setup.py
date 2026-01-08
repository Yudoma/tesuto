#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LoRA Factory セットアップ統合スクリプト（Windows向け）。

このスクリプトは、同梱のセットアップスクリプトを非破壊で統合する入口です。
- setup_lora_env.py
- setup_lora_env_2.py

使い方
  py setup.py
  py setup.py --only base
  py setup.py --only optional
  py setup.py --only all

ログ
  logs/setup_YYYYMMDD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import os
import datetime as dt
import subprocess
import sys

# --- Windows console encoding safety (avoid cp932 UnicodeEncodeError) ---
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
# ----------------------------------------------------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _python_version_check() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        raise SystemExit("Python 3.10 以上が必要です。Python を更新してから再実行してください。")


def _run(cmd: list[str], log_path: Path) -> None:
    # Force UTF-8 for child process + avoid Windows cp932 console issues
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    with log_path.open("a", encoding="utf-8", errors="backslashreplace") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            env=env,
        )

        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)

        code = p.wait()
        if code != 0:
            raise SystemExit(code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["base", "optional", "all"],
        default="all",
        help="base=venv作成と必須導入 / optional=追加導入 / all=両方",
    )
    args = parser.parse_args()

    _python_version_check()

    log_path = LOGS_DIR / f"setup_{_stamp()}.log"
    log_path.write_text("", encoding="utf-8")
    print(f"[setup] log: {log_path}")

    base_script = ROOT / "setup_lora_env.py"
    optional_script = ROOT / "setup_lora_env_2.py"

    if args.only in ("base", "all"):
        if not base_script.exists():
            raise SystemExit("setup_lora_env.py が見つかりません。")
        _run([sys.executable, str(base_script)], log_path)

    if args.only in ("optional", "all"):
        if not optional_script.exists():
            raise SystemExit("setup_lora_env_2.py が見つかりません。")
        _run([sys.executable, str(optional_script)], log_path)

    print("[setup] 完了しました。次は run_lora.bat を実行してください。")


if __name__ == "__main__":
    main()
