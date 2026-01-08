# -*- coding: utf-8 -*-
"""backend/core/dataset_report.py
データセット検査結果の詳細レポート（保存用）。
/api/datasets/validate の判定ロジックを“詳細化”したもの。

- 目的：ジョブに紐づく dataset_report.json を保存し、UIで再表示できるようにする
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import os

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
TEXT_EXTS = {".txt", ".jsonl", ".json"}

def _count_files(root: Path, exts: Optional[set[str]] = None) -> Dict[str, Any]:
    total = 0
    total_bytes = 0
    by_ext: Dict[str, int] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if exts is not None and ext not in exts:
            continue
        total += 1
        try:
            total_bytes += p.stat().st_size
        except Exception:
            pass
        by_ext[ext] = by_ext.get(ext, 0) + 1
    return {"count": total, "bytes": total_bytes, "by_ext": dict(sorted(by_ext.items(), key=lambda x: x[0]))}

def build_dataset_report(mode: str, path: Path, kind: Optional[str] = None) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    report: Dict[str, Any] = {
        "mode": mode,
        "kind": kind,
        "path": str(path),
        "created_at": now,
        "ok": True,
        "messages": [],
        "stats": {},
    }

    def warn(msg: str) -> None:
        report["messages"].append(msg)

    def fail(msg: str) -> None:
        report["ok"] = False
        report["messages"].append(msg)

    if not path.exists():
        fail(f"データセットが見つかりません: {path}")
        return report

    # file / dir
    if path.is_file():
        try:
            size = path.stat().st_size
        except Exception:
            size = -1
        report["stats"]["file_size_bytes"] = size
        if size == 0:
            fail("データセットファイルが空です。")
        ext = path.suffix.lower()
        report["stats"]["file_ext"] = ext
        # text single file check
        if mode == "text":
            if ext not in TEXT_EXTS:
                warn(f"想定外の拡張子です（テキスト）: {ext}")
        return report

    # directory
    report["stats"]["dir_total_files"] = _count_files(path)["count"]

    if mode == "image":
        img = _count_files(path, IMAGE_EXTS)
        report["stats"]["images"] = img
        if img["count"] == 0:
            fail("画像ファイルが見つかりません（png/jpg/webp等）。")
        # captions are optional
        cap = _count_files(path, {".txt", ".caption"})
        report["stats"]["captions"] = cap
        if img["count"] > 0 and cap["count"] == 0:
            warn("キャプション（.txt/.caption）が見つかりません。モデルによっては必須です。")
        return report

    if mode == "audio":
        aud = _count_files(path, AUDIO_EXTS)
        report["stats"]["audio_files"] = aud
        if aud["count"] == 0:
            fail("音声ファイルが見つかりません（wav/flac/mp3等）。")
        # kind-specific hints
        if kind == "tts":
            # optional text pairs
            txt = _count_files(path, {".txt"})
            report["stats"]["text_pairs"] = txt
            if txt["count"] == 0:
                warn("TTS向けのテキストペア（.txt）が見つかりません。学習方式によっては必須です。")
        return report

    # text dir
    txt = _count_files(path, TEXT_EXTS)
    report["stats"]["text_files"] = txt
    if txt["count"] == 0:
        fail("テキストデータ（txt/jsonl/json）が見つかりません。")
    return report
