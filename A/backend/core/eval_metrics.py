# -*- coding: utf-8 -*-
"""backend/core/eval_metrics.py
軽量な品質指標（最小構成 -> 設計Aの足場）。

注意:
- 本番の厳密評価（CLIP/ASR/WER等）は依存が重くなりやすい。
  本ファイルは「導入コストが低い指標」を提供し、導線（meta保存）を整える。
- 依存が無い場合は必ず {} を返す（fail-soft / No Regression）。
"""
from __future__ import annotations

import io
import math
import wave
import json
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple

# ============================================================
# System制約 fingerprint（A品質を“維持・再現・証明”するための固定）
# ------------------------------------------------------------
# - system制約は「回答本文に残る短い断片（fingerprint）」として扱う。
# - 回帰テスト等で fingerprint が欠落した場合は、warn / fail を固定ルールで判定する。
# - ここで定義する fingerprint は BK43 の“運用前提”要件（日本語UI / 省略禁止 / 危険操作禁止）を想定。
# ============================================================
DEFAULT_SYSTEM_CONSTRAINT_FINGERPRINTS: List[str] = [
    # 日本語強制（UI/運用の前提）
    "日本語",
    # 省略禁止（ユーザー要望/運用品質）
    "省略",
    # 危険操作禁止（運用安全性）
    "危険",
    "禁止",
]

# fingerprint 判定ポリシー（固定）
# - critical: 欠落したら fail
# - important: 欠落したら warn
SYSTEM_CONSTRAINT_FINGERPRINT_POLICY: Dict[str, str] = {
    "日本語": "critical",
    "省略": "critical",
    "危険": "important",
    "禁止": "important",
}

def _fingerprint_status(expected: List[str], missing: List[str]) -> str:
    """fingerprint の欠落から status を返す（ok/warn/fail）。"""
    # critical 欠落があれば fail
    for fp in missing:
        if SYSTEM_CONSTRAINT_FINGERPRINT_POLICY.get(fp) == "critical":
            return "fail"
    # それ以外の欠落があれば warn
    return "warn" if missing else "ok"

def _stable_sha256_text(s: str) -> str:
    try:
        b = (s or "").encode("utf-8", errors="ignore")
        return hashlib.sha256(b).hexdigest()
    except Exception:
        return ""


def text_constraint_metrics(
    text: str,
    *,
    forbidden_words: Optional[List[str]] = None,
    forbidden_phrases: Optional[List[str]] = None,
    required_patterns: Optional[List[str]] = None,
    require_json: bool = False,
    required_json_keys: Optional[List[str]] = None,
    system_constraint_fingerprints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """テキストの制約逸脱を簡易検知します（fail-soft）。

    目的:
    - 実運用で致命的になりやすい「禁止語/禁止表現」「形式逸脱（JSON必須等）」
      「system制約（絶対条件）の欠落」を軽量に検知し、meta/metrics に残す。
    - 依存を増やさず、例外は飲み込み、必ず dict を返す。

    補足:
    - require_json は「JSON オブジェクト/配列として parse できるか」を検査します。
      コードフェンス ```json ... ``` が付く場合も考慮して剥がします。
    """
    try:
        s = str(text or "")
        s_norm = " ".join(s.split())

        forb_words = [w for w in (forbidden_words or []) if isinstance(w, str) and w.strip()]
        forb_phr = [p for p in (forbidden_phrases or []) if isinstance(p, str) and p.strip()]
        req = [p for p in (required_patterns or []) if isinstance(p, str) and p.strip()]
        req_keys = [k for k in (required_json_keys or []) if isinstance(k, str) and k.strip()]
        fps = [fp for fp in (system_constraint_fingerprints or []) if isinstance(fp, str) and fp.strip()]

        violations: List[Dict[str, Any]] = []

        # 1) 禁止語（部分一致）
        for w in forb_words:
            if w in s:
                violations.append({"type": "forbidden_word", "value": w})

        # 2) 禁止表現（正規表現 or 部分一致）
        for p in forb_phr:
            try:
                # /.../ 形式は正規表現として扱う
                if len(p) >= 2 and p.startswith("/") and p.endswith("/"):
                    rx = p[1:-1]
                    if re.search(rx, s, flags=re.IGNORECASE):
                        violations.append({"type": "forbidden_phrase", "value": p})
                else:
                    if p in s:
                        violations.append({"type": "forbidden_phrase", "value": p})
            except Exception:
                # 正規表現が壊れていても落とさない
                if p in s:
                    violations.append({"type": "forbidden_phrase", "value": p})

        # 3) 必須パターン
        for p in req:
            try:
                if not re.search(p, s):
                    violations.append({"type": "missing_pattern", "value": p})
            except Exception:
                # 壊れた正規表現は fail-soft: 何もしない
                pass

        # 4) system制約 fingerprint の欠落（要約後に消える等）
        #    fingerprint は短い断片文字列を想定
        missing_fps = []
        for fp in fps:
            if fp and (fp not in s_norm):
                missing_fps.append(fp)
        if missing_fps:
            violations.append({
                "type": "missing_system_constraints",
                "count": int(len(missing_fps)),
                "fingerprints": missing_fps[:32],  # 32件まで
            })

        # 5) JSON 必須チェック
        parsed_json = None
        json_ok = None
        json_err = None
        missing_keys: List[str] = []
        if bool(require_json):
            t = s.strip()

            # code fence を剥がす（```json ... ``` / ``` ... ```）
            # ただし本文全体が fence の場合のみ対象にする
            fence_m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```\s*$", t, flags=re.IGNORECASE)
            if fence_m:
                t = fence_m.group(1).strip()

            try:
                parsed_json = json.loads(t)
                json_ok = True
            except Exception as e:
                json_ok = False
                json_err = str(e)
                violations.append({"type": "format_deviation", "value": "json_required_but_invalid", "detail": json_err})

            if json_ok and req_keys:
                try:
                    if isinstance(parsed_json, dict):
                        for k in req_keys:
                            if k not in parsed_json:
                                missing_keys.append(k)
                    else:
                        # dict 以外の場合は keys 不足扱い
                        missing_keys = req_keys[:]
                except Exception:
                    missing_keys = req_keys[:]
                if missing_keys:
                    violations.append({"type": "missing_json_keys", "value": missing_keys})

        return {
            "text_len": int(len(s)),
            "constraint_violations": violations,
            "constraint_ok": (len(violations) == 0),
            "json_required": bool(require_json),
            "json_ok": json_ok,
            "json_error": json_err,
            # system制約 fingerprint の固定評価
            "system_constraint_fingerprints_expected": fps,
            "system_constraint_fingerprints_missing": missing_fps,
            "system_constraint_status": _fingerprint_status(fps, missing_fps),
            # 総合判定（運用で扱いやすいように ok/warn/fail を固定）
            "constraint_level": (
                "fail"
                if (not json_ok and bool(require_json))
                else _fingerprint_status(fps, missing_fps)
                if missing_fps
                else ("ok" if len(violations) == 0 else "warn")
            ),
            # 回帰の安定比較用（同一出力判定用のハッシュ）
            "output_sha256": _stable_sha256_text(s),
        }
    except Exception:
        return {}


def image_basic_metrics(png_bytes: bytes) -> Dict[str, Any]:
    """画像の基本統計（サイズ/簡易ブラー指標/ハッシュ）"""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        arr = np.array(img).astype("float32") / 255.0
        mean = float(arr.mean())
        std = float(arr.std())
        # ブラー指標: Laplacian variance（OpenCVがあれば） / なければ簡易エッジ平均との差
        blur = None
        try:
            import cv2  # type: ignore
            g = cv2.cvtColor((arr*255).astype("uint8"), cv2.COLOR_RGB2GRAY)
            blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
        except Exception:
            # fallback: 隣接差分の分散
            dx = (arr[:, 1:, :] - arr[:, :-1, :]).reshape(-1, 3)
            dy = (arr[1:, :, :] - arr[:-1, :, :]).reshape(-1, 3)
            blur = float(((dx*dx).mean() + (dy*dy).mean()) / 2.0)
        dhash = image_dhash(png_bytes)
        return {"width": int(w), "height": int(h), "mean": mean, "std": std, "blur": blur, "dhash": dhash}
    except Exception:
        return {}

def image_dhash(png_bytes: bytes, *, hash_size: int = 8) -> str:
    """差分ハッシュ（dHash）"""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes)).convert("L")
        # (hash_size+1, hash_size) にリサイズ
        img = img.resize((hash_size + 1, hash_size))
        pixels = list(img.getdata())
        # row-major
        bits = []
        for row in range(hash_size):
            for col in range(hash_size):
                left = pixels[row * (hash_size + 1) + col]
                right = pixels[row * (hash_size + 1) + col + 1]
                bits.append(1 if left > right else 0)
        # hex
        v = 0
        out = []
        for i, b in enumerate(bits):
            v = (v << 1) | b
            if (i % 4) == 3:
                out.append(format(v, "x"))
                v = 0
        return "".join(out)
    except Exception:
        return ""

def image_hash_similarity(dhash_a: str, dhash_b: str) -> Dict[str, Any]:
    """dHash のハミング距離（小さいほど類似）"""
    try:
        if not dhash_a or not dhash_b or len(dhash_a) != len(dhash_b):
            return {}
        # hex -> bits
        a = int(dhash_a, 16)
        b = int(dhash_b, 16)
        x = a ^ b
        dist = int(x.bit_count()) if hasattr(int, "bit_count") else int(bin(x).count("1"))
        return {"dhash_hamming": dist}
    except Exception:
        return {}

def audio_rms_metrics(wav_bytes: bytes) -> Dict[str, Any]:
    """PCM16 wav の peak/rms を返す（既存互換）"""
    try:
        import numpy as np
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            fr = wf.getframerate()
        if sampwidth != 2:
            return {}
        x = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if n_channels > 1:
            x = x.reshape(-1, n_channels).mean(axis=1)
        peak = float(abs(x).max()) / 32768.0 if x.size else 0.0
        rms = float(math.sqrt((x * x).mean())) / 32768.0 if x.size else 0.0
        return {
            "audio_peak": peak,
            "audio_rms": rms,
            "frames": int(n_frames),
            "channels": int(n_channels),
            "sample_rate": int(fr),
        }
    except Exception:
        return {}

def audio_lufs_estimate(wav_bytes: bytes) -> Dict[str, Any]:
    """pyloudnorm があれば LUFS を推定（無ければ空）。"""
    try:
        import numpy as np
        import pyloudnorm as pyln  # type: ignore
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            fr = wf.getframerate()
            frames = wf.readframes(n_frames)
        if sampwidth != 2:
            return {}
        x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            x = x.reshape(-1, n_channels).mean(axis=1)
        meter = pyln.Meter(fr)
        lufs = float(meter.integrated_loudness(x))
        return {"lufs": lufs}
    except Exception:
        return {}