# -*- coding: utf-8 -*-
"""backend/engines/audio_post.py

音声生成後の後処理ユーティリティ。

- WAV(PCM16) の簡易正規化（RMS or LUFS）
- 失敗時は必ず元データを返す（No Regression / fail-soft）

注意:
- LUFS 正規化は `pyloudnorm` がインストールされている場合のみ有効。
  未導入の場合は自動的に RMS 正規化へフォールバックします。
"""

from __future__ import annotations

import io
import wave
from typing import Any, Dict, Tuple, Optional


def _read_wav_pcm16(wav_bytes: bytes):
    import numpy as np

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"Unsupported sample width: {sampwidth} (only PCM16 supported)")

    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return x, n_channels, framerate


def _write_wav_pcm16(x_int16, n_channels: int, framerate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf2:
        wf2.setnchannels(int(n_channels))
        wf2.setsampwidth(2)
        wf2.setframerate(int(framerate))
        wf2.writeframes(x_int16.tobytes())
    return out.getvalue()


def normalize_wav_volume(
    wav_bytes: bytes,
    *,
    target_rms: float = 0.10,
    peak_limit: float = 0.98,
    target_lufs: Optional[float] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """WAV(PCM16) の音量を正規化します。

    - target_lufs が指定された場合: LUFS 正規化（pyloudnorm がある場合）
    - 未指定/失敗時: RMS 正規化

    返り値:
      (wav_bytes, meta)
    """
    meta: Dict[str, Any] = {}
    try:
        import numpy as np

        x, n_channels, framerate = _read_wav_pcm16(wav_bytes)

        # int16 -> float(-1..1)
        xf = x / 32768.0

        # silence guard
        peak = float(np.max(np.abs(xf))) if xf.size else 0.0
        if peak <= 1e-7:
            meta.update({"post": "skip", "reason": "silence"})
            return wav_bytes, meta

        # ------------------------------------------------------------
        # LUFS normalization (optional)
        # ------------------------------------------------------------
        if target_lufs is not None:
            try:
                import pyloudnorm as pyln

                meter = pyln.Meter(int(framerate))
                # pyloudnorm expects mono or multi-channel shaped (n_samples,) or (n_samples, channels)
                if int(n_channels) > 1:
                    # interleaved -> (n_samples, channels)
                    xf2 = xf.reshape((-1, int(n_channels)))
                else:
                    xf2 = xf

                loudness = float(meter.integrated_loudness(xf2))
                gain_db = float(target_lufs) - loudness
                gain = float(10.0 ** (gain_db / 20.0))

                yf = xf * gain

                # peak limiter
                peak2 = float(np.max(np.abs(yf))) if yf.size else 0.0
                if peak2 > float(peak_limit):
                    limiter_gain = float(peak_limit) / max(peak2, 1e-9)
                    yf = yf * limiter_gain
                    meta["limiter_gain"] = limiter_gain

                y_int16 = np.clip(yf * 32768.0, -32768.0, 32767.0).astype(np.int16)

                meta.update(
                    {
                        "post": "lufs",
                        "target_lufs": float(target_lufs),
                        "measured_lufs": loudness,
                        "gain_db": gain_db,
                        "peak_limit": float(peak_limit),
                    }
                )
                return _write_wav_pcm16(y_int16, n_channels, framerate), meta
            except Exception as e:
                meta["lufs_error"] = str(e)
                # fallthrough to RMS

        # ------------------------------------------------------------
        # RMS normalization (fallback)
        # ------------------------------------------------------------
        rms = float(np.sqrt(np.mean(xf * xf)))
        if rms <= 1e-7:
            meta.update({"post": "skip", "reason": "rms_too_small"})
            return wav_bytes, meta

        gain = float(target_rms) / rms
        # peak limiter
        if peak * gain > float(peak_limit):
            gain = float(peak_limit) / max(peak, 1e-9)

        yf = xf * gain
        y_int16 = np.clip(yf * 32768.0, -32768.0, 32767.0).astype(np.int16)

        meta.update({"post": "rms", "gain": gain, "target_rms": float(target_rms), "peak_limit": float(peak_limit)})
        return _write_wav_pcm16(y_int16, n_channels, framerate), meta

    except Exception as e:
        meta["error"] = str(e)
        return wav_bytes, meta


# ============================================================
# 設計A: LUFS 既定値（用途別）
# ============================================================
DEFAULT_TARGET_LUFS_BY_PRESET = {
    "natural": -16.0,
    "mimic": -18.0,
    "clear": -14.0,
}

def resolve_target_lufs(preset_id: str, fallback: float = -16.0) -> float:
    try:
        pid = str(preset_id or "").strip().lower()
        return float(DEFAULT_TARGET_LUFS_BY_PRESET.get(pid, fallback))
    except Exception:
        return float(fallback)
