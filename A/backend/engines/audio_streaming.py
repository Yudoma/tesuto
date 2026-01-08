# -*- coding: utf-8 -*-
"""backend/engines/audio_streaming.py"""
from __future__ import annotations
import io, re, wave
from typing import List, Tuple, Dict, Any

def split_text(text: str, max_len: int = 120) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[。！？])\s+", t)
    chunks=[]; buf=""
    for p in parts:
        p=p.strip()
        if not p: continue
        if len(buf)+len(p)+1 <= max_len:
            buf=(buf+" "+p).strip()
        else:
            if buf: chunks.append(buf)
            buf=p
    if buf: chunks.append(buf)
    return chunks

def concat_wav(wavs: List[bytes]) -> Tuple[bytes, Dict[str, Any]]:
    meta: Dict[str, Any] = {"chunks": len(wavs)}
    if not wavs:
        return b"", meta
    try:
        def read_w(b):
            with wave.open(io.BytesIO(b),"rb") as wf:
                params = wf.getparams(); frames = wf.readframes(wf.getnframes())
            return params, frames
        base_params, base_frames = read_w(wavs[0])
        out_frames = [base_frames]
        for b in wavs[1:]:
            p, fr = read_w(b)
            if (p.nchannels,p.sampwidth,p.framerate)!=(base_params.nchannels,base_params.sampwidth,base_params.framerate):
                meta["format_mismatch"]=True
                return wavs[0], meta
            out_frames.append(fr)
        out=io.BytesIO()
        with wave.open(out,"wb") as wf2:
            wf2.setnchannels(base_params.nchannels); wf2.setsampwidth(base_params.sampwidth); wf2.setframerate(base_params.framerate)
            for fr in out_frames:
                wf2.writeframes(fr)
        return out.getvalue(), meta
    except Exception:
        return wavs[0], meta
