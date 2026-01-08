# -*- coding: utf-8 -*-
"""backend/engines/audio_prosody.py"""
from __future__ import annotations
import re
from typing import Tuple, Dict

def apply_prosody_rules(text: str, strength: float = 1.0) -> Tuple[str, Dict]:
    t = (text or "").strip()
    s = float(strength)
    t = re.sub(r"\s+"," ",t)
    if s >= 0.75:
        t = re.sub(r"(そして|しかし|また|つまり|なので)", r"\1、", t)
    t = re.sub(r"\s*([、。！？])\s*", r"\1 ", t)
    t = re.sub(r"\s+"," ",t).strip()
    return t, {"prosody_strength": s}
