# -*- coding: utf-8 -*-
"""backend/engines/audio_text_norm.py

日本語音声向けのテキスト正規化（G2Pの前処理）。

目的:
- 数字/英字/記号の読みを整え、TTSの破綻（棒読み/無音/暴走）を減らす
- UI入力の改行やASCII句読点を日本語音声向けに揃える
- 変換は「安全側（読めないよりはそのまま）」に倒す

注:
- 日本語の数詞変換は文脈依存が強い（例: 110番、3D、2026年）ため、
  本実装は「過剰に賢くしすぎない」方針で、誤読リスクが高い部分は桁読みへ退避する。
"""
from __future__ import annotations

import re
from typing import Tuple, Dict

# 0-9の単体読み（桁読み fallback）
_DIGIT_MAP = {
    "0": "ゼロ",
    "1": "いち",
    "2": "に",
    "3": "さん",
    "4": "よん",
    "5": "ご",
    "6": "ろく",
    "7": "なな",
    "8": "はち",
    "9": "きゅう",
}

# A-Z 読み（ローマ字が混ざる現実入力を想定）
_ALPHA_MAP = {
    "A": "エー", "B": "ビー", "C": "シー", "D": "ディー", "E": "イー",
    "F": "エフ", "G": "ジー", "H": "エイチ", "I": "アイ", "J": "ジェー",
    "K": "ケー", "L": "エル", "M": "エム", "N": "エヌ", "O": "オー",
    "P": "ピー", "Q": "キュー", "R": "アール", "S": "エス", "T": "ティー",
    "U": "ユー", "V": "ヴィー", "W": "ダブリュー", "X": "エックス",
    "Y": "ワイ", "Z": "ズィー",
}

# ASCII句読点→日本語寄りに統一
_ASCII_PUNC = {
    ",": "、",
    ".": "。",
    "!": "！",
    "?": "？",
    ":": "：",
    ";": "；",
}

# 数詞の読み（0-9999まで）
# ここだけは「誤読の少ない範囲」で対応し、それ以上は桁読みへ逃がす。
_UNITS = ["", "じゅう", "ひゃく", "せん"]
_SPECIAL = {
    300: "さんびゃく",
    600: "ろっぴゃく",
    800: "はっぴゃく",
    3000: "さんぜん",
    8000: "はっせん",
}
_HUNDREDS = {1: "ひゃく", 2: "にひゃく", 3: "さんびゃく", 4: "よんひゃく", 5: "ごひゃく", 6: "ろっぴゃく", 7: "ななひゃく", 8: "はっぴゃく", 9: "きゅうひゃく"}
_THOUSANDS = {1: "せん", 2: "にせん", 3: "さんぜん", 4: "よんせん", 5: "ごせん", 6: "ろくせん", 7: "ななせん", 8: "はっせん", 9: "きゅうせん"}
_TENS = {1: "じゅう", 2: "にじゅう", 3: "さんじゅう", 4: "よんじゅう", 5: "ごじゅう", 6: "ろくじゅう", 7: "ななじゅう", 8: "はちじゅう", 9: "きゅうじゅう"}

def _normalize_punct(text: str) -> str:
    t = text or ""
    for k, v in _ASCII_PUNC.items():
        t = t.replace(k, v)
    # 連続改行は句点 + 空白へ寄せる
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n+", "\n", t)
    t = t.replace("\n", "。 ")
    # 連続スペースを整理
    t = re.sub(r"\s+", " ", t).strip()
    # 句読点の連続を整理
    t = re.sub(r"[。]{2,}", "。", t)
    t = re.sub(r"[、]{2,}", "、", t)
    return t

def _alpha_to_kana(text: str) -> str:
    def repl(m: re.Match) -> str:
        s = m.group(0)
        out = []
        for ch in s:
            out.append(_ALPHA_MAP.get(ch.upper(), ch))
        return " ".join(out)
    # 2文字以上の英字列のみ変換（1文字は略語/変数名等で誤爆しやすいため控えめ）
    return re.sub(r"[A-Za-z]{2,}", repl, text)

def _int_to_kana(n: int) -> str:
    if n == 0:
        return _DIGIT_MAP["0"]
    if n in _SPECIAL:
        return _SPECIAL[n]
    if n < 0:
        return "マイナス " + _int_to_kana(-n)

    if n > 9999:
        # 桁読み fallback（誤読回避）
        s = str(n)
        return " ".join(_DIGIT_MAP.get(ch, ch) for ch in s)

    parts = []
    thousands = n // 1000
    hundreds = (n // 100) % 10
    tens = (n // 10) % 10
    ones = n % 10

    if thousands:
        parts.append(_THOUSANDS[thousands])
    if hundreds:
        parts.append(_HUNDREDS[hundreds])
    if tens:
        parts.append(_TENS[tens])
    if ones:
        parts.append(_DIGIT_MAP[str(ones)])
    return "".join(parts)

def _numbers_to_kana(text: str) -> str:
    # 例: 3.14 -> さんてんいちよん（小数点は「てん」）
    def repl(m: re.Match) -> str:
        s = m.group(0)
        if "." in s:
            a, b = s.split(".", 1)
            try:
                ai = int(a) if a else 0
                left = _int_to_kana(ai)
            except Exception:
                left = " ".join(_DIGIT_MAP.get(ch, ch) for ch in a)
            right = " ".join(_DIGIT_MAP.get(ch, ch) for ch in b)
            return f"{left}てん{right}"
        try:
            return _int_to_kana(int(s))
        except Exception:
            return " ".join(_DIGIT_MAP.get(ch, ch) for ch in s)
    # 3.14 / 2026 / 110 など
    return re.sub(r"\d+(?:\.\d+)?", repl, text)

def normalize_ja(text: str) -> Tuple[str, Dict[str, str]]:
    """日本語TTS向けにテキストを正規化する。

    Returns:
        normalized_text, meta
    """
    raw = text or ""
    t = raw.strip()
    t = _normalize_punct(t)
    t = _alpha_to_kana(t)
    t = _numbers_to_kana(t)

    # 読みやすさ: 句点/感嘆/疑問の後ろにスペースを入れる（XTTS等の安定用）
    t = re.sub(r"([。！？])\s*", r"\1 ", t).strip()

    if t and t[-1] not in ("。", "！", "？"):
        t += "。"

    return t, {"raw_text": raw, "normalized_text": t}
