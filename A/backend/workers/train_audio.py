# -*- coding: utf-8 -*-
"""Audio (GPT-SoVITS) 学習ワーカー（前処理 + 学習委譲）。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

TRAIN_PRESETS = {
    'jp_standard': {'sr': 44100, 'min_sec': 2.0, 'max_sec': 10.0},
    'jp_quality':  {'sr': 48000, 'min_sec': 2.0, 'max_sec': 12.0},
    'jp_fast':     {'sr': 32000, 'min_sec': 2.0, 'max_sec': 8.0},
}

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

def _log(msg: str):
    print(msg, flush=True)


def _iter_audio_files(root: Path) -> List[Path]:
    res: List[Path] = []
    if not root.exists():
        return res
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            res.append(p)
    res.sort()
    return res

def _try_load_audio(path: Path, sr: int):
    """librosa が読めないケースを安全に救済する（ffmpeg があれば一時WAV変換）。"""
    import librosa
    try:
        y, _sr = _try_load_audio(str(path), sr=sr, mono=True)
        return y, _sr
    except Exception:
        ff = shutil.which("ffmpeg")
        if not ff:
            raise
        tmp = path.with_suffix(".tmp_convert.wav")
        cmd = [ff, "-y", "-i", str(path), "-ar", str(sr), "-ac", "1", str(tmp)]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        y, _sr = _try_load_audio(str(tmp), sr=sr, mono=True)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return y, _sr

def _slice_and_save(src: Path, out_dir: Path, sr: int, min_sec: float, max_sec: float) -> List[Path]:
    import librosa
    import numpy as np
    import soundfile as sf

    y, _sr = _try_load_audio(str(src), sr=sr, mono=True)
    if y is None:
        return []

    min_len = int(min_sec * sr)
    max_len = int(max_sec * sr)
    if y.shape[0] < min_len:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    wavs: List[Path] = []

    start = 0
    idx = 0
    hop = max_len
    while start < y.shape[0]:
        end = min(start + max_len, y.shape[0])
        if end - start < min_len:
            break
        seg = y[start:end].astype(np.float32)
        if float(np.mean(np.abs(seg))) < 1e-4:
            start += hop
            continue
        idx += 1
        out = out_dir / f"{src.stem}_{idx:04d}.wav"
        sf.write(str(out), seg, sr, subtype='PCM_16')
        wavs.append(out)
        start += hop

    return wavs

def _asr(wav: Path, model_name: str, language: str, device: str) -> str:
    from faster_whisper import WhisperModel
    global _WHISPER
    try:
        m = _WHISPER
    except Exception:
        m = None
    if m is None:
        compute_type = 'int8'
        _log(f"[ASR] モデルロード: {model_name} (device={device})")
        m = WhisperModel(model_name, device=device, compute_type=compute_type)
        _WHISPER = m
    segs, _ = m.transcribe(str(wav), language=language or None)
    parts: List[str] = []
    for s in segs:
        t = (s.text or '').strip()
        if t:
            parts.append(t)
    return ''.join(parts).strip()


def _run_xtts_finetune(xtts_repo: Path, prepared_dir: Path, output_dir: Path, config_rel: str, extra_args: str) -> int:
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['PYTHONPATH'] = str(xtts_repo) + (os.pathsep + env.get('PYTHONPATH','') if env.get('PYTHONPATH') else '')

    # Build minimal dataset (LJSpeech-like)
    ds_dir = prepared_dir / "xtts_dataset"
    wavs_dst = ds_dir / "wavs"
    ds_dir.mkdir(parents=True, exist_ok=True)
    wavs_dst.mkdir(parents=True, exist_ok=True)

    train_list = prepared_dir / "train.list"
    meta = ds_dir / "metadata.csv"
    if not train_list.exists():
        _log("[ERROR] train.list が見つかりません。前処理が失敗しています。")
        return 3

    import shutil
    lines = train_list.read_text(encoding="utf-8", errors="replace").splitlines()
    out_lines = []
    for ln in lines:
        if "|" not in ln:
            continue
        wav_path, text = ln.split("|", 1)
        wp = Path(wav_path)
        if not wp.exists():
            continue
        dst = wavs_dst / wp.name
        if not dst.exists():
            try:
                shutil.copy2(wp, dst)
            except Exception:
                pass
        out_lines.append(f"wavs/{dst.name}|{text.strip()}")
    if not out_lines:
        _log("[ERROR] XTTS 用メタデータが作成できませんでした。")
        return 4
    meta.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    cfg = (xtts_repo / config_rel).resolve()
    if not cfg.exists():
        for c in [
            xtts_repo / "recipes" / "xtts" / "finetune" / "xtts_v2_finetune_config.json",
            xtts_repo / "recipes" / "xtts" / "finetune" / "config.json",
        ]:
            if c.exists():
                cfg = c.resolve()
                break
    if not cfg.exists():
        _log(f"[ERROR] XTTS finetune config が見つかりません: {cfg}")
        return 5

    train_py = xtts_repo / "TTS" / "bin" / "train_tts.py"
    xtts_out = output_dir / "xtts_finetune"
    xtts_out.mkdir(parents=True, exist_ok=True)

    base_cmds = []
    if train_py.exists():
        base_cmds.append([sys.executable, str(train_py), "--config_path", str(cfg), "--output_path", str(xtts_out)])
    base_cmds.append([sys.executable, "-m", "TTS.bin.train_tts", "--config_path", str(cfg), "--output_path", str(xtts_out)])

    data_flags = [
        ["--data_path", str(ds_dir)],
        ["--dataset_path", str(ds_dir)],
    ]

    extra = (extra_args or "").strip()
    for base_cmd in base_cmds:
        for df in data_flags:
            cmd = base_cmd + df
            if extra:
                cmd = cmd + extra.split()
            _log("[XTTS] 実行: " + " ".join(cmd))
            try:
                return subprocess.call(cmd, cwd=str(xtts_repo), env=env)
            except Exception as e:
                _log(f"[XTTS] 実行失敗: {e}")
                continue

    _log("[ERROR] XTTS 学習コマンドの実行に失敗しました（train_tts の引数差異の可能性）。")
    _log("[HINT] coqui-ai/TTS の recipes/xtts/finetune を確認し、UIの「XTTS 追加引数」で調整してください。")
    return 6


def _run_external(gpt_sovits_repo: Path, prepared_dir: Path, output_dir: Path, custom_cmd: str) -> int:
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LORA_FACTORY_AUDIO_PREPARED_DIR'] = str(prepared_dir)
    env['LORA_FACTORY_AUDIO_OUTPUT_DIR'] = str(output_dir)
    if custom_cmd:
        cmd = custom_cmd.format(prepared_dir=str(prepared_dir), output_dir=str(output_dir))
        _log(f"[TRAIN] カスタムコマンド: {cmd}")
        return subprocess.call(cmd, cwd=str(gpt_sovits_repo), env=env, shell=True)
    # 既知候補（バージョン差対策のため複数）
    candidates = [
        gpt_sovits_repo / 'cli' / 'train.py',
        gpt_sovits_repo / 'train.py',
        gpt_sovits_repo / 'tools' / 'train.py',
        gpt_sovits_repo / 'GPT_SoVITS' / 'train.py',
    ]
    for c in candidates:
        if c.exists():
            cmd = [sys.executable, str(c), '--prepared_dir', str(prepared_dir), '--output_dir', str(output_dir)]
            _log(f"[TRAIN] 実行: {' '.join(cmd)}")
            return subprocess.call(cmd, cwd=str(gpt_sovits_repo), env=env)
    _log('[ERROR] GPT-SoVITS 学習スクリプトが見つかりません。')
    return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--base_model_path', default='')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--train_batch_size', type=int, default=1)
    ap.add_argument('--whisper_model', default='small')
    ap.add_argument('--language', default='ja')
    ap.add_argument('--gpt_sovits_repo', default='')
    ap.add_argument('--xtts_repo', default='')
    ap.add_argument('--train_type', default='gpt_sovits')
    ap.add_argument('--xtts_config', default='recipes/xtts/finetune/xtts_v2_finetune_config.json')
    ap.add_argument('--xtts_extra_args', default='')
    ap.add_argument('--custom_train_cmd', default='')
    ap.add_argument('--slice_min_sec', type=float, default=3.0)
    ap.add_argument('--slice_max_sec', type=float, default=10.0)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_dir = output_dir / 'prepared'
    wav_out_dir = prepared_dir / 'wavs'
    list_path = prepared_dir / 'train.list'

    # 依存チェック
    missing = []
    for mod in ['librosa', 'soundfile', 'faster_whisper']:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        _log('[ERROR] 依存が不足しています: ' + ', '.join(missing))
        sys.exit(1)

    if prepared_dir.exists():
        import shutil
        shutil.rmtree(prepared_dir, ignore_errors=True)
    wav_out_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_audio_files(dataset_dir)
    if not files:
        _log(f'[ERROR] 音声ファイルが見つかりません: {dataset_dir}')
        sys.exit(2)

    device = 'cuda'
    try:
        import torch
        if not torch.cuda.is_available():
            device = 'cpu'
    except Exception:
        device = 'cpu'

    entries: List[Tuple[Path, str]] = []
    for i, src in enumerate(files, start=1):
        _log(f'[PREP] ({i}/{len(files)}) {src.name}')
        wavs = _slice_and_save(src, wav_out_dir, 44100, args.slice_min_sec, args.slice_max_sec)
        for w in wavs:
            txt = _asr(w, args.whisper_model, args.language, device)
            if txt:
                entries.append((w, txt))
        if i % 2 == 0:
            _log(f'[PREP] 進捗: files={i}/{len(files)} samples={len(entries)}')

    if not entries:
        _log('[ERROR] 学習サンプルが作成できませんでした（ASR結果が空の可能性）。')
        sys.exit(3)

    with list_path.open('w', encoding='utf-8') as f:
        for w, t in entries:
            f.write(f"{w.as_posix()}|{t}\n")

    _log(f'[PREP] 完了: samples={len(entries)} list={list_path}')

    # 学習委譲（任意）
if args.train_type == 'xtts_finetune':
    if not args.xtts_repo:
        _log('[ERROR] xtts_repo 未指定です。third_party/XTTS を配置してください。')
        sys.exit(10)
    code = _run_xtts_finetune(Path(args.xtts_repo), prepared_dir, output_dir, args.xtts_config, args.xtts_extra_args)
    if code != 0:
        sys.exit(code)
elif args.gpt_sovits_repo:
    code = _run_external(Path(args.gpt_sovits_repo), prepared_dir, output_dir, args.custom_train_cmd)
    if code != 0:
        sys.exit(code)
else:
    _log('[INFO] gpt_sovits_repo 未指定のため、前処理のみで終了します。')

_log('[OK] 完了')


if __name__ == '__main__':
    main()
