# -*- coding: utf-8 -*-
"""backend/engines/audio.py

Audio (XTTS中心 + VC(RVC/GPT-SoVITS)) モダリティ用エンジン。
- JobManager で train_audio.py をサブプロセス起動
- 推論は「外部 GPT-SoVITS 推論スクリプト」に委譲（任意）

※ GPT-SoVITS 本体は同梱しないため、UI/params で repo パスを渡せる設計。
"""

from __future__ import annotations

import base64
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Optional: Coqui TTS (XTTS)
try:
    from TTS.api import TTS  # type: ignore
except Exception:
    TTS = None  # type: ignore


from lora_config import settings
from backend.core.dataset_report import build_dataset_report
from backend.core.job_manager import job_manager
from backend.core.artifact_store import artifact_store
from backend.core.eval_metrics import audio_rms_metrics
from backend.engines.base import BaseEngine


# ============================================================
# 設計A: 用途別プリセット（XTTS -> VC -> Post）
# ============================================================
VOICE_PRESETS = {
    # 自然さ重視（日本語読み上げの破綻を減らす）
    "natural": {
        "label": "自然（推奨）",
        "tts_backend": "xtts",
        "vc_backend": "none",
        "postprocess": True,
        "target_lufs": -16.0,
    },
    # 似せ重視（参照音声の雰囲気をVCで寄せる）
    "mimic": {
        "label": "似せ重視（VCあり）",
        "tts_backend": "xtts",
        "vc_backend": "rvc",
        "postprocess": True,
        "target_lufs": -18.0,
    },
}

def _apply_voice_preset(preset_id: str, params: dict) -> dict:
    """preset_id に一致する値があれば params を上書き適用します。

    BK43運用方針:
    - VOICE_PRESETS は「勝ち設定」として凍結し、preset_id 指定時は再現性を最優先します。
    - 呼び出し側の指定値があっても、preset の定義値で上書きします（freeze）。
    """
    try:
        pid = str(preset_id or "").strip().lower()
        p = VOICE_PRESETS.get(pid)
        if not p:
            return params
        # freeze: preset定義を優先するキー
        freeze_keys = {
            "tts_backend",
            "vc_backend",
            "postprocess",
            "target_lufs",
            "xtts_language",
            "xtts_model_id",
        }
        for k, v in p.items():
            if k == "label":
                continue
            if k in freeze_keys:
                params[k] = v
            else:
                # 後方互換: 未指定のみ補完
                if params.get(k) in (None, "", False):
                    params[k] = v
        params["_preset_frozen"] = True
        return params
    except Exception:
        return params

class AudioEngine(BaseEngine):
    def __init__(self):
        self.worker_script = settings.base_dir / 'backend' / 'workers' / 'train_audio.py'
        self._history_file = settings.logs_dir / 'history_audio.json'

        # 推論設定（外部委譲）
        self._infer_repo: Optional[Path] = None
        self._infer_model_dir: Optional[Path] = None
        self._infer_custom_cmd: Optional[str] = None
        self._lock = threading.Lock()

        # XTTS (TTS) / VC 設定
        self._tts_backend: str = 'xtts'  # 'xtts' or 'gpt_sovits'
        self._vc_backend: str = 'none'   # 'none' or 'rvc' or 'gpt_sovits_vc'
        self._xtts_model_id: Optional[str] = None
        self._xtts_instance = None  # lazy cache (Coqui TTS)
        self._rvc_repo: Optional[Path] = None
        self._rvc_custom_cmd: Optional[str] = None
        self._gpt_sovits_vc_repo: Optional[Path] = None
        self._gpt_sovits_vc_custom_cmd: Optional[str] = None

    # =========================================================================
    # Training Job Management
    # =========================================================================

    def start_training(self, base_model: str, dataset: str, params: Dict[str, Any]) -> Dict[str, Any]:
        dataset_path = settings.dirs['audio']['datasets'] / dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f'Dataset not found: {dataset_path}')

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        out_dir = settings.dirs['audio']['output'] / f'{dataset}_{timestamp}'
        out_dir.mkdir(parents=True, exist_ok=True)

        worker_script = self.worker_script
        if not worker_script.exists():
            worker_script = Path('backend/workers/train_audio.py')

        cmd = [
            sys.executable,
            str(worker_script),
            '--dataset_dir', str(dataset_path),
            '--output_dir', str(out_dir),
        ]

        # UI/params のキー揺れを吸収
        def _get(key: str, default=None):
            v = (params or {}).get(key)
            return default if v is None else v

        # 前処理/ASR
        cmd += ['--whisper_model', str(_get('whisper_model', 'small'))]
        cmd += ['--language', str(_get('language', 'ja'))]
        cmd += ['--slice_min_sec', str(_get('slice_min_sec', 3.0))]
        cmd += ['--slice_max_sec', str(_get('slice_max_sec', 10.0))]
        cmd += ['--slice_target_sec', str(_get('slice_target_sec', 8.0))]
        cmd += ['--slice_hop_sec', str(_get('slice_hop_sec', 6.0))]

        # 学習委譲
        cmd += ['--train_type', train_type]
        gpt_repo = str(_get('gpt_sovits_repo', '') or '')
        custom_train_cmd = str(_get('custom_train_cmd', '') or '')
        train_type = str(_get('train_type', 'gpt_sovits') or 'gpt_sovits')
        xtts_repo = str(_get('xtts_repo', '') or '')
        xtts_config = str(_get('xtts_config', 'recipes/xtts/finetune/xtts_v2_finetune_config.json') or 'recipes/xtts/finetune/xtts_v2_finetune_config.json')
        xtts_extra_args = str(_get('xtts_extra_args', '') or '')
        # XTTS（finetune）: 未指定なら third_party/XTTS を使う（clone前提B）
        if (not xtts_repo) and train_type == 'xtts_finetune':
            xtts_repo = str((settings.base_dir / 'third_party' / 'XTTS').resolve())
        if xtts_repo:
            cmd += ['--xtts_repo', xtts_repo]
        if xtts_config:
            cmd += ['--xtts_config', xtts_config]
        if xtts_extra_args:
            cmd += ['--xtts_extra_args', xtts_extra_args]

        if gpt_repo:
            cmd += ['--gpt_sovits_repo', gpt_repo]
        if custom_train_cmd:
            cmd += ['--custom_train_cmd', custom_train_cmd]
        ui_params = dict(params or {})
        ui_params.update({
            'base_model': base_model,
            'dataset': dataset,
            'output_dir': str(out_dir),
        })

        ret = job_manager.start_job(
            cmd=cmd,
            params=ui_params,
            cwd=Path(__file__).resolve().parents[2],
            env=os.environ.copy(),
            log_prefix='train_audio',
        )

        # データセット検査結果を runs/<job_id>/dataset_report.json に保存（UIで再表示）
        try:
            job_id = ret.get("job_id") or ret.get("id") or ret.get("jobId")
            if job_id:
                run_dir = settings.runs_root / str(job_id)
                run_dir.mkdir(parents=True, exist_ok=True)
                rep = build_dataset_report("audio", Path(str(dataset_path)).resolve())
                (run_dir / "dataset_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        # 履歴
        self._append_history(ret.get('job_id'), base_model, dataset, ui_params, 'running')
        return ret

    def stop_training(self) -> Dict[str, str]:
        job_manager.stop_job()
        return {'status': 'stopped'}

    def get_training_status(self) -> Dict[str, Any]:
        st = job_manager.get_status()
        if st.get('status') in ['completed', 'failed', 'stopped']:
            self._update_history_status(st.get('job_id'), st.get('status'))
        return st

    
    def rerun_training(self, job_id: str) -> Dict[str, Any]:
        """履歴から同一条件で再学習を開始する"""
        hist = self.get_training_history().get("history") or []
        target = None
        for item in hist:
            if str(item.get("id")) == str(job_id):
                target = item
                break
        if not target:
            raise RuntimeError("履歴が見つかりませんでした。")
        model = target.get("model")
        dataset = target.get("dataset")
        params = (target.get("params") or {})
        return self.start_training(model, dataset, params)

    def get_training_history(self) -> Dict[str, Any]:
        if not self._history_file.exists():
            return {'history': []}
        try:
            hist = json.loads(self._history_file.read_text(encoding='utf-8'))
            hist.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return {'history': hist}
        except Exception:
            return {'history': []}

    def _append_history(self, job_id: Optional[str], model: str, dataset: str, params: Dict[str, Any], status: str):
        hist = []
        if self._history_file.exists():
            try:
                hist = json.loads(self._history_file.read_text(encoding='utf-8'))
            except Exception:
                hist = []
        hist.append({
            'id': job_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': model,
            'dataset': dataset,
            'status': status,
            'params': params,
        })
        self._history_file.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding='utf-8')

    def _update_history_status(self, job_id: Optional[str], status: str):
        if not job_id or not self._history_file.exists():
            return
        try:
            hist = json.loads(self._history_file.read_text(encoding='utf-8'))
            changed = False
            for it in hist:
                if it.get('id') != job_id:
                    continue
                if it.get('status') not in ['running', 'idle']:
                    continue
                it['status'] = status
                changed = True
            if changed:
                self._history_file.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

    # =========================================================================
    # Inference Model Management
    # =========================================================================

    def load_inference_model(self, base_model: str, adapter_path: Optional[str] = None) -> Dict[str, Any]:
        """推論用モデルをロードする。

        GPT-SoVITS はモデル構成が複雑で、本リポジトリ側で完全なロード実装を固定すると
        将来のバージョン差で破綻しやすいため、基本は「外部推論スクリプト」に委譲する。

        使い方:
        - base_model: models/audio 配下のフォルダ名、または任意パス
        - adapter_path: (任意) 将来用（未使用）
        - 追加の repo パスやコマンドは generate_audio 側の引数（API）で上書きも可能
        """
        with self._lock:
            self.unload_inference_model()

            model_dir = settings.dirs['audio']['models'] / base_model
            if not model_dir.exists():
                model_dir = Path(base_model)

            if not model_dir.exists():
                return {'status': 'error', 'message': f'Model not found: {model_dir}'}

            self._infer_model_dir = model_dir
            # repo は環境変数でも受け取れる
            repo = os.environ.get('GPT_SOVITS_DIR', '').strip()
            self._infer_repo = Path(repo) if repo else None
            self._infer_custom_cmd = None
            return {'status': 'loaded', 'base_model': base_model, 'model_dir': str(model_dir)}

    def unload_inference_model(self) -> Dict[str, str]:
        with self._lock:
            self._infer_repo = None
            self._infer_model_dir = None
            self._infer_custom_cmd = None
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return {'status': 'unloaded'}

    def is_inference_model_loaded(self) -> bool:
        return self._infer_model_dir is not None

    
    # =========================================================================
    # Internal helpers (XTTS / External CLI / VC)
    # =========================================================================

    def _ensure_xtts_loaded(self, model_id: Optional[str] = None) -> None:
        """XTTS を可能ならロードしてキャッシュする。TTSが無い場合は何もしない。"""
        mid = (model_id or self._xtts_model_id or '').strip() or 'tts_models/multilingual/multi-dataset/xtts_v2'
        self._xtts_model_id = mid
        if self._xtts_instance is not None:
            return
        if TTS is None:
            return
        try:
            self._xtts_instance = TTS(mid, progress_bar=False, gpu=torch.cuda.is_available())
        except Exception:
            self._xtts_instance = None

    @staticmethod
    def _run_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Optional[str]:
        """外部コマンドを実行し、失敗時はエラーメッセージを返す（成功時は None）。

        - stdout/stderr を capture してログとして返す（UI/運用の原因特定を容易にする）
        - 例外は文字列化して返す（fail-soft）
        """
        try:
            p = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if p.returncode != 0:
                out = (p.stdout or "").strip()
                err = (p.stderr or "").strip()
                msg = f"コマンド実行に失敗しました (returncode={p.returncode})\ncmd: {' '.join(cmd)}"
                if out:
                    msg += f"\n--- stdout ---\n{out}"
                if err:
                    try:
                        meta.setdefault("run_logs", []).append(err)
                    except Exception:
                        pass
                    msg += f"\n--- stderr ---\n{err}"
                return msg
            return None
        except Exception as e:
            return f"コマンド実行中に例外が発生しました: {e}\ncmd: {' '.join(cmd)}"

    def _pick_first_existing(self, paths: list[Path]) -> Optional[Path]:
        for p in paths:
            if p and p.exists():
                return p
        return None

# =========================================================================
    # Audio-specific API
    # =========================================================================

        def generate_audio(
            self,
            text: str,
            reference_audio_path: str,
            preset_id: Optional[str] = None,
            gpt_sovits_repo: Optional[str] = None,
            custom_infer_cmd: Optional[str] = None,
            output_format: str = 'wav',
            # v2: XTTS / VC
            tts_backend: Optional[str] = None,
            vc_backend: Optional[str] = None,
            xtts_model_id: Optional[str] = None,
            xtts_language: str = 'ja',
            rvc_repo: Optional[str] = None,
            rvc_custom_cmd: Optional[str] = None,
            gpt_sovits_vc_repo: Optional[str] = None,
            gpt_sovits_vc_custom_cmd: Optional[str] = None,
            # post process
            postprocess: bool = True,
            target_lufs: Optional[float] = None,
        ) -> Dict[str, Any]:
            """テキストと参照音声から音声を生成し、Base64(Data URI)で返す。
    
            方針（修正後）:
            1) まず **TTS**（推奨: XTTS）でテキストから音声を生成
            2) 必要なら **VC**（RVC / GPT-SoVITS VC）で「声質」を変換
            3) 後処理（WAV音量正規化）→ artifact保存 → base64返却
    
            互換性:
            - 旧UI/旧APIは gpt_sovits_repo / custom_infer_cmd を送るだけで従来通り動作します。
            - XTTS が利用できない/ロードできない場合は、外部CLI委譲 or GPT-SoVITS推論へフォールバックします。
            """
            with self._lock:
                # プリセット適用（任意・後方互換）
                try:
                    _params = {
                        "tts_backend": tts_backend,
                        "vc_backend": vc_backend,
                        "postprocess": postprocess,
                        "target_lufs": target_lufs,
                    }
                    _params = _apply_voice_preset(preset_id or "", _params)
                    tts_backend = _params.get("tts_backend") or tts_backend
                    vc_backend = _params.get("vc_backend") or vc_backend
                    postprocess = bool(_params.get("postprocess", postprocess))
                    target_lufs = _params.get("target_lufs", target_lufs)
                except Exception:
                    pass

                # 参照音声の解決
                ref = Path(reference_audio_path)
                if not ref.exists():
                    cand = settings.dirs['audio']['datasets'] / reference_audio_path
                    if cand.exists():
                        ref = cand
                if not ref.exists():
                    return {'status': 'error', 'message': f'Reference audio not found: {reference_audio_path}'}
    
                # 日本語前処理（正規化/韻律）
                normalized_text, norm_meta = normalize_ja(text)
                prosody_text, prosody_meta = apply_prosody_rules(normalized_text, strength=1.0)
    
                work = settings.logs_dir / 'audio_infer_tmp'
                work.mkdir(parents=True, exist_ok=True)
                text_file = work / 'text.txt'
                text_file.write_text(prosody_text or '', encoding='utf-8')
    
                tts_out = work / 'tts.wav'
                if tts_out.exists():
                    try:
                        tts_out.unlink()
                    except Exception:
                        pass
    
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
    
                tb = (tts_backend or self._tts_backend or 'xtts').strip().lower()
                vb = (vc_backend or self._vc_backend or 'none').strip().lower()
    
                # 設定を記憶（UIが毎回送らない場合のため）
                self._tts_backend = tb
                self._vc_backend = vb
                if xtts_model_id:
                    self._xtts_model_id = xtts_model_id
                if rvc_repo:
                    self._rvc_repo = Path(rvc_repo)
                if rvc_custom_cmd:
                    self._rvc_custom_cmd = rvc_custom_cmd
                if gpt_sovits_vc_repo:
                    self._gpt_sovits_vc_repo = Path(gpt_sovits_vc_repo)
                if gpt_sovits_vc_custom_cmd:
                    self._gpt_sovits_vc_custom_cmd = gpt_sovits_vc_custom_cmd
    
                tts_ok = False
                tts_meta: Dict[str, Any] = {'tts_backend': tb}
    
                # 1) XTTS
                if tb == 'xtts':
                    self._ensure_xtts_loaded(model_id=xtts_model_id)
                    if self._xtts_instance is not None:
                        try:
                            self._xtts_instance.tts_to_file(
                                text=prosody_text,
                                file_path=str(tts_out),
                                speaker_wav=str(ref),
                                language=xtts_language or 'ja',
                            )
                            tts_ok = tts_out.exists()
                            tts_meta.update({
                                'xtts_model_id': self._xtts_model_id,
                                'xtts_language': xtts_language,
                                'speaker_wav': str(ref),
                            })
                        except Exception as e:
                            tts_meta['xtts_error'] = str(e)
                            tts_ok = False
    
                    # 外部委譲（環境変数）
                    if not tts_ok:
                        xtts_repo = os.environ.get('XTTS_DIR', '').strip()
                        cmd_str = os.environ.get('XTTS_CUSTOM_INFER_CMD', '').strip()
                        if xtts_repo and cmd_str:
                            repo = Path(xtts_repo)
                            if repo.exists():
                                cmd = cmd_str.format(
                                    repo=str(repo),
                                    model_id=str(xtts_model_id or self._xtts_model_id or ''),
                                    ref=str(ref),
                                    text_file=str(text_file),
                                    out=str(tts_out),
                                    language=str(xtts_language or 'ja'),
                                )
                                err = self._run_cmd(cmd, cwd=repo, env=env)
                                if err is None and tts_out.exists():
                                    tts_ok = True
                                    tts_meta.update({'xtts_repo': str(repo), 'xtts_custom_cmd': cmd_str})
    
                    # フォールバック（GPT-SoVITS）
                    if not tts_ok:
                        tb = 'gpt_sovits'
                        tts_meta['fallback'] = 'gpt_sovits'
    
                # 2) GPT-SoVITS（従来）
                if tb == 'gpt_sovits':
                    if not self._infer_model_dir:
                        return {'status': 'error', 'message': 'Inference model is not loaded. Please call /audio/inference/load first.'}
    
                    model_dir = self._infer_model_dir
                    repo = Path(gpt_sovits_repo) if gpt_sovits_repo else self._infer_repo
                    if repo is None:
                        return {'status': 'error', 'message': 'GPT-SoVITS の repo パスが未指定です（環境変数 GPT_SOVITS_DIR か API 引数で指定してください）。'}
                    if not repo.exists():
                        return {'status': 'error', 'message': f'GPT-SoVITS repo not found: {repo}'}
    
                    cmd_str = (custom_infer_cmd or self._infer_custom_cmd or '').strip()
                    if cmd_str:
                        cmd = cmd_str.format(repo=str(repo), model_dir=str(model_dir), ref=str(ref), text_file=str(text_file), out=str(tts_out))
                        err = self._run_cmd(cmd, cwd=repo, env=env)
                        if err is not None:
                            return {'status': 'error', 'message': f'Inference failed (custom_cmd): {err}'}
                    else:
                        candidates = [
                            repo / 'cli' / 'infer.py',
                            repo / 'infer.py',
                            repo / 'tools' / 'infer.py',
                            repo / 'GPT_SoVITS' / 'infer.py',
                        ]
                        script = self._pick_first_existing(candidates)
                        if script is None:
                            return {'status': 'error', 'message': 'GPT-SoVITS 推論スクリプトが見つかりません。custom_infer_cmd を指定してください。'}
    
                        cmd = f'{sys.executable} "{script}" --model_dir "{model_dir}" --ref "{ref}" --text_file "{text_file}" --out "{tts_out}"'
                        err = self._run_cmd(cmd, cwd=repo, env=env)
                        if err is not None:
                            return {'status': 'error', 'message': f'Inference failed: {err}'}
    
                    if not tts_out.exists():
                        return {'status': 'error', 'message': f'Output not created: {tts_out}'}
                    tts_ok = True
    
                if not tts_ok:
                    return {'status': 'error', 'message': 'TTS stage failed. XTTS未導入の場合は、TTSライブラリ導入または XTTS_DIR/XTTS_CUSTOM_INFER_CMD の設定、もしくは GPT-SoVITS 推論設定を確認してください。'}
    
                # 2) VC（任意）
                data: bytes
                vc_meta: Dict[str, Any] = {'vc_backend': vb}
    
                if vb in ('none', '', 'off', 'disabled'):
                    data = tts_out.read_bytes()
                    vc_meta['applied'] = False
                else:
                    out_wav = work / 'vc.wav'
                    if out_wav.exists():
                        try:
                            out_wav.unlink()
                        except Exception:
                            pass
    
                    if vb == 'rvc':
                        repo = self._rvc_repo or (Path(os.environ.get('RVC_DIR', '').strip()) if os.environ.get('RVC_DIR','').strip() else None)
                        cmd_str = (self._rvc_custom_cmd or os.environ.get('RVC_CUSTOM_VC_CMD', '').strip())
                        if repo is None or not Path(repo).exists():
                            vc_meta.update({'applied': False, 'warning': 'RVC の repo パスが未指定のため、VC をスキップしました。'}); data = tts_out.read_bytes()
                        if not cmd_str:
                            vc_meta.update({'applied': False, 'warning': 'RVC_CUSTOM_VC_CMD が未設定のため、VC をスキップしました。'}); data = tts_out.read_bytes()
                        cmd = cmd_str.format(repo=str(repo), in_wav=str(tts_out), ref=str(ref), out=str(out_wav))
                        err = self._run_cmd(cmd, cwd=Path(repo), env=env)
                        if err is not None or not out_wav.exists():
                            vc_meta.update({'applied': False, 'warning': f'VC failed (RVC). fallback to TTS: {err or "output not created"}'}); data = tts_out.read_bytes()
                        vc_meta.update({'applied': True, 'rvc_repo': str(repo), 'rvc_custom_cmd': cmd_str})
                        data = out_wav.read_bytes()
    
                    elif vb == 'gpt_sovits_vc':
                        repo = self._gpt_sovits_vc_repo or (Path(os.environ.get('GPT_SOVITS_VC_DIR', '').strip()) if os.environ.get('GPT_SOVITS_VC_DIR','').strip() else None)
                        cmd_str = (self._gpt_sovits_vc_custom_cmd or os.environ.get('GPT_SOVITS_VC_CUSTOM_CMD', '').strip())
                        if repo is None or not Path(repo).exists():
                            return {'status': 'error', 'message': 'GPT-SoVITS(VC) の repo パスが未指定です（環境変数 GPT_SOVITS_VC_DIR か API 引数で指定してください）。'}
                        if not cmd_str:
                            vc_meta.update({'applied': False, 'warning': 'GPT_SOVITS_VC_CUSTOM_CMD が未設定のため、VC をスキップしました。'}); data = tts_out.read_bytes()
                        cmd = cmd_str.format(repo=str(repo), in_wav=str(tts_out), ref=str(ref), out=str(out_wav))
                        err = self._run_cmd(cmd, cwd=Path(repo), env=env)
                        if err is not None or not out_wav.exists():
                            return {'status': 'error', 'message': f'VC failed (GPT-SoVITS VC): {err or "output not created"}'}
                        vc_meta.update({'applied': True, 'gpt_sovits_vc_repo': str(repo), 'gpt_sovits_vc_custom_cmd': cmd_str})
                        data = out_wav.read_bytes()
                    else:
                        return {'status': 'error', 'message': f'Unknown vc_backend: {vb}'}
    
                # 後処理
                post_meta = {
            "run_logs": [],}
                if output_format.lower() == 'wav' and bool(postprocess):
                    data, post_meta = normalize_wav_volume(data, target_lufs=target_lufs)
                metrics = audio_rms_metrics(data) if output_format.lower() == 'wav' else {}
    
                meta = {
                    'request_params': {
                        'reference_audio_path': str(reference_audio_path),
                        'output_format': output_format,
                        'tts_backend': tb,
                        'vc_backend': vb,
                    },
                    'text': {
                        'raw_text': text,
                        'normalized_text': norm_meta.get('normalized_text'),
                        'prosody_text': prosody_text,
                        'prosody': prosody_meta,
                    },
                    'tts': tts_meta,
                    'vc': vc_meta,
                    'post': post_meta,
                    'metrics': metrics,
                }
                saved = artifact_store.save('audio', data, 'wav' if output_format.lower() == 'wav' else 'bin', meta)
    
                b64 = base64.b64encode(data).decode('utf-8')
                mime = 'audio/wav' if output_format.lower() == 'wav' else 'audio/mpeg'
                return {'status': 'ok', 'artifact_id': saved.get('artifact_id'), 'audio_base64': f'data:{mime};base64,' + b64}

def generate_audio_stream(
        self,
        text: str,
        reference_audio_path: str,
        gpt_sovits_repo: Optional[str] = None,
        custom_infer_cmd: Optional[str] = None,
        output_format: str = "wav",
        max_chunk_len: int = 120,
    ) -> Dict[str, Any]:
        """長文を chunk 分割して順次生成し、結合して返す。
        既存 generate_audio は単発生成として残す。
        """
        # 前処理（正規化/韻律）
        normalized_text, norm_meta = normalize_ja(text)
        prosody_text, prosody_meta = apply_prosody_rules(normalized_text, strength=1.0)

        chunks = split_text(prosody_text, max_len=int(max_chunk_len))
        if not chunks:
            return {'status': 'error', 'message': 'Empty text.'}

        wavs = []
        chunk_artifacts = []
        for i, ch in enumerate(chunks):
            r = self.generate_audio(
                text=ch,
                reference_audio_path=reference_audio_path,
                gpt_sovits_repo=gpt_sovits_repo,
                custom_infer_cmd=custom_infer_cmd,
                output_format=output_format,
            )
            if r.get('status') != 'ok':
                return {'status': 'error', 'message': f'Chunk {i} failed: ' + r.get('message', '')}
            # generate_audio は base64 を返すので、デコードして結合に使う
            import base64 as _b64
            b64 = r.get('audio_base64', '')
            if 'base64,' in b64:
                b64 = b64.split('base64,', 1)[1]
            wav_bytes = _b64.b64decode(b64)
            wavs.append(wav_bytes)
            if r.get('artifact_id'):
                chunk_artifacts.append(r.get('artifact_id'))

        merged, merge_meta = concat_wav(wavs)
        if not merged:
            return {'status': 'error', 'message': 'Failed to merge audio.'}

        # 保存
        meta = {
            'request_params': {'reference_audio_path': str(reference_audio_path), 'output_format': output_format, 'max_chunk_len': int(max_chunk_len)},
            'text': {'raw_text': text, 'normalized_text': norm_meta.get('normalized_text'), 'prosody_text': prosody_text, 'prosody': prosody_meta},
            'merge': merge_meta,
            'chunk_artifacts': chunk_artifacts,
        }
        saved = artifact_store.save('audio', merged, 'wav', meta)

        import base64 as _b64
        b64 = _b64.b64encode(merged).decode('utf-8')
        mime = 'audio/wav'
        return {'status': 'ok', 'artifact_id': saved.get('artifact_id'), 'audio_base64': f'data:{mime};base64,' + b64, 'chunks': len(chunks)}

# シングルトン
audio_engine = AudioEngine()