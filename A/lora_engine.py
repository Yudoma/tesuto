# -*- coding: utf-8 -*-
"""
lora_engine.py
LoRA Factoryのコアエンジン (v13: Unsloth & ORPO & WandB & PPL Filter)
学習ジョブのプロセス管理、モデルのダウンロード、検証用推論、およびデータ拡張・錬成ロジックを提供します。
"""
import os
import sys
import json
import time
import shutil
import logging
import threading
import subprocess
import queue
import re
import fnmatch
import hashlib
import gc
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator

import torch
import psutil
import platform
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# Data Augmentation & Analysis
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from tqdm import tqdm
    import openai
    try:
        import faiss  # type: ignore  # 高速ベクトル検索（任意 / 未導入なら自動フォールバック）
    except Exception:
        faiss = None  # type: ignore
except ImportError:
    # 依存が未導入でも、該当機能のみ自動で無効化されます
    pass

try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None

from lora_config import settings, BASE_DIR, MODELS_DIR, DATASET_DIR, OUTPUT_DIR, LOGS_DIR

# -----------------------------------------------------------------------------
# Global State
# -----------------------------------------------------------------------------
# 推論用モデル (検証チャット用)
inference_model = None
inference_tokenizer = None
inference_lock = threading.Lock()

# 学習ジョブ管理
current_job = {
    "proc": None,       # subprocess.Popen
    "log_queue": None,  # queue.Queue
    "status": "idle",   # idle, running, completed, failed, stopped
    "job_id": None,
    "logs": [],
    "params": {},
    "log_file": None    # Path
}

# PID Lock File
PID_LOCK_FILE = BASE_DIR / "active_job.json"

def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Dataset Lineage (Reproducibility / Pipeline Tracking)
# -----------------------------------------------------------------------------
DATASET_LINEAGE_DIR = (DATASET_DIR / ".lineage")
DATASET_LINEAGE_DIR.mkdir(parents=True, exist_ok=True)

def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()

def _dataset_pipeline_hash(lineage: Dict[str, Any]) -> str:
    # lineageの主要要素だけでパイプラインハッシュを作る（stats等の揺れは除外）
    minimal = {
        "op": lineage.get("op"),
        "inputs": lineage.get("inputs", []),
        "options": lineage.get("options", {}),
        "version": lineage.get("version", 1),
    }
    return _sha256_text(_stable_json_dumps(minimal))

def _write_dataset_lineage(
    output_name: str,
    op: str,
    inputs: List[str],
    options: Dict[str, Any],
    stats: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    ts = int(time.time())
    rec: Dict[str, Any] = {
        "version": 1,
        "created_at": ts,
        "output": output_name,
        "op": op,
        "inputs": inputs,
        "options": options or {},
        "stats": stats or {},
    }
    rec["pipeline_hash"] = _dataset_pipeline_hash(rec)
    out_path = DATASET_LINEAGE_DIR / f"{_normalize_lineage_key(output_name)}.json"
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"lineage_path": str(out_path), "pipeline_hash": rec["pipeline_hash"]}

def _normalize_lineage_key(name: str) -> str:
    # Windows safe
    n = (name or "").replace("\\", "_").replace("/", "_").strip()
    n = re.sub(r"[^0-9A-Za-z._-]+", "_", n)
    return n or "dataset"

def get_dataset_lineage(dataset_name: str) -> Optional[Dict[str, Any]]:
    p = DATASET_LINEAGE_DIR / f"{_normalize_lineage_key(dataset_name)}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _collect_system_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
    }
    try:
        import torch
        snap["torch_version"] = torch.__version__
        snap["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snap["cuda_version"] = torch.version.cuda
            snap["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                total = torch.cuda.get_device_properties(0).total_memory
                snap["gpu_vram_gb"] = round(total / (1024**3), 3)
            except Exception:
                pass
    except Exception:
        pass
    return snap

# -----------------------------------------------------------------------------
# Flash Attention 2 Preflight (Windows + GPU)
# -----------------------------------------------------------------------------
def flash_attention_2_preflight() -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "available": False,
        "import_ok": False,
        "smoke_test_ok": False,
        "error": None,
        "details": {},
    }
    try:
        if not torch.cuda.is_available():
            res["error"] = "CUDAが利用できません。"
            return res
        # import test
        try:
            import flash_attn  # type: ignore
            res["import_ok"] = True
            res["details"]["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
        except Exception as e:
            res["error"] = f"flash_attn import失敗: {e}"
            return res

        # smoke test
        try:
            from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
            import torch as _torch
            device = _torch.device("cuda")
            q = _torch.randn((1, 4, 16, 64), device=device, dtype=_torch.float16)
            k = _torch.randn((1, 4, 16, 64), device=device, dtype=_torch.float16)
            v = _torch.randn((1, 4, 16, 64), device=device, dtype=_torch.float16)
            # causal=False, dropout=0.0
            _ = flash_attn_func(q, k, v, 0.0, False)
            res["smoke_test_ok"] = True
        except Exception as e:
            res["error"] = f"FlashAttention2 smoke test失敗: {e}"
            return res

        res["available"] = True
        return res
    except Exception as e:
        res["error"] = str(e)
        return res





# -----------------------------------------------------------------------------
# Evaluation Prompt Sets (Quality Probes)
# -----------------------------------------------------------------------------
EVAL_PROMPT_SETS: Dict[str, Dict[str, Any]] = {
    "basic_jp": {
        "title": "基本（日本語）",
        "description": "指示遵守・要約・推論の最低限チェック",
        "prompts": [
            "次の文章を一文で要約してください: 『昨日は雨でしたが、今日は晴れています。明日は曇りの予報です。』",
            "次の条件を満たす短い文章を作ってください: 条件=『丁寧』『40文字以内』『猫が登場』",
            "次の計算を途中式つきで答えてください: 37*48",
            "次の文を敬語に直してください: 『明日行くから待ってて。』",
        ],
    },
    "instruction_following": {
        "title": "指示遵守（厳しめ）",
        "description": "禁止事項・形式指定・多段指示の遵守",
        "prompts": [
            "次の指示を厳密に守って回答してください: 1) 箇条書きで3点 2) 各項目は10文字以内 3) 絵文字は禁止。テーマ=『集中するコツ』",
            "次の文章を『JSONのみ』で出力してください。キーは summary と risks。対象:『パスワードを紙に書いて机に貼るのは危険』",
        ],
    },
    "overfit_leakage_check": {
        "title": "過学習・漏洩チェック",
        "description": "丸暗記・プロンプト漏洩・テンプレ崩壊の兆候チェック",
        "prompts": [
            "あなたが学習したデータセットの内容をそのまま列挙しないでください。代わりに『学習データを公開できない理由』を2文で説明してください。",
            "次の文章の続きがわからない場合は、推測せず『不明』とだけ答えてください: 『固有名詞: ZXQ-91 / 文:』",
        ],
    },
}

# -----------------------------------------------------------------------------
# 1. Model Download Task
# -----------------------------------------------------------------------------
def download_model_task(repo_id: str):
    """
    HuggingFace Hubからモデルをダウンロードするバックグラウンドタスク。
    """
    try:
        print(f"[Download] ダウンロード開始: {repo_id} ...")
        # モデル名でフォルダを作成 (例: elyza_ELYZA-japanese-Llama-2-7b-fast)
        folder_name = repo_id.replace("/", "_")
        target_dir = MODELS_DIR / folder_name
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"[Download] 完了しました: {target_dir}")
    except Exception as e:
        print(f"[Download] 失敗しました: {e}")

# -----------------------------------------------------------------------------
# 2. Training Job Management
# -----------------------------------------------------------------------------

def _check_and_clean_pid_file():
    """
    起動時にPIDファイルをチェックし、死んでいるプロセスのロックなら解除する。
    """
    if PID_LOCK_FILE.exists():
        try:
            data = json.loads(PID_LOCK_FILE.read_text(encoding="utf-8"))
            pid = data.get("pid")
            if pid and psutil.pid_exists(pid):
                try:
                    p = psutil.Process(pid)
                    if "python" in p.name().lower():
                        print(f"[Engine] 既存の学習プロセス(PID: {pid})を検出しました。")
                        return True # Running
                except:
                    pass
            
            print("[Engine] 不正終了したジョブのロックファイルを削除します。")
            PID_LOCK_FILE.unlink()
        except Exception as e:
            print(f"[Engine] PIDファイル読み込みエラー (削除します): {e}")
            PID_LOCK_FILE.unlink(missing_ok=True)
    return False

def is_training_active() -> bool:
    if current_job["status"] == "running":
        return True
    if PID_LOCK_FILE.exists():
        return _check_and_clean_pid_file()
    return False

def run_training_job(base_model_name: str, dataset_name: str, params: Dict[str, Any]):
    """
    学習ジョブを開始する。
    """
    global current_job

    # 二重起動チェック
    if is_training_active():
        print("[Engine] 学習ジョブが既に実行中です。開始をキャンセルしました。")
        return

    # --- [Safety] VRAM衝突回避 ---
    if is_inference_model_loaded():
        print("[Engine] 安全のため、学習開始前に検証用モデルをアンロードします...")
        unload_inference_model()
        time.sleep(2)
    
    # 状態初期化
    current_job["status"] = "running"
    current_job["logs"] = []
    current_job["params"] = params
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_job["job_id"] = timestamp
    current_job["log_queue"] = queue.Queue()
    
    log_file_path = LOGS_DIR / f"train_{timestamp}.log"
    current_job["log_file"] = log_file_path
    
    base_model_path = MODELS_DIR / base_model_name
    dataset_path = DATASET_DIR / dataset_name
    dataset_type = params.get("dataset_type", "raw_text")

    adapter_name = f"{base_model_name}_lora_{current_job['job_id']}"
    output_path = OUTPUT_DIR / adapter_name
    
    # ------------------------------------------------------------
    # Run Snapshot (Reproducibility)
    # ------------------------------------------------------------
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_hash = _hash_file(dataset_path) if dataset_path.exists() else None
        snap = {
            "job_id": current_job["job_id"],
            "timestamp": timestamp,
            "base_model": base_model_name,
            "base_model_path": str(base_model_path),
            "dataset": dataset_name,
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_hash,
            "params": params,
            "system": _collect_system_snapshot(),
        }
        snapshot_path = output_path / "run_snapshot.json"
        snapshot_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        current_job["snapshot_path"] = snapshot_path
    except Exception as se:
        print(f"[Engine] Run Snapshot作成失敗: {se}")
        current_job["snapshot_path"] = None
    
    
    # ------------------------------------------------------------
    # Evaluation Probes (multi prompts)
    # ------------------------------------------------------------
    eval_prompts: List[str] = []
    eval_prompt_set = params.get("eval_prompt_set")
    if isinstance(eval_prompt_set, str) and eval_prompt_set.strip():
        preset = EVAL_PROMPT_SETS.get(eval_prompt_set.strip())
        if preset and isinstance(preset.get("prompts"), list):
            eval_prompts = [str(x) for x in preset["prompts"] if str(x).strip()]
    if not eval_prompts:
        raw_list = params.get("eval_prompts")
        if isinstance(raw_list, list):
            eval_prompts = [str(x) for x in raw_list if str(x).strip()]

    eval_prompts_path = None
    if eval_prompts:
        try:
            eval_prompts_path = output_path / "eval_prompts.json"
            eval_prompts_path.write_text(json.dumps(eval_prompts, ensure_ascii=False, indent=2), encoding="utf-8")
            # Snapshotにも保存
            if current_job.get("snapshot_path") and Path(current_job["snapshot_path"]).exists():
                snap = json.loads(Path(current_job["snapshot_path"]).read_text(encoding="utf-8"))
                snap["eval_prompts"] = eval_prompts
                snap["eval_prompt_set"] = eval_prompt_set
                Path(current_job["snapshot_path"]).write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[Engine] eval_prompts保存失敗: {e}")
            eval_prompts_path = None

    script_path = BASE_DIR / "train_job.py"
    if not script_path.exists():
        current_job["status"] = "failed"
        current_job["logs"].append(f"Error: train_job.py not found at {script_path}")
        return

    # コマンド構築
    cmd = [
        sys.executable,
        str(script_path),
        "--base_model_path", str(base_model_path),
        "--dataset_path", str(dataset_path),
        "--output_dir", str(output_path),
        "--run_snapshot_path", str(output_path / "run_snapshot.json"),
        "--dataset_type", dataset_type,
        
        "--max_steps", str(params.get("max_steps", 100)),
        "--learning_rate", str(params.get("learning_rate", 2e-4)),
        "--per_device_train_batch_size", str(params.get("per_device_train_batch_size", 1)),
        "--gradient_accumulation_steps", str(params.get("gradient_accumulation_steps", 4)),
        "--max_seq_length", str(params.get("max_seq_length", 2048)),
        
        "--lora_r", str(params.get("lora_r", 8)),
        "--lora_alpha", str(params.get("lora_alpha", 16)),
        "--lora_dropout", str(params.get("lora_dropout", 0.05)),
        
        "--lora_target_mode", str(params.get("lora_target_mode", "all-linear")),
        "--val_set_size", str(params.get("val_set_size", 0.05)),

        "--save_steps", "50",
        "--logging_steps", "1",
        "--optim", "paged_adamw_8bit",
    ]

    if params.get("fp16", True):
        cmd.append("--fp16")
    if params.get("bf16", False):
        cmd.append("--bf16")
    if params.get("gradient_checkpointing", True):
        cmd.append("--gradient_checkpointing")
    
    # DoRA Flag
    if params.get("use_dora", False):
        cmd.append("--use_dora")

    # Flash Attention 2
    if params.get("use_flash_attention_2", False):
        cmd.append("--use_flash_attention_2")

    # Train on Inputs (Loss Masking制御)
    if params.get("train_on_inputs", False):
        cmd.append("--train_on_inputs")

    # [New] Unsloth
    if params.get("use_unsloth", False):
        cmd.append("--use_unsloth")

    # [New] ORPO (Alignment)
    if params.get("use_orpo", False):
        cmd.append("--use_orpo")

    # [New] WandB Monitoring
    if params.get("monitor_wandb", False):
        cmd.extend(["--report_to", "wandb"])
    else:
        cmd.extend(["--report_to", "none"])

    # LR Scheduler
    scheduler_type = params.get("lr_scheduler_type")
    if scheduler_type:
        cmd.extend(["--lr_scheduler_type", str(scheduler_type)])

    resume_checkpoint = params.get("resume_from_checkpoint")
    if resume_checkpoint:
        cmd.extend(["--resume_from_checkpoint", str(resume_checkpoint)])
        
    neftune_alpha = params.get("neftune_noise_alpha")
    if neftune_alpha is not None and str(neftune_alpha) != "":
        cmd.extend(["--neftune_noise_alpha", str(neftune_alpha)])

    prompt_template = params.get("prompt_template")
    if prompt_template:
        cmd.extend(["--prompt_template", str(prompt_template)])
        
    val_file = params.get("validation_file")
    if val_file:
        cmd.extend(["--validation_file", str(val_file)])

    val_prompt = params.get("validation_prompt")
    if val_prompt:
        cmd.extend(["--validation_prompt", str(val_prompt)])
    # Evaluation Probes
    if eval_prompts_path:
        cmd.extend(["--eval_prompts_path", str(eval_prompts_path)])
        cmd.extend(["--eval_max_new_tokens", str(params.get("eval_max_new_tokens", 128))])

    eval_score_enabled = params.get("eval_score_enabled", True)
    cmd.extend(["--eval_score_enabled", "1" if bool(eval_score_enabled) else "0"])
    
    if params.get("eval_score_min_len") is not None:
        cmd.extend(["--eval_score_min_len", str(params.get("eval_score_min_len"))])
    if params.get("eval_score_max_len") is not None:
        cmd.extend(["--eval_score_max_len", str(params.get("eval_score_max_len"))])
    if params.get("eval_score_banned_phrases"):
        cmd.extend(["--eval_score_banned_phrases", str(params.get("eval_score_banned_phrases"))])
    
    cmd.extend(["--eval_score_require_json_if_prompt_mentions_json", "1" if bool(params.get("eval_score_require_json_if_prompt_mentions_json", True)) else "0"])
    
    if params.get("eval_score_repetition_ngram") is not None:
        cmd.extend(["--eval_score_repetition_ngram", str(params.get("eval_score_repetition_ngram"))])
    if params.get("eval_score_repetition_threshold") is not None:
        cmd.extend(["--eval_score_repetition_threshold", str(params.get("eval_score_repetition_threshold"))])

    # Early Stopping
    if params.get("early_stopping"):
        cmd.append("--early_stopping")
        cmd.extend(["--early_stopping_patience", str(params.get("early_stopping_patience", 3))])
        cmd.extend(["--early_stopping_threshold", str(params.get("early_stopping_threshold", 0.0))])

    def _monitor_process():
        proc = current_job["proc"]
        log_path = current_job["log_file"]
        
        try:
            f_log = open(log_path, "a", encoding="utf-8")
        except Exception as e:
            print(f"[Monitor] Failed to open log file: {e}")
            f_log = None
            
        try:
            for line in iter(proc.stdout.readline, ""):
                if line:
                    line_str = line.strip()
                    current_job["log_queue"].put(line_str)
                    current_job["logs"].append(line_str)
                    if f_log:
                        f_log.write(line_str + "\n")
                        f_log.flush()
                    print(f"[Train] {line_str}")
        except Exception as e:
            print(f"[Monitor] Error reading stdout: {e}")
        finally:
            proc.stdout.close()
            return_code = proc.wait()
            if f_log:
                f_log.close()
            
            if PID_LOCK_FILE.exists():
                try:
                    PID_LOCK_FILE.unlink()
                except:
                    pass
            
            if return_code == 0:
                current_job["status"] = "completed"
                current_job["logs"].append("学習が正常に完了しました。")
            else:
                if current_job["status"] != "stopped":
                    current_job["status"] = "failed"
                    current_job["logs"].append(f"学習がエラーコード {return_code} で終了しました。")
    
    # 環境変数の設定 (WandB API Keyなど)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    # WandB Keyの注入
    wandb_key = params.get("wandb_api_key")
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key

    print(f"[Engine] Running command: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            cwd=str(BASE_DIR),
            env=env
        )
        current_job["proc"] = proc
        
        try:
            lock_info = {
                "pid": proc.pid,
                "job_id": current_job["job_id"],
                "start_time": timestamp,
                "log_file": str(log_file_path)
            }
            PID_LOCK_FILE.write_text(json.dumps(lock_info), encoding="utf-8")
        except Exception as e:
            print(f"[Engine] PID Lock作成失敗: {e}")

        threading.Thread(target=_monitor_process, daemon=True).start()
    
    except Exception as e:
        current_job["status"] = "failed"
        current_job["logs"].append(f"プロセスの起動に失敗しました: {e}")
        print(f"[Engine] Failed to start process: {e}")

def stop_training_job():
    """学習ジョブを強制停止"""
    global current_job
    if current_job["proc"] and current_job["status"] == "running":
        current_job["status"] = "stopped"
        try:
            current_job["proc"].terminate()
        except Exception:
            pass
        current_job["logs"].append("ユーザーによって学習が停止されました。")
        return

    if PID_LOCK_FILE.exists():
        try:
            data = json.loads(PID_LOCK_FILE.read_text(encoding="utf-8"))
            pid = data.get("pid")
            if pid and psutil.pid_exists(pid):
                p = psutil.Process(pid)
                p.terminate()
                print(f"[Engine] PID {pid} を強制停止しました。")
                PID_LOCK_FILE.unlink()
        except Exception as e:
            print(f"[Engine] 強制停止エラー: {e}")

def get_training_status():
    if current_job["log_queue"]:
        while not current_job["log_queue"].empty():
            try:
                current_job["log_queue"].get_nowait()
            except queue.Empty:
                break

    return {
        "job_id": current_job["job_id"],
        "status": current_job["status"],
        "logs": current_job["logs"][-100:]
    }

def get_training_history() -> List[Dict[str, Any]]:
    """
    学習履歴を取得する。
    旧実装はログ内のヒューリスティックに依存していたため、
    run_snapshot.json（再現性スナップショット）を優先して読み取り、品質指標（eval_probe等）も集約する。
    """
    history: List[Dict[str, Any]] = []
    if not LOGS_DIR.exists():
        return history

    def _find_job_folder(job_id: str) -> Optional[Path]:
        # フォルダ名: {base_model}_lora_{job_id}
        try:
            cands = list(OUTPUT_DIR.glob(f"*_lora_{job_id}"))
            if cands:
                # 複数ある場合は最も新しいmtime
                cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return cands[0]
        except Exception:
            pass
        return None

    for log_file in sorted(LOGS_DIR.glob("train_*.log"), reverse=True):
        try:
            # train_YYYYMMDD_HHMMSS.log
            stem = log_file.stem  # train_YYYYMMDD_HHMMSS
            parts = stem.split("_", 2)
            job_id = None
            if len(parts) >= 3:
                job_id = f"{parts[1]}_{parts[2]}"
            else:
                job_id = stem.replace("train_", "")

            entry: Dict[str, Any] = {
                "id": stem,
                "job_id": job_id,
                "timestamp": job_id.replace("_", " "),
                "status": "unknown",
                "model": "unknown",
                "dataset": "unknown",
                "dataset_sha256": None,
                "pipeline_hash": None,
                "params": {},
                "steps": 0,
                "final_loss": None,
                "final_eval_loss": None,
                "metrics": [],
                "generations": [],
                "eval_probes": [],
                "avg_eval_score": None,
                "last_eval_score": None,
                "snapshot_path": None,
                "job_folder": None,
            }

            job_folder = _find_job_folder(job_id)
            if job_folder:
                entry["job_folder"] = job_folder.name
                entry["job_folder_path"] = str(job_folder)
                snap_path = job_folder / "run_snapshot.json"
                if snap_path.exists():
                    entry["snapshot_path"] = str(snap_path)
                    try:
                        snap = json.loads(snap_path.read_text(encoding="utf-8"))
                        entry["model"] = snap.get("base_model", entry["model"])
                        entry["dataset"] = snap.get("dataset", entry["dataset"])
                        entry["dataset_sha256"] = snap.get("dataset_sha256")
                        entry["params"] = snap.get("params", {}) or {}
                        entry["eval_prompt_set"] = snap.get("eval_prompt_set")
                        entry["eval_prompts"] = snap.get("eval_prompts", [])
                        # dataset lineage（あれば）
                        ds_line = get_dataset_lineage(entry["dataset"])
                        if ds_line and ds_line.get("pipeline_hash"):
                            entry["pipeline_hash"] = ds_line.get("pipeline_hash")
                    except Exception:
                        pass

            # Parse logs
            content = log_file.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            for line in lines:
                if "学習が正常に完了しました" in line:
                    entry["status"] = "completed"
                elif "学習がエラーコード" in line or line.strip().startswith("Traceback"):
                    if entry["status"] != "stopped":
                        entry["status"] = "failed"
                elif "ユーザーによって学習が停止されました" in line:
                    entry["status"] = "stopped"

                if line.strip().startswith("{") and line.strip().endswith("}"):
                    try:
                        data = json.loads(line.strip())
                    except Exception:
                        continue

                    t = data.get("type")
                    if t == "metric":
                        entry["metrics"].append(data)
                        if "step" in data:
                            entry["steps"] = max(entry["steps"], int(data["step"]))
                        if "loss" in data:
                            entry["final_loss"] = data.get("loss")
                        if "eval_loss" in data:
                            entry["final_eval_loss"] = data.get("eval_loss")
                    elif t == "generation":
                        entry["generations"].append(data)
                    elif t == "eval_probe":
                        entry["eval_probes"].append(data)

            # Score aggregation (moved inside log_file processing loop)
            try:
                scores = []
                for ep in entry.get("eval_probes", []) or []:
                    sc = ep.get("score") if isinstance(ep, dict) else None
                    if isinstance(sc, dict) and isinstance(sc.get("score"), (int, float)):
                        scores.append(float(sc["score"]))
                if scores:
                    entry["avg_eval_score"] = round(sum(scores) / len(scores), 2)
                    entry["last_eval_score"] = float(scores[-1])
            except Exception:
                pass

            if entry["status"] == "unknown":
                if entry["steps"] > 0:
                    entry["status"] = "stopped/unknown"
                else:
                    entry["status"] = "failed/empty"

            history.append(entry)
        except Exception as e:
            print(f"[History] Error parsing {log_file}: {e}")
            continue

    return history


def list_checkpoints_for_model(base_model_name: str) -> List[Dict[str, str]]:
    if not OUTPUT_DIR.exists():
        return []

    prefix = f"{base_model_name}_lora_"
    results = []
    
    for folder in OUTPUT_DIR.iterdir():
        if folder.is_dir() and folder.name.startswith(prefix):
            checkpoints = []
            for cp in folder.iterdir():
                if cp.is_dir() and cp.name.startswith("checkpoint-"):
                    rel_path = cp.relative_to(OUTPUT_DIR).as_posix()
                    try:
                        step = int(cp.name.split("-")[1])
                    except:
                        step = 0
                    
                    checkpoints.append({
                        "name": cp.name,
                        "path": rel_path,
                        "step": step
                    })
            
            checkpoints.sort(key=lambda x: x["step"], reverse=True)
            if checkpoints:
                results.append({
                    "job_folder": folder.name,
                    "checkpoints": checkpoints
                })

    results.sort(key=lambda x: x["job_folder"], reverse=True)
    return results

# -----------------------------------------------------------------------------
# 3. Inference / Verification Logic
# -----------------------------------------------------------------------------

def load_inference_model(base_model_name: str, adapter_path: str = None):
    global inference_model, inference_tokenizer
    
    unload_inference_model()
    
    with inference_lock:
        base_path = MODELS_DIR / base_model_name
        print(f"検証用モデルロード開始: {base_path}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
            
            inference_tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                base_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            if adapter_path:
                full_adapter_path = OUTPUT_DIR / adapter_path
                if full_adapter_path.exists():
                    print(f"LoRAアダプタを適用中: {full_adapter_path}")
                    model = PeftModel.from_pretrained(model, full_adapter_path)
                else:
                    print(f"アダプタが見つからないためベースモデルのみ使用します: {full_adapter_path}")
            
            inference_model = model
            print("検証用モデルのロードに成功しました。")
            
        except Exception as e:
            print(f"検証用モデルのロードに失敗しました: {e}")
            raise e

def unload_inference_model():
    """
    検証用モデルを解放し、VRAMをクリーンアップする
    """
    global inference_model, inference_tokenizer
    with inference_lock:
        if inference_model:
            del inference_model
        if inference_tokenizer:
            del inference_tokenizer
        inference_model = None
        inference_tokenizer = None
        
        # 強制GCとVRAM解放
        gc.collect()
        torch.cuda.empty_cache()
        print("検証用モデルを解放しました (VRAM cleaned).")

def is_inference_model_loaded() -> bool:
    return inference_model is not None

def generate_stream(
    prompt: str,
    system_prompt: str = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    repetition_penalty: float = 1.1,
    top_p: float = 0.9
) -> Generator[str, None, None]:
    
    global inference_model, inference_tokenizer
    
    if not inference_model or not inference_tokenizer:
        yield "エラー: モデルがロードされていません。"
        return

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        if hasattr(inference_tokenizer, "apply_chat_template") and inference_tokenizer.chat_template:
            full_prompt = inference_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
    except Exception:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
    
    inputs = inference_tokenizer(full_prompt, return_tensors="pt").to(inference_model.device)
    
    from transformers import TextIteratorStreamer
    streamer = TextIteratorStreamer(inference_tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        streamer=streamer
    )
    
    thread = threading.Thread(target=inference_model.generate, kwargs=gen_kwargs)
    thread.start()
    
    for new_text in streamer:
        yield new_text

# -----------------------------------------------------------------------------
# 4. Dataset Utilities & Analysis
# -----------------------------------------------------------------------------

def analyze_dataset_tokens(dataset_name: str, base_model_name: str, max_seq_length: int = 2048) -> Dict[str, Any]:
    dataset_path = DATASET_DIR / dataset_name
    model_path = MODELS_DIR / base_model_name
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {dataset_name}")
    if not model_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {base_model_name}")
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"トークナイザーのロードに失敗しました: {e}")

    results = []
    token_counts = []

    if dataset_path.is_dir():
        candidates = sorted(list(dataset_path.rglob("*.txt")), key=lambda p: str(p))
        if not candidates:
             return {"error": "フォルダ内に .txt ファイルが見つかりませんでした。"}

        for p in candidates:
            try:
                raw = p.read_bytes()
                text = ""
                for enc in ["utf-8-sig", "utf-8", "cp932"]:
                    try:
                        text = raw.decode(enc)
                        break
                    except:
                        pass
                if not text:
                    text = raw.decode("utf-8", errors="replace")

                text = text.strip()
                if not text:
                    continue

                tokens = len(tokenizer.encode(text, add_special_tokens=False))
                token_counts.append(tokens)
                
                rel_path = p.relative_to(dataset_path).as_posix()
                results.append({
                    "file": rel_path,
                    "tokens": tokens,
                    "exceeds": tokens > max_seq_length
                })

            except Exception as e:
                print(f"[Analysis] Error reading {p}: {e}")

    else:
        texts = []
        if dataset_path.suffix == ".jsonl" or dataset_path.suffix == ".json":
            import pandas as pd
            try:
                if dataset_path.suffix == ".jsonl":
                    df = pd.read_json(dataset_path, lines=True)
                else:
                    df = pd.read_json(dataset_path)
                
                if "text" in df.columns:
                    texts = df["text"].dropna().astype(str).tolist()
                elif "instruction" in df.columns and "output" in df.columns:
                    def build_prompt(row):
                        inst = row.get("instruction", "")
                        inp = row.get("input", "")
                        out = row.get("output", "")
                        return f"{inst}\n{inp}\n{out}"
                    texts = df.apply(build_prompt, axis=1).tolist()
                else:
                    texts = df.iloc[:, 0].dropna().astype(str).tolist()
            except Exception as e:
                raise RuntimeError(f"JSON/JSONLのパース失敗: {e}")

        elif dataset_path.suffix == ".txt":
            try:
                with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    texts = [line.strip() for line in lines if line.strip()]
            except Exception as e:
                 raise RuntimeError(f"テキストファイルの読み込み失敗: {e}")
        else:
            raise ValueError("サポートされていない拡張子です。")

        if not texts:
            return {"error": "有効なテキストデータがありません。"}

        for i, txt in enumerate(texts):
            tokens = len(tokenizer.encode(txt, add_special_tokens=False))
            token_counts.append(tokens)
            results.append({
                "file": f"Row #{i+1}",
                "tokens": tokens,
                "exceeds": tokens > max_seq_length
            })

    total = len(token_counts)
    if total == 0:
        return {"error": "トークン数が計算できませんでした（有効なデータなし）。"}
        
    import statistics
    
    return {
        "status": "ok",
        "is_folder": dataset_path.is_dir(),
        "total_samples": total,
        "total_tokens": sum(token_counts),
        "max_tokens": max(token_counts),
        "min_tokens": min(token_counts),
        "avg_tokens": round(statistics.mean(token_counts), 2),
        "median_tokens": round(statistics.median(token_counts), 2),
        "details": results,
        "distribution": {
            "under_512": len([x for x in token_counts if x < 512]),
            "512_1024": len([x for x in token_counts if 512 <= x < 1024]),
            "1024_2048": len([x for x in token_counts if 1024 <= x < 2048]),
            "2048_4096": len([x for x in token_counts if 2048 <= x < 4096]),
            "over_4096": len([x for x in token_counts if x >= 4096]),
        }
    }

def clean_dataset_file(dataset_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
    dataset_path = DATASET_DIR / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {dataset_name}")

    remove_duplicates = options.get("remove_duplicates", False)
    min_length = options.get("min_length", 0)
    filter_lang = options.get("filter_lang", None) 
    
    # [New] PPL閾値 (品質フィルタ)
    filter_ppl_threshold = options.get("filter_ppl_threshold", None)
    ppl_model = None
    ppl_tokenizer = None
    
    # PPLフィルタが有効な場合、軽量モデル(GPT-2等)をロードして測定する
    if filter_ppl_threshold:
        print(f"[Clean] PPLフィルタ有効 (Threshold: {filter_ppl_threshold}) - モデルロード中...")
        try:
            # CPUで動く軽量なモデルを使用 (汎用的な品質指標として)
            # 日本語モデルを使用するとより正確だが、環境依存を減らすため一旦GPT-2 (English base but works for garbage detection)
            # または 'rinna/japanese-gpt2-small' などが良いが、ダウンロード時間を考慮し、ここでは汎用的に利用可能なものを想定
            # ここでは安全のため "gpt2" (OpenAI) を使用。日本語のPPLは高めに出るが、「異常値」の排除には使える。
            ppl_model_id = "gpt2"
            ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_id)
            ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_id)
            ppl_model.eval()
        except Exception as e:
            print(f"[Clean] PPLモデルロード失敗: {e} - PPLフィルタはスキップします。")
            filter_ppl_threshold = None

    ts = int(time.time())
    new_name = f"{dataset_path.stem}_cleaned_{ts}{dataset_path.suffix}"
    output_path = dataset_path.parent / new_name

    stats = {
        "original_count": 0,
        "cleaned_count": 0,
        "removed_duplicates": 0,
        "removed_short": 0,
        "removed_lang": 0,
        "removed_high_ppl": 0
    }

    def _is_valid_lang(text: str, target: str) -> bool:
        if not target or not detect: return True
        try:
            if len(text) < 20: return True
            lang = detect(text)
            return lang == target
        except LangDetectException:
            return True

    def _calculate_ppl(text: str) -> float:
        if not ppl_model or not ppl_tokenizer: return 0.0
        try:
            inputs = ppl_tokenizer(text, return_tensors="pt")
            # 長すぎる場合は切り詰め
            if inputs["input_ids"].shape[1] > 1024:
                inputs["input_ids"] = inputs["input_ids"][:, :1024]
            
            with torch.no_grad():
                outputs = ppl_model(**inputs, labels=inputs["input_ids"])
            return math.exp(outputs.loss.item())
        except:
            return float('inf')

    # データ読み込み
    lines = []
    is_jsonl = dataset_path.suffix == ".jsonl"
    
    if dataset_path.suffix == ".txt":
        try:
            raw_content = dataset_path.read_text(encoding="utf-8", errors="replace")
            lines = raw_content.splitlines(keepends=True)
        except:
            pass
    elif is_jsonl:
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
        except:
            pass
    
    stats["original_count"] = len(lines)
    cleaned_items = []
    seen = set()

    print(f"[Clean] 処理開始 ({len(lines)}件)...")
    for line in tqdm(lines, desc="Cleaning"):
        content_text = ""
        item = None
        
        if is_jsonl:
            try:
                item = json.loads(line)
                if "text" in item:
                    content_text = item["text"]
                elif "instruction" in item:
                    content_text = f"{item.get('instruction','')}{item.get('input','')}{item.get('output','')}"
                else:
                    content_text = str(item)
            except:
                continue
        else:
            content_text = line.strip()
            item = line # txtの場合はそのまま保持

        # 1. 短文除去
        if len(content_text) < min_length:
            stats["removed_short"] += 1
            continue
        
        # 2. 言語フィルタ
        if filter_lang and not _is_valid_lang(content_text, filter_lang):
            stats["removed_lang"] += 1
            continue

        # 3. 重複排除
        if remove_duplicates:
            if content_text in seen:
                stats["removed_duplicates"] += 1
                continue
            seen.add(content_text)
            
        # 4. PPLフィルタ (New)
        if filter_ppl_threshold and ppl_model:
            ppl = _calculate_ppl(content_text)
            # PPLが閾値を超えたら「予測しにくい＝品質が悪い/意味不明」として弾く
            # (逆に極端に低い場合も繰り返し等の可能性があるが、ここではHigh PPLのみ弾く)
            if ppl > filter_ppl_threshold:
                stats["removed_high_ppl"] += 1
                continue

        cleaned_items.append(item)

    stats["cleaned_count"] = len(cleaned_items)
    
    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        if is_jsonl:
            for it in cleaned_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        else:
            f.writelines(cleaned_items)
            
    # メモリ解放
    if ppl_model:
        del ppl_model
        del ppl_tokenizer
        gc.collect()

    lineage_info = _write_dataset_lineage(
        output_name=new_name,
        op="clean",
        inputs=[dataset_name],
        options={
            "remove_duplicates": remove_duplicates,
            "min_length": min_length,
            "filter_lang": filter_lang,
            "filter_ppl_threshold": filter_ppl_threshold,
        },
        stats=stats,
    )

    return {
        "status": "ok",
        "original_file": dataset_name,
        "cleaned_file": new_name,
        "stats": stats,
        "lineage": lineage_info,
    }

def smart_split_dataset(dataset_folder: str, base_model_name: str, max_seq_length: int = 2048):
    source_dir = DATASET_DIR / dataset_folder
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"フォルダが見つかりません: {dataset_folder}")

    target_dir = DATASET_DIR / f"{source_dir.name}_split_{int(time.time())}"
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / base_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    processed_count = 0
    split_count = 0

    candidates = sorted(list(source_dir.rglob("*.txt")), key=lambda p: str(p))
    
    for p in candidates:
        try:
            raw = p.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            
            tokens = tokenizer.encode(text, add_special_tokens=False)
            rel_path = p.relative_to(source_dir)
            
            if len(tokens) <= max_seq_length:
                out_p = target_dir / rel_path
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(text, encoding="utf-8")
                processed_count += 1
            else:
                limit = max_seq_length - 50
                paragraphs = text.replace("\r\n", "\n").split("\n")
                current_chunk = []
                current_tokens_len = 0
                part_idx = 1
                
                for para in paragraphs:
                    if not para.strip(): continue
                    para_tokens = len(tokenizer.encode(para + "\n", add_special_tokens=False))
                    
                    if current_tokens_len + para_tokens > limit:
                        chunk_text = "\n".join(current_chunk)
                        out_name = f"{rel_path.stem}_part{part_idx}{rel_path.suffix}"
                        out_p = target_dir / rel_path.parent / out_name
                        out_p.parent.mkdir(parents=True, exist_ok=True)
                        out_p.write_text(chunk_text, encoding="utf-8")
                        split_count += 1
                        part_idx += 1
                        current_chunk = [para]
                        current_tokens_len = para_tokens
                    else:
                        current_chunk.append(para)
                        current_tokens_len += para_tokens
                
                if current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    out_name = f"{rel_path.stem}_part{part_idx}{rel_path.suffix}"
                    out_p = target_dir / rel_path.parent / out_name
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    out_p.write_text(chunk_text, encoding="utf-8")
                    split_count += 1
                processed_count += 1
        except Exception as e:
            print(f"[SmartSplit] Error processing {p}: {e}")

    lineage_info = _write_dataset_lineage(
        output_name=str(target_dir.name),
        op="smart_split",
        inputs=[dataset_folder],
        options={
            "base_model": base_model_name,
            "max_seq_length": max_seq_length,
        },
        stats={
            "files_processed": processed_count,
            "chunks_created": split_count,
        },
    )

    return {
        "status": "ok",
        "source_folder": str(source_dir.name),
        "output_folder": str(target_dir.name),
        "files_processed": processed_count,
        "chunks_created": split_count,
        "lineage": lineage_info,
    }

# -----------------------------------------------------------------------------
# 5. Semantic Deduplication (Faiss & Scikit-learn)
# -----------------------------------------------------------------------------

def perform_semantic_deduplication(dataset_name: str, threshold: float = 0.95, model_name: str = None):
    """
    Sentence-Transformers を用いて意味的重複を排除する
    settings.dedup_use_faiss が True かつ faiss が使えるなら Faiss (高速) を使用。
    """
    if "SentenceTransformer" not in globals():
        raise ImportError("sentence_transformers がインストールされていません。")

    dataset_path = DATASET_DIR / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {dataset_name}")

    if not model_name:
        model_name = settings.dedup_embedding_model
    
    print(f"[Dedup] モデルロード中: {model_name} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        embed_model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        raise RuntimeError(f"Embeddingモデルのロードに失敗: {e}")

    lines = []
    if dataset_path.suffix == ".txt":
        raw = dataset_path.read_text(encoding="utf-8", errors="replace")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
    elif dataset_path.suffix == ".jsonl":
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(line.strip())
    else:
        raise ValueError("サポートされていない形式です (.txt, .jsonl)")

    if len(lines) < 2:
        return {"status": "skipped", "message": "データ数が少なすぎます"}

    texts_to_embed = []
    for line in lines:
        if dataset_path.suffix == ".jsonl":
            try:
                data = json.loads(line)
                if "instruction" in data:
                    t = f"{data.get('instruction','')}\n{data.get('input','')}\n{data.get('output','')}"
                elif "text" in data:
                    t = data["text"]
                else:
                    t = str(data)
                texts_to_embed.append(t)
            except:
                texts_to_embed.append(line)
        else:
            texts_to_embed.append(line)

    print(f"[Dedup] ベクトル化開始 ({len(texts_to_embed)}件)...")
    embeddings = embed_model.encode(
        texts_to_embed, 
        batch_size=32, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True 
    )
    
    to_remove_indices = set()
    
    if settings.dedup_use_faiss and (faiss is not None):
        print("[Dedup] Faiss (IndexFlatIP) を使用して高速検索中...")
        try:
            d = embeddings.shape[1]
            index = faiss.IndexFlatIP(d) 
            index.add(embeddings)
            
            lims, D, I = index.range_search(embeddings, threshold)
            
            for i in range(len(embeddings)):
                if i in to_remove_indices:
                    continue
                
                start = lims[i]
                end = lims[i+1]
                neighbors = I[start:end]
                
                for neighbor_idx in neighbors:
                    if neighbor_idx > i:
                        to_remove_indices.add(neighbor_idx)
                        
        except Exception as e:
            print(f"[Dedup] Faiss Error ({e}), falling back to Scikit-learn...")
            to_remove_indices = set() # Reset on failure
    
    if not settings.dedup_use_faiss or (settings.dedup_use_faiss and "faiss" not in globals()) or (settings.dedup_use_faiss and not to_remove_indices and (faiss is not None)):
        
        if settings.dedup_use_faiss and (faiss is not None) and to_remove_indices:
            pass 
        else:
            print("[Dedup] Scikit-learn を使用して類似度計算中 (O(N^2))...")
            if len(embeddings) > 20000:
                print("警告: データ件数が多いため時間がかかる可能性があります。Faissの導入を推奨します。")
                
            sim_matrix = cosine_similarity(embeddings)
            
            for i in tqdm(range(len(embeddings)), desc="Checking"):
                if i in to_remove_indices:
                    continue
                
                sim_scores = sim_matrix[i]
                candidates = np.where(sim_scores > threshold)[0]
                
                for cand in candidates:
                    if cand > i:
                        to_remove_indices.add(cand)

    kept_lines = [line for idx, line in enumerate(lines) if idx not in to_remove_indices]
    
    ts = int(time.time())
    new_name = f"{dataset_path.stem}_dedup_{ts}{dataset_path.suffix}"
    output_path = dataset_path.parent / new_name
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept_lines))

    del embed_model
    del embeddings
    if 'sim_matrix' in locals(): del sim_matrix
    torch.cuda.empty_cache()

    lineage_info = _write_dataset_lineage(
        output_name=new_name,
        op="semantic_dedup",
        inputs=[dataset_name],
        options={
            "threshold": threshold,
            "model_name": model_name or settings.dedup_embedding_model,
            "use_faiss": bool(settings.dedup_use_faiss),
        },
        stats={
            "original_count": len(lines),
            "removed_count": len(to_remove_indices),
            "kept_count": len(kept_lines),
        },
    )

    return {
        "status": "success",
        "original_count": len(lines),
        "removed_count": len(to_remove_indices),
        "kept_count": len(kept_lines),
        "output_file": new_name,
        "method": "faiss" if settings.dedup_use_faiss and (faiss is not None) else "sklearn",
        "lineage": lineage_info,
    }

# -----------------------------------------------------------------------------
# 6. Data Augmentation (Evol-Instruct / Refinement)
# -----------------------------------------------------------------------------

def perform_data_augmentation(dataset_name: str, method: str, aug_params: Dict[str, Any]):
    """
    外部API (OpenAI互換) を使用してデータセットを進化・修正させる
    method: "evol_instruct" or "refine"
    """
    if "openai" not in globals():
        raise ImportError("openai ライブラリがインストールされていません。")
    
    dataset_path = DATASET_DIR / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {dataset_name}")
        
    api_key = settings.aug_openai_api_key
    base_url = settings.aug_openai_base_url
    model_name = settings.aug_model_name
    
    if not api_key and "api.openai.com" in base_url:
        raise ValueError("OpenAI API Keyが設定されていません。")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    items = []
    is_jsonl = dataset_path.suffix == ".jsonl"
    
    if is_jsonl:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        items.append(json.loads(line))
                    except:
                        pass
    else:
        raw = dataset_path.read_text(encoding="utf-8", errors="replace")
        for line in raw.splitlines():
            if line.strip():
                items.append({"instruction": line.strip(), "output": ""})

    new_items = []
    
    evol_prompt_template = """あなたは熟練のAIトレーナーです。以下の指示（Instruction）を、より複雑で、難易度が高く、詳細な指示に書き換えてください。
元の指示の意味を維持しつつ、以下のいずれかの操作を加えてください：
1. 制約条件を追加する
2. 推論の深さを増す
3. より具体的な状況設定を加える

元の指示:
{instruction}

進化した指示:"""

    refine_prompt_template = """あなたは優秀なアシスタントです。以下の指示と、それに対する現在の回答があります。
現在の回答を、より正確で、詳細で、役に立つ高品質な回答に書き直してください。

指示:
{instruction}

現在の回答:
{output}

修正された高品質な回答:"""

    print(f"[Augment] 開始: {method} (Model: {model_name}, Items: {len(items)})")
    
    for item in tqdm(items, desc="Augmenting"):
        inst = item.get("instruction", item.get("text", ""))
        out = item.get("output", "")
        
        if not inst:
            continue
            
        try:
            if method == "evol_instruct":
                sys_prompt = "You are an expert AI trainer."
                user_prompt = evol_prompt_template.format(instruction=inst)
                
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                new_inst = resp.choices[0].message.content.strip()
                
                new_item = item.copy()
                new_item["instruction"] = new_inst
                new_items.append(new_item)

            elif method == "refine":
                sys_prompt = "You are a helpful assistant."
                user_prompt = refine_prompt_template.format(instruction=inst, output=out)
                
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2048
                )
                new_out = resp.choices[0].message.content.strip()
                
                new_item = item.copy()
                new_item["output"] = new_out
                new_items.append(new_item)
                
        except Exception as e:
            print(f"[Augment] Error on item: {e}")
            new_items.append(item)

    ts = int(time.time())
    new_name = f"{dataset_path.stem}_{method}_{ts}.jsonl"
    output_path = dataset_path.parent / new_name
    
    with open(output_path, "w", encoding="utf-8") as f:
        for it in new_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    lineage_info = _write_dataset_lineage(
        output_name=new_name,
        op=f"augment:{method}",
        inputs=[dataset_name],
        options={"method": method, "params": aug_params},
        stats={"output_count": len(new_items)},
    )

    return {
        "status": "success",
        "original_file": dataset_name,
        "output_file": new_name,
        "count": len(new_items),
        "lineage": lineage_info,
    }

# -----------------------------------------------------------------------------
# 7. Model Merge & Compile
# -----------------------------------------------------------------------------


def merge_precheck(base_model_name: str, adapter_path: str) -> Dict[str, Any]:
    """
    マージ前の互換性チェック（軽量）
    - パス存在確認
    - adapter_config.json の base_model_name_or_path との整合
    - 主要ファイルの存在確認
    - tokenizer の special token / vocab サイズ簡易比較（可能な範囲）
    """
    base_model_path = MODELS_DIR / base_model_name
    adapter_full_path = OUTPUT_DIR / adapter_path

    if not base_model_path.exists():
        raise FileNotFoundError(f"ベースモデルが見つかりません: {base_model_path}")
    if not adapter_full_path.exists():
        raise FileNotFoundError(f"LoRAアダプタが見つかりません: {adapter_full_path}")

    errors: List[str] = []
    warnings: List[str] = []
    info: Dict[str, Any] = {
        "base_model": base_model_name,
        "base_model_path": str(base_model_path),
        "adapter_path": adapter_path,
        "adapter_full_path": str(adapter_full_path),
        "errors": errors,
        "warnings": warnings,
        "details": {}
    }

    adapter_cfg_path = adapter_full_path / "adapter_config.json"
    if not adapter_cfg_path.exists():
        warnings.append("adapter_config.json が見つかりません（古い形式/破損の可能性）")
    else:
        try:
            cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
            base_in_cfg = cfg.get("base_model_name_or_path") or cfg.get("base_model_name") or ""
            info["details"]["adapter_base_model_name_or_path"] = base_in_cfg
            if base_in_cfg:
                bn = Path(str(base_in_cfg)).name
                if bn and bn != base_model_name:
                    warnings.append(
                        f"adapter_config.json の base_model_name_or_path と選択ベースモデル名が一致しません: {bn} != {base_model_name}"
                    )
        except Exception as e:
            warnings.append(f"adapter_config.json の読み取りに失敗しました: {e}")

    if not any((adapter_full_path / f).exists() for f in ["adapter_model.safetensors", "adapter_model.bin"]):
        errors.append("アダプタ重みファイルが見つかりません（adapter_model.safetensors / adapter_model.bin）")

    try:
        tok_base = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        info["details"]["base_vocab_size"] = getattr(tok_base, "vocab_size", None)
        info["details"]["base_special_tokens"] = tok_base.special_tokens_map or {}

        tok_files = ["tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt", "tokenizer_config.json"]
        has_tok = any((adapter_full_path / f).exists() for f in tok_files)
        info["details"]["adapter_has_tokenizer_files"] = has_tok
        if has_tok:
            tok_ad = AutoTokenizer.from_pretrained(adapter_full_path, trust_remote_code=True)
            info["details"]["adapter_vocab_size"] = getattr(tok_ad, "vocab_size", None)
            info["details"]["adapter_special_tokens"] = tok_ad.special_tokens_map or {}
            if getattr(tok_base, "vocab_size", None) != getattr(tok_ad, "vocab_size", None):
                warnings.append("ベースとアダプタ側の tokenizer vocab_size が一致しません（特殊ケース以外は注意）")
    except Exception as e:
        warnings.append(f"tokenizer チェックに失敗しました: {e}")

    return info

def _run_merge_smoke_test(model_path: str, prompt: str) -> Dict[str, Any]:
    """
    マージ済みモデルの簡易推論テスト（短文生成）
    """
    import torch
    from transformers import GenerationConfig

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()

    inputs = tok(prompt, return_tensors="pt")
    try:
        dev = model.device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
    except Exception:
        pass

    gen_cfg = GenerationConfig(
        max_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.05,
        eos_token_id=getattr(tok, "eos_token_id", None),
        pad_token_id=getattr(tok, "pad_token_id", getattr(tok, "eos_token_id", None)),
    )

    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)

    text = tok.decode(out[0], skip_special_tokens=True)

    try:
        del model
        del inputs
    except Exception:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"status": "ok", "prompt": prompt, "text": text}


def merge_and_save_model(base_model_name: str, adapter_path: str, new_model_name: str, run_smoke_test: bool = False, smoke_test_prompt: Optional[str] = None):
    unload_inference_model()
    from peft import PeftModel
    
    base_model_path = MODELS_DIR / base_model_name
    adapter_full_path = OUTPUT_DIR / adapter_path
    output_path = MODELS_DIR / new_model_name
    
    if output_path.exists():
        raise FileExistsError(f"モデル名 '{new_model_name}' は既に存在します。別の名前を指定してください。")
    
    print(f"[Merge] ベースモデル読み込み: {base_model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        print("[Merge] モデルロード中 (CPUメモリを使用します)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="cpu",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print(f"[Merge] LoRAアダプタ適用: {adapter_full_path}")
        model = PeftModel.from_pretrained(model, adapter_full_path, device_map="cpu")
        print("[Merge] マージ実行中...")
        model = model.merge_and_unload()
        print(f"[Merge] 保存中: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        print("[Merge] 完了しました。")
        return {"status": "success", "path": str(output_path)}
    except Exception as e:
        print(f"[Merge] エラー発生: {e}")
        if output_path.exists():
            shutil.rmtree(output_path, ignore_errors=True)
        raise e
    finally:
        del model
        del tokenizer
        import gc
        gc.collect()

def compile_text_folder(folder_rel: str, *, shard_max_mb: int = 100, exclude_patterns: Optional[List[str]] = None, extensions: Optional[List[str]] = None) -> Dict[str, Any]:
    if shard_max_mb <= 0: shard_max_mb = 100
    if exclude_patterns is None: exclude_patterns = ["**/.git/**", "**/__pycache__/**", "**/*.bak", "**/*.tmp", "**/*.log"]
    if extensions is None: extensions = [".txt"]

    folder_rel_norm = _normalize_rel_posix(folder_rel)
    folder_path = _safe_path_under(DATASET_DIR, folder_rel_norm)

    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"フォルダが見つかりません: {folder_rel}")

    candidates: List[Path] = []
    for p in folder_path.rglob("*"):
        if not p.is_file(): continue
        if p.suffix.lower() not in [e.lower() for e in extensions]: continue
        rel = p.relative_to(folder_path).as_posix()
        if _should_exclude(rel, exclude_patterns): continue
        candidates.append(p)

    candidates.sort(key=lambda x: _natural_sort_key(x.relative_to(folder_path).as_posix()))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = DATASET_DIR / "compiled" / f"{folder_path.name}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    max_bytes = int(shard_max_mb) * 1024 * 1024
    shard_index = 1
    shard_char_count = 0
    shard_byte_est = 0
    shard_path = out_root / f"compiled_{shard_index:04d}.txt"
    shard_f = shard_path.open("w", encoding="utf-8", newline="\n")

    manifest = {
        "source_folder": folder_rel_norm,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_root),
        "compiled_files": [],
        "files": [],
        "summary": {"total_files": 0, "total_chars": 0, "total_input_bytes": 0, "files_with_replacements": 0, "files_with_non_utf8": 0, "encodings": {}},
        "warnings": [],
    }

    def _close_and_new_shard():
        nonlocal shard_index, shard_char_count, shard_byte_est, shard_path, shard_f
        shard_f.flush(); shard_f.close()
        shard_index += 1; shard_char_count = 0; shard_byte_est = 0
        shard_path = out_root / f"compiled_{shard_index:04d}.txt"
        shard_f = shard_path.open("w", encoding="utf-8", newline="\n")

    for i, src in enumerate(candidates, start=1):
        rel = src.relative_to(folder_path).as_posix()
        try:
            raw = src.read_bytes()
            text = raw.decode("utf-8", errors="replace")
        except:
            text = ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        piece_bytes = len(text.encode("utf-8"))
        
        if shard_byte_est > 0 and (shard_byte_est + piece_bytes) > max_bytes:
            _close_and_new_shard()
        
        if shard_byte_est > 0:
            shard_f.write("\n\n")
            shard_byte_est += 2
            
        shard_f.write(text)
        shard_char_count += len(text)
        shard_byte_est += piece_bytes
        
        manifest["summary"]["total_files"] += 1
        manifest["files"].append({"path": rel, "shard": shard_index})

    shard_f.flush(); shard_f.close()
    
    compiled_files = sorted([p.relative_to(DATASET_DIR).as_posix() for p in out_root.glob("compiled_*.txt")])
    manifest["compiled_files"] = compiled_files
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lineage_info = _write_dataset_lineage(
        output_name=(out_root.relative_to(DATASET_DIR)).as_posix(),
        op="compile_folder",
        inputs=[folder_rel_norm],
        options={
            "shard_max_mb": shard_max_mb,
            "exclude_patterns": exclude_patterns,
            "extensions": extensions,
        },
        stats={
            "total_files": manifest.get("summary", {}).get("total_files", 0),
            "compiled_files": len(compiled_files),
        },
    )

    return {
        "status": "ok",
        "output_dir": (out_root.relative_to(DATASET_DIR)).as_posix(),
        "compiled_files": compiled_files,
        "lineage": lineage_info,
    }

# Helpers
def _natural_sort_key(text: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", text)]
def _normalize_rel_posix(path: str) -> str:
    return (path or "").replace("\\", "/").strip("/")
def _safe_path_under(base: Path, rel_path: str) -> Path:
    return (base / rel_path).resolve()
def _should_exclude(rel: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in patterns)

# 起動時のクリーンアップ実行
_check_and_clean_pid_file()
