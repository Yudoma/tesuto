# -*- coding: utf-8 -*-
"""
lora_routes.py
LoRA Factory APIルート定義 (v13: Unsloth & ORPO & WandB Support)
モデル管理、データセット、学習制御、推論検証のエンドポイントを提供します。
"""
import re
import shutil
import subprocess
import sys
import platform
import psutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lora_config import settings

# エンジンモジュール
import lora_engine

router = APIRouter()

# ============================================================
# Helpers (安全なパス解決)
# ============================================================

def _normalize_rel_posix(path: str) -> str:
    p = (path or "").replace("\\", "/").lstrip("/")
    p = re.sub(r"^[A-Za-z]:", "", p).lstrip("/")
    p = re.sub(r"/+", "/", p)
    return p

def _safe_dataset_path(rel_path: str) -> Path:
    rel = _normalize_rel_posix(rel_path)
    if rel in ("", "."):
        raise HTTPException(400, "不正なパスです。")
    base = settings.dataset_dir.resolve()
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except Exception:
        raise HTTPException(400, "不正なパスが指定されました（パストラバーサルの疑い）")
    return candidate

def _sanitize_folder_name(name: str) -> str:
    # Windowsでも扱いやすいフォルダ名へ
    n = (name or "").strip()
    n = n.replace("\\", "_").replace("/", "_")
    n = n.replace(":", "_").replace("*", "_").replace("?", "_")
    n = n.replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")
    if not n:
        n = "folder_upload"
    return n

# ============================================================
# Pydantic Schemas
# ============================================================

class DownloadModelRequest(BaseModel):
    repo_id: str

class MergeModelRequest(BaseModel):
    base_model: str
    adapter_path: str
    new_model_name: str
    run_smoke_test: bool = False
    smoke_test_prompt: Optional[str] = None

class MergePrecheckRequest(BaseModel):
    base_model: str
    adapter_path: str

class TrainStartRequest(BaseModel):
    base_model: str         # モデルフォルダ名
    dataset: str            # データセットファイル名
    dataset_type: str = "raw_text" 
    params: Dict[str, Any]  # 基本学習パラメータ
    
    # 再開・安定化・テンプレート
    resume_from_checkpoint: Optional[str] = None
    neftune_noise_alpha: Optional[float] = None
    prompt_template: Optional[str] = None
    
    # 評価
    validation_file: Optional[str] = None
    validation_prompt: Optional[str] = None

    # [New] Evaluation Probes (複数プロンプトの品質チェック)
    eval_prompt_set: Optional[str] = None          # 例: "basic_jp" (プリセット名)
    eval_prompts: Optional[List[str]] = None       # 任意のプロンプト配列（プリセットより優先）
    eval_max_new_tokens: int = 128                 # 評価生成の最大トークン数

    # Early Stopping (eval_lossベース)
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    # 高品質学習オプション (v11/v12)
    use_dora: bool = False
    lr_scheduler_type: str = "cosine"
    
    # v12 Options
    use_flash_attention_2: bool = False
    train_on_inputs: bool = False

    # [New] v13 Options (High Performance & Quality)
    use_unsloth: bool = False       # Unslothによる高速化
    use_orpo: bool = False          # ORPO (Odds Ratio Preference Optimization)
    monitor_wandb: bool = False     # WandBへのログ送信
    wandb_api_key: Optional[str] = None

class InferenceLoadRequest(BaseModel):
    base_model: str
    adapter_path: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    repetition_penalty: float = 1.1
    top_p: float = 0.9

class CompileFolderRequest(BaseModel):
    folder: str = Field(..., description="datasets配下のフォルダ相対パス")
    shard_max_mb: int = Field(100, ge=1, le=2048, description="出力シャードの最大サイズ(MB)目安")
    exclude_patterns: List[str] = Field(default_factory=list, description="除外パターン(glob)の配列")

class TokenAnalyzeRequest(BaseModel):
    dataset: str
    base_model: str
    max_seq_length: int = 2048

class SmartSplitRequest(BaseModel):
    dataset_folder: str
    base_model: str
    max_seq_length: int = 2048

class CleanDatasetRequest(BaseModel):
    dataset: str
    remove_duplicates: bool = False
    min_length: int = 0
    filter_lang: Optional[str] = None
    # [New] 品質フィルタ
    filter_ppl_threshold: Optional[float] = None # PPL閾値 (これ以上悪いと捨てる)

# Data Augmentation & Dedup
class DedupRequest(BaseModel):
    dataset: str
    threshold: float = 0.95
    model_name: Optional[str] = None

class AugmentRequest(BaseModel):
    dataset: str
    method: str = "evol_instruct"
    params: Optional[Dict[str, Any]] = None

# ============================================================
# System Info
# ============================================================

@router.get("/system_info")
def get_system_info():
    """システム詳細情報を取得"""
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "cpu": {
            "model": platform.processor(),
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=None)
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "percent": psutil.virtual_memory().percent
        },
        "gpu_info": "N/A"
    }

    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, encoding='utf-8')
        if res.returncode == 0:
            info["gpu_info"] = res.stdout
        else:
            info["gpu_info"] = "nvidia-smi returned error."
    except FileNotFoundError:
        info["gpu_info"] = "nvidia-smi not found."
    except Exception as e:
        info["gpu_info"] = f"Error: {str(e)}"

    return info

# ============================================================
# Capabilities / Utilities
# ============================================================

@router.get("/capabilities")
def get_capabilities():
    """このツールが提供する機能（モード）を返す。

    UI 側はこの情報を見て「未対応モード」を表示しない/無効化します。
    """
    # 現状の実装は Text(LLM) 学習/推論が中心です。
    # Image(Audio) は UI 側の骨組みがあっても、API/実装が揃っていない場合は false を返します。
    caps = {
        "text": True,
        "image": False,
        "audio": False,
    }
    return {"capabilities": caps}

@router.get("/system_info/{mode}")
def get_system_info_by_mode(mode: str):
    """モード別の system_info 互換エンドポイント。

    旧UIが /system_info/image のように呼び出すケースがあるため互換として残します。
    実態は /system_info と同じ情報を返しつつ mode を付与します。
    """
    base = get_system_info()
    base["mode"] = mode
    return base

class OpenPathRequest(BaseModel):
    key: Optional[str] = Field(None, description="既知のキー（models/datasets/output/logs 等）を指定して開く")
    path: Optional[str] = Field(None, description="開きたいパス（ファイル or フォルダ）。key 未指定時に使用")
    allow_file: bool = Field(True, description="ファイルパスも許可するか")
    allow_dir: bool = Field(True, description="ディレクトリパスも許可するか")

def _is_path_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

@router.post("/utils/open_path")
def open_path(req: OpenPathRequest):
    """Windows の Explorer でパスを開く（ローカル運用向けユーティリティ）。

    安全のため、以下の配下のみ許可します:
    - base_dir
    - models_root / dataset_root / output_root / logs_dir
    """
    # key 指定がある場合は既知のフォルダへ解決
    if req.key:
        key = req.key.strip().lower()
        mapping = {
            "base": settings.base_dir,
            "models": settings.models_root,
            "datasets": settings.dataset_root,
            "output": settings.output_root,
            "logs": settings.logs_dir,
            # text/image/audio のショートカット
            "models_text": settings.dirs["text"]["models"],
            "models_image": settings.dirs["image"]["models"],
            "models_audio": settings.dirs["audio"]["models"],
            "datasets_text": settings.dirs["text"]["datasets"],
            "datasets_image": settings.dirs["image"]["datasets"],
            "datasets_audio": settings.dirs["audio"]["datasets"],
            "output_text": settings.dirs["text"]["output"],
            "output_image": settings.dirs["image"]["output"],
            "output_audio": settings.dirs["audio"]["output"],
        }
        if key not in mapping:
            raise HTTPException(400, f"不明な key です: {req.key}")
        target = Path(mapping[key])
    else:
        if not req.path:
            raise HTTPException(400, "path または key を指定してください。")
        target = Path(req.path).expanduser()

    # 相対パスは base_dir 基準に解決
    if not target.is_absolute():
        target = (settings.base_dir / target)

    # 許可ルート（必要に応じて増やす）
    allowed_roots = [
        settings.base_dir,
        settings.models_root,
        settings.dataset_root,
        settings.output_root,
        settings.logs_dir,
    ]

    # 存在チェック
    if not target.exists():
        raise HTTPException(404, f"指定パスが見つかりません: {target}")

    # 種別チェック
    if target.is_file() and not req.allow_file:
        raise HTTPException(400, "ファイルパスは許可されていません。")
    if target.is_dir() and not req.allow_dir:
        raise HTTPException(400, "フォルダパスは許可されていません。")

    # ルート制限
    if not any(_is_path_under(target, r) for r in allowed_roots):
        raise HTTPException(400, "安全のため、このパスは開けません（ツール配下のみ許可）。")

    # Windows のみ対応
    if platform.system().lower() != "windows":
        raise HTTPException(400, "open_path は Windows 専用です。")

    try:
        if target.is_dir():
            subprocess.Popen(["explorer", str(target)])
        else:
            # ファイルの場合は所在フォルダを開いて選択状態にする
            subprocess.Popen(["explorer", "/select,", str(target)])
        return {"ok": True, "path": str(target)}
    except Exception as e:
        raise HTTPException(500, f"フォルダを開けませんでした: {e}")

@router.get("/system/flash_attention2_preflight")
def flash_attention2_preflight():
    """FlashAttention2 の導入確認（import + GPU smoke test）"""
    return lora_engine.flash_attention_2_preflight()



# ============================================================
# Model Management
# ============================================================

@router.get("/models")
def list_models():
    """modelsディレクトリ内のモデル一覧"""
    models = []
    if settings.models_dir.exists():
        for d in settings.models_dir.iterdir():
            if d.is_dir():
                is_hf = (d / "config.json").exists()
                models.append({
                    "name": d.name,
                    "path": str(d),
                    "type": "hf" if is_hf else "unknown"
                })
    return {"models": models}

@router.post("/models/download")
def download_model(req: DownloadModelRequest, background_tasks: BackgroundTasks):
    """HuggingFaceからモデルをダウンロード"""
    background_tasks.add_task(lora_engine.download_model_task, req.repo_id)
    return {"status": "started", "repo_id": req.repo_id, "message": "ダウンロードをバックグラウンドで開始しました。"}

@router.post("/models/merge_precheck")
def merge_precheck(req: MergePrecheckRequest):
    """マージ前の互換性チェック（軽量）"""
    try:
        return lora_engine.merge_precheck(req.base_model, req.adapter_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"マージ事前チェック失敗: {e}")

@router.post("/models/merge")
def merge_model(req: MergeModelRequest):
    """LoRAアダプタをマージして保存"""
    try:
        result = lora_engine.merge_and_save_model(req.base_model, req.adapter_path, req.new_model_name, run_smoke_test=req.run_smoke_test, smoke_test_prompt=req.smoke_test_prompt)
        return result
    except FileExistsError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"マージ失敗: {e}")

@router.delete("/models/{name}")
def delete_model(name: str):
    """モデル削除"""
    target = settings.models_dir / name
    if target.exists() and target.is_dir():
        try:
            shutil.rmtree(target)
            return {"status": "deleted", "name": name}
        except Exception as e:
            raise HTTPException(500, f"削除に失敗しました: {e}")
    raise HTTPException(404, "モデルが見つかりません")

# ============================================================
# Dataset Management
# ============================================================

@router.get("/datasets")
def list_datasets():
    """データセット一覧"""
    files = []
    base = settings.dataset_dir
    if base.exists():
        for f in base.rglob("*"):
            if f.is_file() and f.suffix.lower() in [".jsonl", ".json", ".txt"]:
                rel = f.relative_to(base).as_posix()
                files.append({
                    "name": rel,
                    "size": f.stat().st_size,
                    "path": str(f)
                })
    files.sort(key=lambda x: x["name"].lower())
    return {"datasets": files}

@router.get("/datasets/folders")
def list_dataset_folders():
    """datasets配下のフォルダ一覧"""
    base = settings.dataset_dir
    folders: List[str] = []
    if base.exists():
        for d in base.rglob("*"):
            if d.is_dir():
                rel = d.relative_to(base).as_posix()
                if rel and not rel.startswith("compiled"):
                    folders.append(rel)
    folders = sorted(set(folders), key=lambda s: s.lower())
    return {"folders": folders}

@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """単一ファイルアップロード"""
    dest = settings.dataset_dir / file.filename
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "uploaded", "filename": file.filename}
    except Exception as e:
        raise HTTPException(500, f"アップロード失敗: {e}")

@router.post("/datasets/upload_folder")
async def upload_dataset_folder(files: List[UploadFile] = File(...)):
    """フォルダアップロード"""
    if not files:
        raise HTTPException(400, "ファイルがありません。")

    rel_paths = [_normalize_rel_posix(f.filename) for f in files]
    roots = []
    for rp in rel_paths:
        segs = [s for s in rp.split("/") if s]
        if segs:
            roots.append(segs[0])
    root = roots[0] if roots and all(r == roots[0] for r in roots) else "folder_upload"
    root = _sanitize_folder_name(root)

    dest_root = settings.dataset_dir / root
    if dest_root.exists():
        dest_root = settings.dataset_dir / f"{root}_{int(time.time())}"
    dest_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    for up, rp in zip(files, rel_paths):
        segs = [s for s in rp.split("/") if s]
        inner = "/".join(segs[1:]) if segs and segs[0] == root else "/".join(segs)
        if not inner:
            continue
        dest = (dest_root / inner).resolve()
        try:
            dest.relative_to(dest_root.resolve())
        except Exception:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as buffer:
            shutil.copyfileobj(up.file, buffer)
        saved += 1

    rel_folder = dest_root.relative_to(settings.dataset_dir).as_posix()
    return {"status": "uploaded", "folder": rel_folder, "files": saved}

@router.get("/datasets/{filename:path}")
def preview_dataset(filename: str):
    """プレビュー (先頭部分)"""
    path = _safe_dataset_path(filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "ファイルが見つかりません")

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            content = f.read(2000)
        return {"filename": filename, "content": content}
    except Exception as e:
        raise HTTPException(500, f"読み込みエラー: {e}")

@router.post("/datasets/compile_folder")
def compile_dataset_folder(req: CompileFolderRequest):
    """フォルダ内ファイルを連結/分割"""
    try:
        result = lora_engine.compile_text_folder(
            req.folder,
            shard_max_mb=req.shard_max_mb,
            exclude_patterns=req.exclude_patterns if req.exclude_patterns else None,
            extensions=[".txt"],
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"コンパイルに失敗しました: {e}")

@router.post("/datasets/analyze_tokens")
def analyze_tokens(req: TokenAnalyzeRequest):
    """トークン数分布解析"""
    try:
        result = lora_engine.analyze_dataset_tokens(
            req.dataset, 
            req.base_model, 
            max_seq_length=req.max_seq_length
        )
        if "error" in result:
             raise HTTPException(400, result["error"])
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"解析エラー: {e}")

@router.post("/datasets/smart_split")
def smart_split(req: SmartSplitRequest):
    """自動分割"""
    try:
        result = lora_engine.smart_split_dataset(
            req.dataset_folder,
            req.base_model,
            max_seq_length=req.max_seq_length
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"分割エラー: {e}")

@router.post("/datasets/clean")
def clean_dataset(req: CleanDatasetRequest):
    """データセットクリーニング (重複/短文/言語/PPL)"""
    try:
        result = lora_engine.clean_dataset_file(
            req.dataset,
            {
                "remove_duplicates": req.remove_duplicates,
                "min_length": req.min_length,
                "filter_lang": req.filter_lang,
                "filter_ppl_threshold": req.filter_ppl_threshold # [New]
            }
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"クリーニング失敗: {e}")

# Data Augmentation & Semantic Dedup

@router.post("/datasets/dedup")
def semantic_dedup(req: DedupRequest):
    """意味的重複排除"""
    try:
        result = lora_engine.perform_semantic_deduplication(
            req.dataset,
            threshold=req.threshold,
            model_name=req.model_name
        )
        return result
    except ImportError as e:
        raise HTTPException(500, f"必要なライブラリが見つかりません: {e}")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"重複排除処理エラー: {e}")

@router.post("/datasets/augment")
def augment_dataset(req: AugmentRequest):
    """データ拡張 (Evol-Instruct / Refine)"""
    try:
        result = lora_engine.perform_data_augmentation(
            req.dataset,
            method=req.method,
            aug_params=req.params if req.params else {}
        )
        return result
    except ImportError as e:
        raise HTTPException(500, f"必要なライブラリが見つかりません (openai): {e}")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"データ拡張処理エラー: {e}")

# ============================================================
# Training Control
# ============================================================


@router.get("/train/eval_prompt_sets")
def list_eval_prompt_sets():
    """品質評価用のプロンプトセット一覧"""
    return {"sets": lora_engine.EVAL_PROMPT_SETS}

class TrainCompareRequest(BaseModel):
    job_id_a: str
    job_id_b: str

@router.post("/train/compare")
def compare_train_runs(req: TrainCompareRequest):
    """2つの学習ジョブを比較（loss推移/評価プローブ/主要パラメータ）"""
    hist = lora_engine.get_training_history()
    def pick(job_id: str):
        for h in hist:
            if h.get("job_id") == job_id or h.get("id") == job_id or (isinstance(h.get("id"), str) and h["id"].endswith(job_id)):
                return h
        return None
    a = pick(req.job_id_a)
    b = pick(req.job_id_b)
    if not a or not b:
        raise HTTPException(404, "指定されたjob_idが見つかりません。/train/history を確認してください。")

    # 重要項目だけ抽出
    def slim(x: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "job_id": x.get("job_id"),
            "status": x.get("status"),
            "model": x.get("model"),
            "dataset": x.get("dataset"),
            "dataset_sha256": x.get("dataset_sha256"),
            "pipeline_hash": x.get("pipeline_hash"),
            "params": x.get("params"),
            "steps": x.get("steps"),
            "final_loss": x.get("final_loss"),
            "final_eval_loss": x.get("final_eval_loss"),
            "avg_eval_score": x.get("avg_eval_score"),
            "last_eval_score": x.get("last_eval_score"),
            "eval_probes": x.get("eval_probes", [])[-50:],  # 最後の方だけ
        }
    return {"a": slim(a), "b": slim(b)}

@router.post("/train/start")
def start_training(req: TrainStartRequest, background_tasks: BackgroundTasks):
    """学習開始"""
    if lora_engine.is_training_active():
        raise HTTPException(400, "学習ジョブが既に実行中です。")

    train_params = req.params.copy()
    train_params["dataset_type"] = req.dataset_type
    
    # マージ: Resume / NEFTune / Prompt Template / Eval
    if req.resume_from_checkpoint:
        train_params["resume_from_checkpoint"] = req.resume_from_checkpoint
    if req.neftune_noise_alpha:
        train_params["neftune_noise_alpha"] = req.neftune_noise_alpha
    if req.prompt_template:
        train_params["prompt_template"] = req.prompt_template
    
    if req.validation_file:
        train_params["validation_file"] = req.validation_file
    if req.validation_prompt:
        train_params["validation_prompt"] = req.validation_prompt
    # Evaluation Probes
    if req.eval_prompts:
        train_params["eval_prompts"] = req.eval_prompts
    if req.eval_prompt_set:
        train_params["eval_prompt_set"] = req.eval_prompt_set
    train_params["eval_max_new_tokens"] = req.eval_max_new_tokens



    # Early Stopping
    train_params["early_stopping"] = req.early_stopping
    train_params["early_stopping_patience"] = req.early_stopping_patience
    train_params["early_stopping_threshold"] = req.early_stopping_threshold

    # DoRA & Scheduler
    train_params["use_dora"] = req.use_dora
    train_params["lr_scheduler_type"] = req.lr_scheduler_type
    
    # v12 Options
    train_params["use_flash_attention_2"] = req.use_flash_attention_2
    train_params["train_on_inputs"] = req.train_on_inputs

    # [New] v13 Options
    train_params["use_unsloth"] = req.use_unsloth
    train_params["use_orpo"] = req.use_orpo
    train_params["monitor_wandb"] = req.monitor_wandb
    if req.wandb_api_key:
        train_params["wandb_api_key"] = req.wandb_api_key

    background_tasks.add_task(
        lora_engine.run_training_job,
        req.base_model,
        req.dataset,
        train_params
    )
    return {"status": "started", "params": train_params}

@router.post("/train/stop")
def stop_training():
    """学習中断"""
    lora_engine.stop_training_job()
    return {"status": "stop_requested"}

@router.get("/train/status")
def get_train_status():
    """学習進捗"""
    return lora_engine.get_training_status()

@router.get("/train/history")
def get_train_history():
    """学習履歴"""
    return {"history": lora_engine.get_training_history()}

@router.get("/train/checkpoints")
def list_checkpoints(base_model: str):
    """チェックポイント一覧"""
    try:
        checkpoints = lora_engine.list_checkpoints_for_model(base_model)
        return {"checkpoints": checkpoints}
    except Exception as e:
        raise HTTPException(500, f"チェックポイント取得失敗: {e}")

# ============================================================
# Inference / Verification
# ============================================================

@router.post("/inference/load")
def load_inference_model(req: InferenceLoadRequest):
    """検証用モデルロード"""
    try:
        lora_engine.load_inference_model(req.base_model, req.adapter_path)
        return {"status": "loaded", "base": req.base_model, "adapter": req.adapter_path}
    except Exception as e:
        raise HTTPException(500, f"ロード失敗: {e}")

@router.post("/inference/unload")
def unload_inference_model():
    """モデルアンロード"""
    lora_engine.unload_inference_model()
    return {"status": "unloaded"}

@router.post("/inference/chat")
def chat_inference(req: ChatRequest):
    """チャット生成 (Streaming)"""
    if not lora_engine.is_inference_model_loaded():
        raise HTTPException(400, "モデルがロードされていません。")

    return StreamingResponse(
        lora_engine.generate_stream(
            prompt=req.message,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            repetition_penalty=req.repetition_penalty,
            top_p=req.top_p
        ),
        media_type="text/event-stream"
    )