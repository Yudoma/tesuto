# -*- coding: utf-8 -*-
"""
backend/routes/text_routes.py
テキスト（LLM）モダリティ専用のAPIルート定義。
学習、推論、データセット操作、モデル管理のエンドポイントを提供します。
"""
import shutil
import sys
import json
import time
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

# 新しい構成からのインポート
from lora_config import settings
from backend.core.job_manager import job_manager
from backend.engines.text import text_engine
import lora_engine

router = APIRouter()

# ===========================================================
# Request Models (Pydantic)
# ===========================================================

class TrainParams(BaseModel):
    # 基本
    max_steps: int = 100
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_mode: str = "all-linear"
    max_seq_length: int = 2048
    val_set_size: float = 0.05
    prompt_template: Optional[str] = None
    
    # 高度な設定
    use_dora: bool = False
    lr_scheduler_type: str = "cosine"
    use_flash_attention_2: bool = False
    train_on_inputs: bool = False
    neftune_noise_alpha: Optional[float] = None
    
    # 評価・停止
    eval_score_enabled: bool = True
    eval_score_min_len: int = 40
    eval_score_max_len: int = 800
    eval_score_repetition_ngram: int = 6
    eval_score_repetition_threshold: float = 0.35
    eval_score_require_json_if_prompt_mentions_json: bool = True
    eval_score_banned_phrases: Optional[str] = None

class TrainStartRequest(BaseModel):
    base_model: str
    dataset: str
    dataset_type: str = "raw_text"
    params: TrainParams
    resume_from_checkpoint: Optional[str] = None
    validation_file: Optional[str] = None
    validation_prompt: Optional[str] = None
    eval_prompts: Optional[List[str]] = None
    eval_max_new_tokens: int = 128

class InferenceLoadRequest(BaseModel):
    base_model: str
    adapter_path: Optional[str] = None

class ChatRequest(BaseModel):
    """推論チャット要求（互換維持しつつ安全に拡張）"""
    message: str
    system_prompt: Optional[str] = None
    developer_prompt: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None  # [{"role":"user|assistant|system","content":"..."}]
    preset_id: Optional[str] = None

    # 推論設定（安全なデフォルト）
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_tokens: int = 512
    # ストリーミング再試行（ネットワーク瞬断等の保険）
    retry_count: int = 0
    retry_delay_ms: int = 250
class DownloadModelRequest(BaseModel):
    repo_id: str

class MergeRequest(BaseModel):
    base_model: str
    adapter_path: str
    new_model_name: str
    run_smoke_test: bool = False
    smoke_test_prompt: Optional[str] = None

class DatasetCleanRequest(BaseModel):
    dataset: str
    remove_duplicates: bool = True
    min_length: int = 10
    filter_lang: Optional[str] = None

class DatasetDedupRequest(BaseModel):
    dataset: str
    threshold: float = 0.95
    model_name: Optional[str] = None

class DatasetAugmentRequest(BaseModel):
    dataset: str
    method: str  # 'evol_instruct' or 'refine'

class TokenAnalyzeRequest(BaseModel):
    dataset: str
    base_model: str
    max_seq_length: int = 2048

class SmartSplitRequest(BaseModel):
    dataset_folder: str
    base_model: str
    max_seq_length: int = 2048

class CompileFolderRequest(BaseModel):
    folder: str
    shard_max_mb: int = 100
    exclude_patterns: List[str] = []

# ===========================================================
# Helpers
# ===========================================================

def _normalize_rel_posix(path: str) -> str:
    p = (path or "").replace("\\", "/").lstrip("/")
    p = re.sub(r"^[A-Za-z]:", "", p).lstrip("/")
    p = re.sub(r"/+", "/", p)
    return p

def _safe_dataset_path(rel_path: str) -> Path:
    # データセットディレクトリ内の安全なパスを返す
    rel = _normalize_rel_posix(rel_path)
    if rel in ("", "."):
        raise HTTPException(400, "Invalid path.")
    base = settings.dirs["text"]["datasets"].resolve()
    candidate = (base / rel).resolve()
    if not str(candidate).startswith(str(base)):
         raise HTTPException(400, "Path traversal detected.")
    return candidate

def _safe_model_path(rel_path: str) -> Path:
    # モデルディレクトリ内の安全なパスを返す
    rel = _normalize_rel_posix(rel_path)
    base = settings.dirs["text"]["models"].resolve()
    candidate = (base / rel).resolve()
    if not str(candidate).startswith(str(base)):
         raise HTTPException(400, "Path traversal detected.")
    return candidate

# ===========================================================
# Routes: Models
# ===========================================================

@router.get("/models")
def list_models():
    """モデル一覧取得"""
    models_dir = settings.dirs["text"]["models"]
    res = []
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir():
                # HF形式判定 (config.json)
                if (d / "config.json").exists():
                    res.append({"name": d.name, "type": "hf", "path": str(d)})
                # GGUF判定 (簡易)
                elif list(d.glob("*.gguf")):
                    res.append({"name": d.name, "type": "gguf", "path": str(d)})
    return {"models": res}


@router.get("/models/{model_name}/meta")
def model_meta(model_name: str):
    """モデルのメタ情報（UI表示用）"""
    model_dir = settings.dirs["text"]["models"] / model_name
    if not model_dir.exists():
        raise HTTPException(404, "Model not found")
    # directory size
    total = 0
    files = []
    for p in model_dir.rglob("*"):
        if p.is_file():
            try:
                sz = p.stat().st_size
                total += sz
                files.append({"path": str(p.relative_to(model_dir)), "size": sz})
            except Exception:
                pass
    try:
        mtime = model_dir.stat().st_mtime
    except Exception:
        mtime = None
    return {
        "name": model_name,
        "path": str(model_dir.resolve()),
        "total_bytes": total,
        "file_count": len(files),
        "mtime": mtime,
        "files_top": sorted(files, key=lambda x: x.get("size", 0), reverse=True)[:20],
        "modality": "text"
    }

@router.get("/models/{model_name}/predelete_check")
def model_predelete_check(model_name: str):
    """モデル削除前の参照チェック（軽量）"""
    refs = []
    # history files（存在する範囲）
    candidates = [
        settings.logs_dir / "history.json",
        settings.logs_dir / "history_text.json",
        settings.logs_dir / "history_text.json"
    ]
    for hf in candidates:
        try:
            if hf.exists():
                data = json.loads(hf.read_text(encoding="utf-8"))
                for item in data:
                    if str(item.get("model","")) == model_name:
                        refs.append({"where": str(hf.name), "job_id": item.get("id"), "timestamp": item.get("timestamp"), "status": item.get("status")})
        except Exception:
            pass

    active = False
    try:
        st = text_engine.get_training_status()
        if (st or {}).get("job_id") and (st or {}).get("status") == "running":
            if ((st or {}).get("params") or {}).get("base_model") == model_name or (st or {}).get("base_model") == model_name:
                active = True
    except Exception:
        pass

    return {
        "model": model_name,
        "reference_count": len(refs),
        "references": refs,
        "active_job_using": active
    }


@router.delete("/models/{model_name}")
def delete_model(model_name: str):
    """モデル削除"""
    path = _safe_model_path(model_name)
    if not path.exists():
        raise HTTPException(404, "Model not found")
    try:
        shutil.rmtree(path)
        return {"status": "deleted", "name": model_name}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete: {e}")


@router.delete("/models/{name}")
def delete_model_compat(name: str):
    """互換: BK18 の /models/{name} をサポート"""
    return delete_model(name)

@router.post("/models/download")
def download_model(req: DownloadModelRequest, background_tasks: BackgroundTasks):
    """HuggingFaceからモデルをダウンロード (バックグラウンド)"""
    from huggingface_hub import snapshot_download
    
    def _bg_download(repo_id):
        try:
            local_dir = settings.dirs["text"]["models"] / repo_id.split("/")[-1]
            print(f"[Download] Starting download: {repo_id} -> {local_dir}")
            snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
            print(f"[Download] Completed: {repo_id}")
        except Exception as e:
            print(f"[Download] Failed: {e}")

    background_tasks.add_task(_bg_download, req.repo_id)
    return {"status": "started", "repo_id": req.repo_id}

@router.post("/models/merge")
def merge_model(req: MergeRequest):
    """LoRAアダプタのマージ"""
    base_path = _safe_model_path(req.base_model)
    
    # アダプタは output_dir または models_dir にある可能性がある
    adapter_path = settings.dirs["text"]["output"] / req.adapter_path
    if not adapter_path.exists():
        adapter_path = settings.dirs["text"]["models"] / req.adapter_path
    
    if not adapter_path.exists():
        raise HTTPException(404, f"Adapter not found: {req.adapter_path}")

    save_path = settings.dirs["text"]["models"] / req.new_model_name
    if save_path.exists():
        raise HTTPException(400, f"Model '{req.new_model_name}' already exists.")

    print(f"[Merge] Base: {base_path}, Adapter: {adapter_path}")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # CPUでマージ実行 (VRAM節約)
        print("[Merge] Loading base model (CPU)...")
        base = AutoModelForCausalLM.from_pretrained(
            str(base_path),
            device_map="cpu",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)

        print("[Merge] Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base, str(adapter_path), device_map="cpu")
        
        print("[Merge] Merging weights...")
        model = model.merge_and_unload()

        print(f"[Merge] Saving to {save_path}...")
        model.save_pretrained(str(save_path), safe_serialization=True)
        tokenizer.save_pretrained(str(save_path))
        
        # Smoke Test
        smoke_res = None
        if req.run_smoke_test:
            print("[Merge] Running Smoke Test...")
            prompt = req.smoke_test_prompt or "Hello, who are you?"
            inputs = tokenizer(prompt, return_tensors="pt")
            # 簡易生成
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)
            res_text = tokenizer.decode(out[0], skip_special_tokens=True)
            smoke_res = {"status": "ok", "text": res_text}

        # メモリ解放
        del model
        del base
        import gc; gc.collect()

        return {"status": "success", "path": req.new_model_name, "smoke_test": smoke_res}

    except Exception as e:
        print(f"[Merge] Error: {e}")
        raise HTTPException(500, f"Merge failed: {e}")

# ===========================================================
# Routes: Datasets
# ===========================================================

@router.get("/datasets")
def list_datasets():
    """データセット一覧"""
    ds_dir = settings.dirs["text"]["datasets"]
    datasets = []
    if ds_dir.exists():
        for f in ds_dir.glob("*"):
            if f.is_file() and f.suffix in [".txt", ".json", ".jsonl"]:
                datasets.append({"name": f.name, "size": f.stat().st_size})
    return {"datasets": datasets}

@router.get("/datasets/folders")
def list_dataset_folders():
    """データセットフォルダ一覧"""
    ds_dir = settings.dirs["text"]["datasets"]
    folders = []
    if ds_dir.exists():
        for f in ds_dir.iterdir():
            if f.is_dir() and not f.name.startswith("."):
                folders.append(f.name)
    return {"folders": folders}

@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """単一ファイルアップロード"""
    path = settings.dirs["text"]["datasets"] / file.filename
    try:
        with path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "size": path.stat().st_size}
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")

@router.post("/datasets/upload_folder")
async def upload_folder_files(files: List[UploadFile] = File(...)):
    """フォルダアップロード (複数ファイル)"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"upload_{timestamp}"
    folder_path = settings.dirs["text"]["datasets"] / folder_name
    folder_path.mkdir(exist_ok=True)
    
    count = 0
    for file in files:
        # ファイル名に含まれるパス区切り文字を正規化（サブフォルダ対応は簡易的にフラット化、または構造維持）
        # ここではwebkitRelativePath相当の情報がないため、ファイル名のみで保存
        clean_name = Path(file.filename).name
        dest = folder_path / clean_name
        try:
            with dest.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            count += 1
        except: pass
        
    return {"folder": folder_name, "count": count}

@router.get("/datasets/{filename:path}")
def get_dataset_preview(filename: str):
    """データセットプレビュー"""
    path = _safe_dataset_path(filename)
    if not path.exists():
        raise HTTPException(404, "File not found")
    try:
        # 先頭 2KB だけ返す
        content = path.read_text(encoding="utf-8")[:2048]
        return {"content": content}
    except Exception as e:
        raise HTTPException(500, f"Read failed: {e}")

@router.post("/datasets/clean")
def clean_dataset(req: DatasetCleanRequest):
    """データセットクリーニング"""
    try:
        res = text_engine.clean_dataset(
            req.dataset, 
            remove_duplicates=req.remove_duplicates,
            min_length=req.min_length,
            filter_lang=req.filter_lang
        )
        if "error" in res: raise Exception(res["error"])
        return res
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/datasets/dedup")
def dedup_dataset(req: DatasetDedupRequest):
    """意味的重複排除"""
    try:
        res = text_engine.deduplicate_dataset(
            req.dataset,
            threshold=req.threshold,
            model_name=req.model_name
        )
        if "error" in res: raise Exception(res["error"])
        return res
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/datasets/analyze_tokens")
def analyze_tokens(req: TokenAnalyzeRequest):
    """トークン数解析"""
    # フォルダかファイルかを判定
    ds_path = settings.dirs["text"]["datasets"] / req.dataset
    is_folder = ds_path.is_dir()
    
    res = text_engine.analyze_tokens(
        req.dataset,
        req.base_model,
        max_seq_length=req.max_seq_length,
        is_folder=is_folder
    )
    if "error" in res: raise HTTPException(500, res["error"])
    return res

@router.post("/datasets/smart_split")
def smart_split_dataset(req: SmartSplitRequest):
    """超過ファイルの自動分割"""
    # これはEngineに入れるにはロジックが細かいのでRouteで実装（またはUtility化）
    # 簡易実装：analyze結果に基づき分割を行う
    folder_path = settings.dirs["text"]["datasets"] / req.dataset_folder
    if not folder_path.is_dir():
        raise HTTPException(400, "Dataset is not a folder.")
    
    # Engineの機能を使ってまず解析しても良いが、ここでは直接処理
    from transformers import AutoTokenizer
    try:
        base_path = settings.dirs["text"]["models"] / req.base_model
        tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    except:
        raise HTTPException(400, "Tokenizer load failed.")

    output_folder = folder_path.parent / (folder_path.name + "_split")
    output_folder.mkdir(exist_ok=True)
    
    files = sorted([f for f in folder_path.glob("**/*") if f.is_file() and f.suffix in [".txt", ".json", ".jsonl"]])
    
    processed_count = 0
    chunks_created = 0

    for f in files:
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            # 簡易分割: トークン化してチャンク分け
            ids = tokenizer.encode(content, add_special_tokens=False)
            
            if len(ids) <= req.max_seq_length:
                # コピー
                (output_folder / f.name).write_text(content, encoding="utf-8")
                processed_count += 1
                chunks_created += 1
            else:
                # 分割
                chunk_size = req.max_seq_length - 50 # マージン
                for i in range(0, len(ids), chunk_size):
                    chunk_ids = ids[i:i+chunk_size]
                    chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                    fname = f"{f.stem}_part{i//chunk_size:03d}{f.suffix}"
                    (output_folder / fname).write_text(chunk_text, encoding="utf-8")
                    chunks_created += 1
                processed_count += 1
        except: pass

    return {
        "files_processed": processed_count,
        "chunks_created": chunks_created,
        "output_folder": output_folder.name
    }

# ===========================================================
# Routes: Training
# ===========================================================

@router.post("/train/start")
def start_training(req: TrainStartRequest):
    """学習開始"""
    try:
        # パラメータdict変換
        params_dict = req.params.dict()
        
        # Pydanticモデルに含まれない追加パラメータを統合
        extra_args = {
            "resume_from_checkpoint": req.resume_from_checkpoint,
            "validation_file": req.validation_file,
            "validation_prompt": req.validation_prompt,
            "eval_prompts_path": None, # ファイルパスとして渡す必要があるため後述
            "eval_max_new_tokens": req.eval_max_new_tokens
        }
        
        # 評価プロンプトの保存
        if req.eval_prompts:
            import tempfile
            # 一時ファイルではなく logs に保存
            eval_file = settings.logs_dir / f"eval_prompts_{int(time.time())}.json"
            eval_file.write_text(json.dumps(req.eval_prompts, ensure_ascii=False), encoding="utf-8")
            extra_args["eval_prompts_path"] = str(eval_file)

        params_dict.update(extra_args)

        res = text_engine.start_training(
            base_model=req.base_model,
            dataset=req.dataset,
            params=params_dict
        )
        return res
    except Exception as e:
        raise HTTPException(500, f"Training start failed: {e}")

@router.post("/train/stop")
def stop_training():
    """学習停止"""
    return text_engine.stop_training()

@router.get("/train/status")
def get_training_status():
    """学習ステータス取得"""
    return text_engine.get_training_status()


@router.post("/train/rerun/{job_id}")
def train_rerun(job_id: str):
    """過去ジョブの設定で再実行（履歴から再現）"""
    return text_engine.rerun_training(job_id)

@router.get("/train/history")
def get_training_history():
    """学習履歴取得"""
    return text_engine.get_training_history()

@router.get("/train/checkpoints")
def list_checkpoints(base_model: str):
    """チェックポイント一覧"""
    # output_dir 内を検索
    output_root = settings.dirs["text"]["output"]
    res = []
    
    if output_root.exists():
        for job_dir in output_root.iterdir():
            if not job_dir.is_dir(): continue
            if base_model in job_dir.name: # ベースモデル名が含まれるフォルダを候補とする簡易フィルタ
                cps = []
                for cp in job_dir.glob("checkpoint-*"):
                    if cp.is_dir():
                        try:
                            step = int(cp.name.split("-")[1])
                            cps.append({"name": cp.name, "step": step, "path": str(cp.relative_to(output_root))})
                        except: pass
                if cps:
                    cps.sort(key=lambda x: x["step"], reverse=True)
                    res.append({"job_folder": job_dir.name, "checkpoints": cps})
    
    return {"checkpoints": res}

# ===========================================================
# Routes: Inference
# ===========================================================

@router.post("/inference/load")
def load_inference(req: InferenceLoadRequest):
    """推論モデルロード"""
    try:
        return text_engine.load_inference_model(req.base_model, req.adapter_path)
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/inference/unload")
def unload_inference():
    """推論モデルアンロード"""
    return text_engine.unload_inference_model()


@router.get("/inference/presets")
def get_inference_presets():
    """推論プリセット一覧（再現性のため）"""
    try:
        from backend.engines.text import get_text_presets
        return {"status": "ok", "presets": get_text_presets()}
    except Exception as e:
        raise HTTPException(500, f"プリセット取得に失敗しました: {str(e)}")

def _stream_with_retry(gen_factory, *, retry_count: int = 0, retry_delay_ms: int = 250):
    """Streaming generator をクライアント切断/例外に強くする薄いラッパー。"""
    attempt = 0
    while True:
        try:
            for chunk in gen_factory():
                yield chunk
            break
        except GeneratorExit:
            # クライアントが切断した場合は静かに終了
            break
        except BrokenPipeError:
            break
        except Exception as e:
            if attempt >= int(retry_count or 0):
                # SSE としてエラーを返して終了（日本語統一）
                msg = f"エラーが発生しました: {str(e)}"
                yield f"data: {msg}\n\n"
                break
            attempt += 1
            try:
                time.sleep(max(0.0, float(retry_delay_ms or 0) / 1000.0))
            except Exception:
                pass
            continue

@router.post("/inference/chat")
def chat_inference(req: ChatRequest):
    """チャット生成 (Streaming)"""
    if not text_engine.is_inference_model_loaded():
        raise HTTPException(400, "モデルがロードされていません。")
    
    return StreamingResponse(
        _stream_with_retry(
            lambda: text_engine.generate_stream(
            prompt=req.message,
            system_prompt=req.system_prompt,
            developer_prompt=req.developer_prompt,
            history=req.history,
            preset_id=req.preset_id,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            repetition_penalty=req.repetition_penalty,
            top_p=req.top_p
        ),
            retry_count=req.retry_count,
            retry_delay_ms=req.retry_delay_ms,
        ),
        media_type="text/event-stream"
    )

# ===========================================================
# Backward compatibility endpoints (ported from BK18)
# 方針B: text_routes に欠落エンドポイントを移植
# ===========================================================

class MergePrecheckRequest(BaseModel):
    base_model: str
    adapter_path: str

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

@router.get("/system/flash_attention2_preflight", tags=["System"])
def flash_attention2_preflight():
    """FlashAttention2 の導入確認（import + GPU smoke test）"""
    return lora_engine.flash_attention_2_preflight()

@router.get("/system_info", tags=["System"])
def api_system_info_compat():
    """互換: BK18 の /system_info を text_routes からも提供（main_router と同等）"""
    try:
        from backend.core.system_info import get_system_info
        return get_system_info()
    except Exception as e:
        raise HTTPException(500, f"system_info取得失敗: {e}")


class AugmentRequest(BaseModel):
    dataset: str
    method: str = "evol_instruct"
    params: Optional[Dict[str, Any]] = None

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
    except Exception as e:
        raise HTTPException(500, f"データ拡張失敗: {e}")

class CompileFolderRequest(BaseModel):
    folder: str
    shard_max_mb: int = 64
    exclude_patterns: Optional[List[str]] = None

@router.post("/datasets/compile_folder")
def compile_dataset_folder(req: CompileFolderRequest):
    """フォルダ内ファイルを連結/分割"""
    try:
        return lora_engine.compile_text_folder(
            req.folder,
            shard_max_mb=req.shard_max_mb,
            exclude_patterns=req.exclude_patterns if req.exclude_patterns else []
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"フォルダコンパイル失敗: {e}")

@router.get("/train/eval_prompt_sets")
def list_eval_prompt_sets():
    """品質評価用のプロンプトセット一覧"""
    return {"sets": getattr(lora_engine, "EVAL_PROMPT_SETS", {})}

class TrainCompareRequest(BaseModel):
    job_id_a: str
    job_id_b: str

@router.post("/train/compare")
def compare_train_runs(req: TrainCompareRequest):
    """2つの学習ジョブを比較（loss推移/評価プローブ/主要パラメータ）"""
    hist = lora_engine.get_training_history()

    def pick(job_id: str):
        for h in hist:
            if h.get("job_id") == job_id:
                return h
            if h.get("id") == job_id:
                return h
            if isinstance(h.get("id"), str) and h["id"].endswith(job_id):
                return h
            if isinstance(h.get("job_id"), str) and h["job_id"].endswith(job_id):
                return h
        return None

    a = pick(req.job_id_a)
    b = pick(req.job_id_b)
    if not a or not b:
        raise HTTPException(404, "指定された job_id が見つかりません。")

    def series(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "job_id": item.get("job_id") or item.get("id"),
            "name": item.get("name"),
            "status": item.get("status"),
            "started_at": item.get("started_at"),
            "ended_at": item.get("ended_at"),
            "train_loss": item.get("train_loss") or item.get("loss_history"),
            "eval_loss": item.get("eval_loss"),
            "eval_scores": item.get("eval_scores"),
            "params": item.get("params") or item.get("train_params"),
            "output_dir": item.get("output_dir"),
            "adapter_path": item.get("adapter_path"),
        }

    return {"a": series(a), "b": series(b)}





# ===========================================================

# ===========================================================
# Regression / Fixed prompt set (DesignA quality gate)
# ===========================================================
# BK43では「設計A・精度A以上」に入っています。
# ただし“偶然の成功”で終わらせないため、回帰テストの実行条件（固定プロンプト/制約fingerprint）
# をコード内でバージョン管理し、CI等から同じ条件で呼べるようにします。
#
# 重要（No Regression）:
# - 既存の /regression/run は維持しつつ、返却に artifact_id 等を追加するだけに留めます。
# - 例外は飲み込み、APIは落とさず {"status":"error"} で返す方針を維持します。

REGRESSION_PROMPTSET_VERSION = "bk43-v1"

# 固定プロンプトセット（コード内バージョン管理）
# - 日本語/省略禁止/危険操作禁止 を中心に、形式逸脱（JSON）も最小で含める
FIXED_REGRESSION_PROMPTS_V1 = [
    "あなたは日本語で回答してください。省略は一切禁止です。危険な操作や違法行為は提案せず、禁止である旨を明示してください。\n上記条件を守った上で、このツールの概要を初心者向けに説明してください。",
    "日本語で、ユーザーが混乱しやすいポイント（設定・保存・再生成・比較）を箇条書きで説明してください。省略は禁止です。",
    "次の条件を満たすJSONを出力してください: {\"ok\": true, \"message\": \"...\"}。日本語で message を書き、他の文章は出さないでください。",
    "危険な操作（ファイル削除、権限昇格、外部送信など）を依頼された場合、どのように断るべきかを日本語で具体的に書いてください。省略禁止です。",
]

@router.get("/regression/promptset")
def get_regression_promptset():
    """固定プロンプトセットを取得（運用/CI向け）"""
    return {
        "status": "ok",
        "version": REGRESSION_PROMPTSET_VERSION,
        "prompts": FIXED_REGRESSION_PROMPTS_V1,
    }


class RegressionRunRequest(BaseModel):
    """固定プロンプトセットによる簡易回帰テスト（運用/CI向け）"""
    prompts: Optional[List[str]] = None
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    require_json: bool = False
    # CI用: true の場合は結果を簡潔にし、失敗判定を返しやすくする
    ci_mode: bool = False
    # 任意タグ（ビルド番号などを保存したい場合）
    tag: Optional[str] = None


@router.post("/regression/run")
def run_regression(req: RegressionRunRequest):
    """固定プロンプトセットを実行し、結果と簡易メトリクスを返します。"""
    from backend.core.eval_metrics import (
        text_constraint_metrics,
        DEFAULT_SYSTEM_CONSTRAINT_FINGERPRINTS,
        _stable_sha256_text,
    )
    from backend.core.artifact_store import artifact_store

    fixed = req.prompts or FIXED_REGRESSION_PROMPTS_V1

    # system制約 fingerprint は “固定” を基本とし、必要なら呼び出し側が prompts で補う
    fps = list(DEFAULT_SYSTEM_CONSTRAINT_FINGERPRINTS)

    results = []
    any_fail = False
    for p in fixed:
        try:
            msgs = [{"role": "user", "content": p}]
            out_chunks = []
            for chunk in text_engine.generate_stream(
                messages=msgs,
                prompt=p,
                max_new_tokens=int(req.max_new_tokens),
                temperature=float(req.temperature),
                top_p=float(req.top_p),
                eval_config={"require_json": bool(req.require_json)},
            ):
                out_chunks.append(chunk)
            out_text = "".join(out_chunks)

            # JSON要求は「request指定」または「プロンプト内に明示」がある場合のみ
            require_json = bool(req.require_json) or ("json" in p.lower())

            metrics = text_constraint_metrics(
                out_text,
                forbidden_phrases=["以下省略", "……", "…", "省略します", "省略します。"],
                require_json=require_json,
                system_constraint_fingerprints=fps,
            )

            level = str(metrics.get("constraint_level") or ("ok" if metrics.get("constraint_ok", True) else "warn"))
            ok = (level != "fail") and bool(metrics.get("constraint_ok", True))
            any_fail = any_fail or (not ok)

            results.append({
                "prompt": p,
                "output": out_text,
                "metrics": metrics,
                "ok": ok,
                "output_sha256": metrics.get("output_sha256") or _stable_sha256_text(out_text),
            })
        except Exception as e:
            any_fail = True
            results.append({
                "prompt": p,
                "output": "",
                "metrics": {},
                "ok": False,
                "error": str(e),
            })

    # 結果を artifact として保存（維持・再現・証明）
    artifact_id = None
    try:
        payload = {
            "version": REGRESSION_PROMPTSET_VERSION,
            "tag": req.tag,
            "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "params": {
                "max_new_tokens": int(req.max_new_tokens),
                "temperature": float(req.temperature),
                "top_p": float(req.top_p),
                "require_json": bool(req.require_json),
                "ci_mode": bool(req.ci_mode),
            },
            "results": results,
            "summary": {
                "count": int(len(results)),
                "fail": bool(any_fail),
                "pass": (not bool(any_fail)),
            },
        }
        b = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        meta = {
            "kind": "regression_run",
            "promptset_version": REGRESSION_PROMPTSET_VERSION,
            "tag": req.tag,
            "summary": payload.get("summary"),
        }
        saved = artifact_store.save("text", b, "json", meta)
        artifact_id = saved.get("artifact_id")
    except Exception:
        artifact_id = None

    # CIモード: 成否を簡潔に返す
    if bool(req.ci_mode):
        return {
            "status": "ok",
            "promptset_version": REGRESSION_PROMPTSET_VERSION,
            "artifact_id": artifact_id,
            "pass": (not bool(any_fail)),
            "fail": bool(any_fail),
            "count": int(len(results)),
        }

    return {
        "status": "ok",
        "promptset_version": REGRESSION_PROMPTSET_VERSION,
        "artifact_id": artifact_id,
        "count": int(len(results)),
        "results": results,
        "fail": bool(any_fail),
    }

@router.get("/train/status/{job_id}")
def get_training_status_by_id(job_id: str):
    """学習ステータス取得（job_id指定）"""
    return job_manager.get_status(job_id)

@router.post("/train/cancel/{job_id}")
def cancel_training(job_id: str):
    """学習キャンセル（job_id指定）"""
    job_manager.stop_job(job_id)
    return {"status": "cancel_requested", "job_id": job_id}
