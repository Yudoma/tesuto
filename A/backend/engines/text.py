# -*- coding: utf-8 -*-
"""
backend/engines/text.py
テキスト（LLM）モダリティ専用のエンジン。
学習ジョブのパラメータ構築、推論モデルの管理、データセット操作（解析・錬成）を担当します。
"""
import os
import sys
import json
import time
import shutil
import threading
import subprocess
import re
import math
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# 設定と共通基盤のインポート
from lora_config import settings
from backend.core.dataset_report import build_dataset_report
from backend.core.job_manager import job_manager
from backend.engines.base import BaseEngine


# ============================================================
# 設計A: system / developer / user の固定テンプレ（日本語運用・再現性）
# ============================================================

SYSTEM_FIXED_TEMPLATE = (
    "あなたは日本語で回答するアシスタントです。\n"
    "推測で断定せず、不明な点は不明と述べます。\n"
    "ユーザーの要求が危険/違法/プライバシー侵害に該当する場合は安全な代替案を提示します。\n"
)

DEVELOPER_FIXED_TEMPLATE = (
    "【開発者指示】\n"
    "・既存仕様/既存I/Fを壊さず、後方互換を維持すること。\n"
    "・例外で落とさず fail-soft を優先すること。\n"
)

# Alchemy用ライブラリ (存在確認付きインポート)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import openai
    import faiss
except ImportError:
    pass

try:
    from langdetect import detect
except ImportError:
    detect = None


# =============================================================================
# Text: Prompt templates / Presets (Design A)
# =============================================================================
from typing import Generator  # keep for older imports

TEXT_GEN_PRESETS = [
    {
        "id": "jp_default",
        "label": "日本語・標準",
        "description": "日本語の一般会話/作業支援向け（安定）",
        "defaults": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1, "max_tokens": 512},
        "system_prompt": "あなたは日本語で丁寧に回答するアシスタントです。事実は推測せず、不明な点は不明と述べます。",
        "developer_prompt": "",
    },
    {
        "id": "jp_precise",
        "label": "日本語・精密",
        "description": "推測を抑え、根拠重視で回答（業務向け）",
        "defaults": {"temperature": 0.2, "top_p": 0.8, "repetition_penalty": 1.15, "max_tokens": 768},
        "system_prompt": "あなたは日本語で簡潔かつ正確に回答するアシスタントです。根拠のない断定を避け、必要なら確認質問をします。",
        "developer_prompt": "",
    },
    {
        "id": "jp_creative",
        "label": "日本語・創作",
        "description": "文章生成/アイデア出し向け（創造性重視）",
        "defaults": {"temperature": 0.95, "top_p": 0.95, "repetition_penalty": 1.05, "max_tokens": 768},
        "system_prompt": "あなたは日本語の創作支援アシスタントです。多様な案を提示し、表現の幅を広げます。",
        "developer_prompt": "",
    },
]

def get_text_presets():
    return TEXT_GEN_PRESETS

def find_text_preset(preset_id: str):
    for p in TEXT_GEN_PRESETS:
        if p.get("id") == preset_id:
            return p
    return None



def _extract_constraint_fingerprints(msgs):
    """system メッセージから、要約/剪定後に保持すべき「重要制約」の指紋を抽出します（fail-soft）。

    方針:
    - 先頭32文字だけだと、system が長い場合に重要な制約が落ちる可能性がある。
    - ここでは「禁止/必須/絶対/厳守」などの制約ワード周辺を優先的に抽出し、
      それ以外は先頭断片を保険として残す。
    """
    fps = []
    try:
        for mm in msgs or []:
            try:
                role = (mm.get("role") or "").strip()
                if role != "system":
                    continue
                c = (mm.get("content") or "").strip()
                if not c:
                    continue
                # 正規化（比較耐性）
                norm = " ".join(c.split())

                # 1) 制約っぽい行を優先抽出（日本語想定）
                #    例: "禁止: xxx", "必ず xxx", "絶対に xxx", "厳守", "No Regression" 等
                lines = [ln.strip() for ln in c.splitlines() if ln.strip()]
                constraint_lines = []
                for ln in lines:
                    if re.search(r"(禁止|必須|必ず|絶対|厳守|してはなら|してはいけ|No\s*Regression|作り直し禁止|省略禁止)", ln, flags=re.IGNORECASE):
                        constraint_lines.append(ln)
                for ln in constraint_lines[:16]:
                    s = " ".join(ln.split())
                    fps.append(s[:48])

                # 2) 保険: system 全体の先頭断片
                fps.append(norm[:48])
            except Exception:
                continue

        # 重複除去（順序維持）
        seen = set()
        out = []
        for fp in fps:
            if not fp:
                continue
            if fp in seen:
                continue
            seen.add(fp)
            out.append(fp)
        return out[:64]
    except Exception:
        return []


def _validate_summary_constraints_detail(summary_text: str, removed_msgs):
    """要約文に重要な system 制約が残っているかを判定し、詳細を返します（fail-soft）。"""
    try:
        if not summary_text:
            return {"ok": False, "reason": "empty_summary", "matched": 0, "total": 0, "fingerprints": []}
        fps = _extract_constraint_fingerprints(removed_msgs)
        if not fps:
            return {"ok": True, "reason": "no_constraints", "matched": 0, "total": 0, "fingerprints": []}

        s = " ".join(str(summary_text).split())
        matched = []
        missing = []
        for fp in fps:
            if fp and fp in s:
                matched.append(fp)
            else:
                missing.append(fp)

        # 判定: 従来は「半分以上」だったが、制約系の行が含まれている場合はより厳密にする。
        total = len(fps)
        ratio_ok = (len(matched) >= max(1, (total + 1) // 2))

        # 制約行（禁止/必須/厳守など）が存在する場合、最低1つは必ず含む
        critical = [fp for fp in fps if re.search(r"(禁止|必須|必ず|絶対|厳守|No\s*Regression|省略禁止)", fp, flags=re.IGNORECASE)]
        critical_ok = True
        if critical:
            critical_ok = any(fp in matched for fp in critical)

        ok = bool(ratio_ok and critical_ok)
        reason = "ok" if ok else ("missing_critical" if not critical_ok else "low_match_ratio")
        return {
            "ok": ok,
            "reason": reason,
            "matched": int(len(matched)),
            "total": int(total),
            "matched_fps": matched[:16],
            "missing_fps": missing[:16],
        }
    except Exception:
        return {"ok": True, "reason": "fail_soft", "matched": 0, "total": 0}

def _validate_summary_constraints(summary_text: str, removed_msgs) -> bool:
    """互換: bool を返す（詳細は _validate_summary_constraints_detail を使用）。"""
    try:
        d = _validate_summary_constraints_detail(summary_text, removed_msgs)
        return bool(d.get("ok", True))
    except Exception:
        return True


class TextEngine(BaseEngine):
    """
    Text (LLM) モダリティの処理を行うエンジンクラス。
    """
    def __init__(self):
        self.inference_model = None
        self.inference_tokenizer = None
        self.inference_lock = threading.Lock()
        
        # ワーカー・スクリプトのパス
        self.worker_script = settings.base_dir / "backend" / "workers" / "train_text.py"

    # =========================================================================
    # 1. Training Job Management (BaseEngine Implementation)
    # =========================================================================

    def start_training(self, base_model: str, dataset: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        テキスト学習ジョブを開始する。
        """
        # パス解決
        # base_model がパスかモデル名かを判定（簡易的に models/text 配下をチェック）
        base_model_path = settings.dirs["text"]["models"] / base_model
        if not base_model_path.exists():
             # パスが存在しないならHuggingFace IDや絶対パスとみなす（そのまま渡す）
             # ただし、ローカルにあるならそれを優先
             base_model_path = Path(base_model)
        
        # データセットパス
        dataset_path = settings.dirs["text"]["datasets"] / dataset
        
        # 出力ディレクトリ
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_name = f"{base_model}_{timestamp}"
        output_dir = settings.dirs["text"]["output"] / job_name

        # コマンド構築
        cmd = [
            sys.executable,
            str(self.worker_script),
            "--base_model_path", str(base_model_path),
            "--dataset_path", str(dataset_path),
            "--output_dir", str(output_dir),
        ]

        # パラメータをコマンドライン引数に変換
        # params dict: { "learning_rate": 2e-4, "use_flash_attention_2": True, ... }
        for key, val in params.items():
            if val is None:
                continue
            if isinstance(val, bool):
                if val:
                    cmd.append(f"--{key}")  # フラグとして追加
            elif isinstance(val, (list, dict)):
                # リストや辞書はJSON文字列化して渡す（受け側でパース）
                cmd.extend([f"--{key}", json.dumps(val, ensure_ascii=False)])
            else:
                cmd.extend([f"--{key}", str(val)])

        # JobManager経由で実行
        job_info = job_manager.start_job(
            cmd=cmd,
            params={
                "base_model": base_model,
                "dataset": dataset,
                "params": params
            },
            cwd=settings.base_dir,
            log_prefix="train_text"
        )

        # データセット検査結果を runs/<job_id>/dataset_report.json に保存（UIで再表示）
        try:
            job_id = job_info.get("job_id") or job_info.get("id") or job_info.get("jobId")
            if job_id:
                run_dir = settings.runs_root / str(job_id)
                run_dir.mkdir(parents=True, exist_ok=True)
                rep = build_dataset_report("text", Path(str(dataset_path)).resolve())
                (run_dir / "dataset_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


        
        # 履歴ファイルへの追記（開始状態）
        self._append_history(job_info["job_id"], base_model, dataset, params.get("max_steps", 0), "running", params=params, log_file=job_info.get("log_file"))
        
        return job_info

    def stop_training(self) -> Dict[str, str]:
        job_manager.stop_job()
        return {"status": "stopped"}

    def get_training_status(self) -> Dict[str, Any]:
        status = job_manager.get_status()
        
        # 完了または失敗時に履歴を更新
        if status["status"] in ["completed", "failed", "stopped"]:
            self._update_history_status(status["job_id"], status["status"])
            
        return status

    
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
        # steps は履歴にある場合は尊重
        if target.get("steps") and "max_steps" not in params:
            params["max_steps"] = target.get("steps")
        return self.start_training(model, dataset, params)


    def get_training_history(self) -> Dict[str, Any]:
        history_file = settings.logs_dir / "history.json"
        if not history_file.exists():
            return {"history": []}
        try:
            hist = json.loads(history_file.read_text(encoding="utf-8"))
            # 新しい順にソート
            hist.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return {"history": hist}
        except Exception:
            return {"history": []}

    def _append_history(self, job_id, model, dataset, steps, status, params: Dict[str, Any] | None = None, log_file: str | None = None):
        history_file = settings.logs_dir / "history.json"
        hist = []
        if history_file.exists():
            try:
                hist = json.loads(history_file.read_text(encoding="utf-8"))
            except: pass
        
        entry = {
            "id": job_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "dataset": dataset,
            "steps": steps,
            "status": status,
            "final_loss": None,
            "params": params or {},
            "log_file": log_file
        }
        hist.append(entry)
        history_file.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")

    def _update_history_status(self, job_id, status):
        # 履歴ファイルの該当IDのステータスを更新
        history_file = settings.logs_dir / "history.json"
        if not history_file.exists(): return
        
        try:
            hist = json.loads(history_file.read_text(encoding="utf-8"))
            updated = False
            for item in hist:
                if item.get("id") == job_id:
                    # 既に完了状態なら更新しない（多重更新防止）
                    if item["status"] not in ["running", "idle"]:
                        continue
                        
                    item["status"] = status
                    
                    # ログから最終Lossを取得してみる
                    job_logs = job_manager.current_job["logs"]
                    loss = None
                    for line in reversed(job_logs):
                        if '"loss":' in line:
                            try:
                                data = json.loads(line)
                                if "loss" in data:
                                    loss = data["loss"]
                                    break
                            except: pass
                    item["final_loss"] = loss
                    updated = True
                    break
            
            if updated:
                history_file.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
        except: pass

    # =========================================================================
    # 2. Inference / Verification (BaseEngine Implementation)
    # =========================================================================

    def load_inference_model(self, base_model: str, adapter_path: Optional[str] = None) -> Dict[str, Any]:
        """
        推論用モデルをロードする。
        """
        with self.inference_lock:
            # メモリ解放
            self.unload_inference_model()
            
            # パス解決
            base_model_path = settings.dirs["text"]["models"] / base_model
            if not base_model_path.exists():
                base_model_path = Path(base_model) # 外部パス

            print(f"[TextEngine] Loading inference model: {base_model_path}")
            
            try:
                # トークナイザー
                self.inference_tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
                
                # モデル
                # VRAM節約のため 4bit ロードをデフォルトとする
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                
                self.inference_model = AutoModelForCausalLM.from_pretrained(
                    str(base_model_path),
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # LoRAアダプタ適用
                if adapter_path:
                    adapter_full_path = settings.dirs["text"]["output"] / adapter_path
                    if not adapter_full_path.exists():
                         # models直下にある場合も考慮
                         adapter_full_path = settings.dirs["text"]["models"] / adapter_path
                    
                    if adapter_full_path.exists():
                        print(f"[TextEngine] Loading adapter: {adapter_full_path}")
                        from peft import PeftModel
                        self.inference_model = PeftModel.from_pretrained(
                            self.inference_model,
                            str(adapter_full_path)
                        )
                    else:
                        print(f"[TextEngine] Adapter not found: {adapter_path} (skipping)")

                self.inference_model.eval()
                return {"status": "loaded", "base_model": base_model}

            except Exception as e:
                print(f"[TextEngine] Load failed: {e}")
                self.unload_inference_model()
                raise e

    def unload_inference_model(self) -> Dict[str, str]:
        with self.inference_lock:
            if self.inference_model is not None:
                del self.inference_model
                self.inference_model = None
            if self.inference_tokenizer is not None:
                del self.inference_tokenizer
                self.inference_tokenizer = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {"status": "unloaded"}

    def is_inference_model_loaded(self) -> bool:
        return self.inference_model is not None

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        developer_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        preset_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        repetition_penalty: float = 1.1,
        top_p: float = 0.9,
        eval_config: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """ストリーミング生成を行うジェネレータ。

        互換維持:
          - 既存呼び出し (prompt, system_prompt, temperature, max_tokens...) はそのまま動作します。
        設計A:
          - system / developer / user を分離し、履歴(history)を安全に結線します。
          - コンテキスト超過時は古い履歴から縮退し、必要ならユーザー入力も切り詰めます（日本語優先）。
          - 生成プリセット(preset_id)で推論設定を固定し再現性を上げます。
        """
        if not self.inference_model or not self.inference_tokenizer:
            yield "モデルがロードされていません (Model not loaded)."
            return

        # -----------------------------
        # 1) Preset override (optional)
        # -----------------------------
        try:
            if preset_id:
                preset = find_text_preset(preset_id)
                if preset:
                    d = preset.get("defaults") or {}
                    temperature = float(d.get("temperature", temperature))
                    top_p = float(d.get("top_p", top_p))
                    repetition_penalty = float(d.get("repetition_penalty", repetition_penalty))
                    max_tokens = int(d.get("max_tokens", max_tokens))
                    if not system_prompt and preset.get("system_prompt"):
                        system_prompt = str(preset["system_prompt"])
                    if not developer_prompt and preset.get("developer_prompt"):
                        developer_prompt = str(preset["developer_prompt"])
        except Exception:
            # preset が壊れていても推論を止めない（互換維持）
            pass

        # -----------------------------
        # 2) Build chat messages
        # -----------------------------
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": SYSTEM_FIXED_TEMPLATE + str(system_prompt)})
        if developer_prompt:
            # OpenAI互換の "developer" が無いモデルも多いため system と区別しつつ role=system として併記
            messages.append({"role": "system", "content": DEVELOPER_FIXED_TEMPLATE + str(developer_prompt)})
        if history:
            for m in history:
                try:
                    r = str(m.get("role", "")).strip()
                    c = str(m.get("content", ""))
                    if not r or c is None:
                        continue
                    if r not in ("system", "user", "assistant"):
                        # 想定外は user 扱い（安全側）
                        r = "user"
                    messages.append({"role": r, "content": c})
                except Exception:
                    continue
        messages.append({"role": "user", "content": str(prompt)})

        # -----------------------------
        # 3) Apply chat template or fallback
        # -----------------------------
        def _render_messages_to_text(msgs: List[Dict[str, str]]) -> str:
            tok = self.inference_tokenizer
            if hasattr(tok, "chat_template") and getattr(tok, "chat_template", None):
                try:
                    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    pass
            # Fallback: deterministic plain text format (JP friendly)
            out = []
            for mm in msgs:
                role = mm.get("role", "user")
                content = mm.get("content", "")
                if role == "system":
                    out.append(f"### 指示\n{content}")
                elif role == "assistant":
                    out.append(f"### アシスタント\n{content}")
                else:
                    out.append(f"### ユーザー\n{content}")
            out.append("### アシスタント\n")
            return "\n\n".join(out)

        text_input = _render_messages_to_text(messages)

        # -----------------------------
        # 4) Context overflow handling
        # -----------------------------
        try:
            tok = self.inference_tokenizer
            # max_position_embeddings が取れないモデルもあるため安全に
            model_max = int(getattr(getattr(self.inference_model, "config", None), "max_position_embeddings", 0) or 0)
            if model_max <= 0:
                model_max = int(getattr(tok, "model_max_length", 0) or 0)
            # 0/inf/very large のときは制限なし扱い
            if model_max and model_max < 100000:
                # 入力側に割り当てる上限（生成分も確保）
                max_input_tokens = max(256, model_max - max(64, int(max_tokens)))
                enc = tok(text_input, return_tensors=None, add_special_tokens=False)
                in_len = len(enc.get("input_ids", [])) if isinstance(enc, dict) else len(tok.encode(text_input))
                if in_len > max_input_tokens:
                    # 古い履歴から削りつつ再レンダリング
                    pruned = messages[:]
                    removed_msgs = []
                    summary_text = None
                    summary_detail = None

                    def _summarize_removed_msgs(msgs):
                        # 決定論的（非モデル依存）の簡易要約。日本語優先。
                        # 重要: fail-soft（例外は飲み込む）
                        try:
                            parts = []
                            for mm in msgs:
                                role = (mm.get('role') or 'user')
                                c = (mm.get('content') or '').strip()
                                if not c:
                                    continue
                                if role == 'assistant':
                                    parts.append('A: ' + c)
                                elif role == 'system':
                                    parts.append('S: ' + c)
                                else:
                                    parts.append('U: ' + c)
                            text = '\n'.join(parts)
                            # 長すぎる場合は末尾優先で切る
                            limit = 1400
                            if len(text) > limit:
                                text = text[-limit:]
                            summary = '【これまでの要約】\n' + text
                            # 評価フェーズ: system 制約が欠落していないか検証
                            if not _validate_summary_constraints(summary, msgs):
                                return ''
                            return summary
                        except Exception:
                            return ''

                    # system/developer は残すため、先頭2件まで保護
                    protected = 0
                    for mm in pruned:
                        if mm.get("role") == "system":
                            protected += 1
                        else:
                            break
                    # 履歴が無い場合はユーザー入力を切る
                    while True:
                        # 現在の候補メッセージをテキスト化して長さを確認
                        text_input = _render_messages_to_text(pruned)
                        enc = tok(text_input, return_tensors=None, add_special_tokens=False)
                        in_len = len(enc.get("input_ids", [])) if isinstance(enc, dict) else len(tok.encode(text_input, add_special_tokens=False))
                        if in_len <= max_input_tokens:
                            # 収まったタイミングで、削除済み履歴があれば要約を注入（ただし長さ超過はさせない）
                            if removed_msgs:
                                summary = _summarize_removed_msgs(removed_msgs)
                                summary_text = summary
                                summary_detail = _validate_summary_constraints_detail(summary, removed_msgs)
                                if summary:
                                    # 既存の text_input（要約なし）に対して、要約分の許容トークン数を見積もる
                                    # ※ 余白を少し残して、生成側（max_tokens）に食い込まないようにする
                                    buffer_tokens = 8
                                    allowed = max_input_tokens - in_len - buffer_tokens
                                    if allowed > 0:
                                        try:
                                            sum_ids = tok.encode(summary, add_special_tokens=False)
                                        except Exception:
                                            sum_ids = None
                                        if sum_ids is not None and len(sum_ids) > allowed:
                                            # 先頭を優先して deterministic に切り詰める
                                            summary = tok.decode(sum_ids[:allowed], skip_special_tokens=True).strip()
                                        if summary:
                                            # system メッセージとして保護領域の直後に要約を注入
                                            pruned.insert(protected, {"role": "system", "content": summary})
                                            # 要約を含めた最終入力を再構築（ここが重要：text_input を更新しないと要約が推論に反映されない）
                                            text_input2 = _render_messages_to_text(pruned)
                                            try:
                                                in_len2 = len(tok.encode(text_input2, add_special_tokens=False))
                                            except Exception:
                                                enc2 = tok(text_input2, return_tensors=None, add_special_tokens=False)
                                                in_len2 = len(enc2.get("input_ids", [])) if isinstance(enc2, dict) else len(tok.encode(text_input2, add_special_tokens=False))
                                            if in_len2 <= max_input_tokens:
                                                text_input = text_input2
                                                in_len = in_len2
                                            else:
                                                # 要約で超過する場合は、要約を外してこのまま剪定を続行する
                                                pruned.pop(protected)
                            messages = pruned
                            break
                        # まず protected 以降の最古メッセージを削る
                        if len(pruned) > protected + 1:
                            removed_msgs.append(pruned.pop(protected))  # oldest after system/developer
                            continue
                        # それでも超過なら、最後の user 入力を切る
                        if pruned and pruned[-1].get("role") == "user":
                            u = pruned[-1].get("content", "")
                            # 末尾優先で切ると指示が消えるので、先頭を残して末尾を削る
                            pruned[-1]["content"] = u[: max(200, int(len(u) * 0.6))]
                            continue
                        break
        except Exception:
            pass

        # -----------------------------
        # 5) Generate with streamer
        # -----------------------------
        try:
            from transformers import TextIteratorStreamer
            import threading

            inputs = self.inference_tokenizer(
                text_input,
                return_tensors="pt",
                truncation=False,
            )
            if hasattr(self.inference_model, "device"):
                try:
                    inputs = {k: v.to(self.inference_model.device) for k, v in inputs.items()}
                except Exception:
                    pass

            streamer = TextIteratorStreamer(self.inference_tokenizer, skip_prompt=True, skip_special_tokens=True)

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                do_sample=True if float(temperature) > 0 else False,
            )

            # 生成ログ（再現性のため）
            try:
                from backend.core.artifact_store import ArtifactStore
                store = ArtifactStore()
            except Exception:
                store = None

            meta = {
                "modality": "text",
                "preset_id": preset_id,
                "system_prompt": system_prompt,
                "developer_prompt": developer_prompt,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "max_new_tokens": int(max_tokens),
            }

            # --- Context pruning / summarization meta（再現性・精度保証） ---
            try:
                if 'removed_msgs' in locals() and removed_msgs:
                    meta.setdefault("context_pruning", {})
                    meta["context_pruning"].update({
                        "removed_count": int(len(removed_msgs)),
                        "summary_injected": bool(summary_text),
                        "summary_text": str(summary_text) if summary_text else "",
                        "summary_detail": summary_detail or {},
                    })
            except Exception:
                pass

            # --- 評価設定（必要なら meta に残す） ---
            try:
                if eval_config:
                    meta.setdefault("eval_config", {})
                    # パスワード等の機微情報を含めない前提だが、念のため長すぎる値は切らない（No summary）
                    meta["eval_config"].update({k: v for k, v in dict(eval_config).items() if k not in ("api_key", "token")})
            except Exception:
                pass

            buf_parts: List[str] = []

            thread = threading.Thread(target=self.inference_model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()

            for new_text in streamer:
                if new_text:
                    buf_parts.append(new_text)
                    yield new_text

            thread.join()

            # 保存（失敗しても返却は完了させる）
            if store is not None:
                try:
                    out_text = "".join(buf_parts)
                    data = out_text.encode("utf-8", errors="replace")
                    
                    # --- lightweight quality metrics（fail-soft） ---
                    try:
                        from backend.core.eval_metrics import text_constraint_metrics
                        banned_phrases = ["以下省略", "……", "…", "省略します", "省略します。"]
                        # ユーザー/ルートからの追加
                        extra_banned = []
                        if eval_config and isinstance(eval_config, dict):
                            bp = eval_config.get("banned_phrases") or eval_config.get("eval_score_banned_phrases")
                            if isinstance(bp, str) and bp.strip():
                                extra_banned = [x.strip() for x in bp.split("\n") if x.strip()]
                        banned_phrases = banned_phrases + extra_banned

                        require_json = False
                        if eval_config and isinstance(eval_config, dict):
                            require_json = bool(eval_config.get("require_json") or eval_config.get("eval_score_require_json_if_prompt_mentions_json"))
                        # prompt が JSON を要求していると推測できる場合も保険でオン
                        if (not require_json) and (re.search(r"\bjson\b", str(prompt or ""), flags=re.IGNORECASE) or "JSON" in str(prompt or "")):
                            require_json = True

                        metrics = text_constraint_metrics(
                            out_text,
                            forbidden_phrases=banned_phrases,
                            require_json=require_json,
                        )
                        meta.setdefault("metrics", {})
                        meta["metrics"].update({"text_constraints": metrics})
                    except Exception:
                        pass

                    store.save("text", data, "txt", meta)
                except Exception:
                    pass

        except Exception as e:
            yield f"\n[エラー: {str(e)}]"
    def analyze_tokens(self, dataset_name: str, base_model: str, max_seq_length: int = 2048, is_folder: bool = False) -> Dict[str, Any]:
        """
        データセットのトークン数を解析する。
        """
        # トークナイザーロード
        try:
            model_path = settings.dirs["text"]["models"] / base_model
            if not model_path.exists(): model_path = base_model
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        except Exception as e:
            return {"error": f"Tokenizer load failed: {e}"}

        target_path = settings.dirs["text"]["datasets"] / dataset_name
        files = []
        if is_folder and target_path.is_dir():
            files = sorted([f for f in target_path.glob("**/*") if f.is_file() and f.suffix in [".txt", ".json", ".jsonl"]])
        else:
            files = [target_path]

        total_tokens = 0
        total_samples = 0
        details = []
        token_counts = []

        for f in files:
            if not f.exists(): continue
            try:
                text_content = f.read_text(encoding="utf-8", errors="ignore")
                
                # jsonl/json対応
                samples = []
                if f.suffix == ".jsonl":
                    for line in text_content.splitlines():
                        if line.strip(): samples.append(line) # 簡易カウント
                elif f.suffix == ".json":
                    try:
                        d = json.loads(text_content)
                        if isinstance(d, list): samples = d
                        else: samples = [text_content]
                    except: samples = [text_content]
                else:
                    samples = [text_content]

                for s in samples:
                    # 実際はプロンプトフォーマット後の長さが重要だが、ここでは生テキストで概算
                    s_str = str(s)
                    tokens = len(tokenizer.encode(s_str, add_special_tokens=False))
                    token_counts.append(tokens)
                    total_tokens += tokens
                    total_samples += 1
                    
                    if tokens > max_seq_length:
                        details.append({
                            "file": f.name,
                            "tokens": tokens,
                            "exceeds": True
                        })
            except Exception:
                pass

        avg = sum(token_counts) / max(1, len(token_counts))
        median = sorted(token_counts)[len(token_counts)//2] if token_counts else 0
        max_val = max(token_counts) if token_counts else 0

        return {
            "is_folder": is_folder,
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "avg_tokens": int(avg),
            "median_tokens": median,
            "max_tokens": max_val,
            "details": details[:100] # 多すぎると重いので制限
        }

    def deduplicate_dataset(self, dataset_name: str, threshold: float = 0.95, model_name: str = None) -> Dict[str, Any]:
        """
        意味的重複排除 (Semantic Deduplication) を実行する。
        """
        if "SentenceTransformer" not in sys.modules:
            return {"error": "Required libraries (sentence-transformers, etc.) not installed."}

        target_file = settings.dirs["text"]["datasets"] / dataset_name
        if not target_file.exists(): return {"error": "File not found"}

        # 行単位で読み込み
        lines = []
        try:
            content = target_file.read_text(encoding="utf-8")
            # jsonl か txt か
            if target_file.suffix == ".jsonl":
                lines = [line.strip() for line in content.splitlines() if line.strip()]
            else:
                # txtの場合、空行区切りなどを想定するか、行ごとにするか。ここでは行ごと。
                lines = [line.strip() for line in content.splitlines() if line.strip()]
        except Exception as e:
            return {"error": f"Read error: {e}"}

        if len(lines) < 2:
            return {"message": "Not enough lines to deduplicate."}

        # Embedding計算
        emb_model_name = model_name or settings.dedup_embedding_model
        print(f"Loading embedding model: {emb_model_name}")
        model = SentenceTransformer(emb_model_name)
        
        # テキスト抽出 (JSONLの場合は "text" や "output" フィールドなどを結合して判定に使う)
        texts_to_embed = []
        for line in lines:
            if target_file.suffix == ".jsonl":
                try:
                    obj = json.loads(line)
                    # 簡易的に全値を結合
                    t = " ".join([str(v) for v in obj.values()])
                    texts_to_embed.append(t)
                except:
                    texts_to_embed.append(line)
            else:
                texts_to_embed.append(line)

        print(f"Encoding {len(texts_to_embed)} lines...")
        embeddings = model.encode(texts_to_embed, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalize
        faiss.normalize_L2(embeddings)

        # 重複検出
        remove_indices = set()
        
        if settings.dedup_use_faiss and "faiss" in sys.modules:
            print("Using Faiss for similarity search...")
            d = embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            
            # 自身を含む上位2件を検索 (1位は自分自身)
            D, I = index.search(embeddings, 2)
            
            for i in range(len(embeddings)):
                if i in remove_indices: continue
                # 相手が自分より後ろのインデックスで、かつ類似度が高い場合
                neighbor_idx = I[i][1] # 0は自分
                score = D[i][1]
                
                if neighbor_idx != -1 and score >= threshold:
                    # インデックスが大きい方を削除候補にする（順序依存だが簡易的）
                    if i < neighbor_idx:
                        remove_indices.add(neighbor_idx)
                    else:
                        remove_indices.add(i)
        else:
            # SciKit-Learn fallback (slow for large data)
            print("Using Cosine Similarity (slow)...")
            sim_matrix = cosine_similarity(embeddings)
            for i in range(len(embeddings)):
                if i in remove_indices: continue
                for j in range(i + 1, len(embeddings)):
                    if j in remove_indices: continue
                    if sim_matrix[i][j] >= threshold:
                        remove_indices.add(j)

        # 保存
        new_lines = [line for i, line in enumerate(lines) if i not in remove_indices]
        
        new_filename = f"{target_file.stem}_dedup{target_file.suffix}"
        new_path = target_file.parent / new_filename
        new_path.write_text("\n".join(new_lines), encoding="utf-8")

        return {
            "original_count": len(lines),
            "removed_count": len(remove_indices),
            "remaining_count": len(new_lines),
            "output_file": new_filename
        }

    def clean_dataset(self, dataset_name: str, remove_duplicates: bool = True, min_length: int = 10, filter_lang: str = None) -> Dict[str, Any]:
        """
        簡易クリーニング (重複行削除、短文削除、言語判定)
        """
        target_path = settings.dirs["text"]["datasets"] / dataset_name
        if not target_path.exists(): return {"error": "File not found"}
        
        original_content = target_path.read_text(encoding="utf-8")
        lines = original_content.splitlines()
        
        cleaned = []
        seen = set()
        
        stats = {
            "original_count": len(lines),
            "removed_duplicates": 0,
            "removed_short": 0,
            "removed_lang": 0,
            "cleaned_count": 0
        }

        for line in lines:
            line_str = line.strip()
            if not line_str: continue
            
            # 1. 短文削除
            if len(line_str) < min_length:
                stats["removed_short"] += 1
                continue
            
            # 2. 言語判定
            if filter_lang and detect:
                try:
                    lang = detect(line_str)
                    if lang != filter_lang:
                        stats["removed_lang"] += 1
                        continue
                except:
                    pass # 判定不能はスルー（残す）
            
            # 3. 重複削除 (完全一致)
            if remove_duplicates:
                if line_str in seen:
                    stats["removed_duplicates"] += 1
                    continue
                seen.add(line_str)
            
            cleaned.append(line_str)

        stats["cleaned_count"] = len(cleaned)
        
        new_filename = f"{target_path.stem}_clean{target_path.suffix}"
        new_path = target_path.parent / new_filename
        new_path.write_text("\n".join(cleaned), encoding="utf-8")
        
        return {
            "stats": stats,
            "original_file": dataset_name,
            "cleaned_file": new_filename
        }

# シングルトンインスタンス
text_engine = TextEngine()
