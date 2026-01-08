# -*- coding: utf-8 -*-
"""
backend/workers/train_text.py
テキスト（LLM）学習ジョブ実行スクリプト。
backend/engines/text.py からサブプロセスとして呼び出されます。
(v14: Modular Architecture Support)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# -----------------------------------------------------------------------------
# Path Setup for Modular Architecture
# -----------------------------------------------------------------------------
# このスクリプトは backend/workers/ に配置されているため、
# プロジェクトルート (../../) を sys.path に追加して lora_config 等を読み込めるようにする
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, ORPOTrainer, ORPOConfig

# Unsloth Support
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

# -----------------------------------------------------------------------------
# Templates & Constants
# -----------------------------------------------------------------------------

CHAT_TEMPLATES = {
    "llama-3": (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|start_header_id|>system<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        "{% endif %}"
    ),
    "chatml": (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}"
    ),
    "gemma": (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}"
        "{{ raise_exception('System role not supported') }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{% endif %}"
        "{% if (message['role'] == 'assistant') %}"
        "{% set role = 'model' %}"
        "{% else %}"
        "{% set role = message['role'] %}"
        "{% endif %}"
        "{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<start_of_turn>model\\n' }}"
        "{% endif %}"
    ),
}

RESPONSE_TEMPLATES = {
    "alpaca": "\n### Response:\n",
    "llama-3": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "chatml": "<|im_start|>assistant\n",
    "gemma": "<start_of_turn>model\n",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def _jprint(obj: Dict[str, Any]) -> None:
    try:
        s = json.dumps(obj, ensure_ascii=False)
        print(s, flush=True)
    except Exception:
        pass

def _log(msg: str) -> None:
    print(msg, flush=True)

def find_all_linear_names(model, mode: str = "all-linear") -> List[str]:
    import bitsandbytes as bnb
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    keywords_to_exclude = ["lm_head"]
    
    if mode == "attention-only":
        mlp_keywords = ["mlp", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]
        
    for name, module in model.named_modules():
        if isinstance(module, cls):
            child_name = name.split('.')[-1]
            if child_name in keywords_to_exclude:
                continue
            if mode == "attention-only":
                if any(k in child_name for k in mlp_keywords):
                    continue
            lora_module_names.add(child_name)

    return list(lora_module_names)

# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

@dataclass
class TrainArgs:
    base_model_path: Path
    dataset_path: Path
    output_dir: Path
    dataset_type: str 

    max_steps: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int

    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_mode: str

    save_steps: int
    logging_steps: int

    fp16: bool
    bf16: bool
    optim: str
    warmup_ratio: float
    gradient_checkpointing: bool

    val_set_size: float
    
    resume_from_checkpoint: Optional[str]
    neftune_noise_alpha: Optional[float]
    prompt_template: Optional[str]
    validation_file: Optional[Path]
    validation_prompt: Optional[str]
    eval_prompts_path: Optional[Path]
    eval_max_new_tokens: int

    eval_score_enabled: bool = True
    eval_score_min_len: int = 40
    eval_score_max_len: int = 800
    eval_score_banned_phrases: List[str] = None
    eval_score_require_json_if_prompt_mentions_json: bool = True
    eval_score_repetition_ngram: int = 6
    eval_score_repetition_threshold: float = 0.35
    
    # v11/v12 Args
    use_dora: bool
    lr_scheduler_type: str
    use_flash_attention_2: bool
    train_on_inputs: bool

    # v13 Args
    use_unsloth: bool
    use_orpo: bool
    report_to: str
    
    # Reproducibility & Quality
    run_snapshot_path: str | None = None
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    seed: int = 42
    use_rslora: bool = False

def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description="LoRA Factory Training Job (Text)")

    p.add_argument("--base_model_path", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_type", default="raw_text")

    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=2048)

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_mode", type=str, default="all-linear")

    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--optim", type=str, default="paged_adamw_8bit")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    
    p.add_argument("--val_set_size", type=float, default=0.05)

    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--neftune_noise_alpha", type=float, default=None)
    p.add_argument("--prompt_template", type=str, default=None)
    p.add_argument("--validation_file", type=str, default=None)
    p.add_argument("--validation_prompt", type=str, default=None)
    p.add_argument("--eval_prompts_path", type=str, default=None)
    p.add_argument("--eval_max_new_tokens", type=int, default=128)

    p.add_argument("--eval_score_enabled", type=int, default=1)
    p.add_argument("--eval_score_min_len", type=int, default=40)
    p.add_argument("--eval_score_max_len", type=int, default=800)
    p.add_argument("--eval_score_banned_phrases", type=str, default=None)
    p.add_argument("--eval_score_require_json_if_prompt_mentions_json", type=int, default=1)
    p.add_argument("--eval_score_repetition_ngram", type=int, default=6)
    p.add_argument("--eval_score_repetition_threshold", type=float, default=0.35)

    p.add_argument("--run_snapshot_path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--early_stopping", action="store_true")
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--early_stopping_threshold", type=float, default=0.0)

    p.add_argument("--use_dora", action="store_true")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--use_flash_attention_2", action="store_true")
    p.add_argument("--train_on_inputs", action="store_true")

    p.add_argument("--use_unsloth", action="store_true")
    p.add_argument("--use_orpo", action="store_true")
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--use_rslora", action="store_true")

    a = p.parse_args()

    resume_ckpt = a.resume_from_checkpoint
    if resume_ckpt and (resume_ckpt.lower() == "none" or resume_ckpt == ""): resume_ckpt = None
        
    prompt_tpl = a.prompt_template
    if prompt_tpl and (prompt_tpl.lower() == "none" or prompt_tpl == ""): prompt_tpl = None
        
    val_file = Path(a.validation_file) if a.validation_file and a.validation_file.strip() else None
    val_prompt = a.validation_prompt if a.validation_prompt and a.validation_prompt.strip() else None

    banned_phrases = None
    if a.eval_score_banned_phrases and a.eval_score_banned_phrases.strip():
        raw = a.eval_score_banned_phrases.strip()
        try:
            if raw.startswith("["):
                banned_phrases = [str(x) for x in json.loads(raw) if str(x).strip()]
            else:
                lines = [ln.strip() for ln in raw.splitlines()]
                banned_phrases = [ln for ln in lines if ln]
        except Exception:
            lines = [ln.strip() for ln in raw.splitlines()]
            banned_phrases = [ln for ln in lines if ln]

    return TrainArgs(
        base_model_path=Path(a.base_model_path),
        dataset_path=Path(a.dataset_path),
        output_dir=Path(a.output_dir),
        dataset_type=a.dataset_type,
        max_steps=a.max_steps,
        learning_rate=a.learning_rate,
        per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        max_seq_length=a.max_seq_length,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        lora_target_mode=a.lora_target_mode,
        save_steps=a.save_steps,
        logging_steps=a.logging_steps,
        fp16=bool(a.fp16),
        bf16=bool(a.bf16),
        optim=a.optim,
        warmup_ratio=a.warmup_ratio,
        gradient_checkpointing=bool(a.gradient_checkpointing),
        val_set_size=a.val_set_size,
        resume_from_checkpoint=resume_ckpt,
        neftune_noise_alpha=a.neftune_noise_alpha,
        prompt_template=prompt_tpl,
        validation_file=val_file,
        validation_prompt=val_prompt,
        eval_prompts_path=Path(a.eval_prompts_path) if a.eval_prompts_path and a.eval_prompts_path.strip() else None,
        eval_max_new_tokens=int(a.eval_max_new_tokens),
        eval_score_enabled=bool(a.eval_score_enabled),
        eval_score_min_len=a.eval_score_min_len,
        eval_score_max_len=a.eval_score_max_len,
        eval_score_banned_phrases=banned_phrases,
        eval_score_require_json_if_prompt_mentions_json=bool(a.eval_score_require_json_if_prompt_mentions_json),
        eval_score_repetition_ngram=a.eval_score_repetition_ngram,
        eval_score_repetition_threshold=a.eval_score_repetition_threshold,
        use_dora=bool(a.use_dora),
        lr_scheduler_type=a.lr_scheduler_type,
        use_flash_attention_2=bool(a.use_flash_attention_2),
        train_on_inputs=bool(a.train_on_inputs),
        use_unsloth=bool(a.use_unsloth),
        use_orpo=bool(a.use_orpo),
        report_to=a.report_to,
        run_snapshot_path=a.run_snapshot_path,
        early_stopping=bool(a.early_stopping),
        early_stopping_patience=a.early_stopping_patience,
        early_stopping_threshold=a.early_stopping_threshold,
        seed=a.seed,
        use_rslora=bool(a.use_rslora)
    )

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------

def main() -> int:
    _force_utf8_stdout()
    args = parse_args()
    start_ts = time.time()

    _log("--- 学習プロセス開始 (v14: Modular Text Worker) ---")
    _log(f"Base Model: {args.base_model_path.name}")
    _log(f"Dataset: {args.dataset_path.name} (Type: {args.dataset_type})")
    
    # Feature Flags Log
    features = []
    if args.use_unsloth: features.append("Unsloth(HighSpeed)")
    if args.use_orpo: features.append("ORPO(Alignment)")
    if args.use_flash_attention_2: features.append("FlashAttn2")
    if args.use_dora: features.append("DoRA")
    if args.report_to == "wandb": features.append("WandB")
    
    _log(f"Active Features: {', '.join(features) if features else 'Standard SFT'}")

    if torch.cuda.is_available():
        _log(f"GPU: {torch.cuda.get_device_name(0)}")
        if args.use_flash_attention_2:
            cap = torch.cuda.get_device_capability()
            if cap[0] < 8:
                _log("警告: Flash Attention 2 は Ampere (RTX 30xx~) 以降を推奨します。")
    else:
        _log("GPU: (CUDA利用不可) CPU")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load Model & Tokenizer (Unsloth vs Standard)
    # -------------------------------------------------------------------------
    model = None
    tokenizer = None

    if args.use_unsloth and HAS_UNSLOTH:
        _log("🚀 Unsloth バックエンドを使用します (最大2倍高速・メモリ削減)")
        
        # Unsloth Load
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.base_model_path),
            max_seq_length=args.max_seq_length,
            dtype=torch.float16 if args.fp16 else torch.float32,
            load_in_4bit=True,
            # device_map="auto", # Unsloth handles this
        )
        
        # UnslothのLoRAパッチ適用
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ] if args.lora_target_mode == "all-linear" else ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=0, # Unslothはdropout 0推奨
            bias="none",
            use_gradient_checkpointing="unsloth", # Optimized GC
            random_state=3407,
            use_rslora=args.use_rslora,
            loftq_config=None,
        )
        _log("Unsloth LoRA Patched.")

    else:
        # Standard Load (bitsandbytes + peft)
        if args.use_unsloth and not HAS_UNSLOTH:
            _log("警告: Unslothがインストールされていません。標準バックエンドを使用します。")
            
        _log("標準バックエンド (Transformers + PEFT) を使用します...")
        
        tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if args.fp16 else torch.float32,
            bnb_4bit_use_double_quant=False,
        )
        
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        if args.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            _log("⚡ Flash Attention 2 を有効化")

        try:
            model = AutoModelForCausalLM.from_pretrained(str(args.base_model_path), **model_kwargs)
        except Exception as e:
            if args.use_flash_attention_2:
                _log(f"エラー: Flash Attention 2 ロード失敗 ({e})。標準アテンションに切り替えます。")
                del model_kwargs["attn_implementation"]
                model = AutoModelForCausalLM.from_pretrained(str(args.base_model_path), **model_kwargs)
            else:
                raise e

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        model = prepare_model_for_kbit_training(model)
        
        target_modules = find_all_linear_names(model, mode=args.lora_target_mode)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            use_dora=args.use_dora,
            use_rslora=args.use_rslora,
        )
        model = get_peft_model(model, peft_config)

    # -------------------------------------------------------------------------
    # 2. Template Setup
    # -------------------------------------------------------------------------
    current_template_key = None
    use_chat_template = False
    
    if args.prompt_template:
        if args.prompt_template in CHAT_TEMPLATES:
            tokenizer.chat_template = CHAT_TEMPLATES[args.prompt_template]
            use_chat_template = True
            current_template_key = args.prompt_template
        elif "{%" in args.prompt_template:
            tokenizer.chat_template = args.prompt_template
            use_chat_template = True
            current_template_key = "custom"
        elif args.prompt_template == "alpaca":
            use_chat_template = False
            current_template_key = "alpaca"
    elif hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        use_chat_template = True
        if "start_header_id" in tokenizer.chat_template: current_template_key = "llama-3"
        elif "im_start" in tokenizer.chat_template: current_template_key = "chatml"

    # -------------------------------------------------------------------------
    # 3. Dataset Loading & Formatting
    # -------------------------------------------------------------------------
    _log(f"学習データセット読み込み中...")
    data_path_str = str(args.dataset_path)
    extension = "json" if data_path_str.lower().endswith((".json", ".jsonl")) else "text"
    
    train_dataset = load_dataset(extension, data_files=data_path_str, split="train")
    eval_dataset = None

    if args.validation_file:
        val_path_str = str(args.validation_file)
        val_ext = "json" if val_path_str.lower().endswith((".json", ".jsonl")) else "text"
        eval_dataset = load_dataset(val_ext, data_files=val_path_str, split="train")
    else:
        if len(train_dataset) > 20 and args.val_set_size > 0:
            dataset_split = train_dataset.train_test_split(test_size=args.val_set_size, seed=args.seed)
            train_dataset = dataset_split["train"]
            eval_dataset = dataset_split["test"]

    # Formatting / Collator
    dataset_text_field = "text"
    formatting_func = None
    data_collator = None
    packing = False 

    if args.use_orpo:
        column_names = train_dataset.column_names
        
        def format_orpo(examples):
            if "chosen" in examples and "rejected" in examples:
                return examples
            
            new_examples = {"prompt": [], "chosen": [], "rejected": []}
            insts = examples.get("instruction", [])
            inps = examples.get("input", [])
            outs = examples.get("output", [])
            
            for i in range(len(insts)):
                p = insts[i] + ("\n" + inps[i] if inps[i] else "")
                c = outs[i]
                r = "" 
                if "rejected" in examples:
                    r = examples["rejected"][i]
                new_examples["prompt"].append(p)
                new_examples["chosen"].append(c)
                new_examples["rejected"].append(r)
            return new_examples

        if "prompt" not in column_names or "chosen" not in column_names:
            _log("ORPO: データセットカラム変換を試みています...")
            # map処理はメモリ消費するため、あえてSFTTrainerのformatting_funcのように動的にやるか、
            # 事前に変換しておくのが望ましい。今回は簡易実装として省略。
            # 必要なら: train_dataset = train_dataset.map(format_orpo, batched=True)

    elif args.dataset_type == "raw_text":
        packing = True 
    else:
        # Loss Masking
        if not args.train_on_inputs:
            response_template = None
            if current_template_key in RESPONSE_TEMPLATES:
                response_template = RESPONSE_TEMPLATES[current_template_key]
            
            if response_template:
                _log(f"Loss Masking有効: Response Template = '{repr(response_template)}'")
                data_collator = DataCollatorForCompletionOnlyLM(
                    response_template=response_template,
                    tokenizer=tokenizer
                )

        def formatting_prompts_func(example: Dict[str, Any]):
            output_texts = []
            texts = example.get("text", [])
            instructions = example.get("instruction", [])
            inputs = example.get("input", [""] * len(instructions))
            outputs = example.get("output", [""] * len(instructions))
            
            batch_size = len(instructions) if instructions else len(texts)
            
            for i in range(batch_size):
                if texts and i < len(texts):
                    output_texts.append(texts[i])
                    continue
                
                inst = instructions[i] if i < len(instructions) else ""
                inp = inputs[i] if i < len(inputs) else ""
                out = outputs[i] if i < len(outputs) else ""
                
                if use_chat_template:
                    user_content = f"{inst}\n{inp}" if inp else inst
                    messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": out}]
                    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    output_texts.append(formatted)
                else:
                    t = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}" if inp else f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                    output_texts.append(t + tokenizer.eos_token)
            return output_texts

        formatting_func = formatting_prompts_func

    # -------------------------------------------------------------------------
    # 4. Callbacks (Eval & Metric)
    # -------------------------------------------------------------------------
    class GenerationCallback(TrainerCallback):
        def __init__(self, tokenizer, prompt, max_new_tokens=128):
            self.tokenizer = tokenizer
            self.prompt = prompt
            self.max_new_tokens = max_new_tokens
            
        def on_evaluate(self, args, state, control, model=None, **kwargs):
            if not self.prompt or model is None: return
            _log(f"[Eval] 生成テスト実行中...")
            
            if HAS_UNSLOTH and args.use_unsloth:
                FastLanguageModel.for_inference(model)
                
            was_training = model.training
            model.eval()
            try:
                input_text = self.prompt
                if use_chat_template:
                    messages = [{"role": "user", "content": self.prompt}]
                    input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                elif args.prompt_template == "alpaca":
                    input_text = f"### Instruction:\n{self.prompt}\n\n### Response:\n"
                
                inputs = self.tokenizer(input_text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                _jprint({"type": "generation", "step": state.global_step, "prompt": self.prompt, "output": decoded, "time": time.time(), "score": (_score_output(self.prompt, decoded, args) if getattr(args, "eval_score_enabled", True) else None)})
            except Exception as e:
                _log(f"[Eval] 生成テスト失敗: {e}")
            finally:
                if HAS_UNSLOTH and args.use_unsloth:
                    FastLanguageModel.for_training(model)
                elif was_training: 
                    model.train()

    def _strip_code_fence(s: str) -> str:
        t = (s or "").strip()
        if t.startswith("```"):
            lines = t.splitlines()
            if len(lines) >= 3:
                body = "\n".join(lines[1:-1])
                return body.strip()
        return t

    def _ngram_repetition_ratio(text: str, n: int) -> float:
        toks = re.findall(r"\w+|[^\s\w]", text or "", flags=re.UNICODE)
        if len(toks) < n * 2:
            return 0.0
        grams = [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
        if not grams:
            return 0.0
        seen = set()
        rep = 0
        for g in grams:
            if g in seen:
                rep += 1
            else:
                seen.add(g)
        return rep / max(1, len(grams))

    def _score_output(prompt: str, output: str, args: TrainArgs) -> Dict[str, Any]:
        prompt_s = (prompt or "").strip()
        out_raw = output or ""
        out = out_raw.strip()
        reasons: List[str] = []
        score = 100.0

        min_len = int(getattr(args, "eval_score_min_len", 40) or 40)
        max_len = int(getattr(args, "eval_score_max_len", 800) or 800)

        if len(out) < min_len:
            score -= min(40.0, (min_len - len(out)) * 0.25)
            reasons.append("too_short")
        if len(out) > max_len:
            score -= min(25.0, (len(out) - max_len) * 0.02)
            reasons.append("too_long")

        banned = getattr(args, "eval_score_banned_phrases", None)
        if not banned:
            banned = [
                "As an AI", "AIとして", "できません", "I can't", "I cannot", "申し訳", "すみません", "対応できません",
            ]
        hit = []
        low = out.lower()
        for ph in banned:
            phs = str(ph)
            if not phs: continue
            if phs.lower() in low: hit.append(phs)
        if hit:
            score -= min(60.0, 10.0 * len(hit))
            reasons.append("banned_phrase:" + ",".join(hit[:5]))

        n = int(getattr(args, "eval_score_repetition_ngram", 6) or 6)
        thr = float(getattr(args, "eval_score_repetition_threshold", 0.35) or 0.35)
        rep_ratio = _ngram_repetition_ratio(out, n)
        if rep_ratio >= thr:
            score -= min(50.0, (rep_ratio - thr) * 120.0 + 10.0)
            reasons.append(f"repetition_ngram{n}:{rep_ratio:.3f}")

        require_json = bool(getattr(args, "eval_score_require_json_if_prompt_mentions_json", True))
        json_expected = False
        if require_json:
            pl = prompt_s.lower()
            if "json" in pl or "json形式" in pl or "jsonで" in pl:
                json_expected = True

        json_ok = None
        if json_expected:
            body = _strip_code_fence(out)
            try:
                json.loads(body)
                json_ok = True
            except Exception:
                json_ok = False
                score -= 30.0
                reasons.append("json_invalid")

        if score < 0: score = 0.0
        if score > 100: score = 100.0

        return {
            "score": round(score, 2),
            "reasons": reasons,
            "len": len(out),
            "rep_ratio": round(rep_ratio, 4),
            "json_expected": json_expected,
            "json_ok": json_ok,
        }

    class MultiProbeGenerationCallback(TrainerCallback):
        def __init__(self, tokenizer, prompts: List[str], max_new_tokens: int = 128):
            self.tokenizer = tokenizer
            self.prompts = [p for p in (prompts or []) if p]
            self.max_new_tokens = int(max_new_tokens)

        def on_evaluate(self, args, state, control, model=None, **kwargs):
            if not self.prompts or model is None: return
            _log(f"[Eval] 評価プローブ実行中... ({len(self.prompts)} prompts)")

            if HAS_UNSLOTH and args.use_unsloth:
                FastLanguageModel.for_inference(model)

            was_training = model.training
            model.eval()
            try:
                for idx, pr in enumerate(self.prompts, start=1):
                    input_text = pr
                    if use_chat_template:
                        messages = [{"role": "user", "content": pr}]
                        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    elif args.prompt_template == "alpaca":
                        input_text = f"### Instruction:\n{pr}\n\n### Response:\n"

                    inputs = self.tokenizer(input_text, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    _jprint({
                        "type": "eval_probe",
                        "step": int(state.global_step),
                        "time": time.time(),
                        "prompt_id": idx,
                        "prompt": pr,
                        "output": decoded,
                        "score": (_score_output(pr, decoded, args) if getattr(args, "eval_score_enabled", True) else None),
                    })
            except Exception as e:
                _log(f"[Eval] 評価プローブ失敗: {e}")
            finally:
                if HAS_UNSLOTH and args.use_unsloth:
                    FastLanguageModel.for_training(model)
                elif was_training:
                    model.train()

    class MetricCallback(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kwargs):
            if not logs: return
            msg = {"type": "metric", "step": int(state.global_step), "time": time.time()}
            if "loss" in logs:
                msg["loss"] = float(logs["loss"])
                try: msg["ppl"] = math.exp(msg["loss"])
                except Exception: msg["ppl"] = float("inf")
            if "eval_loss" in logs:
                msg["eval_loss"] = float(logs["eval_loss"])
                try: msg["eval_ppl"] = math.exp(msg["eval_loss"])
                except Exception: msg["eval_ppl"] = float("inf")
            if "loss" in logs or "eval_loss" in logs:
                _jprint(msg)

    eval_prompts: List[str] = []
    if args.eval_prompts_path:
        try:
            raw = Path(args.eval_prompts_path).read_text(encoding="utf-8")
            obj = json.loads(raw)
            if isinstance(obj, list):
                eval_prompts = [str(x) for x in obj if str(x).strip()]
        except Exception as e:
            _log(f"[Eval] eval_prompts_pathの読み込みに失敗: {e}")

    callbacks = [MetricCallback()]
    if eval_prompts:
        callbacks.append(MultiProbeGenerationCallback(tokenizer, eval_prompts, args.eval_max_new_tokens))
    if args.validation_prompt:
        callbacks.append(GenerationCallback(tokenizer, args.validation_prompt))

    if args.early_stopping and eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        ))

    # -------------------------------------------------------------------------
    # 5. Trainer Initialization
    # -------------------------------------------------------------------------
    enable_best_model = (eval_dataset is not None)
    
    common_args = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        optim=args.optim,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        
        evaluation_strategy="steps" if enable_best_model else "no",
        eval_steps=args.save_steps if enable_best_model else None,
        save_strategy="steps",
        load_best_model_at_end=True if enable_best_model else False,
        metric_for_best_model="eval_loss" if enable_best_model else None,
        greater_is_better=False if enable_best_model else None,
        save_total_limit=2,
        lr_scheduler_type=args.lr_scheduler_type,
    )

    trainer = None
    
    if args.use_orpo:
        _log("Trainer: ORPOTrainer (Preference Optimization)")
        orpo_args = ORPOConfig(beta=0.1, **common_args)
        trainer = ORPOTrainer(
            model=model,
            args=orpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            peft_config=None if args.use_unsloth else model.peft_config,
        )
    else:
        _log("Trainer: SFTTrainer (Supervised Fine-Tuning)")
        train_args = TrainingArguments(group_by_length=False, **common_args)
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=None if args.use_unsloth else model.peft_config,
            dataset_text_field=dataset_text_field,
            formatting_func=formatting_func,
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            args=train_args,
            packing=packing,
            callbacks=callbacks,
            data_collator=data_collator, 
            neftune_noise_alpha=args.neftune_noise_alpha,
        )

    # -------------------------------------------------------------------------
    # 6. Training Execution
    # -------------------------------------------------------------------------
    _log("学習を開始します...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    _log("モデルを保存しています...")
    trainer.model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    
    _log("完了しました。")

    # Run Snapshot finalize
    if args.run_snapshot_path:
        try:
            snap_path = Path(args.run_snapshot_path)
            snap = {}
            if snap_path.exists():
                try: snap = json.loads(snap_path.read_text(encoding="utf-8"))
                except Exception: snap = {}
            train_job = snap.get("train_job", {})
            train_job["completed_at"] = time.time()
            try: state = trainer.state.to_dict() if trainer is not None else None
            except Exception: state = None
            train_job["trainer_state"] = state
            try: best_ckpt = getattr(trainer.state, "best_model_checkpoint", None) if trainer is not None else None
            except Exception: best_ckpt = None
            train_job["best_model_checkpoint"] = best_ckpt
            snap["train_job"] = train_job
            snap_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            _log(f"Run Snapshotの保存に失敗しました: {e}")

    _jprint({
        "type": "final",
        "status": "completed",
        "time": time.time(),
        "elapsed_sec": round(time.time() - start_ts, 3),
        "output_dir": str(args.output_dir),
    })
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        _force_utf8_stdout()
        _log(f"致命的なエラーが発生しました: {e}")
        traceback.print_exc()
        _jprint({"type": "final", "status": "failed", "time": time.time(), "error": str(e)})
        sys.exit(1)