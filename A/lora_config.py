# -*- coding: utf-8 -*-
"""
lora_config.py
LoRA学習ツール専用の設定管理モジュール。
ディレクトリパス定義と、RTX 30xx/40xx (8GB~) 向けの学習パラメータデフォルト値を管理します。
(v14: Modular Architecture Support - Text/Image/Audio)
"""
import os
from pathlib import Path
from typing import Any, Dict

class LoraSettings:
    """
    LoRA Factoryの設定クラス
    """
    def __init__(self):
        # -------------------------------------------------
        # ディレクトリ設定 (カレントディレクトリ基準)
        # -------------------------------------------------
        self.base_dir = Path(__file__).resolve().parent
        
        # ルートディレクトリ
        self.models_root = self.base_dir / "models"
        self.dataset_root = self.base_dir / "datasets"
        self.output_root = self.base_dir / "lora_adapters"
        self.logs_dir = self.base_dir / "logs"
        self.runs_root = self.base_dir / "runs"
        self.artifacts_root = self.output_root / "artifacts"

        # -------------------------------------------------
        # 設計A: ジョブキュー（SQLite）
        # -------------------------------------------------
        # FastAPIプロセスと生成処理を分離するための永続キュー。
        # 既存の同期APIは維持しつつ、必要に応じて worker プロセスを起動して運用する。
        self.jobs_db_path = self.output_root / "jobs.db"

        # モダリティ別ディレクトリ定義
        # 将来的に Image/Audio を追加する際の基盤
        self.dirs = {
            "text": {
                "models": self.models_root / "text",
                "datasets": self.dataset_root / "text",
                "output": self.output_root / "text",
            },
            "image": {
                "models": self.models_root / "image",
                "datasets": self.dataset_root / "image",
                "output": self.output_root / "image",
                "aux": self.models_root / "image" / "aux",
                "controlnet": self.models_root / "image" / "controlnet",
                "ip_adapter": self.models_root / "image" / "ip_adapter",
                "vae": self.models_root / "image" / "vae",
                "refiner": self.models_root / "image" / "refiner",
            },
            "audio": {
                "models": self.models_root / "audio",
                "datasets": self.dataset_root / "audio",
                "output": self.output_root / "audio",
                "aux": self.models_root / "audio" / "aux",
            }
        }

        # -------------------------------------------------
        # 旧バージョン互換性 / デフォルト設定 (Textモード)
        # -------------------------------------------------
        # 既存のコードが self.models_dir 等を参照しているため、
        # デフォルトで "text" モードのパスを指すように設定
        # これにより、以降のフェーズで作成する text エンジンはここを参照すればOK
        
        self.models_dir = self.dirs["text"]["models"]
        self.dataset_dir = self.dirs["text"]["datasets"]
        self.output_dir = self.dirs["text"]["output"]

        # -------------------------------------------------
        # サーバー設定
        # -------------------------------------------------
        self.host = os.environ.get("LORA_HOST", "0.0.0.0")
        self.port = int(os.environ.get("LORA_PORT", "8081"))

        # -------------------------------------------------
        # データ拡張・錬成設定 (Data Augmentation)
        # -------------------------------------------------
        # 外部LLM API設定 (Evol-Instruct / Refinement用)
        self.aug_openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.aug_openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.aug_model_name = os.environ.get("AUG_MODEL_NAME", "gpt-3.5-turbo")
        
        # -------------------------------------------------
        # 意味的重複排除設定 (Semantic Deduplication)
        # -------------------------------------------------
        # 埋め込みモデル
        self.dedup_embedding_model = "intfloat/multilingual-e5-large"
        self.dedup_threshold = 0.95
        
        # Faiss使用フラグ (高速近似探索)
        # Faiss は Windows 環境では導入が難しい/非推奨なことが多いので、既定はOFFにしています。
        # （入っていれば自動的に使える任意機能：dedup_use_faiss=True にすると Faiss を優先します）
        self.dedup_use_faiss = False

        # ディレクトリの自動作成
        self._ensure_dirs()

    def _ensure_dirs(self):
        # ログディレクトリ作成
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        
        # モダリティごとのディレクトリを一括作成
        for mode, paths in self.dirs.items():
            for key, path in paths.items():
                path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # デフォルト学習パラメータ (RTX 3070/4070 8GB~ VRAM 最適化)
    # ※ Text (LLM) 用のデフォルトパラメータ
    # -------------------------------------------------
    @property
    def default_train_params(self) -> Dict[str, Any]:
        return {
            # --- 基本設定 ---
            "max_steps": 100,            # 短めのステップ数で動作確認推奨
            "save_steps": 50,
            "logging_steps": 1,
            "learning_rate": 2e-4,       # LoRAの標準的な学習率 (QLoRA)
            
            # --- メモリ節約設定 (8GB VRAM向け) ---
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4, # 実質バッチサイズ = 1 * 4 = 4
            
            # コンテキスト長 (VRAM消費に直結)
            "max_seq_length": 2048,

            # --- 量子化・精度設定 ---
            "fp16": True,
            "bf16": False,
            
            # --- LoRAパラメータ ---
            "lora_r": 8,                 # ランク
            "lora_alpha": 16,            # アルファ (通常 r * 2)
            "lora_dropout": 0.05,
            
            # --- ターゲット設定 ---
            "lora_target_mode": "all-linear",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

            # --- 高品質学習設定 (v12) ---
            
            # Loss Masking (Instruction Tuningの基本)
            # False = 回答部分のみ学習 (推奨: オウム返し防止)
            # True = 入力プロンプトも含めて学習
            "train_on_inputs": False,

            # Flash Attention 2 (Windows/Ampere以降推奨)
            # 学習速度とメモリ効率が大幅向上。環境構築が必要なためデフォルトはFalse。
            "use_flash_attention_2": False,

            # DoRA (Weight-Decomposed LoRA)
            # 精度向上のため推奨 (Trueで有効化)。
            "use_dora": False,

            # --- 検証データ ---
            "val_set_size": 0.05,

            # --- 再開・安定化設定 ---
            "resume_from_checkpoint": None,
            "neftune_noise_alpha": None,

            # --- オプティマイザ & スケジューラー ---
            "optim": "paged_adamw_8bit",
            
            # LR Scheduler
            "lr_scheduler_type": "cosine", 
            
            "warmup_ratio": 0.03,
            "gradient_checkpointing": True,
        }

# シングルトンインスタンス
settings = LoraSettings()

# 各所からインポートしやすい定数定義
# 今後は settings.dirs["text"]["models"] のように使うか、
# 以下のエイリアスを使用する（Textモードがデフォルト）
BASE_DIR = settings.base_dir
MODELS_DIR = settings.models_dir
DATASET_DIR = settings.dataset_dir
OUTPUT_DIR = settings.output_dir
LOGS_DIR = settings.logs_dir

# HuggingFace Hub のキャッシュディレクトリ等を固定したい場合は環境変数を設定
# os.environ["HF_HOME"] = str(BASE_DIR / "hf_cache")
