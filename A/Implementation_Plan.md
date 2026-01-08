# LoRA/モデル作成特化ツール 実装計画書（テキスト1.zip 改修版）
- 作成日時: 2026-01-04 14:54:11
- 前提: Windows 10/11, NVIDIA GPU（CUDA）, 日本語UI中心, ローカル運用（ネット無しでもUIが破綻しない）

## 1. ZIP監査（実装事実）: 全階層ツリー
以下は実ファイル構造（プロジェクトルート基準の相対パス）です。

```
__pycache__/
artifacts/
backend/
datasets/
logs/
lora_adapters/
models/
static/
#U4eee#U60f3#U74b0#U5883#U8d77#U52d5.bat
#U5909#U66f4#U70b9#U4e00#U89a7_#U7701#U7565#U306a#U3057.md
#U8d77#U52d5#U6642#U306b#U5fc5#U8981.md
lora_app.py
lora_config.py
lora_engine.py
lora_routes.py
lora_server.py
run_lora.bat
setup.py
setup_lora_env.py
setup_lora_env_2.py
train_job.py
zzz_test.txt
backend/__pycache__/
backend/core/
backend/engines/
backend/routes/
backend/workers/
backend/__init__.py
backend/core/__pycache__/
backend/core/__init__.py
backend/core/artifact_store.py
backend/core/errors.py
backend/core/eval_metrics.py
backend/core/job_manager.py
backend/core/job_spec.py
backend/core/regression_tests.py
backend/core/sqlite_queue.py
backend/core/system_info.py
backend/core/__pycache__/__init__.cpython-311.pyc
backend/core/__pycache__/artifact_store.cpython-311.pyc
backend/core/__pycache__/errors.cpython-311.pyc
backend/core/__pycache__/eval_metrics.cpython-311.pyc
backend/core/__pycache__/job_manager.cpython-311.pyc
backend/core/__pycache__/job_spec.cpython-311.pyc
backend/core/__pycache__/regression_tests.cpython-311.pyc
backend/core/__pycache__/sqlite_queue.cpython-311.pyc
backend/core/__pycache__/system_info.cpython-311.pyc
backend/engines/__pycache__/
backend/engines/__init__.py
backend/engines/audio.py
backend/engines/audio_post.py
backend/engines/audio_prosody.py
backend/engines/audio_streaming.py
backend/engines/audio_text_norm.py
backend/engines/base.py
backend/engines/image.py
backend/engines/image_metadata.py
backend/engines/image_pipelines.py
backend/engines/image_presets.py
backend/engines/text.py
backend/engines/__pycache__/__init__.cpython-311.pyc
backend/engines/__pycache__/audio.cpython-311.pyc
backend/engines/__pycache__/audio_post.cpython-311.pyc
backend/engines/__pycache__/audio_prosody.cpython-311.pyc
backend/engines/__pycache__/audio_streaming.cpython-311.pyc
backend/engines/__pycache__/audio_text_norm.cpython-311.pyc
backend/engines/__pycache__/base.cpython-311.pyc
backend/engines/__pycache__/image.cpython-311.pyc
backend/engines/__pycache__/image_metadata.cpython-311.pyc
backend/engines/__pycache__/image_pipelines.cpython-311.pyc
backend/engines/__pycache__/image_presets.cpython-311.pyc
backend/engines/__pycache__/text.cpython-311.pyc
backend/routes/__pycache__/
backend/routes/__init__.py
backend/routes/audio_routes.py
backend/routes/common_routes.py
backend/routes/image_routes.py
backend/routes/main_router.py
backend/routes/text_routes.py
backend/routes/__pycache__/__init__.cpython-311.pyc
backend/routes/__pycache__/audio_routes.cpython-311.pyc
backend/routes/__pycache__/common_routes.cpython-311.pyc
backend/routes/__pycache__/image_routes.cpython-311.pyc
backend/routes/__pycache__/main_router.cpython-311.pyc
backend/routes/__pycache__/text_routes.cpython-311.pyc
backend/workers/__pycache__/
backend/workers/__init__.py
backend/workers/job_worker.py
backend/workers/train_audio.py
backend/workers/train_image.py
backend/workers/train_text.py
backend/workers/__pycache__/__init__.cpython-311.pyc
backend/workers/__pycache__/job_worker.cpython-311.pyc
backend/workers/__pycache__/train_audio.cpython-311.pyc
backend/workers/__pycache__/train_image.cpython-311.pyc
backend/workers/__pycache__/train_text.cpython-311.pyc
backend/__pycache__/__init__.cpython-311.pyc
datasets/.lineage/
datasets/audio/
datasets/image/
datasets/text/
datasets/audio/#U30c6#U30b9#U30c81/
datasets/image/#U30c6#U30b9#U30c81/
datasets/text/.lineage/
datasets/text/#U30e1#U30e2.txt
datasets/text/#U732b.txt
logs/constraints_torch_2.5.1_cu121_20260104_224952.txt
logs/setup_lora_env_20260104_224651.log
logs/setup_lora_env_optional_next_SAFE_TORCH_NO_XFORMERS_20260104_224951.log
lora_adapters/audio/
lora_adapters/image/
lora_adapters/text/
lora_adapters/jobs.db
models/audio/
models/image/
models/text/
models/audio/_aux/
models/audio/aux/
models/image/_aux/
models/image/aux/
models/image/controlnet/
models/image/ip_adapter/
models/image/refiner/
models/image/vae/
static/js/
static/index.html
static/style.css
static/js/modules/
static/js/api.js
static/js/app.js
static/js/modules/audio_ui.js
static/js/modules/image_ui.js
static/js/modules/text_ui.js
__pycache__/lora_app.cpython-311.pyc
__pycache__/lora_config.cpython-311.pyc
__pycache__/lora_engine.cpython-311.pyc
```

## 2. エントリポイント（起動手順の断定）
- サーバー起動（推奨）: run_lora.bat を実行する
- 直接起動: py lora_server.py（Uvicornで FastAPI を起動）
- UI: static/index.html を FastAPI が配信（ブラウザで http://127.0.0.1:8081 へアクセス）

## 3. 既存機能の実動確認ポイント（断定）
- モデル管理: HuggingFace Repo ID からモデルを models/text 配下へダウンロードできる
- データセット管理: datasets/text 配下へJSONL等を配置し、一覧表示・アップロードできる
- データ錬成: クリーニング、重複排除、Evol-Instruct等の錬成機能を実行できる
- 学習: QLoRA/PEFT 方式の学習ジョブを開始できる（ジョブID方式、ログ保存あり）
- 学習履歴: 過去ジョブの結果を一覧表示できる
- 推論（検証）: 学習したアダプタを読み込み、チャットで簡易検証できる

## 4. 本改修で追加した内容（差分粒度）
### 4.1 UI/UX
- 学習ウィザード（最短導線）タブを追加
  - モデル保存先、データセット保存先、LoRA出力先、ログ保存先の絶対パスを表示
  - それぞれに「フォルダを開く」ボタンを配置
  - 「モデル」「データセット」「学習」への移動ボタンを配置
- 既存タブ構成・API呼び出しは非破壊で維持

### 4.2 API
- GET /utils/paths を追加（既知ディレクトリの絶対パス一覧）
- POST /utils/open_path の key に lora_adapters_root, artifacts_root を追加

### 4.3 セットアップ
- setup.py を追加（既存 setup_lora_env.py / setup_lora_env_2.py の統合入口）
  - logs/setup_YYYYMMDD_HHMMSS.log へログ保存
  - Python 3.10 以上をチェック

## 5. API設計（主要エンドポイントとJSON例）
### 5.1 ジョブ開始
POST /train/start
```json
{
  "base_model": "models/text/<model_dir>",
  "dataset": "datasets/text/<dataset_file>",
  "output_name": "my_lora",
  "preset": "standard"
}
```
レスポンス:
```json
{"job_id":"20260104_123456_abcd"}
```

### 5.2 ステータス
GET /train/status/{job_id}
レスポンス例:
```json
{"status":"running","step":12,"max_steps":100,"loss":1.234,"log_file":"logs/train_<JOB_ID>.log"}
```

### 5.3 キャンセル
POST /train/cancel/{job_id}
```json
{"ok":true}
```

### 5.4 既知パスの取得
GET /utils/paths
```json
{
  "base": "C:/PATH/TO/LoRA_Factory",
  "logs": "C:/PATH/TO/LoRA_Factory/logs",
  "text": {
    "models": "C:/PATH/TO/LoRA_Factory/models/text",
    "datasets": "C:/PATH/TO/LoRA_Factory/datasets/text",
    "output": "C:/PATH/TO/LoRA_Factory/lora_adapters/text"
  }
}
```

## 6. ジョブ/ログ/保存構造（再現性）
- logs/train_<job_id>.log: 学習ログ
- lora_adapters/text/<output_name>/: 出力されたLoRAアダプタ
- （既存）runs/<job_id>/config.json: 学習設定の保存（再実行の基盤）

## 7. 依存関係方針（Windows安定優先）
- torch は既存環境を壊さない方針（setup_lora_env_2.py は torch 固定 constraints を使用）
- xformers / flash-attn / triton は必須にしない（入っていれば使う任意機能）

## 8. テスト観点（手動確認）
- 起動: run_lora.bat でUIが開き、JS例外が出ない
- 学習ウィザード: パス表示が更新され、各フォルダが explorer で開く
- 学習開始: /train/start が job_id を返し、/train/status が running を返す
- キャンセル: /train/cancel が受理され、status が canceled または stopped になる
- 出力: lora_adapters/text に出力が作成され、履歴に表示される
- 削除: UIから削除できる（安全確認あり）
