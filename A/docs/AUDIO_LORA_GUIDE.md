# 音声（TTS/VC）LoRA 作成ガイド（UI追加版）

このビルドでは「音声 → 学習」タブが **学習（LoRA/モデル作成）** として表示され、
「学習タイプ（TTS LoRA / VC LoRA）」と「LoRA Rank/Alpha」を入力できます。

## 重要（現状の挙動）
- バックエンドは既存の `/api/audio/train/start` を利用します。
- `train_type/lora_rank/lora_alpha` は `params` に渡されます（既存互換は維持）。
- 実際のLoRA学習は、利用している音声エンジン（XTTS / GPT-SoVITS / RVC など）に依存します。
  必要なら「カスタム学習コマンド」に学習CLIを指定してください。

## 使い方
1. 音声モード → 学習（LoRA/モデル作成）
2. データセットを選択
3. 学習タイプと LoRA Rank/Alpha を設定
4. 必要に応じて `カスタム学習コマンド` を設定
5. 「開始」

## 出力先（目安）
- TTS LoRA: `outputs/lora/audio/tts`
- VC LoRA: `outputs/lora/audio/vc`
