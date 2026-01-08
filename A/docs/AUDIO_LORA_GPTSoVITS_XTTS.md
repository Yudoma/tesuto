# 音声LoRA（VC: GPT-SoVITS / TTS: XTTS 話者適応）最終ビルド

このビルドは以下を満たします：
- 自動 clone はしない（ユーザーが事前に clone 済み前提）
- clone されていない場合、UI に明確に表示しプリセットを無効化
- XTTS は「話者適応（finetune）」のみを UI に出す
- UI の開始ボタン 1 回で XTTS の学習が走る

## 事前準備（必須）
ツールのルート（テキスト2/）で以下を配置してください。

### VC: GPT-SoVITS
```bat
git clone https://github.com/RVC-Boss/GPT-SoVITS third_party\GPT-SoVITS
```

### TTS: XTTS (Coqui TTS)
```bat
git clone https://github.com/coqui-ai/TTS third_party\XTTS
```

## XTTS データセット形式（LJSpeech）
期待する構成：
```
datasets/audio_tts/
  wavs/
    000001.wav
    ...
  metadata.csv
```
`metadata.csv` は `|` 区切りの 3 列（最低 3 列）を想定します：
```
000001|こんにちは|こんにちは
```
※ wavs/ の中のファイル名と一致させてください（拡張子は行に書かない運用が一般的です）。

## UI の場所
- 音声 → 学習（LoRA/モデル作成）
  - 学習タイプ: TTS LoRA（XTTS） / VC LoRA（GPT-SoVITS）
  - 開始ボタンで学習開始

## 追加API
- `GET  /api/capabilities` : clone 状態を返します（UI がプリセット無効化に使用）
- `POST /api/audio/train/tts-lora` : XTTS 話者適応（finetune）を開始
- `POST /api/audio/train/vc-lora`  : GPT-SoVITS 側の学習開始（既存ブリッジ）
