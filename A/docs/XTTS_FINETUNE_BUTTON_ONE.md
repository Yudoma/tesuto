# XTTS v2 話者適応（finetune）— UIからボタン一発

## 前提（自動cloneしません）
- `third_party/XTTS` に `coqui-ai/TTS` を clone して配置してください。
- `third_party/GPT-SoVITS` に `RVC-Boss/GPT-SoVITS` を clone して配置してください（VC学習を使う場合）。

## 使い方
1. UI → 音声 → 学習（LoRA/モデル作成）
2. 学習タイプで「TTS：XTTS v2（話者適応 finetune）」を選択
3. データセット（datasets/audio/<name>）を選択
4. 開始

## 何が行われるか
- 前処理: 音声をスライスし、faster-whisper で文字起こしして `prepared/train.list` を作ります。
- XTTS: `prepared/train.list` から `prepared/xtts_dataset/metadata.csv` を自動生成し、coqui-ai/TTS の finetune を起動します。

## 追加引数
TTS側の引数差異で失敗する場合は、UI の「XTTS 追加引数」で調整できます。
