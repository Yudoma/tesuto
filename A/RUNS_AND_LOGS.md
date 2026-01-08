# runs と logs の構造

## logs
- logs/setup_YYYYMMDD_HHMMSS.log: セットアップ実行ログ
- logs/train_<job_id>.log: 学習ログ
- logs/server.log（存在する場合）: サーバー実行ログ

## lora_adapters
- lora_adapters/text/<output_name>/: 学習出力（LoRAアダプタ）
- lora_adapters/image/: 画像系の出力先（実装が有効な場合）
- lora_adapters/audio/: 音声系の出力先（実装が有効な場合）

## 再現
- 学習設定は UI が送る JSON として残ります。
- 同一設定で再実行する場合は、学習履歴から設定を読み取り、同じ値で /train/start を再送します。

