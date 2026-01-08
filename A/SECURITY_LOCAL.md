# ローカル運用上の注意

## 1. フォルダを開く（open_path）
- open_path は Windows の explorer を起動します。
- 安全のため、このツール配下（base_dir / models / datasets / lora_adapters / logs）のみ許可しています。

## 2. 個人データ
- datasets に置くデータは個人情報を含む可能性があります。
- 共有や配布を行う場合は、必ず個人情報を除去してから行ってください。

## 3. ネットワーク
- 本ツールはローカル運用が前提です。
- 外部APIを使う機能は、環境変数（OPENAI_API_KEY 等）が無い場合は失敗しますが、UIが破綻しない設計です。

