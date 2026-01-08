# LoRA Factory（LoRA/モデル作成特化ツール）

## 1. 前提
- Windows 10/11
- NVIDIA GPU（CUDA）
- Python 3.10 以上

## 2. セットアップ
1) まずは同梱の setup.py を実行します。

```
py setup.py
```

2) 追加導入を分けたい場合

```
py setup.py --only base
py setup.py --only optional
```

ログは logs/setup_YYYYMMDD_HHMMSS.log に保存されます。

## 3. 起動
```
run_lora.bat
```

起動後、ブラウザで表示されたURL（既定: http://127.0.0.1:8081 ）を開きます。

## 4. 学習フロー（最短）
1) 学習ウィザードで、保存先フォルダを確認し「フォルダを開く」で配置場所を開く
2) 「モデル」タブでモデルをダウンロード（または models/text に手動配置）
3) 「データセット」タブで datasets/text に JSONL 等を配置（アップロード可）
4) 「学習（LLM）」タブでプリセットを選び、開始
5) 完了後、lora_adapters/text に出力が作成される

## 5. トラブルシュート
- UIが開かない: 先に run_lora.bat のコンソール出力を確認し、エラー行を logs に残す
- フォルダを開くが失敗: Windows 以外では open_path は動作しません
- VRAM不足: 学習プリセットを「高速（試験）」へ変更し max_seq_length を下げる

