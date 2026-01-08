# third_party 事前clone 手順（必須）

このツールは **自動 clone を行いません**。ユーザーが事前に clone 済みである前提です。
未 clone の場合は UI に警告が表示され、該当プリセットは選択不可になります。

## GPT-SoVITS（VC）
- clone 先: `third_party/GPT-SoVITS`
- URL: https://github.com/RVC-Boss/GPT-SoVITS

例:
```bat
cd <ツールのフォルダ>
git clone https://github.com/RVC-Boss/GPT-SoVITS third_party\GPT-SoVITS
```

## XTTS（Coqui TTS）
- clone 先: `third_party/XTTS`
- URL: https://github.com/coqui-ai/TTS

例:
```bat
cd <ツールのフォルダ>
git clone https://github.com/coqui-ai/TTS third_party\XTTS
```

## UI での確認
- 音声 → 学習（LoRA/モデル作成）
- 警告が消え、プリセットが選べれば OK
