Heretic Plugin Framework Patch v3 (プラグイン拡張点 + Hereticプラグイン)

目的
- 本体更新時に差分衝突を最小化するため、「拡張点（plugin loader）」のみを本体に追加します。
- Heretic機能は plugins/heretic/ 以下へ分離し、本体更新に強い構造にします。

適用方法
1) このZIPの内容を、ツール本体フォルダのルートへフォルダ構造を保って上書きコピー
2) サーバ再起動
3) Textモードのナビに "Heretic" タブが追加されます（プラグインが読み込まれている場合）

本体側の最小変更
- lora_app.py:
  - /plugins のStaticFilesマウント
  - /api/plugins/frontend の提供
  - pluginsのbackend routerを /api に include
- static/js/modules/text_ui.js:
  - /api/plugins/frontend を参照し、ES module plugin を動的importしてタブを追加

プラグイン追加手順（将来）
- plugins/<name>/__init__.py を作る
- plugins/<name>/plugin.json を作る
- Frontend: plugins/<name>/frontend_plugin.js (ES module) を置く
- Backend (任意): plugins/<name>/backend_plugin.py で get_routers() を返す

Hereticプラグインの提供機能（内容非依存）
- Debug Hooks:
  - 任意module_nameに forward_hook / forward_pre_hook を付け、統計を取得
- SVD Weight Analysis:
  - 2モデルの state_dict 差分を取り、2D weight に対してSVD低ランク化しLoRA相当を safetensors 保存

注意
- UIは任意コード実行はしません（安全側）。
- モデルの改変を目的とした特定用途ロジックは含みません。


Hereticライブラリ配置ガイド（UI）
- Hereticタブ最上部に、配置先フォルダを開くボタンと GitHub を開くボタンがあります。
- 'heretic-master' が検出された場合、赤字で手動リネーム手順を強調表示します。
- さらに条件を満たす場合（heretic_master が未作成・レイアウト妥当）、ボタン押下で自動リネーム（安全確認付き）が可能です。


[NEW v3.3]
- Detects common layout: heretic_master/src/heretic and shows a yellow, non-destructive warning with manual move steps in the Heretic tab.
