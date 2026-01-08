/**
 * static/js/modules/text_ui.js
 * Text (LLM) モード専用のUIロジックモジュール。
 * HTMLのレンダリングとイベントハンドリングを担当します。
 * (v14: Modular Architecture Support)
 */

import { api } from '../api.js';
import { renderSystemInfoTab, fetchPathsSafe } from './system_info_view.js';

// ------------------------------------------------------------
// DOM Helpers
// ------------------------------------------------------------
// このWebUIは jQuery を使いません。
// 旧実装の名残で $()（getElementById の短縮）を使う箇所があるため、
// モジュール単体でも確実に動くようにここで提供します。
var $ = window.$ || function(id){ return document.getElementById(id); };
window.$ = $;

// ------------------------------------------------------------
// Minimal notify (toast alternative)
// ------------------------------------------------------------
function notify(msg) {
  try { alert(msg); } catch { console.log(msg); }
}

// ------------------------------------------------------------
// Open folder helper
// ------------------------------------------------------------
async function openFolder({ key=null, path=null }) {
  try {
    await api.post('/utils/open_path', { key, path });
  } catch (e) {
    notify(`フォルダを開けませんでした: ${e.message || e}`);
  }
}

function addOpenFolderRow(parentEl, buttons) {
  if (!parentEl) return;
  const row = document.createElement('div');
  row.className = 'open-folder-row';
  for (const b of buttons) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'open-folder-btn';
    btn.innerHTML = `<i class="fas fa-folder-open"></i> ${b.label}`;
    btn.addEventListener('click', b.onClick);
    row.appendChild(btn);
  }
  parentEl.appendChild(row);
}


async function loadTextPlugins(navContainer, contentContainer) {
  // Discover frontend plugins for text mode and mount their tabs/panels.
  try {
    const resp = await api.get('/plugins/frontend?mode=text');
    const plugins = (resp && resp.plugins) ? resp.plugins : [];
    if (!plugins.length) {
      console.info('[TextUI] no frontend plugins discovered for text mode');
    }
    for (const p of plugins) {
      if (!p || !p.module_url) continue;
      try {
        const mod = await import(p.module_url);
        const getTabs = mod.getTextModeTabs || (mod.default && mod.default.getTextModeTabs);
        if (typeof getTabs !== 'function') continue;
        const tabs = await getTabs({ api, notify });
        if (!Array.isArray(tabs)) continue;

        for (const t of tabs) {
          if (!t || !t.id || !t.buttonHTML || !t.paneHTML) continue;
          // Append button
          const anchor = document.getElementById('text-plugin-anchor');
          if (anchor) {
            anchor.insertAdjacentHTML('beforebegin', t.buttonHTML);
          } else {
            navContainer.insertAdjacentHTML('beforeend', t.buttonHTML);
          }
          // Append pane
          contentContainer.insertAdjacentHTML('beforeend', t.paneHTML);
          // Optional onMount callback
          if (typeof t.onMount === 'function') {
            try { await t.onMount(); } catch(e) { /* ignore */ }
          }
        }
      } catch(e) {
        console.warn('[TextUI] frontend plugin load failed:', p && p.name ? p.name : p, e);
      }
    }
  } catch(e) {
    // plugin system optional
  }
}

// ============================================================================
// 1. HTML Templates
// ============================================================================

const NAV_HTML = `
  <button class="nav-btn active" data-target="tab-system">
    <i class="fas fa-microchip"></i> システム情報
  </button>
  <button class="nav-btn" data-target="tab-models">
    <i class="fas fa-cubes"></i> モデル
  </button>
  <button class="nav-btn" data-target="tab-datasets">
    <i class="fas fa-database"></i> データセット
  </button>
  <button class="nav-btn" data-target="tab-alchemy">
    <i class="fas fa-flask"></i> データ錬成
  </button>
  <button class="nav-btn" data-target="tab-train">
    <i class="fas fa-fire"></i> 学習（LLM）
  </button>
  <button class="nav-btn" data-target="tab-history">
    <i class="fas fa-history"></i> 学習履歴
  </button>
  <div id="text-plugin-anchor" style="margin:6px 0;"></div>
  <button class="nav-btn" data-target="tab-inference">
    <i class="fas fa-comments"></i> 検証（推論）
  </button>
`;

const CONTENT_HTML = `
    <section id="tab-system" class="tab-pane active">
      <header class="pane-header">
        <h3>システムモニター</h3>
        <button id="refresh-sys-btn" class="action-btn"><i class="fas fa-sync"></i> 更新</button>
      </header>
      <div class="pane-body">
<details class="help-details">
  <summary><i class="fas fa-circle-question"></i> はじめての方へ（基本の流れ）</summary>
  <div class="help-content">
    <ol>
      <li><strong>モデル</strong>：HuggingFace からモデルをダウンロードします（例：ELYZA / Qwen など）。</li>
      <li><strong>データセット</strong>：学習に使う .txt / .jsonl をアップロードし、必要ならクリーニングや分割を行います。</li>
      <li><strong>データ錬成</strong>：複数ファイルの結合やフォーマット変換など、学習用に整えます。</li>
      <li><strong>学習（LLM）</strong>：学習ジョブを開始し、ログで進捗を確認します。</li>
      <li><strong>検証（推論）</strong>：学習した LoRA を読み込み、短いプロンプトで動作確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>用語メモ：</strong>「LoRA」は追加学習データ（差分）です。ベースモデル本体は変更せず、必要に応じて LoRA を読み込んで使います。</p>
      <p><strong>困ったとき：</strong> 学習が止まる／エラーになる場合は、まず <em>Batch Size</em> や <em>seq_len</em> を下げて VRAM 使用量を減らしてください。</p>
    </div>
  </div>
</details>
        <div class="card">
          <h4>NVIDIA-SMI 出力</h4>
          <div id="txt-system-info-container" class="system-info-container"></div>
        </div>
      </div>
    </section>

    <section id="tab-models" class="tab-pane">
      <header class="pane-header">
        <h3>モデル管理（テキスト）</h3>
      </header>
      <div class="pane-body">
<details class="help-details">
  <summary><i class="fas fa-circle-question"></i> はじめての方へ（基本の流れ）</summary>
  <div class="help-content">
    <ol>
      <li><strong>モデル</strong>：HuggingFace からモデルをダウンロードします（例：ELYZA / Qwen など）。</li>
      <li><strong>データセット</strong>：学習に使う .txt / .jsonl をアップロードし、必要ならクリーニングや分割を行います。</li>
      <li><strong>データ錬成</strong>：複数ファイルの結合やフォーマット変換など、学習用に整えます。</li>
      <li><strong>学習（LLM）</strong>：学習ジョブを開始し、ログで進捗を確認します。</li>
      <li><strong>検証（推論）</strong>：学習した LoRA を読み込み、短いプロンプトで動作確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>用語メモ：</strong>「LoRA」は追加学習データ（差分）です。ベースモデル本体は変更せず、必要に応じて LoRA を読み込んで使います。</p>
      <p><strong>困ったとき：</strong> 学習が止まる／エラーになる場合は、まず <em>Batch Size</em> や <em>seq_len</em> を下げて VRAM 使用量を減らしてください。</p>
    </div>
  </div>
</details>
        <div class="card">
          <h4>HuggingFaceからダウンロード</h4>
          <div class="input-group">
            <input type="text" id="hf-repo-id" placeholder="Repo ID（例: elyza/ELYZA-japanese-Llama-2-7b-fast）" />
            <button id="download-btn" class="primary-btn"><i class="fas fa-download"></i> ダウンロード</button>
          </div>
          <small class="hint">※ HuggingFace形式（safetensors等）のモデルを指定してください。</small>
        </div>

        <div class="card">
          <h4>LoRAマージ & 保存 (Export)</h4>
          <div class="form-group">
            <label>ベースモデル</label>
            <select id="merge-base-model"></select>
          </div>
          <div class="form-group">
            <label>LoRAアダプタ (lora_adapters/配下のフォルダ名)</label>
            <input type="text" id="merge-adapter-path" placeholder="例: my_job_2023..." />
          </div>
          <div class="form-group">
            <label>保存名 (models/配下に作成されます)</label>
            <input type="text" id="merge-new-model-name" placeholder="例: my-merged-model-v1" />
          </div>

          <label style="display:flex; align-items:center; gap:8px; cursor:pointer; margin-top:10px;">
            <input type="checkbox" id="merge-smoke-test">
            <span><strong>動作確認（生成テスト）</strong>（マージ後に短い生成テストを実行）</span>
          </label>
          <div class="form-group" style="margin-top:8px;">
            <label>動作確認用プロンプト（任意）</label>
            <textarea id="merge-smoke-test-prompt" rows="2" placeholder="例: こんにちは。自己紹介をしてください。"></textarea>
            <small class="hint">※ 実行すると追加でモデルロードが走るため時間/VRAMを消費します</small>
          </div>

          <button id="merge-btn" class="primary-btn"><i class="fas fa-file-export"></i> マージ実行</button>
          <small class="hint">※ CPUメモリを使用してマージを行います。完了まで数分かかる場合があります。</small>

          <div class="form-group" style="margin-top:10px;">
            <label>動作確認の出力（結果）</label>
            <textarea id="merge-smoke-output" rows="3" readonly placeholder="動作確認を実行するとここに出力します"></textarea>
          </div>

        </div>

        <div class="card">
          <h4>ローカルモデル一覧</h4>
          <div class="row-actions" style="margin:6px 0 10px;">
            <button id="text-models-refresh" class="mini-btn"><i class="fas fa-sync"></i> 更新</button>
            <button id="text-models-open-root" class="mini-btn"><i class="fas fa-folder-open"></i> フォルダ</button>
          </div>
          <table class="data-table">
            <thead>
              <tr>
                <th>モデル名</th>
                <th>種別</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody id="models-list-body"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section id="tab-datasets" class="tab-pane">
      <header class="pane-header">
        <h3>データセット管理（テキスト）</h3>
      </header>

      <div class="pane-body">
<details class="help-details">
  <summary><i class="fas fa-circle-question"></i> はじめての方へ（基本の流れ）</summary>
  <div class="help-content">
    <ol>
      <li><strong>モデル</strong>：HuggingFace からモデルをダウンロードします（例：ELYZA / Qwen など）。</li>
      <li><strong>データセット</strong>：学習に使う .txt / .jsonl をアップロードし、必要ならクリーニングや分割を行います。</li>
      <li><strong>データ錬成</strong>：複数ファイルの結合やフォーマット変換など、学習用に整えます。</li>
      <li><strong>学習（LLM）</strong>：学習ジョブを開始し、ログで進捗を確認します。</li>
      <li><strong>検証（推論）</strong>：学習した LoRA を読み込み、短いプロンプトで動作確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>用語メモ：</strong>「LoRA」は追加学習データ（差分）です。ベースモデル本体は変更せず、必要に応じて LoRA を読み込んで使います。</p>
      <p><strong>困ったとき：</strong> 学習が止まる／エラーになる場合は、まず <em>Batch Size</em> や <em>seq_len</em> を下げて VRAM 使用量を減らしてください。</p>
    </div>
  </div>
</details>
        <div class="columns">

          <div class="column is-4">
            <div class="card">
              <h4>データセット・クリーニング (簡易)</h4>
              <div class="form-group">
                <label>対象ファイル (.txt / .jsonl)</label>
                <select id="clean-dataset-select"></select>
              </div>
              <div class="form-group">
                <label style="display:flex; align-items:center; gap:8px;">
                  <input type="checkbox" id="clean-remove-duplicates" checked> 完全一致行を削除
                </label>
              </div>
              <div class="form-group">
                <label>最小文字数 (これ未満の行を削除)</label>
                <input type="number" id="clean-min-length" value="10">
              </div>
              <div class="form-group">
                <label>言語フィルタ (保持する言語コード)</label>
                <input type="text" id="clean-filter-lang" placeholder="例: ja (空欄で無効)" />
                <small class="hint">※ 'ja'と入力すると日本語以外を削除します。</small>
              </div>

              <button id="clean-btn" class="action-btn full-width"><i class="fas fa-broom"></i> クリーニング実行</button>
              <div class="card-subsection">
                <strong>結果</strong>
                <pre id="clean-result" class="console-log small-log"></pre>
              </div>
            </div>

            <div class="card">
              <h4>データセット解析 (トークン数)</h4>
              <div class="form-group">
                <label>使用するトークナイザー(モデル)</label>
                <select id="analysis-base-model"></select>
              </div>
              
              <div class="form-group">
                <label>対象種別</label>
                <div style="display:flex; gap:15px; margin-bottom:5px;">
                  <label style="display:flex; align-items:center; cursor:pointer;">
                    <input type="radio" name="analysis-target-type" value="file" checked> 単一ファイル
                  </label>
                  <label style="display:flex; align-items:center; cursor:pointer;">
                    <input type="radio" name="analysis-target-type" value="folder"> フォルダ
                  </label>
                </div>
              </div>

              <div class="form-group">
                <label>対象データセット</label>
                <select id="analysis-dataset"></select>
              </div>

              <div class="form-group">
                <label>基準トークン長 (超過警告用)</label>
                <input type="number" id="analysis-max-seq" value="2048" step="128">
              </div>

              <button id="analyze-tokens-btn" class="action-btn full-width"><i class="fas fa-calculator"></i> 解析実行</button>
              
              <div class="card-subsection">
                <strong>解析結果</strong>
                <div id="analysis-result" class="console-log small-log" style="margin-top:5px;">(ここに結果が表示されます)</div>
                
                <button id="smart-split-btn" class="action-btn full-width hidden" style="margin-top:10px; background-color:#238636; color:white;">
                  <i class="fas fa-scissors"></i> トークン超過ファイルを自動分割（スマート分割）
                </button>
              </div>
            </div>

            <div class="card">
              <h4>データセットをアップロード（単一ファイル）</h4>
              <div class="upload-area" id="dataset-upload-area">
                <input type="file" id="dataset-file-input" accept=".jsonl,.json,.txt" hidden />
                <p>クリックまたはドラッグ＆ドロップ（.jsonl / .json / .txt）</p>
              </div>
            </div>

            <div class="card">
              <h4>フォルダをアップロード（複数txt）</h4>
              <div class="form-group">
                <input type="file" id="dataset-folder-input" class="file-input-hidden" webkitdirectory directory multiple />
                <div class="button-row">
                  <button id="dataset-folder-pick-btn" type="button" class="primary-btn">
                    <i class="fas fa-folder-open"></i> フォルダを選択
                  </button>
                  <button id="dataset-folder-upload-btn" class="action-btn" disabled>
                    <i class="fas fa-upload"></i> アップロード
                  </button>
                </div>
                <div class="hint" id="dataset-folder-selected">未選択</div>
              </div>

              <hr>

              <h4 style="margin-top:0.8rem;">フォルダ連結/分割 (Compile)</h4>
              <div class="form-group">
                <label>対象フォルダ（datasets配下）</label>
                <select id="dataset-folder-select"></select>
              </div>
              <div class="form-row">
                <div class="form-group">
                  <label>分割上限（MB/ファイル）</label>
                  <input type="number" id="dataset-shard-max-mb" value="100" min="1" max="2048" />
                </div>
              </div>
              <div class="form-group">
                <label>除外パターン（glob）</label>
                <textarea id="dataset-exclude-patterns" rows="4" spellcheck="false">**/.git/**\n**/__pycache__/**\n**/*.bak\n**/*.log</textarea>
              </div>
              <div class="actions">
                <button id="dataset-compile-btn" class="primary-btn full-width">
                  <i class="fas fa-wand-magic-sparkles"></i> 連結/分割して生成
                </button>
              </div>
              <div class="card-subsection">
                <div class="subheader-row">
                  <strong>結果</strong>
                  <span id="dataset-compile-status" class="badge badge-gray">未実行</span>
                </div>
                <pre id="dataset-compile-result" class="console-log small-log"></pre>
              </div>
            </div>

            <div class="card">
              <h4>利用可能なデータセット</h4>
              <ul id="dataset-list" class="list-group"></ul>
            </div>
          </div>

          <div class="column is-8">
            <div class="card full-height">
              <h4>プレビュー</h4>
              <pre id="dataset-preview" class="code-preview">データセットを選択してください...</pre>
            </div>
          </div>

        </div>
      </div>
    </section>

    <section id="tab-alchemy" class="tab-pane">
      <header class="pane-header">
        <h3>データ錬成 (Data Alchemy)</h3>
      </header>
      <div class="pane-body">
<details class="help-details">
  <summary><i class="fas fa-circle-question"></i> はじめての方へ（基本の流れ）</summary>
  <div class="help-content">
    <ol>
      <li><strong>モデル</strong>：HuggingFace からモデルをダウンロードします（例：ELYZA / Qwen など）。</li>
      <li><strong>データセット</strong>：学習に使う .txt / .jsonl をアップロードし、必要ならクリーニングや分割を行います。</li>
      <li><strong>データ錬成</strong>：複数ファイルの結合やフォーマット変換など、学習用に整えます。</li>
      <li><strong>学習（LLM）</strong>：学習ジョブを開始し、ログで進捗を確認します。</li>
      <li><strong>検証（推論）</strong>：学習した LoRA を読み込み、短いプロンプトで動作確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>用語メモ：</strong>「LoRA」は追加学習データ（差分）です。ベースモデル本体は変更せず、必要に応じて LoRA を読み込んで使います。</p>
      <p><strong>困ったとき：</strong> 学習が止まる／エラーになる場合は、まず <em>Batch Size</em> や <em>seq_len</em> を下げて VRAM 使用量を減らしてください。</p>
    </div>
  </div>
</details>
        
        <div class="columns">
          <div class="column is-6">
            <div class="card full-height">
              <h4><i class="fas fa-clone"></i> Semantic Deduplication (意味的重複排除)</h4>
              <p style="color:var(--text-sub); font-size:0.9rem; margin-bottom:1rem;">
                AI (Embeddingモデル) を使用してテキストの意味ベクトルを計算し、類似度が閾値を超える「意味的に重複しているデータ」を削除します。<br>
                <strong style="color:var(--success);">Faiss (高速近似探索) 対応済み</strong>
              </p>
              
              <div class="form-group">
                <label>対象データセット</label>
                <select id="dedup-dataset-select"></select>
              </div>

              <div class="form-row">
                <div class="form-group">
                  <label>類似度閾値 (0.5 - 1.0)</label>
                  <input type="number" id="dedup-threshold" value="0.95" step="0.01" min="0.5" max="1.0">
                  <small class="hint">※ 0.95以上を推奨。低いと似ているだけの別の文も消えます。</small>
                </div>
                <div class="form-group flex-grow">
                  <label>Embeddingモデル (任意)</label>
                  <input type="text" id="dedup-model" placeholder="intfloat/multilingual-e5-large" />
                </div>
              </div>

              <button id="dedup-btn" class="primary-btn full-width"><i class="fas fa-cut"></i> 重複排除を実行</button>

              <div class="card-subsection">
                <strong>実行ログ</strong>
                <pre id="dedup-log" class="console-log small-log">待機中...</pre>
              </div>
            </div>
          </div>

          <div class="column is-6">
             <div class="card full-height">
               <h4><i class="fas fa-magic"></i> Data Augmentation (データ拡張・進化)</h4>
               <p style="color:var(--text-sub); font-size:0.9rem; margin-bottom:1rem;">
                 外部LLM (OpenAI API等) を使用して、データセットを進化させます。<br>
                 <span style="color:#d29922;">※ 実行にはサーバー環境変数 (OPENAI_API_KEY) が必要です。</span>
               </p>

               <div class="form-group">
                 <label>対象データセット</label>
                 <select id="aug-dataset-select"></select>
               </div>

               <div class="form-group">
                 <label>錬成手法 (Method)</label>
                 <select id="aug-method-select">
                   <option value="evol_instruct">Evol-Instruct (指示の複雑化・進化)</option>
                   <option value="refine">Refinement (回答の品質向上・修正)</option>
                 </select>
                 <small class="hint" id="aug-method-hint">指示をより複雑・具体的に書き換えます。</small>
               </div>

               <button id="aug-btn" class="primary-btn full-width" style="background-color: #8250df; border-color: #8250df;">
                 <i class="fas fa-hat-wizard"></i> 錬成開始 (API実行)
               </button>

               <div class="card-subsection">
                 <strong>実行ログ</strong>
                 <pre id="aug-log" class="console-log small-log">待機中...</pre>
               </div>
             </div>
          </div>
        </div>

      </div>
    </section>

    <section id="tab-train" class="tab-pane">
      <header class="pane-header">
        <h3>LoRA 学習（QLoRA）</h3>
      </header>

      <div class="pane-body">
<details class="help-details">
  <summary><i class="fas fa-circle-question"></i> はじめての方へ（基本の流れ）</summary>
  <div class="help-content">
    <ol>
      <li><strong>モデル</strong>：HuggingFace からモデルをダウンロードします（例：ELYZA / Qwen など）。</li>
      <li><strong>データセット</strong>：学習に使う .txt / .jsonl をアップロードし、必要ならクリーニングや分割を行います。</li>
      <li><strong>データ錬成</strong>：複数ファイルの結合やフォーマット変換など、学習用に整えます。</li>
      <li><strong>学習（LLM）</strong>：学習ジョブを開始し、ログで進捗を確認します。</li>
      <li><strong>検証（推論）</strong>：学習した LoRA を読み込み、短いプロンプトで動作確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>用語メモ：</strong>「LoRA」は追加学習データ（差分）です。ベースモデル本体は変更せず、必要に応じて LoRA を読み込んで使います。</p>
      <p><strong>困ったとき：</strong> 学習が止まる／エラーになる場合は、まず <em>Batch Size</em> や <em>seq_len</em> を下げて VRAM 使用量を減らしてください。</p>
    </div>
  </div>
</details>
        <div class="columns">
          <div class="column is-4">
            <div class="card">
              <h4>設定</h4>

              <div class="form-group">
                <label>ベースモデル</label>
                <select id="train-base-model"></select>
              </div>
              <div class="form-group">
                <label>学習プリセット（事故防止）</label>
                <div class="row">
                  <select id="train-preset-select">
                    <option value="">(選択してください)</option>
                    <option value="style_tone">文体・口調（小〜中）</option>
                    <option value="domain_knowledge">ドメイン知識追加（中〜大）</option>
                    <option value="chat_instruction">対話指示（遵守強化）</option>
                    <option value="jp_novel">日本語小説（地の文中心）</option>
                  </select>
                  <button id="apply-train-preset-btn" class="btn">適用</button>
                </div>
              </div>

              <div class="card subtle">
                <h4 class="card-title">環境ドクター（Windows + GPU 前提）</h4>
                <div id="train-preflight" class="preflight">
                  <div class="preflight-row"><span class="label">GPU / VRAM</span><span id="preflight-gpu" class="value">未チェック</span></div>
                  <div class="preflight-row"><span class="label">推定VRAM</span><span id="preflight-vram-est" class="value">-</span></div>
                  <div class="preflight-row"><span class="label">Flash Attention 2</span><span id="preflight-flash" class="value">-</span></div>
                  <div class="preflight-row"><span class="label">QLoRA (4bit)</span><span id="preflight-qlora" class="value">-</span></div>
                  <div class="preflight-row"><span class="label">注意</span><span id="preflight-notes" class="value">-</span></div>
                </div>
                <div class="row">
                  <button id="run-preflight-btn" class="btn secondary">事前チェック</button>
                </div>
              </div>


              <div class="form-group">
                <label>学習データセット</label>
                <select id="train-dataset"></select>
              </div>

              <div class="form-group">
                <label>検証用データセット (任意)</label>
                <select id="train-val-dataset">
                  <option value="">(自動分割 5%)</option>
                </select>
                <small class="hint">※ 指定すると自動分割は無効化されます。</small>
              </div>
              
              <div class="form-group">
                <label>生成テスト用プロンプト (任意)</label>
                <input type="text" id="train-val-prompt" placeholder="例: 自己紹介してください" />
                <small class="hint">※ 学習中にこのプロンプトで生成テストを行います。</small>
              </div>
              
              <div class="form-group">
                <label>品質評価プローブ (任意)</label>
                <select id="train-eval-prompt-set">
                    <option value="">(未使用)</option>
                </select>
                <small class="hint">※ 既定セットを選ぶか、下のJSON配列でカスタム指定できます。</small>
                <textarea id="train-eval-prompts" rows="4" placeholder='["プローブ1","プローブ2"]' spellcheck="false" style="font-family:monospace; font-size:0.8rem; margin-top:6px;"></textarea>
              </div>

              <div class="form-group">
                <label>評価の自動スコアリング</label>
                <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
                    <input type="checkbox" id="train-eval-score-enabled" checked>
                    <span>有効化</span>
                </label>
                <div class="grid-2" style="margin-top:8px;">
                    <div>
                    <label class="sub-label">最小文字数</label>
                    <input type="number" id="train-eval-score-min-len" value="40" min="0" step="1">
                    </div>
                    <div>
                    <label class="sub-label">最大文字数</label>
                    <input type="number" id="train-eval-score-max-len" value="800" min="0" step="1">
                    </div>
                </div>
                <div class="grid-2" style="margin-top:8px;">
                    <div>
                    <label class="sub-label">繰り返し n-gram</label>
                    <input type="number" id="train-eval-score-rep-ngram" value="6" min="2" step="1">
                    </div>
                    <div>
                    <label class="sub-label">繰り返し閾値</label>
                    <input type="number" id="train-eval-score-rep-threshold" value="0.35" min="0" max="1" step="0.01">
                    </div>
                </div>
                <label style="display:flex; align-items:center; gap:8px; cursor:pointer; margin-top:8px;">
                    <input type="checkbox" id="train-eval-score-require-json" checked>
                    <span>プロンプトがJSONを要求している場合はJSON妥当性を評価</span>
                </label>
                <label class="sub-label" style="margin-top:8px;">禁止フレーズ (JSON配列または1行1フレーズ)</label>
                <textarea id="train-eval-score-banned" rows="3" placeholder='["As an AI","AIとして","できません"]' spellcheck="false" style="font-family:monospace; font-size:0.8rem;"></textarea>
              </div>

              <div class="form-group">
                <label>プロンプトテンプレート</label>
                <select id="train-prompt-template">
                  <option value="">Auto (自動判定 / Alpaca)</option>
                  <option value="llama-3">Llama-3 (Instruct)</option>
                  <option value="chatml">ChatML (Qwen/Yi等)</option>
                  <option value="gemma">Gemma (IT)</option>
                  <option value="alpaca">Alpaca (Legacy)</option>
                  <option value="custom">Custom (Jinja2)</option>
                </select>
              </div>
              
              <div class="form-group hidden" id="train-custom-template-group">
                <label>カスタムテンプレート (Jinja2)</label>
                <textarea id="train-custom-template-content" rows="4" placeholder="{% for message in messages %}..." spellcheck="false" style="font-family:monospace; font-size:0.8rem;"></textarea>
                <small class="hint">※ messages変数が渡されます。</small>
              </div>

              <div class="form-group">
                <label>LoRA適用モード</label>
                <select id="train-lora-target-mode">
                  <option value="attention-only">Standard (Attention層のみ / VRAM節約)</option>
                  <option value="all-linear" selected>High Quality (全線形層 / 高精度・要VRAM)</option>
                </select>
              </div>

              <div class="form-group" style="margin-top: 8px; border-top:1px solid #30363d; paddingTop:8px;">
                 <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
                   <input type="checkbox" id="train-use-dora"> 
                   <span><strong>DoRA</strong> (高精度 / 重み分解)</span>
                 </label>
                 
                 <label style="display:flex; align-items:center; gap:8px; cursor:pointer; margin-top:5px;">
                   <input type="checkbox" id="train-use-flash-attn"> 
                   <span><strong>Flash Attention 2</strong> (高速化 / 要RTX30xx~)</span>
                 </label>

                 <label style="display:flex; align-items:center; gap:8px; cursor:pointer; margin-top:5px;">
                    <input type="checkbox" id="train-train-on-inputs"> 
                    <span>入力プロンプトも学習 (通常はOFF)</span>
                 </label>

                 <label style="display:flex; align-items:center; gap:8px; cursor:pointer; margin-top:10px;">
                    <input type="checkbox" id="train-early-stopping">
                    <span><strong>Early Stopping</strong> (検証loss悪化で早期停止)</span>
                 </label>
                 <div class="row" style="margin-top:8px; gap:12px;">
                   <div class="form-group" style="flex:1; min-width:220px;">
                     <label>Patience (評価回数)</label>
                     <input type="number" id="train-early-stopping-patience" min="1" step="1" value="3">
                   </div>
                   <div class="form-group" style="flex:1; min-width:220px;">
                     <label>Threshold (改善とみなす最小差分)</label>
                     <input type="number" id="train-early-stopping-threshold" step="0.0001" value="0">
                   </div>
                 </div>
              </div>

              <hr>

              <div class="form-group">
                <label style="display:flex; align-items:center; gap:8px;">
                  <input type="checkbox" id="train-resume"> 中断から再開 (Resume)
                </label>
                <select id="train-resume-path" disabled style="margin-top:4px;">
                  <option value="">(ベースモデルを選択してください)</option>
                </select>
                <small class="hint" id="resume-hint"></small>
              </div>

              <div class="form-group">
                <label style="display:flex; align-items:center; gap:8px;">
                  <input type="checkbox" id="train-neftune"> NEFTune (安定化)
                </label>
                <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
                  <span style="font-size:0.85rem;">Alpha:</span>
                  <input type="number" id="train-neftune-alpha" value="5" step="0.1" disabled>
                </div>
              </div>

              <hr>

              <div class="form-row">
                <div class="form-group">
                  <label>Max Steps</label>
                  <input type="number" id="train-max-steps" value="100">
                </div>
                <div class="form-group">
                  <label>Learning Rate</label>
                  <input type="text" id="train-lr" value="2e-4">
                </div>
              </div>

              <div class="form-group">
                <label>LR Scheduler</label>
                <select id="train-lr-scheduler">
                  <option value="cosine" selected>Cosine (推奨)</option>
                  <option value="linear">Linear</option>
                  <option value="constant">Constant</option>
                  <option value="cosine_with_restarts">Cosine with Restarts</option>
                </select>
              </div>

              <div class="form-row">
                <div class="form-group">
                  <label>Batch Size</label>
                  <input type="number" id="train-batch-size" value="1">
                </div>
                <div class="form-group">
                  <label>Grad Acc</label>
                  <input type="number" id="train-grad-acc" value="4">
                </div>
              </div>

              <div class="form-row">
                <div class="form-group">
                  <label>LoRA Rank</label>
                  <input type="number" id="train-lora-r" value="8">
                </div>
                <div class="form-group">
                  <label>Alpha</label>
                  <input type="number" id="train-lora-alpha" value="16">
                </div>
              </div>

              <div class="form-group">
                <label>Max Seq Length</label>
                <input type="number" id="train-seq-len" value="2048">
              </div>

              <div class="actions">
                <button id="start-train-btn" class="primary-btn full-width">
                  <i class="fas fa-play"></i> 学習開始
                </button>
                <button id="stop-train-btn" class="danger-btn full-width hidden">
                  <i class="fas fa-stop"></i> 停止
                </button>
              </div>
            </div>
          </div>

          <div class="column is-8">
            <div class="card full-height">
              <div class="header-row">
                <h4>学習状況 (Loss & PPL & Generation)</h4>
                <span id="train-status-badge" class="badge">Idle</span>
              </div>
              
              <div class="chart-container" style="position: relative; height:250px; width:100%; margin-bottom: 10px; background: #0d1117; border-bottom: 1px solid #30363d;">
                <canvas id="train-loss-chart"></canvas>
              </div>

              <div class="tab-switcher" style="margin-bottom: 5px;">
                <button class="switcher-btn active" data-target="train-console">コンソールログ</button>
                <button class="switcher-btn" data-target="train-gen-log">生成ログ (Gen Test)</button>
              </div>

              <pre id="train-console" class="console-log" style="height: calc(100% - 310px); display:block;"></pre>
              
              <div id="train-gen-log" class="console-log" style="height: calc(100% - 310px); display:none; overflow-y:auto;">
                <div class="system-msg">生成テストの結果がここに表示されます...</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section id="tab-history" class="tab-pane">
      <header class="pane-header">
        <h3>学習履歴</h3>
        <button id="refresh-history-btn" class="action-btn"><i class="fas fa-sync"></i> 更新</button>
      </header>
      <div class="pane-body">
<details class="help-details">
  <summary><i class="fas fa-circle-question"></i> はじめての方へ（基本の流れ）</summary>
  <div class="help-content">
    <ol>
      <li><strong>モデル</strong>：HuggingFace からモデルをダウンロードします（例：ELYZA / Qwen など）。</li>
      <li><strong>データセット</strong>：学習に使う .txt / .jsonl をアップロードし、必要ならクリーニングや分割を行います。</li>
      <li><strong>データ錬成</strong>：複数ファイルの結合やフォーマット変換など、学習用に整えます。</li>
      <li><strong>学習（LLM）</strong>：学習ジョブを開始し、ログで進捗を確認します。</li>
      <li><strong>検証（推論）</strong>：学習した LoRA を読み込み、短いプロンプトで動作確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>用語メモ：</strong>「LoRA」は追加学習データ（差分）です。ベースモデル本体は変更せず、必要に応じて LoRA を読み込んで使います。</p>
      <p><strong>困ったとき：</strong> 学習が止まる／エラーになる場合は、まず <em>Batch Size</em> や <em>seq_len</em> を下げて VRAM 使用量を減らしてください。</p>
    </div>
  </div>
</details>
        <div class="card full-height">
           <table class="data-table">
             <thead>
               <tr>
                 <th>ID (日時)</th>
                 <th>Model</th>
                 <th>Dataset</th>
                 <th>Steps</th>
                 <th>Loss</th>
                 <th>Status</th>
               </tr>
             </thead>
             <tbody id="history-list-body"></tbody>
           </table>
        </div>
      </div>
    </section>

    <section id="tab-inference" class="tab-pane">
      <header class="pane-header">
        <h3>検証 / 推論</h3>
      </header>

      <div class="pane-body">
<details class="help-details">
  <summary><i class="fas fa-circle-question"></i> はじめての方へ（基本の流れ）</summary>
  <div class="help-content">
    <ol>
      <li><strong>モデル</strong>：HuggingFace からモデルをダウンロードします（例：ELYZA / Qwen など）。</li>
      <li><strong>データセット</strong>：学習に使う .txt / .jsonl をアップロードし、必要ならクリーニングや分割を行います。</li>
      <li><strong>データ錬成</strong>：複数ファイルの結合やフォーマット変換など、学習用に整えます。</li>
      <li><strong>学習（LLM）</strong>：学習ジョブを開始し、ログで進捗を確認します。</li>
      <li><strong>検証（推論）</strong>：学習した LoRA を読み込み、短いプロンプトで動作確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>用語メモ：</strong>「LoRA」は追加学習データ（差分）です。ベースモデル本体は変更せず、必要に応じて LoRA を読み込んで使います。</p>
      <p><strong>困ったとき：</strong> 学習が止まる／エラーになる場合は、まず <em>Batch Size</em> や <em>seq_len</em> を下げて VRAM 使用量を減らしてください。</p>
    </div>
  </div>
</details>
        <div class="card compact-card">
          <h4>モデルロード</h4>
          <div class="form-row align-end">
            <div class="form-group flex-grow">
              <label>ベースモデル</label>
              <select id="inf-base-model"></select>
            </div>
            <div class="form-group flex-grow">
              <label>アダプタ（任意）</label>
              <input type="text" id="inf-adapter-path" placeholder="base_model_lora_..." />
            </div>
            <div class="form-group">
              <button id="inf-load-btn" class="primary-btn">ロード</button>
              <button id="inf-unload-btn" class="danger-btn hidden">アンロード</button>
            </div>
          </div>
          <div id="inf-load-status" class="status-text">未ロード</div>
        </div>

        <div class="card compact-card" style="margin-top: 1rem;">
          <h4>生成パラメータ</h4>
          <div class="form-row">
            <div class="form-group">
              <label>Temperature</label>
              <input type="number" id="inf-temp" value="0.7" step="0.1" min="0.1" max="2.0">
            </div>
            <div class="form-group">
              <label>Top P</label>
              <input type="number" id="inf-top-p" value="0.9" step="0.05" min="0.0" max="1.0">
            </div>
            <div class="form-group">
              <label>Repetition Penalty</label>
              <input type="number" id="inf-rep-pen" value="1.1" step="0.05" min="1.0">
            </div>
            <div class="form-group">
              <label>Max Tokens</label>
              <input type="number" id="inf-max-tokens" value="512" step="64">
            </div>
          </div>
        </div>

        <div class="chat-wrapper" style="margin-top: 1rem;">
          <div id="chat-history" class="chat-history">
            <div class="system-msg">モデルをロードして検証を開始してください。</div>
          </div>

          <div class="chat-input-area">
            <textarea id="chat-input" placeholder="メッセージを入力..." rows="3"></textarea>
            <button id="chat-send-btn" class="icon-btn large"><i class="fas fa-paper-plane"></i></button>
          </div>
        </div>
      </div>
    </section>
`;

// ============================================================================
// 2. Constants & State
// ============================================================================

const TRAIN_PRESETS = {
    style_tone: {
        name: '文体・口調（小〜中）',
        lr: 2e-4, scheduler: 'cosine', batch_size: 1, grad_acc: 8, lora_r: 16, lora_alpha: 32,
        seq_len: 2048, lora_target_mode: 'attention', use_dora: false, use_flash_attn: false,
        train_on_inputs: false, neftune: false, neftune_alpha: 5.0, max_steps: 800
    },
    domain_knowledge: {
        name: 'ドメイン知識追加（中〜大）',
        lr: 1e-4, scheduler: 'cosine', batch_size: 1, grad_acc: 16, lora_r: 32, lora_alpha: 64,
        seq_len: 2048, lora_target_mode: 'all-linear', use_dora: true, use_flash_attn: false,
        train_on_inputs: false, neftune: false, neftune_alpha: 5.0, max_steps: 1500
    },
    chat_instruction: {
        name: '対話指示（遵守強化）',
        lr: 1.5e-4, scheduler: 'linear', batch_size: 1, grad_acc: 8, lora_r: 16, lora_alpha: 32,
        seq_len: 1536, lora_target_mode: 'attention', use_dora: false, use_flash_attn: false,
        train_on_inputs: false, neftune: false, neftune_alpha: 5.0, max_steps: 1200
    },
    jp_novel: {
        name: '日本語小説（地の文中心）',
        lr: 1e-4, scheduler: 'cosine', batch_size: 1, grad_acc: 16, lora_r: 32, lora_alpha: 64,
        seq_len: 3072, lora_target_mode: 'attention', use_dora: true, use_flash_attn: false,
        train_on_inputs: false, neftune: true, neftune_alpha: 5.0, max_steps: 2000
    }
};

const state = {
    models: [],
    datasets: [],
    datasetFolders: [],
    datasetFolder: { files: [], lastUploadedFolder: null },
    training: { status: 'idle', logs: [], pollInterval: null },
    history: [],
    inference: { isLoaded: false, isGenerating: false, abortController: null },
    alchemy: { isProcessing: false }
};

let trainChart = null;

// Helpers
const $$ = (sel) => document.querySelectorAll(sel);
const toastOk = (msg) => alert(msg);
const toastWarn = (msg) => alert(msg);
const toastError = (msg) => alert(msg);
const escapeHtml = (str) => {
    if (!str) return '';
    return str.replace(/[&<>"']/g, function(m) {
        switch (m) {
            case '&': return '&amp;';
            case '<': return '&lt;';
            case '>': return '&gt;';
            case '"': return '&quot;';
            case "'": return '&#039;';
            default: return m;
        }
    });
};

function _parseModelSizeB(modelName) {
    const m = String(modelName || '').match(/(\d+(?:\.\d+)?)\s*[bB]\b/);
    if (!m) return null;
    return parseFloat(m[1]);
}

function _estimateVramGB({ modelName, seqLen, batchSize, gradAcc, targetMode, useFlashAttn }) {
    const notes = [];
    const b = _parseModelSizeB(modelName);
    let base;
    if (b == null) { base = 8.0; notes.push('モデル名からパラメータ規模(B)を推定できないため、8B相当で概算しています。'); } 
    else if (b <= 4) base = 5.0;
    else if (b <= 8) base = 7.0;
    else if (b <= 13) base = 11.0;
    else if (b <= 16) base = 13.0;
    else if (b <= 22) base = 17.0;
    else if (b <= 32) base = 23.0;
    else base = 30.0;

    const s = Math.max(256, Math.min(8192, Number(seqLen) || 2048));
    const bs = Math.max(1, Number(batchSize) || 1);
    const ga = Math.max(1, Number(gradAcc) || 1);

    let act = (s / 2048) * (0.9 + Math.log2(bs + 1) * 0.15);
    if (ga >= 16) act *= 1.08;
    else if (ga >= 8) act *= 1.04;

    let modeMul = (targetMode === 'all-linear') ? 1.35 : 1.0;
    let flashMul = useFlashAttn ? 0.92 : 1.0;

    const low = (base * modeMul * flashMul) + (2.0 * act);
    const high = (base * modeMul * 1.05) + (4.0 * act);

    if (targetMode === 'all-linear') notes.push('LoRA適用モードが all-linear のため、VRAM要求が増えます。');
    if (s >= 3072) notes.push('seq_len が長めです。Smart Split / token超過対策とあわせて、OOM時は seq_len を下げてください。');
    if (useFlashAttn) notes.push('Flash Attention 2 は環境依存です。事前チェックの結果に従ってON/OFFしてください。');

    return { low: Math.round(low * 10) / 10, high: Math.round(high * 10) / 10, notes };
}

function _parseNvidiaSmiVramTotalGB(gpuInfoText) {
    const text = String(gpuInfoText || '');
    const m = text.match(/\/\s*(\d{4,6})\s*MiB\b/);
    if (!m) return null;
    return Math.round((parseInt(m[1], 10) / 1024) * 10) / 10;
}

function _guessFlashAttnFeasible(gpuInfoText) {
    const t = String(gpuInfoText || '');
    if (/RTX\s*3\d{3}|RTX\s*4\d{3}|A100|H100|L4\b|L40\b|A40\b|A5000|A6000/i.test(t)) return true;
    return null; 
}


// ============================================================================
// 3. Main Logic (Exported Mount Function)
// ============================================================================

export async function mount(navContainer, contentContainer) {
    // 1. Inject HTML
    navContainer.innerHTML = NAV_HTML;
    contentContainer.innerHTML = CONTENT_HTML;
  contentContainer.insertAdjacentHTML("beforeend", `
  <!-- text log modal -->
  <div id="text-log-modal" class="modal-backdrop hidden">
    <div class="modal-card">
      <div class="modal-header">
        <div style="font-weight:700;">ログ表示</div>
        <button id="text-log-close" class="action-btn">閉じる</button>
      </div>
      <div class="modal-body">
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px;">
          <button id="text-log-refresh" class="action-btn">更新</button>
          <button id="text-log-openfolder" class="action-btn">ログフォルダを開く</button>
          <span id="text-log-path" style="font-size:0.85rem; opacity:0.9;"></span>
        </div>
        <div class="two-col">
          <div>
            <div class="small-title">stdout（標準出力）</div>
            <pre id="text-log-stdout" class="log-pre"></pre>
          </div>
          <div>
            <div class="small-title">stderr（標準エラー）</div>
            <pre id="text-log-stderr" class="log-pre"></pre>
          </div>
        </div>
      </div>
    </div>
  </div>
`);

    // 2. Mount optional frontend plugins (tabs/panels)
    await loadTextPlugins(navContainer, contentContainer);

    // 3. Initialize UI Components
    initTabs();
    initLogSwitcher();
    initTrainUI();
    initAlchemyUI();
    initChart();
    initDatasetFolderUI();

    // フォルダを開くボタンを設置（初心者がファイル配置しやすくする）
    installOpenFolderButtons();
    
    // 3. Attach Event Listeners
    attachEventListeners();

    // Models tab small actions
    const modelsRefreshBtn = $('text-models-refresh');
    if (modelsRefreshBtn && !modelsRefreshBtn.__bound) {
        modelsRefreshBtn.__bound = true;
        modelsRefreshBtn.addEventListener('click', refreshModels);
    }
    const modelsOpenRootBtn = $('text-models-open-root');
    if (modelsOpenRootBtn && !modelsOpenRootBtn.__bound) {
        modelsOpenRootBtn.__bound = true;
        modelsOpenRootBtn.addEventListener('click', async () => {
            await openFolder({ key: 'models_text' });
        });
    }


    // 4. Initial Data Fetch
    await Promise.all([
        refreshModels(),
        refreshDatasets(),
        refreshDatasetFolders(),
        updateSystemInfo()
    ]);
    
    // 5. Start Polling
    startLogPolling();
    
    console.log("Text UI Mounted");
}


function _mkOpenBtn(label, onClick) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'open-folder-btn';
  btn.innerHTML = `<i class="fas fa-folder-open"></i> ${label}`;
  btn.addEventListener('click', onClick);
  return btn;
}

function _appendOpenRow(afterEl, buttons) {
  if (!afterEl || !afterEl.parentElement) return;
  const row = document.createElement('div');
  row.className = 'open-folder-row';
  buttons.forEach(b => row.appendChild(b));
  afterEl.parentElement.appendChild(row);
}

function installOpenFolderButtons() {
  // モデル: HuggingFace ダウンロード先（models/text）
  const dlBtn = $('download-btn');
  if (dlBtn && dlBtn.parentElement) {
    const openModels = _mkOpenBtn('モデル保存先を開く (models/text)', () => openFolder({ key: 'models_text' }));
    dlBtn.parentElement.appendChild(openModels);
  }

  // LoRA マージ: adapter 入力は lora_adapters 配下
  const mergeAdapter = $('merge-adapter-path');
  _appendOpenRow(mergeAdapter, [
    _mkOpenBtn('LoRAフォルダを開く (lora_adapters)', () => openFolder({ key: 'lora_adapters_root' })),
  ]);

  // データセット（テキスト）: datasets/text
  const uploadArea = $('dataset-upload-area');
  if (uploadArea && uploadArea.parentElement) {
    const openDs = _mkOpenBtn('データセットフォルダを開く (datasets/text)', () => openFolder({ key: 'datasets_text' }));
    uploadArea.parentElement.appendChild(openDs);
  }

  // クリーニング対象: 選択中ファイルのあるフォルダ
  const cleanSel = $('clean-dataset-select');
  if (cleanSel) {
    _appendOpenRow(cleanSel, [
      _mkOpenBtn('データセットフォルダを開く (datasets/text)', () => openFolder({ key: 'datasets_text' })),
      _mkOpenBtn('選択ファイルの場所を開く', () => {
        const v = (cleanSel.value || '').trim();
        if (!v) return notify('対象ファイルを選択してください');
        // value はファイル名（相対）想定
        openFolder({ path: `datasets/text/${v}` });
      }),
    ]);
  }
}


// ============================================================================
// 4. Initialization & UI Logic
// ============================================================================

function attachEventListeners() {
    // System
    $('refresh-sys-btn')?.addEventListener('click', updateSystemInfo);
    
    // Models
    $('download-btn')?.addEventListener('click', handleDownload);
    $('merge-btn')?.addEventListener('click', handleMergeModel);
    
    // Datasets
    $('dataset-file-input')?.addEventListener('change', (e) => handleDatasetUpload(e.target.files));
    $('dataset-upload-area')?.addEventListener('click', () => $('dataset-file-input').click());
    
    // Train
    $('start-train-btn')?.addEventListener('click', startTraining);
    $('stop-train-btn')?.addEventListener('click', stopTraining);
    
    // Inference
    $('inf-load-btn')?.addEventListener('click', loadInferenceModel);
    $('inf-unload-btn')?.addEventListener('click', unloadInferenceModel);
    $('chat-send-btn')?.addEventListener('click', sendChatMessage);
    $('chat-input')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });

    // History
    $('refresh-history-btn')?.addEventListener('click', refreshHistory);
}

function initTabs() {
    const navBtns = $$('.nav-btn');
    const panes = $$('.tab-pane');

    if (navBtns.length && panes.length) {
        navBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                navBtns.forEach(b => b.classList.remove('active'));
                panes.forEach(p => p.classList.remove('active'));

                btn.classList.add('active');
                const targetId = btn.dataset.target;
                const targetEl = targetId ? $(targetId) : null;
                if (targetEl) targetEl.classList.add('active');

                if (targetId === 'tab-history') refreshHistory();
    bindHistoryActions();
            });
        });
        
        // Activate first tab if none active
        const hasActive = Array.from(panes).some(p => p.classList.contains('active'));
        if (!hasActive && panes[0]) {
            panes[0].classList.add('active');
            navBtns[0]?.classList.add('active');
        }
    }
}

function initLogSwitcher() {
    $$('.switcher-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.switcher-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const targetId = btn.dataset.target;
            if (targetId === 'train-console') {
                $('train-console').style.display = 'block';
                $('train-gen-log').style.display = 'none';
            } else {
                $('train-console').style.display = 'none';
                $('train-gen-log').style.display = 'block';
            }
        });
    });
}

async function updateSystemInfo() {
  const elLog = document.getElementById('txt-sys-log');
  const elSummary = document.getElementById('txt-system-info-container');
  try {
    const data = await api.get('/system_info');
    const paths = await fetchPathsSafe();

    renderSystemInfoTab(elSummary, data, paths, 'text');
    if (elLog) elLog.textContent = '更新完了';
  } catch (e) {
    if (elLog) elLog.textContent = `システム情報の取得に失敗: ${e?.message || e}`;
    if (elSummary) elSummary.innerHTML = '<div class="system-msg error">取得に失敗しました。ログを確認してください。</div>';
  }
}

// --- Models Logic ---

async function refreshModels() {
    try {
        const data = await api.get('/models');
        state.models = data.models || [];
        renderModelsList();
        renderModelSelects();
    } catch (e) {
        console.error(e);
    }
}

function renderModelsList() {
    const tbody = $('models-list-body');
    if (!tbody) return;
    tbody.innerHTML = '';
    state.models.forEach(m => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${m.name}</td>
            <td><span class="badge ${m.type === 'hf' ? 'badge-blue' : 'badge-gray'}">${m.type}</span></td>
            <td><div class="row-actions">
                <button class="icon-btn meta-model-btn" data-name="${m.name}" title="詳細"><i class="fas fa-circle-info"></i></button>
                <button class="icon-btn open-model-btn" data-name="${m.name}" title="フォルダ"><i class="fas fa-folder-open"></i></button>
                <button class="icon-btn danger-btn delete-model-btn" data-name="${m.name}" title="削除">
                    <i class="fas fa-trash"></i>
                </button>
            </div></td>
        `;
        tbody.appendChild(tr);
    });
    
    tbody.querySelectorAll('.meta-model-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const name = e.currentTarget.dataset.name;
            try {
                const meta = await api.get(`/models/${encodeURIComponent(name)}/meta`);
                const mb = (meta.total_bytes || 0) / (1024*1024);
                alert(`モデル: ${meta.name}\n保存先: ${meta.path}\n合計サイズ: ${mb.toFixed(1)} MB\nファイル数: ${meta.file_count}`);
            } catch (err) {
                alert('詳細取得失敗: ' + (err?.message || err));
            }
        });
    });


    tbody.querySelectorAll('.open-model-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const name = e.currentTarget.dataset.name;
            if (!name) return;
            // モデルフォルダを開く（相対パスで安全に解決）
            const rel = 'models/text/' + name;
            await openFolder({ path: rel });
        });
    });

tbody.querySelectorAll('.delete-model-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const name = e.currentTarget.dataset.name;
            try {
                const chk = await api.get(`/models/${encodeURIComponent(name)}/predelete_check`);
                let msg = `本当にモデル "${name}" を削除しますか？\n参照数: ${(chk.reference_count||0)}`;
                if (chk.active_job_using) msg += `\n※実行中ジョブがこのモデルを使用しています。`;
                if (!confirm(msg)) return;
            } catch (e2) {
                if (!confirm(`本当にモデル "${name}" を削除しますか？\n(参照チェック取得失敗: ${e2.message})`)) return;
            }
            {
                try {
                    await api.delete(`/models/${name}`);
                    refreshModels();
                } catch (err) {
                    alert(err.message);
                }
            }
        });
    });
}

function renderModelSelects() {
    const trainSelect = $('train-base-model');
    const infSelect = $('inf-base-model');
    const analysisSelect = $('analysis-base-model');
    const mergeSelect = $('merge-base-model');

    const optionsHtml = state.models.map(m => `<option value="${m.name}">${m.name}</option>`).join('');
    
    [trainSelect, infSelect, analysisSelect, mergeSelect].forEach(sel => {
        if (sel) {
            const currentVal = sel.value;
            sel.innerHTML = optionsHtml;
            if (currentVal) sel.value = currentVal;
        }
    });
}

async function handleDownload() {
    const repoId = $('hf-repo-id').value.trim();
    if (!repoId) return alert('リポジトリIDを入力してください');
    const btn = $('download-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ダウンロード中...';
    try {
        await api.post('/models/download', { repo_id: repoId });
        alert(`ダウンロードをバックグラウンドで開始しました。完了までコンソールを確認してください。`);
    } catch (e) {
        alert('失敗: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-download"></i> ダウンロード';
    }
}

async function handleMergeModel() {
    const baseModel = $('merge-base-model').value;
    const adapterPath = $('merge-adapter-path').value.trim();
    const newModelName = $('merge-new-model-name').value.trim();
    const smokeTest = $('merge-smoke-test') ? $('merge-smoke-test').checked : false;
    const smokePrompt = $('merge-smoke-test-prompt') ? $('merge-smoke-test-prompt').value.trim() : '';

    if (!baseModel || !adapterPath || !newModelName) return alert('すべての項目を入力してください。');
    if (!confirm(`以下の設定でマージを実行しますか？\n\nBase: ${baseModel}\nAdapter: ${adapterPath}\nNew Name: ${newModelName}\n\n※ 数分かかる場合があります。`)) return;

    const btn = $('merge-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> マージ中...';

    try {
        const res = await api.post('/models/merge', {
            base_model: baseModel,
            adapter_path: adapterPath,
            new_model_name: newModelName,
            run_smoke_test: smokeTest,
            smoke_test_prompt: smokePrompt || null
        });

        toastOk('マージが完了しました: ' + (res.path || ''));
        if (res.smoke_test) {
            if (res.smoke_test.status === 'ok') {
                const out = res.smoke_test.text || '';
                if ($('merge-smoke-output')) $('merge-smoke-output').value = out;
            } else {
                toastWarn('Smoke Test: 失敗');
            }
        }
        await refreshModels();
    } catch (e) {
        toastError('マージに失敗しました: ' + (e?.message || e));
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-file-export"></i> マージ実行';
    }
}

// --- Datasets Logic ---

async function refreshDatasets() {
    try {
        const data = await api.get('/datasets');
        state.datasets = data.datasets || [];
        renderDatasetsList();
        renderDatasetSelect();
        renderAnalysisDatasetSelect();
        renderCleanDatasetSelect();
        renderAlchemyDatasetSelects();
    } catch (e) { console.error(e); }
}

function renderDatasetsList() {
    const ul = $('dataset-list');
    if (!ul) return;
    ul.innerHTML = '';
    state.datasets.forEach(d => {
        const li = document.createElement('li');
        li.className = 'list-item';
        let icon = d.name.endsWith('.jsonl') ? 'fa-code' : 'fa-file-alt';
        li.innerHTML = `<i class="fas ${icon}"></i> ${d.name} <span style="font-size:0.8em; color:#888;">(${Math.round(d.size/1024)}KB)</span>`;
        li.addEventListener('click', () => loadDatasetPreview(d.name));
        ul.appendChild(li);
    });
}

function renderDatasetSelect() {
    const trainSelect = $('train-dataset');
    const valSelect = $('train-val-dataset');
    if (!trainSelect) return;

    const optionsHtml = state.datasets.map(d => `<option value="${d.name}">${d.name}</option>`).join('');
    trainSelect.innerHTML = optionsHtml;
    
    if (valSelect) valSelect.innerHTML = '<option value="">(自動分割 5%)</option>' + optionsHtml;
}

function renderAnalysisDatasetSelect() {
    const analysisSelect = $('analysis-dataset');
    if (!analysisSelect) return;
    const targetType = document.querySelector('input[name="analysis-target-type"]:checked')?.value || 'file';
    let options = [];
    if (targetType === 'file') {
        options = state.datasets.map(d => `<option value="${d.name}">${d.name}</option>`);
    } else {
        options = state.datasetFolders.map(f => `<option value="${f}">${f}</option>`);
    }
    analysisSelect.innerHTML = options.length ? options.join('') : '<option value="">(候補なし)</option>';
}

function renderCleanDatasetSelect() {
    const select = $('clean-dataset-select');
    if (!select) return;
    const options = state.datasets.map(d => `<option value="${d.name}">${d.name}</option>`);
    select.innerHTML = options.length ? options.join('') : '<option value="">(候補なし)</option>';
}

function renderAlchemyDatasetSelects() {
    const dedupSelect = $('dedup-dataset-select');
    const augSelect = $('aug-dataset-select');
    if (!dedupSelect || !augSelect) return;
    const options = state.datasets.map(d => `<option value="${d.name}">${d.name}</option>`);
    const html = options.length > 0 ? options.join('') : '<option value="">(候補なし)</option>';
    dedupSelect.innerHTML = html;
    augSelect.innerHTML = html;
}

async function loadDatasetPreview(filename) {
    const pre = $('dataset-preview');
    if (!pre) return;
    pre.textContent = '読み込み中...';
    try {
        const data = await api.get(`/datasets/${filename}`);
        pre.textContent = data.content;
    } catch (e) { pre.textContent = 'プレビュー失敗'; }
}

async function handleDatasetUpload(files) {
    if (!files || files.length === 0) return;
    try {
        await api.upload('/datasets/upload', files[0]);
        refreshDatasets();
        alert('完了');
    } catch (e) { alert(e.message); }
}

async function refreshDatasetFolders() {
    const sel = $('dataset-folder-select');
    try {
        const data = await api.get('/datasets/folders');
        state.datasetFolders = data.folders || [];
        if (sel) {
            const opts = ['<option value="">（選択してください）</option>']
                .concat(state.datasetFolders.map(f => `<option value="${f}">${f}</option>`))
                .join('');
            sel.innerHTML = opts;
        }
        renderAnalysisDatasetSelect();
    } catch (e) { console.error(e); }
}

function initDatasetFolderUI() {
    const pickBtn = $('dataset-folder-pick-btn');
    const uploadBtn = $('dataset-folder-upload-btn');
    const fileInput = $('dataset-folder-input');
    const selectedInfo = $('dataset-folder-selected');
    const compileBtn = $('dataset-compile-btn');
    const analyzeBtn = $('analyze-tokens-btn');
    const smartSplitBtn = $('smart-split-btn');
    const cleanBtn = $('clean-btn');

    document.getElementsByName('analysis-target-type').forEach(r => {
        r.addEventListener('change', renderAnalysisDatasetSelect);
    });

    pickBtn?.addEventListener('click', () => fileInput.click());
    fileInput?.addEventListener('change', () => {
        const files = Array.from(fileInput.files || []);
        state.datasetFolder.files = files;
        selectedInfo.textContent = files.length ? `選択中: ${files.length}ファイル` : '未選択';
        uploadBtn.disabled = files.length === 0;
    });

    uploadBtn?.addEventListener('click', async () => {
        const files = state.datasetFolder.files;
        uploadBtn.disabled = true;
        try {
            await api.uploadMany('/datasets/upload_folder', files);
            await refreshDatasetFolders();
            await refreshDatasets();
            alert('アップロード完了');
        } catch (e) { alert(e.message); } 
        finally { uploadBtn.disabled = false; }
    });

    compileBtn?.addEventListener('click', async () => {
        const folder = $('dataset-folder-select').value;
        if (!folder) return alert('フォルダを選択してください');
        compileBtn.disabled = true;
        try {
            const res = await api.post('/datasets/compile_folder', {
                folder,
                shard_max_mb: parseInt($('dataset-shard-max-mb').value),
                exclude_patterns: $('dataset-exclude-patterns').value.split('\n').filter(Boolean)
            });
            $('dataset-compile-result').textContent = JSON.stringify(res, null, 2);
            await refreshDatasets();
        } catch (e) { alert(e.message); }
        finally { compileBtn.disabled = false; }
    });

    analyzeBtn?.addEventListener('click', async () => {
        const baseModel = $('analysis-base-model').value;
        const dataset = $('analysis-dataset').value;
        const maxSeq = parseInt($('analysis-max-seq').value) || 2048;
        const resDiv = $('analysis-result');
        const splitBtn = $('smart-split-btn');
        if (!baseModel || !dataset) return alert('モデルとデータセットを選択してください');
        
        analyzeBtn.disabled = true;
        resDiv.innerHTML = '<div style="padding:10px;">解析中... (数分かかる場合があります)</div>';
        if (splitBtn) splitBtn.classList.add('hidden');
        
        try {
            const res = await api.post('/datasets/analyze_tokens', { dataset, base_model: baseModel, max_seq_length: maxSeq });
            if (res.error) { resDiv.textContent = 'エラー: ' + res.error; return; }

            let html = `
                <div style="margin-bottom:10px; border-bottom:1px solid #30363d; padding-bottom:8px;">
                    <strong>タイプ:</strong> ${res.is_folder ? 'フォルダ' : 'ファイル'} <br>
                    <strong>総サンプル:</strong> ${res.total_samples} / <strong>総トークン:</strong> ${res.total_tokens.toLocaleString()} <br>
                    <strong>Avg:</strong> ${res.avg_tokens} / <strong>Max:</strong> ${res.max_tokens}
                </div>`;
            
            if (res.details?.length > 0) {
                const exceedCount = res.details.filter(d => d.exceeds).length;
                if (exceedCount > 0) {
                    html += `<div style="color:#f85149; margin-bottom:5px;">⚠️ ${maxSeq}トークン超が ${exceedCount} 件あります！</div>`;
                    if (splitBtn && res.is_folder) {
                        splitBtn.classList.remove('hidden');
                        splitBtn.dataset.folder = dataset;
                        splitBtn.dataset.baseModel = baseModel;
                        splitBtn.dataset.maxSeq = maxSeq;
                    }
                }
                // Table...
                html += `<div style="max-height:200px; overflow-y:auto; font-size:0.8rem; border:1px solid #30363d;">
                    <table class="data-table" style="width:100%;"><tbody>`;
                res.details.forEach(item => {
                    const style = item.exceeds ? 'color:#f85149; font-weight:bold;' : '';
                    html += `<tr style="${style}"><td style="word-break:break-all;">${item.file}</td><td>${item.tokens}</td></tr>`;
                });
                html += `</tbody></table></div>`;
            }
            resDiv.innerHTML = html;
        } catch (e) { resDiv.textContent = 'エラー: ' + e.message; }
        finally { analyzeBtn.disabled = false; }
    });

    smartSplitBtn?.addEventListener('click', async () => {
        const folder = smartSplitBtn.dataset.folder;
        const baseModel = smartSplitBtn.dataset.baseModel;
        const maxSeq = parseInt(smartSplitBtn.dataset.maxSeq);
        if (!confirm(`フォルダ "${folder}" の超過ファイルを自動分割しますか？`)) return;
        smartSplitBtn.disabled = true;
        try {
            const res = await api.post('/datasets/smart_split', { dataset_folder: folder, base_model: baseModel, max_seq_length: maxSeq });
            alert(`完了: 処理ファイル数 ${res.files_processed}, 生成チャンク数 ${res.chunks_created}`);
            await refreshDatasets();
            await refreshDatasetFolders();
        } catch (e) { alert(e.message); }
        finally { smartSplitBtn.disabled = false; }
    });

    cleanBtn?.addEventListener('click', async () => {
        const dataset = $('clean-dataset-select').value;
        if (!dataset) return alert('データセットを選択してください');
        cleanBtn.disabled = true;
        try {
            const res = await api.post('/datasets/clean', {
                dataset,
                remove_duplicates: $('clean-remove-duplicates').checked,
                min_length: parseInt($('clean-min-length').value) || 0,
                filter_lang: $('clean-filter-lang').value.trim() || null
            });
            const s = res.stats;
            $('clean-result').textContent = `完了: ${res.cleaned_file} (${s.cleaned_count}行)\n削除(重複): ${s.removed_duplicates}, 削除(短文): ${s.removed_short}, 言語: ${s.removed_lang}`;
            await refreshDatasets();
        } catch (e) { $('clean-result').textContent = 'エラー: ' + e.message; }
        finally { cleanBtn.disabled = false; }
    });
}

function initAlchemyUI() {
    const dedupBtn = $('dedup-btn');
    const augBtn = $('aug-btn');

    dedupBtn?.addEventListener('click', async () => {
        const dataset = $('dedup-dataset-select').value;
        const threshold = parseFloat($('dedup-threshold').value);
        const modelName = $('dedup-model').value.trim() || null;
        if (!dataset) return alert('データセットを選択してください');
        if (state.alchemy.isProcessing) return alert('処理中です');
        if (!confirm('重複排除を実行しますか？ (時間がかかる場合があります)')) return;

        state.alchemy.isProcessing = true;
        dedupBtn.disabled = true;
        $('dedup-log').textContent = "実行中...";
        try {
            const res = await api.post('/datasets/dedup', { dataset, threshold, model_name: modelName });
            $('dedup-log').textContent = JSON.stringify(res, null, 2);
            await refreshDatasets();
            alert('完了');
        } catch (e) { $('dedup-log').textContent = `エラー: ${e.message}`; }
        finally { state.alchemy.isProcessing = false; dedupBtn.disabled = false; }
    });

    augBtn?.addEventListener('click', async () => {
        const dataset = $('aug-dataset-select').value;
        const method = $('aug-method-select').value;
        if (!dataset) return alert('データセットを選択してください');
        if (state.alchemy.isProcessing) return alert('処理中です');
        if (!confirm('APIを使用して錬成しますか？ (課金が発生する可能性があります)')) return;

        state.alchemy.isProcessing = true;
        augBtn.disabled = true;
        $('aug-log').textContent = "API実行中...";
        try {
            const res = await api.post('/datasets/augment', { dataset, method });
            $('aug-log').textContent = JSON.stringify(res, null, 2);
            await refreshDatasets();
            alert('完了');
        } catch (e) { $('aug-log').textContent = `エラー: ${e.message}`; }
        finally { state.alchemy.isProcessing = false; augBtn.disabled = false; }
    });
}

// --- Chart & History ---

function initChart() {
    const ctx = document.getElementById('train-loss-chart');
    if (!ctx) return;
    if (trainChart) trainChart.destroy();

    trainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Training Loss', data: [], borderColor: '#36a2eb', borderWidth: 2, tension: 0.1, pointRadius: 1, yAxisID: 'y' },
                { label: 'Validation Loss', data: [], borderColor: '#ff6384', borderWidth: 2, tension: 0.1, pointRadius: 2, yAxisID: 'y' }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: { grid: { color: '#30363d' }, ticks: { color: '#888' } },
                y: { type: 'linear', display: true, position: 'left', grid: { color: '#30363d' }, ticks: { color: '#888' }, min: 0 }
            },
            plugins: { legend: { labels: { color: '#c9d1d9' } } }
        }
    });
}

function updateChart(logs) {
    if (!trainChart) return;
    const dataMap = new Map();
    logs.forEach(line => {
        try {
            if (!line.trim().startsWith('{')) return;
            const data = JSON.parse(line);
            if (data.loss || data.eval_loss) {
                const step = data.step || 0;
                if (!dataMap.has(step)) dataMap.set(step, { step, loss: null, eval_loss: null });
                const entry = dataMap.get(step);
                if (data.loss !== undefined) entry.loss = data.loss;
                if (data.eval_loss !== undefined) entry.eval_loss = data.eval_loss;
            }
        } catch (e) {}
    });
    const sorted = Array.from(dataMap.values()).sort((a, b) => a.step - b.step);
    trainChart.data.labels = sorted.map(d => d.step);
    trainChart.data.datasets[0].data = sorted.map(d => d.loss);
    trainChart.data.datasets[1].data = sorted.map(d => d.eval_loss);
    trainChart.update();
}


function splitStdoutStderr(lines) {
  const out = [];
  const err = [];
  for (const line of lines) {
    if (line.startsWith('[stderr] ')) {
      err.push(line.replace(/^\[stderr\]\s?/, ''));
    } else {
      out.push(line);
    }
  }
  return { stdout: out, stderr: err };
}

let textLogModalState = { jobId: null, logFile: null };

function showTextLogModal(jobId, logFile) {
  textLogModalState = { jobId, logFile };
  const modal = document.getElementById('text-log-modal');
  if (!modal) return;
  modal.classList.remove('hidden');
  document.getElementById('text-log-path').textContent = logFile ? logFile : '';
  refreshTextLogModal();
}


let _cdeModalBound = false;

function bindCDEModals() {
  if (_cdeModalBound) return;
  _cdeModalBound = true;

  // Log modal
  const logClose = document.getElementById('text-log-close');
  logClose?.addEventListener('click', hideTextLogModal);
  document.getElementById('text-log-refresh')?.addEventListener('click', refreshTextLogModal);

  // Dataset report modal
  document.getElementById('text-dataset-close')?.addEventListener('click', () => {
    document.getElementById('text-dataset-modal')?.classList.add('hidden');
  });

  // Env diff modal
  document.getElementById('text-envdiff-close')?.addEventListener('click', () => {
    document.getElementById('text-envdiff-modal')?.classList.add('hidden');
  });

  // Setup logs modal
  document.getElementById('setup-logs-close')?.addEventListener('click', () => {
    document.getElementById('setup-logs-modal')?.classList.add('hidden');
  });
}

async function showDatasetModal(jobId) {
  bindCDEModals();
  const modal = document.getElementById('text-dataset-modal');
  const pre = document.getElementById('text-dataset-content');
  if (!modal || !pre) return;
  pre.textContent = "取得中...";
  modal.classList.remove('hidden');
  try {
    const rep = await api.get(`/runs/dataset_report?job_id=${encodeURIComponent(jobId)}`);
    pre.textContent = JSON.stringify(rep, null, 2);
  } catch(e) {
    pre.textContent = "データ検査結果を取得できませんでした。\n" + e.message;
  }
}

async function showEnvDiffModal(jobId) {
  bindCDEModals();
  const modal = document.getElementById('text-envdiff-modal');
  const pre = document.getElementById('text-envdiff-content');
  if (!modal || !pre) return;
  pre.textContent = "取得中...";
  modal.classList.remove('hidden');
  try {
    const diff = await api.get(`/runs/env_diff?job_id=${encodeURIComponent(jobId)}`);
    pre.textContent = JSON.stringify(diff, null, 2);
  } catch(e) {
    pre.textContent = "環境差分を取得できませんでした。\n" + e.message;
  }
}

async function showSetupLogsModal() {
  bindCDEModals();
  const modal = document.getElementById('setup-logs-modal');
  const list = document.getElementById('setup-logs-list');
  const pre = document.getElementById('setup-logs-content');
  if (!modal || !list || !pre) return;
  list.innerHTML = "取得中...";
  pre.textContent = "";
  modal.classList.remove('hidden');
  try {
    const res = await api.get(`/utils/list_setup_logs`);
    const items = (res && res.items) ? res.items : [];
    if (!items.length) {
      list.innerHTML = "<div style='color:#666;'>setupログが見つかりません。</div>";
      return;
    }
    list.innerHTML = items.map(it => `
      <button class="action-btn js-open-setup-log" data-path="${it.path}">${it.name}</button>
    `).join(" ");
    list.querySelectorAll('.js-open-setup-log').forEach(btn=>{
      btn.addEventListener('click', async ()=>{
        const p = btn.getAttribute('data-path');
        if (!p) return;
        pre.textContent = "読み込み中...";
        try {
          const r = await api.get(`/utils/read_text_file?path=${encodeURIComponent(p)}&max_lines=1200`);
          pre.textContent = (r.lines||[]).join("");
        } catch(e) {
          pre.textContent = "読み込みに失敗しました: " + e.message;
        }
      });
    });
  } catch(e) {
    list.innerHTML = "取得失敗: " + e.message;
  }
}

function hideTextLogModal() {
  const modal = document.getElementById('text-log-modal');
  if (!modal) return;
  modal.classList.add('hidden');
}

async function refreshTextLogModal() {
  const st = textLogModalState;
  if (!st || !st.logFile) {
    setText(document.getElementById('text-log-stdout'), 'ログファイルがありません');
    setText(document.getElementById('text-log-stderr'), '');
    return;
  }
  try {
    const res = await api.get('/utils/read_text_file', { path: st.logFile, max_lines: 800 });
    const lines = res?.lines || [];
    const sp = splitStdoutStderr(lines);
    setText(document.getElementById('text-log-stdout'), sp.stdout.join('\n'));
    setText(document.getElementById('text-log-stderr'), sp.stderr.join('\n'));
  } catch (e) {
    setText(document.getElementById('text-log-stdout'), `取得失敗: ${e.message}`);
    setText(document.getElementById('text-log-stderr'), '');
  }
}


function bindHistoryActions() {
    bindCDEModals();
    const tbody = $('history-list-body');
    if (!tbody || tbody.__bound) return;
    tbody.__bound = true;
    tbody.addEventListener('click', async (ev) => {
        const btn = ev.target.closest('button');
        if (!btn) return;
        const jobId = btn.dataset.jobid;

        try {
            if (btn.classList.contains('js-open-log')) {
                const logFile = btn.dataset.logfile;
                await showTextLogModal(jobId, logFile);
                return;
            }
            if (btn.classList.contains('js-open-dataset-report')) {
                if (!jobId) return;
                await showDatasetModal(jobId);
                return;
            }
            if (btn.classList.contains('js-open-env-diff')) {
                if (!jobId) return;
                await showEnvDiffModal(jobId);
                return;
            }
            if (btn.classList.contains('js-open-setup-logs')) {
                await showSetupLogsModal();
                return;
            }
            if (btn.classList.contains('js-cancel-job')) {
                if (!jobId) return;
                if (!confirm('学習ジョブをキャンセルしますか？')) return;
                await api.post(`/train/cancel/${jobId}`, {});
                await refreshHistory();
                bindHistoryActions();
                return;
            }
            if (btn.classList.contains('js-rerun-job')) {
                if (!jobId) return;
                if (!confirm('この履歴の設定で再実行しますか？')) return;
                const res = await api.post(`/train/rerun/${jobId}`, {});
                const newJobId = res?.job_id || res?.jobId || jobId;
                alert('再実行を開始しました: job_id=' + newJobId);
                await refreshAll();
                return;
            }
        } catch (e) {
            alert('操作に失敗しました: ' + (e.message || e));
        }
    });
}

async function refreshHistory() {
    const tbody = $('history-list-body');
    if (!tbody) return;
    tbody.innerHTML = '<tr><td colspan="7">読み込み中...</td></tr>';
    try {
        const res = await api.get('/train/history');
        state.history = res.history || [];
        renderHistoryList();
    } catch (e) { tbody.innerHTML = `<tr><td colspan="7">エラー: ${e.message}</td></tr>`; }
}

function renderHistoryList() {
    const tbody = $('history-list-body');
    if (!tbody) return;
    tbody.innerHTML = '';
    if (!state.history.length) { tbody.innerHTML = '<tr><td colspan="7">履歴がありません</td></tr>'; return; }
    state.history.forEach(item => {
        const tr = document.createElement('tr');
        const lossVal = item.final_loss !== null ? item.final_loss?.toFixed(4) : '-';
        let statusBadge = item.status === 'completed' ? 'badge-green' : (item.status === 'failed' ? 'badge-red' : 'badge-gray');
        tr.innerHTML = `
            <td style="font-size:0.85rem;">${item.timestamp}</td><td>${item.model}</td><td>${item.dataset}</td>
            <td>${item.steps}</td><td>${lossVal}</td><td><span class="badge ${statusBadge}">${item.status}</span></td>
            <td>
              <div class="row-actions">
                <button class="action-btn js-open-log" data-job-id="${item.id}" title="ログ"><i class="fas fa-file-lines"></i> ログ</button>
                <button class="action-btn js-open-dataset-report" data-job-id="${item.id}" title="データ検査結果"><i class="fas fa-clipboard-check"></i> データ検査</button>
                <button class="action-btn js-open-env-diff" data-job-id="${item.id}" title="環境差分"><i class="fas fa-code-branch"></i> 環境差分</button>
                <button class="action-btn js-rerun-job" data-job-id="${item.id}" title="再実行"><i class="fas fa-rotate-right"></i></button>
                <button class="action-btn js-cancel-job" data-job-id="${item.id}" title="キャンセル"><i class="fas fa-stop"></i></button>
              </div>
            </td>`;
        tbody.appendChild(tr);
    });
}

// --- Training Logic ---

function initTrainUI() {
    const resumeCheck = $('train-resume');
    const resumeSelect = $('train-resume-path');
    const modelSelect = $('train-base-model');
    const promptSelect = $('train-prompt-template');
    const customGroup = $('train-custom-template-group');
    const presetSelect = $('train-preset-select');
    const applyPresetBtn = $('apply-train-preset-btn');
    const preflightBtn = $('run-preflight-btn');

    resumeCheck?.addEventListener('change', (e) => {
        resumeSelect.disabled = !e.target.checked;
        if (e.target.checked) updateCheckpointList();
    });

    modelSelect?.addEventListener('change', () => {
        if (resumeCheck?.checked) updateCheckpointList();
        runTrainPreflight().catch(()=>{});
    });

    applyPresetBtn?.addEventListener('click', () => {
        const id = presetSelect.value;
        if (!id || !TRAIN_PRESETS[id]) return alert('プリセットを選択してください');
        const p = TRAIN_PRESETS[id];
        $('train-lr').value = p.lr;
        $('train-lr-scheduler').value = p.scheduler;
        $('train-batch-size').value = p.batch_size;
        $('train-grad-acc').value = p.grad_acc;
        $('train-lora-r').value = p.lora_r;
        $('train-lora-alpha').value = p.lora_alpha;
        $('train-seq-len').value = p.seq_len;
        $('train-lora-target-mode').value = p.lora_target_mode;
        $('train-use-dora').checked = !!p.use_dora;
        $('train-use-flash-attn').checked = !!p.use_flash_attn;
        $('train-train-on-inputs').checked = !!p.train_on_inputs;
        $('train-neftune').checked = !!p.neftune;
        $('train-neftune-alpha').value = p.neftune_alpha;
        $('train-max-steps').value = p.max_steps;
        alert(`プリセット適用: ${p.name}`);
    });

    preflightBtn?.addEventListener('click', () => runTrainPreflight());
    promptSelect?.addEventListener('change', (e) => {
        if (customGroup) customGroup.classList.toggle('hidden', e.target.value !== 'custom');
    });

    ['train-seq-len', 'train-batch-size', 'train-grad-acc', 'train-lora-target-mode', 'train-use-flash-attn'].forEach(id => {
        $(id)?.addEventListener('change', () => runTrainPreflight().catch(()=>{}));
    });
}

async function updateCheckpointList() {
    const baseModel = $('train-base-model').value;
    const select = $('train-resume-path');
    if (!baseModel || !select) return;
    select.innerHTML = '<option value="">読み込み中...</option>';
    try {
        const res = await api.get(`/train/checkpoints?base_model=${baseModel}`);
        const cps = res.checkpoints || [];
        if (!cps.length) {
            select.innerHTML = '<option value="">(履歴なし)</option>';
        } else {
            let html = '<option value="">最新 (latest)</option>';
            cps.forEach(job => {
                html += `<optgroup label="${job.job_folder}">` + job.checkpoints.map(c => `<option value="${c.path}">${c.name} (Step ${c.step})</option>`).join('') + `</optgroup>`;
            });
            select.innerHTML = html;
        }
    } catch(e) { select.innerHTML = '<option value="">(エラー)</option>'; }
}

async function runTrainPreflight() {
    const gpuEl = $('preflight-gpu');
    const vramEl = $('preflight-vram-est');
    const flashEl = $('preflight-flash');
    const notesEl = $('preflight-notes');
    if (!gpuEl) return;

    const baseModel = $('train-base-model').value;
    const est = _estimateVramGB({
        modelName: baseModel,
        seqLen: $('train-seq-len').value,
        batchSize: $('train-batch-size').value,
        gradAcc: $('train-grad-acc').value,
        targetMode: $('train-lora-target-mode').value,
        useFlashAttn: $('train-use-flash-attn').checked
    });
    
    vramEl.textContent = `${est.low}〜${est.high} GB`;
    notesEl.textContent = est.notes.length ? est.notes.join(' / ') : '-';
    
    try {
        const sys = await api.get('/system_info');
        const vram = _parseNvidiaSmiVramTotalGB(sys.gpu_info);
        gpuEl.textContent = vram ? `VRAM 約${vram}GB` : '不明';
        
        const flashGuess = _guessFlashAttnFeasible(sys.gpu_info);
        flashEl.textContent = $('train-use-flash-attn').checked ? (flashGuess ? 'ON (OK)' : 'ON (注意)') : 'OFF';
    } catch(e) { gpuEl.textContent = '取得失敗'; }
}

async function startTraining() {
    const baseModel = $('train-base-model').value;
    const dataset = $('train-dataset').value;
    if (!baseModel || !dataset) return alert('モデルとデータセットを選択してください');
    
    // Params Construction
    const params = {
        max_steps: parseInt($('train-max-steps').value),
        learning_rate: parseFloat($('train-lr').value),
        per_device_train_batch_size: parseInt($('train-batch-size').value),
        gradient_accumulation_steps: parseInt($('train-grad-acc').value),
        lora_r: parseInt($('train-lora-r').value),
        lora_alpha: parseInt($('train-lora-alpha').value),
        max_seq_length: parseInt($('train-seq-len').value),
        lora_target_mode: $('train-lora-target-mode').value,
        use_dora: $('train-use-dora').checked,
        use_flash_attention_2: $('train-use-flash-attn').checked,
        train_on_inputs: $('train-train-on-inputs').checked,
        lr_scheduler_type: $('train-lr-scheduler').value,
        
        eval_score_enabled: $('train-eval-score-enabled')?.checked ?? true,
        eval_score_min_len: parseInt($('train-eval-score-min-len')?.value || 40),
        eval_score_max_len: parseInt($('train-eval-score-max-len')?.value || 800),
        eval_score_repetition_ngram: parseInt($('train-eval-score-rep-ngram')?.value || 6),
        eval_score_repetition_threshold: parseFloat($('train-eval-score-rep-threshold')?.value || 0.35),
        eval_score_require_json_if_prompt_mentions_json: $('train-eval-score-require-json')?.checked ?? true,
        eval_score_banned_phrases: $('train-eval-score-banned')?.value || null,
        
        prompt_template: ($('train-prompt-template').value === 'custom') ? $('train-custom-template-content').value : $('train-prompt-template').value
    };
    
    if (params.prompt_template === 'custom' && !params.prompt_template) return alert('テンプレートを入力してください');

    const evalPromptsRaw = $('train-eval-prompts')?.value;
    let evalPrompts = null;
    if (evalPromptsRaw?.trim()) {
        try { evalPrompts = JSON.parse(evalPromptsRaw); } catch(e) { return alert('評価プローブのJSONが無効です'); }
    }

    if (!confirm('学習を開始しますか？')) return;
    
    state.training.processedLogIndex = 0;
    if (trainChart) { trainChart.data.labels=[]; trainChart.data.datasets.forEach(d=>d.data=[]); trainChart.update(); }
    $('train-gen-log').innerHTML = '<div class="system-msg">待機中...</div>';
    
    try {
        state.training.status = 'starting...';
        updateTrainUIStatus();
        
        await api.post('/train/start', {
            base_model: baseModel,
            dataset,
            dataset_type: dataset.endsWith('.jsonl') ? 'instruction' : 'raw_text',
            params,
            resume_from_checkpoint: $('train-resume').checked ? ($('train-resume-path').value || "latest") : null,
            neftune_noise_alpha: $('train-neftune').checked ? parseFloat($('train-neftune-alpha').value) : null,
            validation_file: $('train-val-dataset').value || null,
            validation_prompt: $('train-val-prompt').value || null,
            eval_prompts: evalPrompts
        });
        
        state.training.status = 'running';
        updateTrainUIStatus();
        startLogPolling();
        
        if (state.inference.isLoaded) {
            state.inference.isLoaded = false;
            $('inf-load-status').textContent = '未ロード (学習優先のため解放)';
            $('inf-load-btn').classList.remove('hidden');
            $('inf-unload-btn').classList.add('hidden');
        }

    } catch (e) {
        alert(e.message);
        state.training.status = 'failed';
        updateTrainUIStatus();
    }
}

async function stopTraining() {
    if (!confirm('停止しますか？')) return;
    await api.post('/train/stop', {});
    state.training.status = 'stopped';
    updateTrainUIStatus();
}

function updateTrainUIStatus() {
    const badge = $('train-status-badge');
    if (badge) {
        badge.textContent = state.training.status;
        badge.className = `badge ${state.training.status === 'running' ? 'badge-green' : 'badge-gray'}`;
    }
    const startBtn = $('start-train-btn');
    const stopBtn = $('stop-train-btn');
    if (state.training.status === 'running' || state.training.status === 'starting...') {
        startBtn?.classList.add('hidden');
        stopBtn?.classList.remove('hidden');
    } else {
        startBtn?.classList.remove('hidden');
        stopBtn?.classList.add('hidden');
    }
}

function startLogPolling() {
    if (state.training.pollInterval) clearInterval(state.training.pollInterval);
    state.training.pollInterval = setInterval(async () => {
        try {
            const data = await api.get('/train/status');
            state.training.status = data.status;
            processLogs(data.logs || []);
            updateTrainUIStatus();
            if (data.status !== 'running') {
                clearInterval(state.training.pollInterval);
                if (data.status === 'succeeded' || data.status === 'completed') {
                    try { refreshModels(); } catch(_) {}
                    window.dispatchEvent(new CustomEvent('lora:job_succeeded', {detail:{modality:'text'}}));
                }
            }
        } catch(e) {}
    }, 2000);
}

function processLogs(logs) {
    const consoleEl = $('train-console');
    const genLogEl = $('train-gen-log');
    if (!consoleEl) return;
    
    // Console Log
    consoleEl.textContent = logs.join('\n');
    consoleEl.scrollTop = consoleEl.scrollHeight;
    
    // Chart
    updateChart(logs);
    
    // Gen Log
    const gens = [];
    logs.forEach(l => {
        try {
            if (!l.startsWith('{')) return;
            const d = JSON.parse(l);
            if (d.type === 'generation' || d.type === 'eval_probe') gens.push(d);
        } catch(e) {}
    });
    
    if (gens.length && genLogEl) {
        const html = gens.map(g => {
            const score = g.score ? `<span class="badge">Score: ${g.score.score}</span>` : '';
            return `<div class="gen-item" style="border-bottom:1px solid #333; padding:5px;">
                <div style="font-size:0.8rem; color:#888;">Step: ${g.step} ${score}</div>
                <div style="color:#ddd;">${escapeHtml(g.prompt)}</div>
                <div style="color:#aaffaa; font-family:monospace;">${escapeHtml(g.output)}</div>
            </div>`;
        }).join('');
        if (genLogEl.innerHTML !== html) {
            genLogEl.innerHTML = html;
            genLogEl.scrollTop = genLogEl.scrollHeight;
        }
    }
}

// --- Inference Logic ---

async function loadInferenceModel() {
    const base = $('inf-base-model').value;
    const adapter = $('inf-adapter-path').value.trim() || null;
    const statusText = $('inf-load-status');
    statusText.textContent = 'ロード中...';
    try {
        await api.post('/inference/load', { base_model: base, adapter_path: adapter });
        state.inference.isLoaded = true;
        statusText.textContent = 'ロード済み';
        $('inf-load-btn').classList.add('hidden');
        $('inf-unload-btn').classList.remove('hidden');
    } catch (e) { statusText.textContent = '失敗: ' + e.message; }
}

async function unloadInferenceModel() {
    try {
        await api.post('/inference/unload', {});
        state.inference.isLoaded = false;
        $('inf-load-status').textContent = '未ロード';
        $('inf-load-btn').classList.remove('hidden');
        $('inf-unload-btn').classList.add('hidden');
    } catch(e) { alert(e.message); }
}

async function sendChatMessage() {
    const input = $('chat-input');
    const text = input.value.trim();
    if (!text || !state.inference.isLoaded || state.inference.isGenerating) return;
    
    appendMessage('user', text);
    input.value = '';
    state.inference.isGenerating = true;
    
    const assistantMsgDiv = appendMessage('assistant', '');
    const contentDiv = assistantMsgDiv.querySelector('.msg-content');
    
    try {
        const response = await fetch(`${api.API_BASE || '/api'}/inference/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message: text, 
                temperature: parseFloat($('inf-temp').value), 
                max_tokens: parseInt($('inf-max-tokens').value),
                repetition_penalty: parseFloat($('inf-rep-pen').value),
                top_p: parseFloat($('inf-top-p').value)
            })
        });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let generatedText = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            generatedText += decoder.decode(value, { stream: true });
            contentDiv.textContent = generatedText;
            $('chat-history').scrollTop = $('chat-history').scrollHeight;
        }
    } catch (e) { contentDiv.textContent += ` [エラー: ${e.message}]`; }
    finally { state.inference.isGenerating = false; }
}

function appendMessage(role, text) {
    const container = $('chat-history');
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    div.innerHTML = `<div class="msg-role">${role === 'user' ? 'ユーザー' : 'AI'}</div><div class="msg-content">${text}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}





async function validateDatasetAndToggleStart_text() {
  const btn = document.getElementById("start-train-btn");
  const sel = document.getElementById("train-dataset");
  if(!btn || !sel) return;
  const warnId = "text-dataset-validate";
  let warn = document.getElementById(warnId);
  if(!warn) {
    warn = document.createElement("pre");
    warn.id = warnId;
    warn.className = "warn";
    warn.style.whiteSpace = "pre-wrap";
    warn.style.display = "none";
    // place near select if possible
    sel.parentElement?.appendChild(warn);
  }
  const dataset = sel.value;
  if(!dataset) {
    btn.disabled = true;
    warn.textContent = "データセットを選択してください";
    warn.style.display = "block";
    return;
  }
  let kind = null;

  try {
    const res = await api.post("/api/datasets/validate", { mode: "text", dataset, kind });
    if(res && res.ok) {
      btn.disabled = false;
      const msg = (res.messages||[]).join("\n");
      warn.textContent = msg;
      warn.style.display = msg ? "block" : "none";
    } else {
      btn.disabled = true;
      const msg = (res && res.messages) ? res.messages.join("\n") : "データセット検査に失敗しました";
      warn.textContent = msg;
      warn.style.display = "block";
    }
  } catch(e) {
    btn.disabled = true;
    warn.textContent = "データセット検査APIに接続できません";
    warn.style.display = "block";
  }
}



window.textUI = window.textUI || {}; window.textUI.refreshModels = refreshModels;