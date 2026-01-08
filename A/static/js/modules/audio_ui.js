/**
 * static/js/modules/audio_ui.js
 * Audio (Voice / GPT-SoVITS) モード専用UI。
 *
 * - データセット選択（datasets/audio 配下）
 * - 学習（前処理 + 外部 GPT-SoVITS 学習スクリプト委譲）
 * - 推論（テキスト→音声, 参照音声指定）
 */

import { api } from '../api.js';
import { renderSystemInfoTab, fetchPathsSafe } from './system_info_view.js';

// HTMLエスケープ（UI表示用）
function escapeHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}


// ------------------------------------------------------------
// DOM Helpers
// ------------------------------------------------------------
// このWebUIは jQuery を使わない設計です。
// 以前のコード由来で $()（getElementById の短縮）が参照されているため、
// モジュール単体でも確実に動作するようにここで提供します。
// 既存の ID ベース参照の挙動は変更しません。
var $ = window.$ || function(id){ return document.getElementById(id); };
window.$ = $;

function notify(msg) {
  try { alert(msg); } catch { console.log(msg); }
}

async function openFolder({ key=null, path=null }) {
  try {
    await api.post('/utils/open_path', { key, path });
  } catch (e) {
    notify(`フォルダを開けませんでした: ${e.message || e}`);
  }
}

function _mkOpenBtn(label, onClick) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'open-folder-btn';
  btn.innerHTML = `<i class="fas fa-folder-open"></i> ${label}`;
  btn.addEventListener('click', onClick);
  return btn;
}

function installOpenFolderButtons() {
  // 推論（Text → Voice）
  const modelSel = $('aud-model-select');
  if (modelSel) {
    _appendOpenRow(modelSel, [
      _mkOpenBtn('モデルフォルダを開く (models/audio)', () => openFolder({ key: 'models_audio' })),
    ]);
  }

  // データセット（音声）: datasets/audio 配下のフォルダ選択
  const dsSel = $('aud-dataset-select');
  if (dsSel) {
    _appendOpenRow(dsSel, [
      _mkOpenBtn('datasets/audio を開く', () => openFolder({ key: 'datasets_audio' })),
      _mkOpenBtn('選択中フォルダを開く', () => {
        const v = (dsSel.value || '').trim();
        if (!v) return notify('データセットを選択してください');
        openFolder({ path: `datasets/audio/${v}` });
      }),
    ]);
  }

  // 学習（音声）: データセット選択
  const trainDsSel = $('aud-train-dataset');
  if (trainDsSel) {
    _appendOpenRow(trainDsSel, [
      _mkOpenBtn('datasets/audio を開く', () => openFolder({ key: 'datasets_audio' })),
      _mkOpenBtn('選択中フォルダを開く', () => {
        const v = (trainDsSel.value || '').trim();
        if (!v) return notify('データセットを選択してください');
        openFolder({ path: `datasets/audio/${v}` });
      }),
    ]);
  }

  // 参照音声（スタイル）
  const ref = $('aud-ref-audio');
  if (ref) {
    _appendOpenRow(ref, [
      _mkOpenBtn('datasets/audio を開く', () => openFolder({ key: 'datasets_audio' })),
      _mkOpenBtn('入力パスの場所を開く', () => {
        const v = (ref.value || '').trim();
        if (!v) return notify('参照音声（パス）を入力してください');
        openFolder({ path: v });
      }),
    ]);
  }

  // GPT-SoVITS / RVC 関連（任意）
  const maybeRepo = [
    'aud-rvc-repo',
    'aud-gpt-vc-repo',
    'aud-infer-repo',
    'aud-gpt-repo',
  ];
  for (const id of maybeRepo) {
    const el = $(id);
    if (!el) continue;
    _appendOpenRow(el, [
      _mkOpenBtn('このパスのフォルダを開く', () => {
        const v = (el.value || '').trim();
        if (!v) return notify('パスを入力してください');
        openFolder({ path: v });
      })
    ]);
  }
}

function _appendOpenRow(afterEl, buttons) {
  if (!afterEl || !afterEl.parentElement) return;
  const row = document.createElement('div');
  row.className = 'open-folder-row';
  buttons.forEach(b => row.appendChild(b));
  afterEl.parentElement.appendChild(row);
}

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
  <button class="nav-btn" data-target="tab-train">
    <i class="fas fa-microphone"></i> 学習（音声）
  </button>
  <button class="nav-btn" data-target="tab-infer">
    <i class="fas fa-music"></i> 推論（読み上げ）
  </button>
`;

const CONTENT_HTML = `
  <section id="tab-system" class="tab-pane active">
    <header class="pane-header">
      <h3>システムモニター</h3>
      <button id="aud-refresh-sys" class="action-btn"><i class="fas fa-sync"></i> 更新</button>
    </header>
    <div class="pane-body">
      <details class="help-details">
        <summary><i class="fas fa-circle-question"></i> はじめての方へ（音声の基本）</summary>
        <div class="help-content">
          <ol>
            <li><strong>データセット</strong>：<code>datasets/audio</code> に音声フォルダを用意します（wav/mp3 など）。</li>
            <li><strong>学習</strong>：まず前処理（スライス + 文字起こし）が走ります。必要なら GPT-SoVITS 本体へ学習を委譲します。</li>
            <li><strong>推論</strong>：モデルをロードし、参照音声（話し方のスタイル）とテキストを指定して生成します。</li>
          </ol>
          <div class="help-note">
            <p><strong>重要</strong>：音声は <strong>XTTS（推奨）</strong> を中心に扱います。XTTSは Pythonライブラリ（Coqui TTS）として導入するか、外部CLIへ委譲できます。VC（声質変換）として <strong>RVC / GPT-SoVITS(VC)</strong> も併用できます。<br>環境変数例：<code>GPT_SOVITS_DIR</code> / <code>XTTS_DIR</code> / <code>RVC_DIR</code> / <code>GPT_SOVITS_VC_DIR</code>。</p>
          </div>
        </div>
      </details>
      <div class="card">
        <h4>音声（Voice）用システム情報</h4>
        <div class="system-info-container" id="aud-system-info-container">取得中...</div>
        <small class="hint">音声の学習/生成に関係する主な情報（GPU/VRAM, PyTorch/CUDA, torchaudio/librosa, ffmpeg, GPT-SoVITS パス等）を表示します。</small>
      </div>
</div>
    </div>
  </section>


  <section id="tab-models" class="tab-pane">
    <header class="pane-header">
      <h3>モデル管理（音声）</h3>
      <button id="audio-models-refresh" class="action-btn"><i class="fas fa-sync"></i> 更新</button>
      <button id="audio-models-open-root" class="action-btn"><i class="fas fa-folder-open"></i> フォルダ</button>
    </header>
    <div class="pane-body">
      <div class="card" style="margin-top:12px;">
        <h4>HuggingFaceからダウンロード</h4>
        <div class="input-group">
          <input id="audio-hf-repo-id" type="text" placeholder="Repo ID（例: coqui/XTTS-v2）" />
          <button id="audio-download-btn" class="primary-btn"><i class="fas fa-download"></i> ダウンロード</button>
        </div>
        <small class="hint">完了後に「更新」を押すと一覧へ反映されます。</small>
      </div>
      <div class="card">
        <h4>ローカルモデル一覧</h4>
        <table class="data-table">
          <thead><tr><th>モデル名</th><th>種別</th><th>操作</th></tr></thead>
          <tbody id="audio-models-body"><tr><td colspan="3">未取得</td></tr></tbody>
        </table>
        <div id="audio-model-detail" class="note-box" style="display:none;margin-top:10px;"></div>
      </div>
    </div>
  </section>



  <section id="tab-datasets" class="tab-pane">
    <header class="pane-header">
      <h3>データセット（音声）</h3>
      <button id="aud-refresh-datasets" class="action-btn"><i class="fas fa-sync"></i> 更新</button>
    </header>
    <div class="pane-body">
      <div class="card">
        <h4>datasets/audio 配下のフォルダ</h4>
        <div class="form-group">
          <label>データセット</label>
          <select id="aud-dataset-select"></select>
          <small class="hint">音声ファイル（wav/mp3 等）を入れたフォルダを選びます。</small>
        </div>
        <div class="form-group">
          <label>説明</label>
          <div class="console-log small-log" id="aud-dataset-info">(未選択)</div>
        </div>
      </div>
    </div>
  </section>

  <section id="tab-train" class="tab-pane">
    <header class="pane-header">
      <h3>学習（LoRA/モデル作成）</h3>
      <div style="display:flex; gap:8px; align-items:center;">
        <button id="aud-train-start" class="primary-btn"><i class="fas fa-play"></i> 開始</button>
        <button id="aud-train-stop" class="action-btn" style="background:var(--danger); color:white;"><i class="fas fa-stop"></i> 停止</button>
        <button id="aud-train-refresh" class="action-btn"><i class="fas fa-sync"></i> 状態更新</button>
      </div>
    </header>
    <div class="pane-body">
      <div class="columns">
        <div class="card">
          <h4>学習設定（基本）</h4>
          <div class="card" id="aud-thirdparty-warn" style="border:1px solid var(--danger); display:none;">
  <h4 style="margin:0 0 8px 0;">外部リポジトリが見つかりません</h4>
  <div class="console-log small-log" id="aud-thirdparty-warn-text"></div>
</div>

<div class="form-group">
  <label>学習タイプ</label>
  <select id="aud-train-type">
    <option value="xtts_finetune">TTS：XTTS v2（話者適応 finetune）</option>
    <option value="gpt_sovits">VC：GPT-SoVITS（学習委譲）</option>
  </select>
  <small class="hint">自動cloneはしません。事前に <code>third_party</code> に clone 済みである必要があります。</small>
</div>

<div class="form-group">
  <label>データセット</label>
  <select id="aud-train-dataset"></select>
</div>

          <div class="form-group">
            <label>Whisperモデル（文字起こし）</label>
            <input id="aud-whisper-model" type="text" value="small" />
            <small class="hint">例: tiny / base / small / medium / large-v3（大きいほど高精度・重い）</small>
          </div>
          <div class="form-group">
  <label>XTTS 追加引数（任意）</label>
  <input id="aud-xtts-extra" type="text" value="" placeholder="例: --continue_path outputs/..." />
  <small class="hint">TTS側の引数差異吸収用（通常は空でOK）。</small>
</div>

<div class="form-group">
  <label>言語</label>
  <input id="aud-language" type="text" value="ja" />

            <small class="hint">例: ja / en / auto（環境により）</small>
          </div>
          <div class="form-group">
            <label>スライス（最小〜最大 秒）</label>
            <div style="display:flex; gap:8px;">
              <input id="aud-slice-min" type="number" value="3" step="0.5" />
              <input id="aud-slice-max" type="number" value="10" step="0.5" />
            </div>
            <small class="hint">短すぎる音声は学習に不向きです。まずは 3〜10 秒が目安。</small>
          </div>
        </div>

        <div class="card">
          <h4>GPT-SoVITS 連携（任意）</h4>
          <div class="form-group">
            <label>GPT-SoVITS リポジトリのパス</label>
            <input id="aud-gpt-repo" type="text" placeholder="例: C:/AI/GPT-SoVITS" />
            <small class="hint">指定しない場合、前処理（train.list 作成）だけ実行します。</small>
          </div>
          <div class="form-group">
            <label>カスタム学習コマンド（上級者向け）</label>
            <textarea id="aud-custom-train" rows="3" placeholder="例: python cli/train.py --prepared_dir {prepared_dir} --output_dir {output_dir}"></textarea>
            <small class="hint">{prepared_dir} と {output_dir} が置換されます。</small>
          </div>
        </div>
      </div>

      <div class="card">
        <h4>学習ログ</h4>
        <pre id="aud-train-log" class="console-log">(待機中)</pre>
      </div>
    </div>
  </section>

  <section id="tab-infer" class="tab-pane">
    <header class="pane-header">
      <h3>推論（Text → Voice）</h3>
      <div style="display:flex; gap:8px; align-items:center;">
        <button id="aud-model-refresh" class="action-btn"><i class="fas fa-sync"></i> モデル更新</button>
        <button id="aud-model-load" class="primary-btn"><i class="fas fa-cubes"></i> モデルロード</button>
      </div>
    </header>
    <div class="pane-body">
      <div class="columns">
        <div class="card">
          <h4>モデル</h4>
          <div class="form-group">
            <label>models/audio</label>
            <select id="aud-model-select"></select>
          </div>
          <div class="form-group">
            <label>TTSバックエンド</label>
            <select id="aud-tts-backend">
              <option value="xtts">XTTS（推奨）</option>
              <option value="gpt_sovits">GPT-SoVITS（従来）</option>
            </select>
            <small class="hint">XTTSが使えない場合は自動でGPT-SoVITSへフォールバックします。</small>
          </div>
          <div class="form-group">
            <label>XTTS モデルID（任意）</label>
            <input id="aud-xtts-model-id" type="text" placeholder="例: tts_models/multilingual/multi-dataset/xtts_v2" />
            <small class="hint">空欄なら既定モデルを使用します（導入環境による）。</small>
          </div>
          <div class="form-group">
            <label>XTTS 言語</label>
            <input id="aud-xtts-lang" type="text" value="ja" />
            <small class="hint">例: ja / en など。日本語は ja 推奨。</small>
          </div>
          <div class="form-group">
            <label>VC（声質変換）</label>
            <select id="aud-vc-backend">
              <option value="none">なし</option>
              <option value="rvc">RVC（外部委譲）</option>
              <option value="gpt_sovits_vc">GPT-SoVITS(VC)（外部委譲）</option>
            </select>
            <small class="hint">まずXTTSで読み上げを作り、必要ならVCで声質を変換します。</small>
          </div>
          <div class="form-group">
            <label>RVC リポジトリのパス（任意）</label>
            <input id="aud-rvc-repo" type="text" placeholder="例: C:/AI/RVC" />
            <small class="hint">環境変数 <code>RVC_DIR</code> でも指定できます。</small>
          </div>
          <div class="form-group">
            <label>RVC VC カスタムコマンド（任意）</label>
            <textarea id="aud-rvc-cmd" rows="2" placeholder="例: python infer.py --input {in_wav} --ref {ref} --output {out}"></textarea>
            <small class="hint">{repo} {in_wav} {ref} {out} が置換されます。環境変数 <code>RVC_CUSTOM_VC_CMD</code> でも指定できます。</small>
          </div>
          <div class="form-group">
            <label>GPT-SoVITS(VC) リポジトリのパス（任意）</label>
            <input id="aud-gpt-vc-repo" type="text" placeholder="例: C:/AI/GPT-SoVITS" />
            <small class="hint">環境変数 <code>GPT_SOVITS_VC_DIR</code> でも指定できます。</small>
          </div>
          <div class="form-group">
            <label>GPT-SoVITS(VC) カスタムコマンド（任意）</label>
            <textarea id="aud-gpt-vc-cmd" rows="2" placeholder="例: python vc.py --input {in_wav} --ref {ref} --output {out}"></textarea>
            <small class="hint">{repo} {in_wav} {ref} {out} が置換されます。環境変数 <code>GPT_SOVITS_VC_CUSTOM_CMD</code> でも指定できます。</small>
          </div>

          <div class="form-group">
            <label>GPT-SoVITS リポジトリのパス（推論）</label>
            <input id="aud-infer-repo" type="text" placeholder="例: C:/AI/GPT-SoVITS" />
            <small class="hint">学習と同じでもOK。指定しない場合、環境変数 GPT_SOVITS_DIR を参照します。</small>
          </div>
          <div class="form-group">
            <label>カスタム推論コマンド（上級者向け）</label>
            <textarea id="aud-custom-infer" rows="3" placeholder="例: python cli/infer.py --model_dir {model_dir} --ref {ref} --text_file {text_file} --out {out}"></textarea>
            <small class="hint">{model_dir} {ref} {text_file} {out} {repo} が置換されます。</small>
          </div>
        </div>

        <div class="card">
          <h4>生成</h4>
          <div class="form-group">
            <label>参照音声（スタイル）</label>
            <input id="aud-ref-audio" type="text" placeholder="例: datasets/audio/xxx/sample.wav" />
            <small class="hint">モデルの話し方を決める参照音声のパス（ローカルパス）。</small>
          </div>
          <div class="form-group">
            <label>喋らせたいテキスト</label>
            <textarea id="aud-text" rows="4" placeholder="ここにテキストを入力"></textarea>
          </div>
          <div class="form-group">
            <label>プリセット</label>
            <select id="aud-preset">
              <option value="natural">自然（おすすめ）</option>
              <option value="clear">ハキハキ</option>
            </select>
            <small class="hint">バックエンド側で「正規化・韻律・音量正規化」を標準適用します。</small>
          </div>
          <div class="form-group">
            <label style="display:flex; gap:8px; align-items:center;">
              <input type="checkbox" id="aud-stream" checked />
              長文モード（自動分割して安定化）
            </label>
            <small class="hint">長い文章は自動で分割して結合します（WAVのみ）。</small>
          </div>
          <div class="form-group" id="aud-stream-options" style="display:block;">
            <label>1チャンク最大文字数</label>
            <input id="aud-chunk-len" type="number" step="10" value="120" />
          </div>
          <button id="aud-generate" class="primary-btn"><i class="fas fa-wand-magic-sparkles"></i> 生成</button>
          <div style="margin-top:12px;">
            <audio id="aud-player" controls style="width:100%;"></audio>
          </div>
          <pre id="aud-infer-log" class="console-log small-log">(待機中)</pre>
        </div>
      </div>
    </div>
  </section>
`;

function bindTabSwitch(navRoot, contentRoot) {
  const buttons = navRoot.querySelectorAll('.nav-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.target;
      contentRoot.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      const pane = contentRoot.querySelector('#' + target);
      if (pane) pane.classList.add('active');
    });
  });
}

async function refreshSystem() {
  const elSummary = document.getElementById('aud-system-info-container');
  if (elSummary) elSummary.textContent = '取得中...';
  try {
    const data = await api.get('/system_info/audio');
    const paths = await fetchPathsSafe();
    renderSystemInfoTab(elSummary, data, paths, 'audio');
  } catch (e) {
    if (elSummary) {
      elSummary.innerHTML = '<div class="system-msg error">システム情報の取得に失敗しました。ログを確認してください。</div>';
    }
  }
}

async function refreshDatasets() {
  const sel = document.getElementById('aud-dataset-select');
  const sel2 = document.getElementById('aud-train-dataset');
  const info = document.getElementById('aud-dataset-info');
  try {
    const data = await api.get('/audio/datasets');
    const items = data.datasets || [];
    sel.innerHTML = '';
    sel2.innerHTML = '';
    for (const it of items) {
      const opt = document.createElement('option');
      opt.value = it.name;
      opt.textContent = it.name;
      sel.appendChild(opt);
      const opt2 = document.createElement('option');
      opt2.value = it.name;
      opt2.textContent = it.name;
      sel2.appendChild(opt2);
    }
    if (items.length) {
      info.textContent = `選択中: ${items[0].name}\nパス: ${items[0].path}`;
    } else {
      info.textContent = 'datasets/audio にフォルダがありません。';
    }
    sel.addEventListener('change', () => {
      const it = items.find(x => x.name === sel.value);
      info.textContent = it ? `選択中: ${it.name}\nパス: ${it.path}` : '(未選択)';
    });
  } catch (e) {
    info.textContent = e.message;
  }
}

async function refreshModels() {
  const sel = document.getElementById('aud-model-select');
  try {
    const data = await api.get('/audio/models');
    const items = data.models || [];
    sel.innerHTML = '';
    for (const it of items) {
      const opt = document.createElement('option');
      opt.value = it.name;
      opt.textContent = it.name;
      sel.appendChild(opt);
    }
  } catch (e) {
    // ここは UI のみ
    sel.innerHTML = '';
  }
}

async function pollTrainStatus() {
  const log = document.getElementById('aud-train-log');
  try {
    const st = await api.get('/audio/train/status');
    const lines = (st.logs || []).join('\n');
    log.textContent = lines || '(ログなし)';
  } catch (e) {
    log.textContent = e.message;
  }
}

export async function refreshAudioModelManager() {
  const tbody = $('audio-models-body');
  if (!tbody) return;
  tbody.innerHTML = '<tr><td colspan="3">読み込み中...</td></tr>';
  try {
    const res = await api.get('/audio/models');
    const models = res?.models || [];
    if (!models.length) {
      tbody.innerHTML = '<tr><td colspan="3">モデルがありません</td></tr>';
      return;
    }
    tbody.innerHTML = '';
    for (const item of models) {
      const model = (typeof item === 'string') ? { name: item } : (item || {});
      const name = model.name || String(item);
      const type = model.type ? String(model.type) : '';

      const tr = document.createElement('tr');
      tr.innerHTML =
        '<td>' + escapeHtml(name) + '</td>' +
        '<td>' + (type ? ('<span class="badge ' + (type === 'hf' ? 'badge-blue' : 'badge-gray') + '">' + escapeHtml(type) + '</span>') : '<span class="badge badge-gray">local</span>') + '</td>' +
        '<td><div class="row-actions">' +
          '<button class="icon-btn js-model-meta" data-name="' + escapeHtml(name) + '" title="詳細"><i class="fas fa-circle-info"></i></button>' +
          '<button class="icon-btn js-model-open" data-name="' + escapeHtml(name) + '" title="フォルダ"><i class="fas fa-folder-open"></i></button>' +
          '<button class="icon-btn danger-btn js-model-del" data-name="' + escapeHtml(name) + '" title="削除"><i class="fas fa-trash"></i></button>' +
        '</div></td>';
      tbody.appendChild(tr);
}
  } catch(e) {
    tbody.innerHTML = '<tr><td colspan="3">取得失敗: ' + escapeHtml(e.message) + '</td></tr>';
  }
}

function bindAudioModelManager() {
  // allow detail box inline button
  window.__open_audio_model_folder = async (name) => {
    const rel = 'models/audio/' + name;
    await openFolder({ path: rel });
  };

  const tbody = $('audio-models-body');
  const btn = $('audio-models-refresh');
  if (btn && !btn.__bound) {
    btn.__bound = true;
    btn.addEventListener('click', refreshAudioModelManager);
  }
    const openRootBtn = $('audio-models-open-root');
  if (openRootBtn && !openRootBtn.__bound) {
    openRootBtn.__bound = true;
    openRootBtn.addEventListener('click', () => openFolder({ key: 'models_audio' }));
  }
const dlBtn = $('audio-download-btn');
  if (dlBtn && !dlBtn.__bound) {
    dlBtn.__bound = true;
    dlBtn.addEventListener('click', async () => {
      const repoEl = $('audio-hf-repo-id');
      const repoId = (repoEl?.value || '').trim();
      if (!repoId) {
        alert('Repo ID を入力してください。');
        return;
      }
      try {
        await api.post('/audio/models/download', { repo_id: repoId });
        alert('ダウンロードを開始しました。完了後に「更新」を押してください。');
        // 自動更新（軽め）
        setTimeout(() => refreshAudioModelManager(), 1500);
      } catch (e) {
        alert('ダウンロード開始に失敗しました: ' + (e?.message || e));
      }
    });
  }

  if (!tbody || tbody.__bound) return;
  tbody.__bound = true;
  tbody.addEventListener('click', async (ev) => {
    const b = ev.target.closest('button');
    if (!b) return;
    const name = b.dataset.name;
    if (!name) return;
    if (b.classList.contains('js-model-meta')) {
      try {
        const meta = await api.get('/audio/models/' + encodeURIComponent(name) + '/meta');
        const box = $('audio-model-detail');
        if (box) {
          const mb = (meta.total_bytes || 0) / (1024*1024);
          box.innerHTML =
            '<div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">'
          +   '<div><strong>' + escapeHtml(meta.name) + '</strong></div>'
          +   '<button class="mini-btn js-model-open" data-name="' + escapeHtml(meta.name) + '">フォルダを開く</button>'
          + '</div>'
          + '<div>保存先: <code>' + escapeHtml(meta.path) + '</code></div>'
          + '<div>更新日時: ' + escapeHtml(meta.mtime || '') + '</div>'
          + '<div>合計サイズ: ' + mb.toFixed(1) + ' MB / ファイル数: ' + (meta.file_count || 0) + '</div>'
          + (Array.isArray(meta.files_top) && meta.files_top.length
              ? ('<details style="margin-top:6px;"><summary>大きいファイル（上位）</summary>'
                 + '<ul style="margin:6px 0 0 18px;">'
                 + meta.files_top.slice(0, 12).map(f => {
                     const fmb = (f.size || 0) / (1024*1024);
                     return '<li><code>' + escapeHtml(f.path) + '</code> (' + fmb.toFixed(1) + ' MB)</li>';
                   }).join('')
                 + '</ul></details>')
              : '');
          box.style.display = 'block';
        } else {
          alert(JSON.stringify(meta, null, 2));
        }
      } catch(e) {
        alert('詳細取得失敗: ' + e.message);
      }
    } else if (b.classList.contains('js-model-open')) {
      // モデルフォルダを開く（相対パスで安全に解決）
      const rel = 'models/audio/' + name;
      await openFolder({ path: rel });
    
} else if (b.classList.contains('js-model-del')) {
      try {
        const chk = await api.get('/audio/models/' + encodeURIComponent(name) + '/predelete_check');
        let msg = 'モデル「' + name + '」を削除しますか？\n参照数: ' + (chk.reference_count || 0);
        if (chk.active_job_using) msg += '\n※現在の実行中ジョブがこのモデルを使用しています。';
        if (!confirm(msg)) return;
        await api.delete('/audio/models/' + encodeURIComponent(name));
        await refreshAudioModelManager();
      } catch(e) {
        alert('削除失敗: ' + e.message);
      }
    }
  })

  const detailBox = $('audio-model-detail');
  if (detailBox && !detailBox.__bound) {
    detailBox.__bound = true;
    detailBox.addEventListener('click', async (ev) => {
      const b = ev.target.closest('button');
      if (!b) return;
      const name = b.dataset.name;
      if (!name) return;
      if (b.classList.contains('js-model-open')) {
        const rel = 'models/audio/' + name;
        await openFolder({ path: rel });
      }
    });
  }
;
}

export async function mount(navContainer, contentContainer) {
  navContainer.innerHTML = NAV_HTML;
  contentContainer.innerHTML = CONTENT_HTML;
  bindAudioModelManager();
  refreshAudioModelManager();
  contentContainer.insertAdjacentHTML("beforeend", `
  <!-- audio log modal -->
  <div id="audio-log-modal" class="modal-backdrop hidden">
    <div class="modal-card">
      <div class="modal-header">
        <div style="font-weight:700;">ログ表示</div>
        <button id="audio-log-close" class="action-btn">閉じる</button>
      </div>
      <div class="modal-body">
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px;">
          <button id="audio-log-refresh" class="action-btn">更新</button>
          <button id="audio-log-openfolder" class="action-btn">ログフォルダを開く</button>
          <span id="audio-log-path" style="font-size:0.85rem; opacity:0.9;"></span>
        </div>
        <div class="two-col">
          <div>
            <div class="small-title">stdout（標準出力）</div>
            <pre id="audio-log-stdout" class="log-pre"></pre>
          </div>
          <div>
            <div class="small-title">stderr（標準エラー）</div>
            <pre id="audio-log-stderr" class="log-pre"></pre>
          </div>
        </div>
      </div>
    </div>
  </div>
`);

  // フォルダを開くボタンを設置（ファイル配置/移動がしやすいように）
  try { installOpenFolderButtons(); } catch (_) {}

  // inference helpers (presets)
  try {
    const presetRes = await api.get('/audio/inference/presets');
    const presets = (presetRes && presetRes.presets) ? presetRes.presets : [];
    const sel = $('aud-preset');
    if (sel && presets.length) {
      sel.innerHTML = '';
      presets.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = `${p.label} - ${p.description}`;
        sel.appendChild(opt);
      });
    }
  } catch (e) {
    // audio presets are optional
  }

  const streamCb = $('aud-stream');
  const streamBox = $('aud-stream-options');
  if (streamCb && streamBox) {
    const sync = () => { streamBox.style.display = streamCb.checked ? 'block' : 'none'; };
    streamCb.addEventListener('change', sync);
    sync();
  }

  bindTabSwitch(navContainer, contentContainer);

async function applyThirdPartyStatus(){
  try{
    const caps = await api.get('/utils/capabilities');
    const tp = caps?.details?.third_party_status || null;
    const warnCard = document.getElementById('aud-thirdparty-warn');
    const warnText = document.getElementById('aud-thirdparty-warn-text');
    const typeSel = document.getElementById('aud-train-type');
    if(!tp || !warnCard || !warnText || !typeSel) return;

    let msgs = [];
    for (const opt of Array.from(typeSel.options)){
      if(opt.value === 'xtts_finetune' && !tp.xtts_repo_ok){
        opt.disabled = true;
        msgs.push(`XTTS が見つかりません: ${tp.xtts_repo_path}\nclone: ${tp.xtts_repo_url}`);
      }
      if(opt.value === 'gpt_sovits' && !tp.gpt_sovits_repo_ok){
        opt.disabled = true;
        msgs.push(`GPT-SoVITS が見つかりません: ${tp.gpt_sovits_repo_path}\nclone: ${tp.gpt_sovits_repo_url}`);
      }
    }

    if(typeSel.selectedOptions[0]?.disabled){
      const firstEnabled = Array.from(typeSel.options).find(o=>!o.disabled);
      if(firstEnabled) typeSel.value = firstEnabled.value;
    }

    if(msgs.length){
      warnCard.style.display = 'block';
      warnText.textContent = msgs.join('\n\n');
      const anyEnabled = Array.from(typeSel.options).some(o=>!o.disabled);
      const startBtn = document.getElementById('aud-train-start');
      if(startBtn) startBtn.disabled = !anyEnabled;
    }else{
      warnCard.style.display = 'none';
    }
  }catch(e){}
}


  // System
  document.getElementById('aud-refresh-sys').addEventListener('click', refreshSystem);
  await refreshSystem();
  await applyThirdPartyStatus();

  // Datasets
  document.getElementById('aud-refresh-datasets').addEventListener('click', refreshDatasets);
  await refreshDatasets();

  // Train
  document.getElementById('aud-train-start').addEventListener('click', async () => {
    const dataset = document.getElementById('aud-train-dataset').value;
    const whisperModel = document.getElementById('aud-whisper-model').value.trim() || 'small';
    const language = document.getElementById('aud-language').value.trim() || 'ja';
    const sliceMin = parseFloat(document.getElementById('aud-slice-min').value || '3');
    const sliceMax = parseFloat(document.getElementById('aud-slice-max').value || '10');
    const trainType = document.getElementById('aud-train-type')?.value || 'xtts_finetune';
    const gptRepo = document.getElementById('aud-gpt-repo').value.trim();
    const customTrain = document.getElementById('aud-custom-train').value.trim();
    const xttsExtra = (document.getElementById('aud-xtts-extra')?.value || '').trim();

    const req = {
      base_model: '',
      dataset,
      params: {
        epochs: 1,
        train_batch_size: 1,
        whisper_model: whisperModel,
        language,
        slice_min_sec: sliceMin,
        slice_max_sec: sliceMax,
        train_type: trainType,
        xtts_repo: null,
        xtts_config: 'recipes/xtts/finetune/xtts_v2_finetune_config.json',
        xtts_extra_args: xttsExtra || '',
        gpt_sovits_repo: (trainType==='gpt_sovits') ? (gptRepo || null) : null,
        custom_train_cmd: (trainType==='gpt_sovits') ? (customTrain || null) : null
      }
    };

    const log = document.getElementById('aud-train-log');
    log.textContent = '開始中...';
    try {
      const res = await api.post('/audio/train/start', req);
      log.textContent = `開始しました: job_id=${res.job_id || ''}\n`;
      await pollTrainStatus();
    } catch (e) {
      log.textContent = e.message;
    }
  });

  document.getElementById('aud-train-stop').addEventListener('click', async () => {
    const log = document.getElementById('aud-train-log');
    try {
      await api.post('/audio/train/stop', {});
      log.textContent += '\n停止しました。';
      await pollTrainStatus();
    } catch (e) {
      log.textContent = e.message;
    }
  });

  document.getElementById('aud-train-refresh').addEventListener('click', pollTrainStatus);

  // Inference
  document.getElementById('aud-model-refresh').addEventListener('click', refreshModels);
  await refreshModels();

  document.getElementById('aud-model-load').addEventListener('click', async () => {
    const modelDir = document.getElementById('aud-model-select').value;
    const repo = document.getElementById('aud-infer-repo').value.trim();
    const cmd = document.getElementById('aud-custom-infer').value.trim();
    const log = document.getElementById('aud-infer-log');
    log.textContent = 'ロード中...';
    try {
      const res = await api.post('/audio/inference/load', {
        model_dir: modelDir,
        gpt_sovits_repo: repo || null,
        custom_infer_cmd: cmd || null,
        tts_backend: document.getElementById('aud-tts-backend').value,
        vc_backend: document.getElementById('aud-vc-backend').value,
        xtts_model_id: document.getElementById('aud-xtts-model-id').value.trim() || null,
        xtts_language: document.getElementById('aud-xtts-lang').value.trim() || 'ja',
        rvc_repo: document.getElementById('aud-rvc-repo').value.trim() || null,
        rvc_custom_cmd: document.getElementById('aud-rvc-cmd').value.trim() || null,
        gpt_sovits_vc_repo: document.getElementById('aud-gpt-vc-repo').value.trim() || null,
        gpt_sovits_vc_custom_cmd: document.getElementById('aud-gpt-vc-cmd').value.trim() || null,
        tts_backend: document.getElementById('aud-tts-backend').value,
        vc_backend: document.getElementById('aud-vc-backend').value,
        xtts_model_id: document.getElementById('aud-xtts-model-id').value.trim() || null,
        xtts_language: document.getElementById('aud-xtts-lang').value.trim() || 'ja',
        rvc_repo: document.getElementById('aud-rvc-repo').value.trim() || null,
        rvc_custom_cmd: document.getElementById('aud-rvc-cmd').value.trim() || null,
        gpt_sovits_vc_repo: document.getElementById('aud-gpt-vc-repo').value.trim() || null,
        gpt_sovits_vc_custom_cmd: document.getElementById('aud-gpt-vc-cmd').value.trim() || null
      });
      log.textContent = JSON.stringify(res, null, 2);
    } catch (e) {
      log.textContent = e.message;
    }
  });

  document.getElementById('aud-generate').addEventListener('click', async () => {
    const text = document.getElementById('aud-text').value;
    const ref = document.getElementById('aud-ref-audio').value.trim();
    const cmd = document.getElementById('aud-custom-infer').value.trim();
    const log = document.getElementById('aud-infer-log');
    const player = document.getElementById('aud-player');

    if (!text.trim()) {
      log.textContent = 'テキストを入力してください。';
      return;
    }
    if (!ref) {
      log.textContent = '参照音声（パス）を指定してください。';
      return;
    }

    log.textContent = '生成中...';
    try {
      const res = await api.post('/audio/inference/generate', {
        text,
        reference_audio: ref,
        output_format: 'wav',
        preset_id: document.getElementById('aud-preset') ? document.getElementById('aud-preset').value : null,
        custom_infer_cmd: cmd || null,
        tts_backend: document.getElementById('aud-tts-backend').value,
        vc_backend: document.getElementById('aud-vc-backend').value,
        xtts_model_id: document.getElementById('aud-xtts-model-id').value.trim() || null,
        xtts_language: document.getElementById('aud-xtts-lang').value.trim() || 'ja',
        rvc_repo: document.getElementById('aud-rvc-repo').value.trim() || null,
        rvc_custom_cmd: document.getElementById('aud-rvc-cmd').value.trim() || null,
        gpt_sovits_vc_repo: document.getElementById('aud-gpt-vc-repo').value.trim() || null,
        gpt_sovits_vc_custom_cmd: document.getElementById('aud-gpt-vc-cmd').value.trim() || null
      });
      if (res.status !== 'ok') {
        log.textContent = res.message || '生成に失敗しました。';
        return;
      }
      player.src = res.audio_base64;
      log.textContent = '完了';
    } catch (e) {
      log.textContent = e.message;
    }
  });
}






async function validateDatasetAndToggleStart_audio() {
  const btn = document.getElementById("aud-train-start");
  const sel = document.getElementById("aud-train-dataset");
  if(!btn || !sel) return;
  const warnId = "audio-dataset-validate";
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
  const tsel=document.getElementById('aud-train-type'); if(tsel && tsel.value==='tts_lora_xtts'){ kind='tts'; } else { kind='vc'; }
  try {
    const res = await api.post("/api/datasets/validate", { mode: "audio", dataset, kind });
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



window.audioUI = window.audioUI || {}; if (typeof refreshModels==='function') window.audioUI.refreshModels = refreshModels;
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

let audioLogModalState = { jobId: null, logFile: null };

function showAudioLogModal(jobId, logFile) {
  audioLogModalState = { jobId, logFile };
  const modal = document.getElementById('audio-log-modal');
  if (!modal) return;
  modal.classList.remove('hidden');
  document.getElementById('audio-log-path').textContent = logFile ? logFile : '';
  refreshAudioLogModal();
}

function hideAudioLogModal() {
  const modal = document.getElementById('audio-log-modal');
  if (!modal) return;
  modal.classList.add('hidden');
}

async function refreshAudioLogModal() {
  const st = audioLogModalState;
  if (!st || !st.logFile) {
    setText(document.getElementById('audio-log-stdout'), 'ログファイルがありません');
    setText(document.getElementById('audio-log-stderr'), '');
    return;
  }
  try {
    const res = await api.get('/utils/read_text_file', { path: st.logFile, max_lines: 800 });
    const lines = res?.lines || [];
    const sp = splitStdoutStderr(lines);
    setText(document.getElementById('audio-log-stdout'), sp.stdout.join('\n'));
    setText(document.getElementById('audio-log-stderr'), sp.stderr.join('\n'));
  } catch (e) {
    setText(document.getElementById('audio-log-stdout'), `取得失敗: ${e.message}`);
    setText(document.getElementById('audio-log-stderr'), '');
  }
}