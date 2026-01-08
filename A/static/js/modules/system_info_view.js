/**
 * static/js/modules/system_info_view.js
 * システム情報タブ専用: 生JSONを日本語ラベル/カテゴリで整形表示し、
 * パスは [📂 開く] ボタンで Explorer を起動できるようにする。
 *
 * 前提: backend に /api/system_info, /api/system_info/image, /api/system_info/audio,
 *       /api/utils/paths, /api/utils/open_path が存在する。
 */
import { api } from '../api.js';

function safeStr(v, fallback = '-') {
  if (v === null || v === undefined) return fallback;
  const s = String(v);
  return s.trim() === '' ? fallback : s;
}

function get(obj, path, fallback = '-') {
  try {
    const parts = path.split('.');
    let cur = obj;
    for (const p of parts) {
      if (cur === null || cur === undefined) return fallback;
      cur = cur[p];
    }
    return (cur === null || cur === undefined) ? fallback : cur;
  } catch (_) {
    return fallback;
  }
}

function yesno(v, y = 'あり', n = 'なし') {
  return v ? y : n;
}

function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

function makeButton(label, onClick) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'open-folder-btn';
  b.innerHTML = `<i class="fas fa-folder-open"></i> ${label}`;
  b.addEventListener('click', onClick);
  return b;
}

async function openPath(path) {
  await api.post('/utils/open_path', { path });
}


// --- 関連ライブラリの解説（未導入時の影響も含む）
// 表示は「関連ライブラリ一覧（未導入含む）」セクションで使用します。
const LIB_EXPLAIN = {
  image: {
    'torch': {
      ok: '画像LoRA学習/推論の計算基盤として動作します。',
      ng: '画像LoRA学習/推論が実行できません。'
    },
    'diffusers': {
      ok: 'Diffusers系の画像LoRA学習・保存（safetensors等）が実行できます。',
      ng: 'Diffusers系の画像LoRA学習が実行できません。'
    },
    'transformers': {
      ok: 'CLIP/Text Encoder等を用いた画像モデルの学習が安定して実行できます。',
      ng: 'CLIP/Text Encoder等が読み込めず、画像LoRA学習が失敗する可能性が高いです。'
    },
    'accelerate': {
      ok: '学習の実行制御（混合精度/分散/最適化）により安定して学習できます。',
      ng: '学習実行の制御ができず、起動失敗または学習が不安定になります。'
    },
    'safetensors': {
      ok: 'LoRA/モデルを安全な形式（.safetensors）で保存・読み込みできます。',
      ng: '.safetensors 形式の保存/読み込みができない、または互換性問題が起きます。'
    },
    'huggingface-hub': {
      ok: 'モデル取得/キャッシュ管理ができます（オフライン運用でもキャッシュ参照に使われます）。',
      ng: 'モデル取得/キャッシュ参照が制限され、環境によっては学習開始に失敗します。'
    },
    'numpy': {
      ok: '前処理・配列計算が安定して動作します。',
      ng: '前処理が動かず、多くの処理が起動時点で失敗します。'
    },
    'pillow': {
      ok: '画像の読み込み/変換（PNG/JPG等）ができます。',
      ng: '画像の読み込み/変換ができず、データセット検査や前処理が失敗します。'
    },
    'xformers': {
      ok: '任意：注意機構の高速化により速度向上/VRAM削減が期待できます。',
      ng: '任意：無くても動作しますが、速度やVRAM効率が低下する場合があります。'
    },
    'bitsandbytes': {
      ok: '任意：量子化/省VRAM関連機能が使える場合があります（環境依存）。',
      ng: '任意：QLoRA等の省VRAM機能が使えません（通常LoRAは別経路で動く場合があります）。'
    },
    'triton': {
      ok: '任意：一部の高速化カーネルが使える場合があります（Windows制約あり）。',
      ng: '任意：高速化が効かないだけで、通常は致命的ではありません。'
    },
    'torchvision': {
      ok: '任意：画像変換/データローダ補助が利用できます。',
      ng: '任意：一部の画像変換/データ処理が使えない場合があります。'
    },
    'opencv-python': {
      ok: '任意：高度な画像処理（拡張前処理等）が利用できます。',
      ng: '任意：OpenCV前提の前処理を使う場合は実行できません。'
    },
    'einops': {
      ok: '任意：一部モデルのテンソル変形処理が利用できます。',
      ng: '任意：einops前提のモデル/処理を使う場合は失敗します。'
    }
  },
  audio: {
    'torch': {
      ok: '音声学習/推論の計算基盤として動作します。',
      ng: '音声学習/推論が実行できません。'
    },
    'torchaudio': {
      ok: '音声I/Oや特徴量処理が利用でき、音声系の前処理が安定します。',
      ng: '音声I/Oや特徴量処理ができず、音声学習が失敗する可能性が高いです。'
    },
    'librosa': {
      ok: '音声前処理（STFT等）に利用できます。',
      ng: '音声前処理が不足し、外部学習ツールの前提を満たせない場合があります。'
    },
    'soundfile': {
      ok: 'wav等の読み書きができます。',
      ng: 'wav等の読み書きができず、データ検査/前処理が失敗します。'
    },
    'numpy': {
      ok: '前処理・配列計算が安定して動作します。',
      ng: '前処理が動かず、多くの処理が起動時点で失敗します。'
    },
    'scipy': {
      ok: '任意：音声処理で使用される補助関数が利用できます。',
      ng: '任意：一部の音声処理が使えず、外部ツールの一部機能が失敗する場合があります。'
    },
    'faster-whisper': {
      ok: '任意：文字起こし（ASR）を高速に実行できます。',
      ng: '任意：自動文字起こし機能が使えません。'
    },
    'pydub': {
      ok: '任意：音声変換（ffmpeg連携）などの補助が使えます。',
      ng: '任意：pydub前提の変換処理は使えません。'
    },
    'pyworld': {
      ok: '任意：VC系で使われる特徴量処理が利用できる場合があります。',
      ng: '任意：pyworld前提の処理は使えません。'
    },
    'ffmpeg（外部）': {
      ok: '外部：音声変換（wav化/サンプルレート変換等）が実行できます。',
      ng: '外部：自動音声変換ができず、対応形式以外で失敗します。'
    }
  },
  text: {
    'torch': {
      ok: 'テキストLoRA学習/推論の計算基盤として動作します。',
      ng: 'テキストLoRA学習/推論が実行できません。'
    },
    'transformers': {
      ok: 'LLM/Tokenizerの読み込みと学習ができます。',
      ng: 'LLM/Tokenizerが読み込めず、テキストLoRA学習ができません。'
    },
    'peft': {
      ok: 'LoRA/PEFT（アダプタ学習・適用）ができます。',
      ng: 'LoRA（PEFT）学習/適用ができません。'
    },
    'accelerate': {
      ok: '学習の実行制御（混合精度/最適化）により安定して学習できます。',
      ng: '学習実行が不安定になり、起動失敗する場合があります。'
    },
    'datasets': {
      ok: 'データセットの読み込み/前処理/分割ができます。',
      ng: 'データセット処理ができず、学習データの取り込みが失敗します。'
    },
    'bitsandbytes': {
      ok: '任意：QLoRA/量子化でVRAMを抑えた学習ができる場合があります（環境依存）。',
      ng: '任意：QLoRA/量子化が使えず、VRAMが足りない場合に学習できません。'
    },
    'sentencepiece': {
      ok: '任意：SentencePiece系Tokenizerを使うモデルが動きます。',
      ng: '任意：SentencePiece系Tokenizerモデルが読み込めません。'
    },
    'safetensors': {
      ok: '任意：安全な形式でモデル/アダプタを保存できます。',
      ng: '任意：.safetensors 保存/読み込みができない場合があります。'
    },
    'trl': {
      ok: '任意：SFT/DPO等の学習レシピを使う場合に利用できます。',
      ng: '任意：trl前提の学習レシピは使えません。'
    }
  }
};

export async function fetchPathsSafe() {
  try {
    const res = await api.get('/utils/paths');
    return res?.paths || {};
  } catch (_) {
    return {};
  }
}

function section(mount, title) {
  const card = el('div', 'sys-card');
  card.appendChild(el('div', 'sys-card-title', title));
  const body = el('div', 'sys-card-body');
  card.appendChild(body);
  mount.appendChild(card);
  return body;
}

function list(body, items) {
  const ul = el('ul', 'sys-list');
  for (const it of items) {
    const li = el('li', 'sys-item');

    const row = el('div', 'sys-row');
    row.appendChild(el('span', 'sys-key', safeStr(it.label)));
    row.appendChild(el('span', 'sys-val', safeStr(it.value)));
    li.appendChild(row);

    if (it.openPath) {
      const wrap = el('div', 'sys-open-wrap');
      wrap.appendChild(el('div', 'sys-path', safeStr(it.openPath)));
      wrap.appendChild(makeButton('開く', async () => {
        try {
          await openPath(it.openPath);
        } catch (e) {
          alert(`フォルダを開けませんでした。\n${safeStr(e?.message || e)}`);
        }
      }));
      li.appendChild(wrap);
    }

    ul.appendChild(li);
  }
  body.appendChild(ul);
}

function listLibStatus(body, libs, mode) {
  // libs: [{name, note, installed, version, explainOk, explainNg}]
  const ul = el('ul', 'sys-list');
  for (const it of libs) {
    const li = el('li', 'sys-item');

    // 2カラム（左：名称＋状態、右：解説）
    const grid = el('div', 'sys-lib-grid');

    const left = el('div', 'sys-lib-left');
    const name = el('div', 'sys-key', it.name);

    const st = el('span', 'sys-lib-status');
    st.className = 'sys-lib-status';
    if (it.installed) {
      st.classList.add('ok');
      st.textContent = `導入済み (${safeStr(it.version)})`;
    } else {
      st.classList.add('ng');
      st.textContent = '未導入';
    }

    const metaRow = el('div', 'sys-lib-meta');
    metaRow.appendChild(st);

    left.appendChild(name);
    left.appendChild(metaRow);

    if (it.note) {
      left.appendChild(el('div', 'sys-note', it.note));
    }

    const right = el('div', 'sys-lib-right');

    // 解説の決定（優先順位：個別指定 → 辞書 → 汎用文）
    let explain = '';
    if (it.installed) {
      explain = it.explainOk || LIB_EXPLAIN?.[mode]?.[it.name]?.ok || '';
      if (!explain) explain = '導入済みのため、この機能は利用可能です。';
    } else {
      explain = it.explainNg || LIB_EXPLAIN?.[mode]?.[it.name]?.ng || '';
      if (!explain) explain = '未導入のため、関連する機能が使えない可能性があります。';
    }
    right.appendChild(el('div', 'sys-lib-desc', explain));

    grid.appendChild(left);
    grid.appendChild(right);
    li.appendChild(grid);
    ul.appendChild(li);
  }
  body.appendChild(ul);
}

function renderJsonDetails(body, jsonObj) {
  const details = el('details', 'sys-details');
  const summary = el('summary', 'sys-details-summary', '詳細（JSON）');
  details.appendChild(summary);

  const pre = el('pre', 'console-log small-log');
  try {
    pre.textContent = JSON.stringify(jsonObj, null, 2);
  } catch (_) {
    pre.textContent = safeStr(jsonObj);
  }
  details.appendChild(pre);
  body.appendChild(details);
}

/**
 * @param {HTMLElement} mountEl
 * @param {object} sysData
 * @param {object} paths
 * @param {"text"|"image"|"audio"} mode
 */

function ensureSetupLogsModal() {
  if (document.getElementById('global-setup-logs-modal')) return;
  const html = `
  <div id="global-setup-logs-modal" class="modal-backdrop hidden">
    <div class="modal-card modal-wide">
      <div class="modal-header">
        <div style="font-weight:700;">セットアップログ</div>
        <button id="global-setup-logs-close" class="action-btn">閉じる</button>
      </div>
      <div class="modal-body">
        <div id="global-setup-logs-list" class="setup-logs-list"></div>
        <pre id="global-setup-logs-content" class="log-pre"></pre>
      </div>
    </div>
  </div>`;
  document.body.insertAdjacentHTML('beforeend', html);
  document.getElementById('global-setup-logs-close')?.addEventListener('click', ()=>{
    document.getElementById('global-setup-logs-modal')?.classList.add('hidden');
  });
}

async function openSetupLogsModal() {
  ensureSetupLogsModal();
  const modal = document.getElementById('global-setup-logs-modal');
  const list = document.getElementById('global-setup-logs-list');
  const pre = document.getElementById('global-setup-logs-content');
  if (!modal || !list || !pre) return;
  list.innerHTML = "取得中...";
  pre.textContent = "";
  modal.classList.remove('hidden');
  try {
    const res = await api.get('/utils/list_setup_logs');
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

export function renderSystemInfoTab(mountEl, sysData, paths = {}, mode = 'text') {
  if (!mountEl) return;
  mountEl.innerHTML = '';

  // 互換: image/audio は {base, libs, disk, messages...} を返す。text はそのままのことがある。
  const base = sysData?.base ? sysData.base : sysData;
  const tc = sysData?.torch_cuda || base?.torch_cuda || {};
  const libs = sysData?.libs || {};
  const disk = sysData?.disk || {};
  const env = sysData?.env || {};
  const messages = sysData?.messages || [];
  const packages = (base && base.packages) ? base.packages : (sysData?.packages || {});

  function pkgVer(name) {
    const k = String(name || '').toLowerCase();
    return packages && Object.prototype.hasOwnProperty.call(packages, k) ? packages[k] : null;
  }

  // 1) 実行環境
  {
    const body = section(mountEl, '実行環境');
    list(body, [
      { label: 'OS', value: get(base, 'os') },
      { label: 'Python', value: get(base, 'python') },
      { label: 'PyTorch', value: get(tc, 'torch') },
      { label: 'CUDA利用可', value: yesno(get(tc, 'cuda_available', false), '利用可能', '利用不可') },
      { label: 'CUDA', value: get(tc, 'cuda_version') },
      { label: 'cuDNN', value: get(tc, 'cudnn_version') },
    ]);
  }

  // 2) CPU / メモリ
  {
    const body = section(mountEl, 'CPU / メモリ');
    list(body, [
      { label: 'CPU', value: get(base, 'cpu.model') },
      { label: '物理コア', value: get(base, 'cpu.cores_physical') },
      { label: '論理コア', value: get(base, 'cpu.cores_logical') },
      { label: 'CPU使用率', value: `${safeStr(get(base, 'cpu.usage_percent'))}%` },
      { label: 'メモリ総量', value: `${safeStr(get(base, 'memory.total_gb'))}GB` },
      { label: 'メモリ空き', value: `${safeStr(get(base, 'memory.available_gb'))}GB` },
      { label: 'メモリ使用率', value: `${safeStr(get(base, 'memory.percent'))}%` },
    ]);
  }

  // 3) GPU（PyTorch）
  {
    const body = section(mountEl, 'GPU（PyTorch）');
    const vramTotal = get(tc, 'vram_total_gb', null);
    const vramFree = get(tc, 'vram_free_gb', null);
    const vramText = (vramTotal === null || vramTotal === undefined)
      ? '-'
      : `総量 ${safeStr(vramTotal)}GB / 空き ${safeStr(vramFree)}GB`;
    list(body, [
      { label: 'GPU名', value: get(tc, 'gpu_name', 'CUDA未検出') },
      { label: 'Compute Capability', value: get(tc, 'compute_capability') },
      { label: 'VRAM', value: vramText },
      { label: 'bf16', value: yesno(get(tc, 'bf16_supported', false), '対応', '非対応/不明') },
    ]);
  }

  // 4) 主要フォルダ（ボタン化）
  {
    const body = section(mountEl, '主要フォルダ（クリックで開く）');
    const items = [];

    if (paths.base_dir) items.push({ label: 'プロジェクトルート', value: paths.base_dir, openPath: paths.base_dir });
    if (paths.logs_dir) items.push({ label: 'ログ', value: paths.logs_dir, openPath: paths.logs_dir });

    if (mode === 'text') {
      if (paths.models_text) items.push({ label: 'モデル（テキスト）', value: paths.models_text, openPath: paths.models_text });
      if (paths.datasets_text) items.push({ label: 'データセット（テキスト）', value: paths.datasets_text, openPath: paths.datasets_text });
      if (paths.lora_adapters_text) items.push({ label: '出力（テキストLoRA）', value: paths.lora_adapters_text, openPath: paths.lora_adapters_text });
    }
    if (mode === 'image') {
      if (paths.models_image) items.push({ label: 'モデル（画像）', value: paths.models_image, openPath: paths.models_image });
      if (paths.datasets_image) items.push({ label: 'データセット（画像）', value: paths.datasets_image, openPath: paths.datasets_image });
      if (paths.lora_adapters_image) items.push({ label: '出力（画像LoRA）', value: paths.lora_adapters_image, openPath: paths.lora_adapters_image });
    }
    if (mode === 'audio') {
      if (paths.models_audio) items.push({ label: 'モデル（音声）', value: paths.models_audio, openPath: paths.models_audio });
      if (paths.datasets_audio) items.push({ label: 'データセット（音声）', value: paths.datasets_audio, openPath: paths.datasets_audio });
      if (paths.lora_adapters_audio) items.push({ label: '出力（音声LoRA）', value: paths.lora_adapters_audio, openPath: paths.lora_adapters_audio });
    }

    if (!items.length) {
      items.push({ label: '情報', value: 'フォルダ一覧を取得できませんでした（/api/utils/paths が失敗している可能性）' });
    }
    list(body, items);
  }

  // 5) 主要ライブラリ
  {
    const body = section(mountEl, '主要ライブラリ');
    if (mode === 'image') {
      list(body, [
        { label: 'diffusers', value: libs.diffusers || '-' },
        { label: 'transformers', value: libs.transformers || '-' },
        { label: 'accelerate', value: libs.accelerate || '-' },
        { label: 'safetensors', value: libs.safetensors || '-' },
        { label: 'xformers（任意）', value: libs.xformers ? `あり (${libs.xformers})` : 'なし' },
        { label: 'bitsandbytes', value: libs.bitsandbytes ? `あり (${libs.bitsandbytes})` : 'なし' },
      ]);
    } else if (mode === 'audio') {
      list(body, [
        { label: 'torchaudio', value: libs.torchaudio || '-' },
        { label: 'librosa', value: libs.librosa || '-' },
        { label: 'soundfile', value: libs.soundfile || '-' },
        { label: 'faster-whisper', value: libs.faster_whisper || '-' },
        { label: 'ffmpeg', value: env.ffmpeg_available ? 'あり' : 'なし' },
        { label: 'GPT_SOVITS_DIR', value: env.GPT_SOVITS_DIR ? env.GPT_SOVITS_DIR : '（未設定）' },
        { label: 'XTTS_DIR', value: env.XTTS_DIR ? env.XTTS_DIR : '（未設定）' },
      ]);
    } else {
      list(body, [
        { label: 'PyTorch', value: get(tc, 'torch') },
      ]);
    }
  }

  // 5.5) 関連ライブラリ一覧（未導入含む）
  {
    const body = section(mountEl, '関連ライブラリ一覧（未導入含む）');

    /** @type {{name:string, note?:string, installed:boolean, version?:string}[]} */
    const items = [];

    function add(name, note) {
      const v = pkgVer(name);
      items.push({ name, note, installed: !!v, version: v || undefined });
    }

    if (mode === 'image') {
      // コア
      add('torch', '学習/推論の基盤');
      add('diffusers', '画像LoRA（Diffusers）');
      add('transformers', 'テキストエンコーダ/CLIP等');
      add('accelerate', '学習実行の補助（分散/AMP等）');
      add('safetensors', 'モデル/LoRA保存形式');
      add('huggingface-hub', 'モデル取得/キャッシュ（オフライン運用でも利用）');
      add('numpy', '前処理/配列');
      add('pillow', '画像入出力');
      // 任意（高速化/省VRAM）
      add('xformers', '任意：高速化（Windowsでは環境により不安定）');
      add('bitsandbytes', '任意：省VRAM/量子化（環境依存）');
      add('triton', '任意：一部高速化（Windowsでは制約あり）');
      add('torchvision', '任意：画像変換/データセット');
      add('opencv-python', '任意：画像処理（使う場合のみ）');
      add('einops', '任意：テンソル操作（モデルによる）');
    } else if (mode === 'audio') {
      // コア
      add('torch', '学習/推論の基盤');
      add('torchaudio', '音声入出力/特徴量');
      add('librosa', '音声前処理');
      add('soundfile', 'wav等の読み書き');
      add('numpy', '前処理/配列');
      add('scipy', '音声処理（環境による）');
      // 補助
      add('faster-whisper', '任意：ASR（文字起こし）');
      add('pydub', '任意：音声変換（ffmpeg利用）');
      add('pyworld', '任意：特徴量/VC系で使用する場合あり');
      add('phonemizer', '任意：TTSの音素化（モデルによる）');
      add('unidecode', '任意：テキスト正規化（モデルによる）');
      // 外部バイナリはpackagesに出ないので注記のみ
      items.push({ name: 'ffmpeg（外部）', note: '外部バイナリ。システム情報の ffmpeg 欄で別途判定。', installed: !!env.ffmpeg_available, version: env.ffmpeg_available ? 'あり' : undefined });
    } else {
      // text
      add('torch', '学習/推論の基盤');
      add('transformers', 'LLM/Tokenizer');
      add('peft', 'LoRA/PEFT');
      add('accelerate', '学習実行の補助');
      add('datasets', 'データセット処理');
      add('bitsandbytes', '任意：QLoRA/量子化（環境依存）');
      add('sentencepiece', '任意：Tokenizer（モデルによる）');
      add('safetensors', '任意：保存形式');
      add('trl', '任意：SFT/DPO等（使う場合のみ）');
    }

    listLibStatus(body, items, mode);
  }

  // 6) ディスク（空き容量）
  {
    const body = section(mountEl, 'ディスク（空き容量）');
    const items = [];

    // ある場合のみ表示
    const root = disk.project_root;
    if (root?.free_gb != null) items.push({ label: 'プロジェクトルート 空き容量', value: `${root.free_gb}GB` });

    const dkModels = disk?.[`models_${mode}`];
    const dkDatasets = disk?.[`datasets_${mode}`];
    if (dkModels?.free_gb != null) items.push({ label: `models/${mode} 空き容量`, value: `${dkModels.free_gb}GB` });
    if (dkDatasets?.free_gb != null) items.push({ label: `datasets/${mode} 空き容量`, value: `${dkDatasets.free_gb}GB` });

    if (!items.length) items.push({ label: '情報', value: '空き容量の詳細は未取得/未対応です。' });
    list(body, items);
  }

  // 7) 外部リポジトリ（音声のみ）
  if (mode === 'audio') {
    const body = section(mountEl, '外部リポジトリ（音声）');
    const repos = env.external_repos || {};
    const keys = Object.keys(repos || {});
    const items = [];
    if (keys.length) {
      for (const k of keys) {
        const r = repos[k] || {};
        const exists = !!r.exists;
        const path = r.path || '';
        items.push({
          label: k,
          value: exists ? `検出: あり` : `検出: なし`,
          openPath: path || null
        });
      }
    } else {
      items.push({ label: '情報', value: '外部リポジトリ情報は未取得です（環境変数が未設定の可能性）' });
    }
    list(body, items);
  }

  // 8) 通知/警告（任意）
  if (Array.isArray(messages) && messages.length) {
    const body = section(mountEl, '通知 / 警告');
    const ul = el('ul', 'sys-msg-list');
    for (const m of messages) {
      const li = el('li', 'sys-msg-item');
      const title = m?.title ? `【${m.title}】` : '【通知】';
      li.appendChild(el('div', 'sys-msg-title', title));
      li.appendChild(el('div', 'sys-msg-body', safeStr(m?.message || '')));
      ul.appendChild(li);
    }
    body.appendChild(ul);
  }

  // 9) nvidia-smi（生ログ）
  {
    const body = section(mountEl, 'GPU（nvidia-smi 生ログ）');

    // backend/core/system_info.py は nvidia_smi を "文字列" で返す
    const nv = get(base, 'nvidia_smi', '') || get(sysData, 'nvidia_smi', '');
    const raw = (typeof nv === 'string') ? nv : safeStr(nv, '');
    const pre = el('pre', 'console-log small-log');
    pre.textContent = raw ? raw : 'nvidia-smi 情報を取得できませんでした（NVIDIAドライバ / nvidia-smi の有無を確認してください）。';
    body.appendChild(pre);
  }

  

// 10) セットアップログ
{
  const body = section(mountEl, 'セットアップ / 運用');
  const row = el('div', 'sys-actions-row');
  const btn = el('button', 'action-btn js-open-setup-logs', 'セットアップログを見る');
  btn.type = 'button';
  btn.addEventListener('click', ()=> { openSetupLogsModal(); });
    row.appendChild(btn);
  body.appendChild(row);
  body.appendChild(el('div', 'hint', 'setup_lora_env.py 等の実行ログ（logs/ 配下の setup*.log）を表示します。'));
}

// 11) 詳細（JSON）…デフォルト折りたたみ
  {
    const body = section(mountEl, '詳細');
    renderJsonDetails(body, sysData);
  }
}
