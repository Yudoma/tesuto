/**
 * static/js/modules/image_ui.js
 * Image (Diffusers) モード専用UI。
 *
 * - データセット選択（datasets/image 配下）
 * - 学習パラメータ（Resolution/エポック数/バッチサイズ 等）
 * - 推論タブ（プロンプトから画像生成し表示）
 *
 * 既存の Modular UI (text_ui.js) と同様に mount(nav, content) を提供。
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
  <button class="nav-btn" data-target="tab-train">
    <i class="fas fa-fire"></i> 学習（画像）
  </button>
  <button class="nav-btn" data-target="tab-inference">
    <i class="fas fa-image"></i> 検証（生成）
  </button>
  <button class="nav-btn" data-target="tab-history">
    <i class="fas fa-history"></i> 学習履歴
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
      <li><strong>データセット</strong>：<code>datasets/image</code> に画像フォルダを用意します（画像と同名の <code>.txt</code> キャプション推奨）。</li>
      <li><strong>学習</strong>：ベースモデル（<code>models/image</code>）とデータセットを選び、学習設定を調整して開始します。</li>
      <li><strong>検証（生成）</strong>：モデルをロードし、プロンプトから画像を生成して品質を確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>VRAMが足りない場合：</strong>解像度／Batch／Steps を下げる、Gradient Checkpointing を有効にするなどで改善します。</p>
    </div>
  </div>
</details>
      <div class="card">
        <h4>画像（Diffusers）用システム情報</h4>
        <div class="system-info-container" id="img-system-info-container">取得中...</div>
        <small class="hint">画像生成/学習に関係する主な情報（GPU/VRAM, PyTorch/CUDA, diffusers系ライブラリ, キャッシュ/ディスク）を表示します。</small>
      </div>
</div>
    </div>
  </section>


  <section id="tab-models" class="tab-pane">
    <header class="pane-header">
      <h3>モデル管理（画像）</h3>
      <button id="image-models-refresh" class="action-btn"><i class="fas fa-sync"></i> 更新</button>
      <button id="image-models-open-root" class="action-btn"><i class="fas fa-folder-open"></i> フォルダ</button>
    </header>
    <div class="pane-body">
      <div class="card" style="margin-top:12px;">
        <h4>HuggingFaceからダウンロード</h4>
        <div class="input-group">
          <input id="image-hf-repo-id" type="text" placeholder="Repo ID（例: stabilityai/sdxl-turbo）" />
          <button id="image-download-btn" class="primary-btn"><i class="fas fa-download"></i> ダウンロード</button>
        </div>
        <small class="hint">完了後に「更新」を押すと一覧へ反映されます。</small>
      </div>
      <div class="card">
        <h4>ローカルモデル一覧</h4>
        <table class="data-table">
          <thead><tr><th>モデル名</th><th>種別</th><th>操作</th></tr></thead>
          <tbody id="image-models-body"><tr><td colspan="3">未取得</td></tr></tbody>
        </table>
        <div id="image-model-detail" class="note-box" style="display:none;margin-top:10px;"></div>
      </div>
    </div>
  </section>



  <section id="tab-datasets" class="tab-pane">
    <header class="pane-header">
      <h3>データセット（画像）</h3>
      <button id="refresh-datasets" class="action-btn"><i class="fas fa-sync"></i> 更新</button>
    </header>
    <div class="pane-body">
<details class="help-details">
  <summary><i class="fas fa-circle-question"></i> はじめての方へ（基本の流れ）</summary>
  <div class="help-content">
    <ol>
      <li><strong>データセット</strong>：<code>datasets/image</code> に画像フォルダを用意します（画像と同名の <code>.txt</code> キャプション推奨）。</li>
      <li><strong>学習</strong>：ベースモデル（<code>models/image</code>）とデータセットを選び、学習設定を調整して開始します。</li>
      <li><strong>検証（生成）</strong>：モデルをロードし、プロンプトから画像を生成して品質を確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>VRAMが足りない場合：</strong>解像度／Batch／Steps を下げる、Gradient Checkpointing を有効にするなどで改善します。</p>
    </div>
  </div>
</details>
      <div class="card">
        <h4>datasets/image 配下のフォルダ</h4>
        <div class="form-group">
          <label>データセット</label>
          <select id="image-dataset-select"></select>
          <small class="hint">各画像ファイルと同名の .txt キャプションがある構成を想定しています。</small>
        </div>
        <div class="form-group">
          <label>説明</label>
          <div class="console-log small-log" id="image-dataset-info">(未選択)</div>
        </div>
      </div>
    </div>
  </section>

  <section id="tab-train" class="tab-pane">
    <header class="pane-header">
      <h3>学習 (Image LoRA)</h3>
      <div style="display:flex; gap:8px; align-items:center;">
        <button id="image-train-start" class="primary-btn"><i class="fas fa-play"></i> 開始</button>
        <button id="image-train-stop" class="action-btn" style="background:var(--danger); color:white;"><i class="fas fa-stop"></i> 停止</button>
        <button id="image-train-refresh" class="action-btn"><i class="fas fa-sync"></i> 状態更新</button>
      </div>
    </header>
    <div class="pane-body">
<details class="help-details">
  <summary><i class="fas fa-circle-question"></i> はじめての方へ（基本の流れ）</summary>
  <div class="help-content">
    <ol>
      <li><strong>データセット</strong>：<code>datasets/image</code> に画像フォルダを用意します（画像と同名の <code>.txt</code> キャプション推奨）。</li>
      <li><strong>学習</strong>：ベースモデル（<code>models/image</code>）とデータセットを選び、学習設定を調整して開始します。</li>
      <li><strong>検証（生成）</strong>：モデルをロードし、プロンプトから画像を生成して品質を確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>VRAMが足りない場合：</strong>解像度／Batch／Steps を下げる、Gradient Checkpointing を有効にするなどで改善します。</p>
    </div>
  </div>
</details>
      <div class="columns">
        <div class="column is-4">
          <div class="card">
            <h4>学習設定</h4>
            <div class="form-group">
              <label>ベースモデル (models/image)</label>
              <select id="image-base-model"></select>
              <small class="hint">Diffusers形式のフォルダ、または単一モデルファイル（.safetensors / .ckpt）に対応します。</small>
            </div>
            <div class="form-group">
              <label>モデルタイプ</label>
              <select id="image-model-type">
                <option value="sdxl" selected>SDXL</option>
                <option value="sd15">SD 1.5</option>
              </select>
            </div>
            <div class="form-group">
              <label>解像度（基準）</label>
              <input type="number" id="image-resolution" value="1024" step="64" min="256" />
              <small class="hint">SDXL=1024 / SD1.5=512 推奨。Aspect Ratio Bucketing で複数解像度を自動生成します。</small>
            </div>
            <div class="form-group">
              <label>エポック数</label>
              <input type="number" id="image-epochs" value="1" min="1" />
            </div>
            <div class="form-group">
              <label>バッチサイズ</label>
              <input type="number" id="image-batch" value="1" min="1" />
            </div>
            <div class="form-group">
              <label>勾配累積（Gradient Accumulation）</label>
              <input type="number" id="image-grad-acc" value="4" min="1" />
            </div>
            <div class="form-group">
              <label>学習率（Learning Rate）</label>
              <input type="number" id="image-lr" value="0.0001" step="0.00001" />
            </div>
            <div class="form-group">
              <label>LoRAランク（r）</label>
              <input type="number" id="image-lora-r" value="16" min="1" />
            </div>
            <div class="form-group">
              <label>LoRAアルファ（Alpha）</label>
              <input type="number" id="image-lora-alpha" value="32" min="1" />
            </div>
            <div class="form-group">
              <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
                <input type="checkbox" id="image-use-8bit" checked>
                <span>8bit AdamW（bitsandbytes）</span>
              </label>
            </div>
            <div class="form-group">
              <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
                <input type="checkbox" id="image-grad-ckpt" checked>
                <span>勾配チェックポイント（Gradient Checkpointing）</span>
              </label>
            </div>
            <div class="form-group">
              <label>混合精度（Mixed Precision）</label>
              <select id="image-mixed">
                <option value="fp16" selected>fp16</option>
                <option value="bf16">bf16</option>
                <option value="no">no</option>
              <div class="form-group">
  <label>チェックポイント保存間隔（ステップ）</label>
  <input type="number" id="image-save-every" value="200" min="0" step="1" />
  <small class="hint">0 の場合は途中保存しません。サンプルプレビューは「チェックポイント保存時」に更新されます。</small>
</div>
<div class="form-group">
  <label>サンプルプロンプト（任意）</label>
  <textarea id="image-sample-prompt" rows="3" placeholder="例: 1girl, masterpiece, best quality ...（空欄なら生成しません）"></textarea>
  <small class="hint">学習中の品質確認用に、チェックポイント保存のタイミングで画像を生成します（OOM回避のため生成時は一時的にVRAM解放を行います）。</small>
</div>
</select>
            </div>
          </div>
        </div>

        <div class="column is-8">
          <div class="card">
            <h4>ジョブ状態</h4>
            <div class="console-log small-log" id="image-train-status">(未開始)</div>
            <h4 style="margin-top:12px;">学習中サンプル（最新）</h4>
<div class="console-log small-log" id="image-train-sample-meta">(未生成)</div>
<div style="margin-top:8px; display:flex; justify-content:center;">
  <img id="image-train-sample-img" alt="学習中サンプル" style="max-width:100%; border-radius:12px; display:none;" />
</div>
<h4 style="margin-top:12px;">ログ</h4>
            <pre id="image-train-log" class="console-log" style="max-height:420px; overflow:auto;">(ログなし)</pre>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section id="tab-inference" class="tab-pane">
    <header class="pane-header">
      <h3>画像生成（推論）</h3>
      <div style="display:flex; gap:8px; align-items:center;">
        <button id="image-infer-load" class="action-btn"><i class="fas fa-bolt"></i> モデルロード</button>
        <button id="image-infer-unload" class="action-btn"><i class="fas fa-eject"></i> アンロード</button>
        <button id="image-generate" class="primary-btn"><i class="fas fa-magic"></i> 生成</button>
      </div>
    </header>
    <div class="pane-body">
<details class="help-details">
  <summary><i class="fas fa-circle-question"></i> はじめての方へ（基本の流れ）</summary>
  <div class="help-content">
    <ol>
      <li><strong>データセット</strong>：<code>datasets/image</code> に画像フォルダを用意します（画像と同名の <code>.txt</code> キャプション推奨）。</li>
      <li><strong>学習</strong>：ベースモデル（<code>models/image</code>）とデータセットを選び、学習設定を調整して開始します。</li>
      <li><strong>検証（生成）</strong>：モデルをロードし、プロンプトから画像を生成して品質を確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>VRAMが足りない場合：</strong>解像度／Batch／Steps を下げる、Gradient Checkpointing を有効にするなどで改善します。</p>
    </div>
  </div>
</details>
      <div class="columns">
        <div class="column is-5">
          <div class="card">
            <h4>推論設定</h4>
            <div class="form-group">
              <label>ベースモデル</label>
              <select id="infer-base-model"></select>
            </div>
            <div class="form-group">
              <label>LoRA アダプタ（任意）</label>
              <input type="text" id="infer-adapter" placeholder="例: my_job_20260101... (lora_adapters/image/配下)" />
              <small class="hint">空欄ならベースモデルのみで生成します。</small>
            </div>
            <div class="form-group">
              <label>プリセット（迷ったらこれ）</label>
              <select id="infer-preset">
                <option value="">（プリセットなし）</option>
              </select>
              <small class="hint">用途別に「安定しやすい設定」を自動適用します。必要なら後から個別調整できます。</small>
            </div>

            <div class="form-group">
              <label style="display:flex; gap:8px; align-items:center;">
                <input type="checkbox" id="infer-advanced" />
                高品質モード（Hi-Res/Refiner を使って破綻を減らす）
              </label>
              <small class="hint">ONにすると「二段生成」で画作りを安定させます（少し時間がかかります）。</small>
            </div>

            <div id="infer-advanced-options" class="form-grid" style="display:none; grid-template-columns:repeat(4, minmax(0,1fr)); gap:10px;">
              <div class="form-group">
                <label>Scheduler</label>
                <select id="infer-scheduler">
                  <option value="">（自動）</option>
                  <option value="dpmpp_2m">DPM++ 2M（おすすめ）</option>
                  <option value="euler">Euler</option>
                  <option value="euler_a">Euler a</option>
                </select>
              </div>
              <div class="form-group">
                <label>Hi-Res倍率</label>
                <input id="infer-hires-scale" type="number" step="0.1" value="1.5" />
                <small class="hint">1.0で無効（通常は1.4〜1.6）</small>
              </div>
              <div class="form-group">
                <label>Hi-Res Steps</label>
                <input id="infer-hires-steps" type="number" step="1" value="15" />
              </div>
              <div class="form-group">
                <label>Hi-Res Denoise</label>
                <input id="infer-hires-denoise" type="number" step="0.05" value="0.35" />
                <small class="hint">0.25〜0.45が無難</small>
              </div>
              <div class="form-group" style="grid-column:span 4;">
                <label style="display:flex; gap:8px; align-items:center;">
                  <input type="checkbox" id="infer-use-refiner" />
                  Refiner（SDXLのみ）を使う（用意している場合）
                </label>
                <small class="hint"><code>models/image/refiner</code> 配下に refiner モデルを置くと自動検出します。</small>
              </div>
            </div>

            <details class="panel" style="margin-top:10px;">
              <summary><i class="fas fa-sitemap"></i> ControlNet（任意：破綻抑制/構図固定）</summary>
              <div class="form-grid" style="margin-top:10px;">
                <div class="form-group">
                  <label>ControlNet種類</label>
                  <select id="infer-controlnet-type">
                    <option value="">なし</option>
                    <option value="canny">Canny（輪郭）</option>
                    <option value="depth">Depth（深度）</option>
                    <option value="openpose">OpenPose（姿勢）</option>
                    <option value="custom">Custom（手動指定）</option>
                  </select>
                  <small class="hint">有効化するには「Control画像」も指定してください。</small>
                </div>
                <div class="form-group">
                  <label>ControlNetモデル（任意）</label>
                  <input id="infer-controlnet-model" placeholder="例）canny_sdxl（models/image/controlnet配下）" />
                  <small class="hint">空欄の場合、種類に応じた既定名を探します（例：canny_sdxl）。</small>
                </div>
                <div class="form-group" style="grid-column:span 2;">
                  <label>Control画像（必須）</label>
                  <input type="file" id="infer-controlnet-file" accept="image/*" />
                  <small class="hint">輪郭/深度/姿勢などのガイド画像を指定します（JPG/PNG）。</small>
                </div>
                <div class="form-group" style="grid-column:span 4;">
                  <img id="infer-controlnet-preview" style="max-width:100%; display:none; border:1px solid #ddd; border-radius:8px;" />
                </div>
              </div>
            </details>
            <details class="panel" style="margin-top:10px;">
              <summary><i class="fas fa-brush"></i> 画像編集（Img2Img / Inpaint / Outpaint）</summary>
              <div class="help-note" style="margin-top:10px;">
                <p><strong>ポイント：</strong>「元画像（init）」があると <strong>img2img</strong>（雰囲気変換）や <strong>inpaint</strong>（部分修復）ができます。</p>
                <ul>
                  <li><strong>img2img</strong>：元画像のみ（mask不要）</li>
                  <li><strong>inpaint/outpaint</strong>：元画像 + マスク画像が必須</li>
                  <li><strong>注意：</strong>ControlNet と Inpaint（mask指定）の併用は現在サポートしていません。</li>
                </ul>
              </div>
              <div class="form-grid" style="margin-top:10px;">
                <div class="form-group">
                  <label>編集モード</label>
                  <select id="infer-inpaint-mode">
                    <option value="">自動（おすすめ）</option>
                    <option value="img2img">img2img（元画像から変換）</option>
                    <option value="inpaint">inpaint（部分修復）</option>
                    <option value="outpaint">outpaint（外側拡張）</option>
                  </select>
                  <small class="hint">自動の場合：init+maskなら inpaint、initのみなら img2img。</small>
                </div>

                <div class="form-group">
                  <label>元画像（init）</label>
                  <input id="infer-init-image" type="file" accept="image/*" />
                  <small class="hint">img2img/inpaint/outpaint を使う場合に指定します。</small>
                </div>

                <div class="form-group">
                  <label>マスク画像（mask）</label>
                  <input id="infer-mask-image" type="file" accept="image/*" />
                  <small class="hint">inpaint/outpaint の場合に必須です。白=編集、黒=保持（一般的）。</small>
                </div>
              </div>

              <div class="form-grid" style="margin-top:10px;">
                <div class="form-group">
                  <label>プレビュー（init）</label>
                  <img id="infer-init-preview" style="max-width:100%; border:1px solid #ddd; border-radius:8px;" />
                </div>
                <div class="form-group">
                  <label>プレビュー（mask）</label>
                  <img id="infer-mask-preview" style="max-width:100%; border:1px solid #ddd; border-radius:8px;" />
                </div>
              </div>
            </details>


            <div class="form-group">
              <label>プロンプト（Prompt）</label>
              <textarea id="infer-prompt" rows="4" placeholder="masterpiece, best quality, ..."></textarea>
            </div>
            <div class="form-group">
              <label>Negative プロンプト（Prompt）</label>
              <textarea id="infer-neg" rows="2" placeholder="worst quality, lowres, ..."></textarea>
            </div>
            <div class="form-group">
              <label>幅（Width）</label>
              <input type="number" id="infer-w" value="1024" step="64" min="256" />
            </div>
            <div class="form-group">
              <label>高さ（Height）</label>
              <input type="number" id="infer-h" value="1024" step="64" min="256" />
            </div>
            <div class="form-group">
              <label>ステップ数（Steps）</label>
              <input type="number" id="infer-steps" value="28" min="1" />
            </div>
            <div class="form-group">
              <label>ガイダンス（CFG Scale）</label>
              <input type="number" id="infer-guidance" value="5" step="0.5" min="0" />
            </div>
            <div class="form-group">
              <label>Seed (空欄=ランダム)</label>
              <input type="number" id="infer-seed" placeholder="例: 1234" />
            </div>
          </div>
        </div>
        <div class="column is-7">
          <div class="card">
            <h4>生成結果</h4>
            <div class="console-log small-log" id="infer-meta">(まだ生成していません)</div>
            <div style="margin-top:10px; display:flex; justify-content:center;">
              <img id="infer-image" style="max-width:100%; border-radius:12px; display:none;" />
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section id="tab-history" class="tab-pane">
    <header class="pane-header">
      <h3>学習履歴（画像）</h3>
      <button id="image-history-refresh" class="action-btn"><i class="fas fa-sync"></i> 更新</button>
    </header>
    <div class="pane-body">
<details class="help-details">
  <summary><i class="fas fa-circle-question"></i> はじめての方へ（基本の流れ）</summary>
  <div class="help-content">
    <ol>
      <li><strong>データセット</strong>：<code>datasets/image</code> に画像フォルダを用意します（画像と同名の <code>.txt</code> キャプション推奨）。</li>
      <li><strong>学習</strong>：ベースモデル（<code>models/image</code>）とデータセットを選び、学習設定を調整して開始します。</li>
      <li><strong>検証（生成）</strong>：モデルをロードし、プロンプトから画像を生成して品質を確認します。</li>
    </ol>
    <div class="help-note">
      <p><strong>VRAMが足りない場合：</strong>解像度／Batch／Steps を下げる、Gradient Checkpointing を有効にするなどで改善します。</p>
    </div>
  </div>
</details>
      <div class="card">
        <table class="data-table">
          <thead>
            <tr>
              <th>日時</th>
              <th>モデル</th>
              <th>データセット</th>
              <th>状態</th>
              <th>Notes</th><th>操作</th>
            </tr>
          </thead>
          <tbody id="image-history-body"></tbody>
        </table>
      </div>
    </div>
  </section>
`;

// ============================================================================
// 2. State
// ============================================================================

let _pollTimer = null;

// ============================================================================
// 3. Utilities
// ============================================================================

function switchTab(targetId) {
  // nav
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.target === targetId);
  });
  // panes
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === targetId);
  });
}

function setText(el, text) {
  if (!el) return;
  el.textContent = text;
}

function escapeHtml(s) {
  return (s ?? '').toString()
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

// モデル一覧の要素が文字列/オブジェクトどちらでも安全に名前を取り出す
function getModelName(modelItem) {
  if (!modelItem) return '';
  if (typeof modelItem === 'string') return modelItem;
  return modelItem.name || modelItem.model || modelItem.id || String(modelItem);
}


function toast(msg) {
  // 既存システムにトーストが無い前提でも落ちない簡易通知
  try {
    // eslint-disable-next-line no-alert
    alert(msg);
  } catch (e) {
    console.log(msg);
  }
}

// ------------------------------------------------------------
// Open folder helper (Windows Explorer / Finder)
// ------------------------------------------------------------
async function openFolder({ key=null, path=null }) {
  try {
    await api.post('/utils/open_path', { key, path });
  } catch (e) {
    toast(`フォルダを開けませんでした: ${e.message || e}`);
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

function _appendOpenRow(afterEl, buttons) {
  if (!afterEl || !afterEl.parentElement) return;
  const row = document.createElement('div');
  row.className = 'open-folder-row';
  buttons.forEach(b => row.appendChild(b));
  afterEl.parentElement.appendChild(row);
}

// ============================================================================
// 4. API Calls
// ============================================================================

async function refreshSystemInfo() {
  const elSummary = document.getElementById('img-system-info-container');
  if (elSummary) elSummary.textContent = '取得中...';
  try {
    const data = await api.get('/system_info/image');
    const paths = await fetchPathsSafe();
    renderSystemInfoTab(elSummary, data, paths, 'image');
  } catch (e) {
    if (elSummary) {
      elSummary.innerHTML = '<div class="system-msg error">システム情報の取得に失敗しました。ログを確認してください。</div>';
    }
  }
}

async function refreshモデルs() {
  const res = await api.get('/image/models');
  const models = res?.models || [];
  const sel1 = $('image-base-model');
  const sel2 = $('infer-base-model');
  [sel1, sel2].forEach(sel => {
    if (!sel) return;
    sel.innerHTML = '';
    for (const m of models) {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.name;
      sel.appendChild(opt);
    }
  });
}

async function refreshデータセットs() {
  const res = await api.get('/image/datasets');
  const datasets = res?.datasets || [];
  const sel = $('image-dataset-select');
  sel.innerHTML = '';
  for (const d of datasets) {
    const opt = document.createElement('option');
    opt.value = d.name;
    opt.textContent = `${d.name} (${d.count || 0} files)`;
    sel.appendChild(opt);
  }
  if (datasets.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(datasets/image にデータセットがありません)';
    sel.appendChild(opt);
  }

  // 付随情報
  sel.addEventListener('change', () => {
    const cur = datasets.find(x => x.name === sel.value);
    setText($('image-dataset-info'), cur ? `選択中: ${cur.name}\nパス: ${cur.path || ''}\n画像数: ${cur.count ?? 0}\n例: ${cur.example || ''}` : '(未選択)');
  }, { once: true });
  const cur = datasets.find(x => x.name === sel.value);
  setText($('image-dataset-info'), cur ? `選択中: ${cur.name}\nパス: ${cur.path || ''}\n画像数: ${cur.count ?? 0}\n例: ${cur.example || ''}` : '(未選択)');
}

async function pollTrain状態() {
  try {
    const st = await api.get('/image/train/status');
    const statusEl = $('image-train-status');
    const logEl = $('image-train-log');
    setText(statusEl, `status=${st.status}  job_id=${st.job_id || ''}`);
    if (logEl) {
      const logs = (st.logs || []).join('\n');
      logEl.textContent = logs || '(ログなし)';
      // 末尾まで自動スクロール
      logEl.scrollTop = logEl.scrollHeight;
    }

    // 学習中サンプル（最新）を取得して表示
    try {
      const samp = await api.get('/image/train/latest_sample');
      const imgEl = $('image-train-sample-img');
      const metaEl = $('image-train-sample-meta');
      if (samp && samp.exists && samp.image_base64) {
        if (imgEl) {
          imgEl.src = samp.image_base64;
          imgEl.style.display = 'block';
        }
        if (metaEl) {
          setText(metaEl, `表示中: ${samp.filename || ''}`);
        }
      } else {
        if (metaEl) setText(metaEl, '(未生成)');
        if (imgEl) {
          imgEl.style.display = 'none';
          imgEl.removeAttribute('src');
        }
      }
    } catch (e2) {
      // サンプル取得失敗は無視（学習は継続させる）
    }
  } catch (e) {
    setText($('image-train-status'), `status=error (${e.message})`);
  }
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

let imageLogModalState = { jobId: null, logFile: null };

function showImageLogModal(jobId, logFile) {
  imageLogModalState = { jobId, logFile };
  const modal = document.getElementById('image-log-modal');
  if (!modal) return;
  modal.classList.remove('hidden');
  document.getElementById('image-log-path').textContent = logFile ? logFile : '';
  refreshImageLogModal();
}


function bindImageExtraModals() {
  document.getElementById('image-dataset-close')?.addEventListener('click', ()=>{
    document.getElementById('image-dataset-modal')?.classList.add('hidden');
  });
  document.getElementById('image-envdiff-close')?.addEventListener('click', ()=>{
    document.getElementById('image-envdiff-modal')?.classList.add('hidden');
  });
}

async function showImageDatasetModal(jobId) {
  bindImageExtraModals();
  const modal = document.getElementById('image-dataset-modal');
  const pre = document.getElementById('image-dataset-content');
  if (!modal || !pre) return;
  pre.textContent = "取得中...";
  modal.classList.remove('hidden');
  try {
    const rep = await api.get(`/runs/dataset_report?job_id=${encodeURIComponent(jobId)}`);
    pre.textContent = JSON.stringify(rep, null, 2);
  } catch(e) {
    pre.textContent = "取得できません: " + e.message;
  }
}

async function showImageEnvDiffModal(jobId) {
  bindImageExtraModals();
  const modal = document.getElementById('image-envdiff-modal');
  const pre = document.getElementById('image-envdiff-content');
  if (!modal || !pre) return;
  pre.textContent = "取得中...";
  modal.classList.remove('hidden');
  try {
    const diff = await api.get(`/runs/env_diff?job_id=${encodeURIComponent(jobId)}`);
    pre.textContent = JSON.stringify(diff, null, 2);
  } catch(e) {
    pre.textContent = "取得できません: " + e.message;
  }
}

function hideImageLogModal() {
  const modal = document.getElementById('image-log-modal');
  if (!modal) return;
  modal.classList.add('hidden');
}

async function refreshImageLogModal() {
  const st = imageLogModalState;
  if (!st || !st.logFile) {
    setText(document.getElementById('image-log-stdout'), 'ログファイルがありません');
    setText(document.getElementById('image-log-stderr'), '');
    return;
  }
  try {
    const res = await api.get('/utils/read_text_file', { path: st.logFile, max_lines: 800 });
    const lines = res?.lines || [];
    const sp = splitStdoutStderr(lines);
    setText(document.getElementById('image-log-stdout'), sp.stdout.join('\n'));
    setText(document.getElementById('image-log-stderr'), sp.stderr.join('\n'));
  } catch (e) {
    setText(document.getElementById('image-log-stdout'), `取得失敗: ${e.message}`);
    setText(document.getElementById('image-log-stderr'), '');
  }
}


function bindImageHistoryActions() {
  const tbody = $('image-history-body');
  if (!tbody || tbody.__bound) return;
  tbody.__bound = true;

  tbody.addEventListener('click', async (ev) => {
    const btn = ev.target.closest('button');
    if (!btn) return;

    const jobId = btn.dataset.jobid || btn.getAttribute('data-jobid');

    try {
      if (btn.classList.contains('js-open-log')) {
        const logFile = btn.dataset.logfile;
        await showImageLogModal(jobId, logFile);
        return;
      }

      if (btn.classList.contains('js-open-dataset-report')) {
        if (!jobId) return;
        await showImageDatasetModal(jobId);
        return;
      }

      if (btn.classList.contains('js-open-env-diff')) {
        if (!jobId) return;
        await showImageEnvDiffModal(jobId);
        return;
      }

      if (btn.classList.contains('js-cancel-job')) {
        if (!jobId) return;
        if (!confirm('学習ジョブをキャンセルしますか？')) return;
        await api.post('/image/train/cancel/' + jobId, {});
        await refreshHistory();
        return;
      }

      if (btn.classList.contains('js-rerun-job')) {
        if (!jobId) return;
        if (!confirm('この履歴の設定で再実行しますか？')) return;
        const res = await api.post('/image/train/rerun/' + jobId, {});
        alert('再実行を開始しました: job_id=' + (res?.job_id || jobId));
        await refreshHistory();
        return;
      }
    } catch (e) {
      alert('操作に失敗しました: ' + (e?.message || e));
    }
  });
}

async function refreshHistory() {
  const res = await api.get('/image/train/history');
  const hist = res?.history || [];
  const tbody = $('image-history-body');
  tbody.innerHTML = '';
  for (const h of hist) {
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
  bindImageHistoryActions();
}

// ============================================================================
// 5. Mount
// ============================================================================

export async function refreshImageModelManager() {
  const tbody = $('image-models-body');
  if (!tbody) return;
  tbody.innerHTML = '<tr><td colspan="3">読み込み中...</td></tr>';
  try {
    const res = await api.get('/image/models');
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
      const badge = type
        ? ('<span class="badge ' + (type === 'hf' ? 'badge-blue' : 'badge-gray') + '">' + escapeHtml(type) + '</span>')
        : '<span class="badge badge-gray">local</span>';

      tr.innerHTML =
        '<td>' + escapeHtml(name) + '</td>' +
        '<td>' + badge + '</td>' +
        '<td><div class="row-actions">' +
          '<button class="icon-btn js-model-meta" data-name="' + escapeHtml(name) + '" title="詳細"><i class="fas fa-circle-info"></i></button>' +
          '<button class="icon-btn js-model-open" data-name="' + escapeHtml(name) + '" title="フォルダ"><i class="fas fa-folder-open"></i></button>' +
          '<button class="icon-btn danger-btn js-model-del" data-name="' + escapeHtml(name) + '" title="削除"><i class="fas fa-trash"></i></button>' +
        '</div></td>';
      tbody.appendChild(tr);
    }
  } catch(e) {
    tbody.innerHTML = '<tr><td colspan="3">取得失敗: ' + escapeHtml(e.message || String(e)) + '</td></tr>';
  }
}

function bindImageModelManager() {
  // allow detail box inline button
  window.__open_image_model_folder = async (name) => {
    const rel = 'models/image/' + name;
    await openFolder({ path: rel });
  };

  const tbody = $('image-models-body');
  const btn = $('image-models-refresh');
  if (btn && !btn.__bound) {
    btn.__bound = true;
    btn.addEventListener('click', refreshImageModelManager);
  }
    const openRootBtn = $('image-models-open-root');
  if (openRootBtn && !openRootBtn.__bound) {
    openRootBtn.__bound = true;
    openRootBtn.addEventListener('click', () => openFolder({ key: 'models_image' }));
  }
const dlBtn = $('image-download-btn');
  if (dlBtn && !dlBtn.__bound) {
    dlBtn.__bound = true;
    dlBtn.addEventListener('click', async () => {
      const repoEl = $('image-hf-repo-id');
      const repoId = (repoEl?.value || '').trim();
      if (!repoId) {
        alert('Repo ID を入力してください。');
        return;
      }
      try {
        await api.post('/image/models/download', { repo_id: repoId });
        alert('ダウンロードを開始しました。完了後に「更新」を押してください。');
        // 自動更新（軽め）
        setTimeout(() => refreshImageModelManager(), 1500);
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
        const meta = await api.get('/image/models/' + encodeURIComponent(name) + '/meta');
        const box = $('image-model-detail');
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
      const rel = 'models/image/' + name;
      await openFolder({ path: rel });
    
} else if (b.classList.contains('js-model-del')) {
      try {
        const chk = await api.get('/image/models/' + encodeURIComponent(name) + '/predelete_check');
        let msg = 'モデル「' + name + '」を削除しますか？\n参照数: ' + (chk.reference_count || 0);
        if (chk.active_job_using) msg += '\n※現在の実行中ジョブがこのモデルを使用しています。';
        if (!confirm(msg)) return;
        await api.delete('/image/models/' + encodeURIComponent(name));
        await refreshImageModelManager();
      } catch(e) {
        alert('削除失敗: ' + e.message);
      }
    }
  })

  const detailBox = $('image-model-detail');
  if (detailBox && !detailBox.__bound) {
    detailBox.__bound = true;
    detailBox.addEventListener('click', async (ev) => {
      const b = ev.target.closest('button');
      if (!b) return;
      const name = b.dataset.name;
      if (!name) return;
      if (b.classList.contains('js-model-open')) {
        const rel = 'models/image/' + name;
        await openFolder({ path: rel });
      }
    });
  }
;
}

export async function mount(navContainer, contentContainer) {
  navContainer.innerHTML = NAV_HTML;
  contentContainer.innerHTML = CONTENT_HTML;
  bindImageModelManager();
  refreshImageModelManager();
  contentContainer.insertAdjacentHTML("beforeend", `
  <!-- image log modal -->
  <div id="image-log-modal" class="modal-backdrop hidden">
    <div class="modal-card">
      <div class="modal-header">
        <div style="font-weight:700;">ログ表示</div>
        <button id="image-log-close" class="action-btn">閉じる</button>
      </div>
      <div class="modal-body">
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px;">
          <button id="image-log-refresh" class="action-btn">更新</button>
          <button id="image-log-openfolder" class="action-btn">ログフォルダを開く</button>
          <span id="image-log-path" style="font-size:0.85rem; opacity:0.9;"></span>
        </div>
        <div class="two-col">
          <div>
            <div class="small-title">stdout（標準出力）</div>
            <pre id="image-log-stdout" class="log-pre"></pre>
          </div>
          <div>
            <div class="small-title">stderr（標準エラー）</div>
            <pre id="image-log-stderr" class="log-pre"></pre>
          </div>
        </div>
      </div>
    </div>
  </div>
`);

  // log modal binds
  const imageLogClose = document.getElementById('image-log-close');
  const imageLogRefresh = document.getElementById('image-log-refresh');
  const imageLogOpen = document.getElementById('image-log-openfolder');
  if (imageLogClose) imageLogClose.addEventListener('click', hideImageLogModal);
  if (imageLogRefresh) imageLogRefresh.addEventListener('click', refreshImageLogModal);
  if (imageLogOpen) imageLogOpen.addEventListener('click', async () => {
    const st = imageLogModalState;
    if (!st || !st.logFile) return;
    await api.post('/utils/open_path', { path: st.logFile });
  });


  // BK43運用: 危険/破綻しやすい“自由調整・学習系”UIをデフォルトで隠す（プリセット運用を優先）
  // - 画像学習/LoRA作成系は不要（ロードのみの方針）
  // - 既存DOMがあれば非表示にする（HTML互換のため削除はしない）
  try {
    const hideIds = [
      'image-train-start','image-train-stop','image-train-refresh','image-train-status','image-train-log',
      'image-lora-r','image-lora-alpha','image-lr','image-epochs','image-batch','image-grad-acc','image-grad-ckpt',
      'image-use-8bit','image-mixed','image-save-every',
      'image-train-sample-img','image-train-sample-meta','image-sample-prompt','image-base-model','image-model-type'
    ];
    for (const id of hideIds) {
      const el = document.getElementById(id);
      if (el) {
        // 入力要素は無効化し、表示は隠す
        el.disabled = true;
        el.style.display = 'none';
      }
    }
  } catch (_) {}

  // フォルダを開くボタンを設置（ファイル配置/移動がしやすいように）
  try {
    installOpenFolderButtons();
  } catch (_) {}

  // ControlNet（任意）: Control画像の読み込み
  let _controlnetDataUrl = null;
  const cnFile = $('infer-controlnet-file');
  const cnPrev = $('infer-controlnet-preview');
  if (cnFile) {
    cnFile.addEventListener('change', () => {
      const f = cnFile.files && cnFile.files[0];
      if (!f) { _controlnetDataUrl = null; if (cnPrev) cnPrev.style.display='none'; return; }
      const r = new FileReader();
      r.onload = () => {
        _controlnetDataUrl = String(r.result || '');
        if (cnPrev) { cnPrev.src = _controlnetDataUrl; cnPrev.style.display = 'block'; }
      };
      r.readAsDataURL(f);
    });
  }


  // inference UI helpers (presets / advanced options)
  try {
    
  // Img2Img / Inpaint: init/mask 画像の読み込み
  let _initImageDataUrl = null;
  let _maskImageDataUrl = null;
  const initFile = $('infer-init-image');
  const maskFile = $('infer-mask-image');
  const initPrev = $('infer-init-preview');
  const maskPrev = $('infer-mask-preview');

  if (initPrev) { initPrev.style.display = 'none'; }
  if (maskPrev) { maskPrev.style.display = 'none'; }

  if (initFile) {
    initFile.addEventListener('change', () => {
      const f = initFile.files && initFile.files[0];
      if (!f) { _initImageDataUrl = null; if (initPrev) initPrev.style.display='none'; return; }
      const r = new FileReader();
      r.onload = () => {
        _initImageDataUrl = String(r.result || '');
        if (initPrev) { initPrev.src = _initImageDataUrl; initPrev.style.display = 'block'; }
      };
      r.readAsDataURL(f);
    });
  }

  if (maskFile) {
    maskFile.addEventListener('change', () => {
      const f = maskFile.files && maskFile.files[0];
      if (!f) { _maskImageDataUrl = null; if (maskPrev) maskPrev.style.display='none'; return; }
      const r = new FileReader();
      r.onload = () => {
        _maskImageDataUrl = String(r.result || '');
        if (maskPrev) { maskPrev.src = _maskImageDataUrl; maskPrev.style.display = 'block'; }
      };
      r.readAsDataURL(f);
    });
  }


async function loadImageInferencePresets() {
  try {
    const sel = $('infer-preset');
    if (!sel) return;
    const presetRes = await api.get('/image/inference/presets');
    const presets = (presetRes && presetRes.presets) ? presetRes.presets : [];
    // remove old preset options (keep first default option)
    Array.from(sel.querySelectorAll('option[data-is-preset="1"]')).forEach(o => o.remove());
    if (presets.length) {
      presets.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = `${p.label} - ${p.description}`;
        opt.dataset.defaults = JSON.stringify(p.defaults || {});
        opt.dataset.negative = p.negative_prompt || '';
        opt.dataset.isPreset = '1';
        sel.appendChild(opt);
      });
    }
    if (!sel.dataset.boundChange) {
      sel.addEventListener('change', () => {
        const opt = sel.options[sel.selectedIndex];
        if (!opt) return;
        const neg = opt?.dataset?.negative || '';
        const negEl = $('infer-neg');
        if (neg && negEl && !negEl.value) negEl.value = neg;
        // defaults -> fields (only when advanced ON)
        let d = {};
        try { d = opt?.dataset?.defaults ? JSON.parse(opt.dataset.defaults) : {}; } catch (_) { d = {}; }
        if ($('infer-advanced')?.checked) {
          if (d.scheduler && $('infer-scheduler')) $('infer-scheduler').value = d.scheduler;
          if (d.hires_scale && $('infer-hires-scale')) $('infer-hires-scale').value = d.hires_scale;
          if (d.hires_steps && $('infer-hires-steps')) $('infer-hires-steps').value = d.hires_steps;
          if (d.hires_denoise && $('infer-hires-denoise')) $('infer-hires-denoise').value = d.hires_denoise;
          if (typeof d.use_refiner === 'boolean' && $('infer-use-refiner')) $('infer-use-refiner').checked = d.use_refiner;
          if (d.steps && $('infer-steps')) $('infer-steps').value = d.steps;
          if (d.cfg && $('infer-guidance')) $('infer-guidance').value = d.cfg;
        }
      });
      sel.dataset.boundChange = '1';
    }
  } catch (e) {
    // presets are optional (no regression)
  }
}

  const advCb = $('infer-advanced');
  const advBox = $('infer-advanced-options');
  if (advCb && advBox) {
    const sync = () => {
      advBox.style.display = advCb.checked ? 'grid' : 'none';
    };
    advCb.addEventListener('change', sync);
    sync();
  }


  // inference presets (optional)
  loadImageInferencePresets();
  // tab switching
  navContainer.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.target));
  });

  // system
  $('refresh-sys-btn')?.addEventListener('click', refreshSystemInfo);

  // datasets
  $('refresh-datasets')?.addEventListener('click', refreshデータセットs);

  // train status poll
  $('image-train-refresh')?.addEventListener('click', pollTrain状態);

  $('image-train-start')?.addEventListener('click', async () => {
    try {
      const base_model = $('image-base-model').value;
      const dataset = $('image-dataset-select').value;
      if (!base_model) return toast('ベースモデルを選択してください');
      if (!dataset) return toast('データセットを選択してください');

      const params = {
        model_type: $('image-model-type').value,
        resolution: Number($('image-resolution').value || 1024),
        epochs: Number($('image-epochs').value || 1),
        train_batch_size: Number($('image-batch').value || 1),
        gradient_accumulation_steps: Number($('image-grad-acc').value || 1),
        learning_rate: Number($('image-lr').value || 0.0001),
        lora_r: Number($('image-lora-r').value || 16),
        lora_alpha: Number($('image-lora-alpha').value || 32),
        use_8bit_adam: Boolean($('image-use-8bit').checked),
        gradient_checkpointing: Boolean($('image-grad-ckpt').checked),
        mixed_precision: $('image-mixed').value,
        save_every_n_steps: Number($('image-save-every')?.value || 0),
        sample_prompt: String($('image-sample-prompt')?.value || '').trim(),
      };

      const res = await api.post('/image/train/start', { base_model, dataset, params });
      toast(`学習を開始しました: job_id=${res.job_id}`);

      // poll start
      if (_pollTimer) clearInterval(_pollTimer);
      _pollTimer = setInterval(pollTrain状態, 1500);
      await pollTrain状態();
    } catch (e) {
      toast(`開始失敗: ${e.message}`);
    }
  });

  $('image-train-stop')?.addEventListener('click', async () => {
    try {
      await api.post('/image/train/stop', {});
      toast('停止要求を送信しました');
      await pollTrain状態();
      if (_pollTimer) {
        clearInterval(_pollTimer);
        _pollTimer = null;
      }
    } catch (e) {
      toast(`停止失敗: ${e.message}`);
    }
  });

  // history
  $('image-history-refresh')?.addEventListener('click', refreshHistory);

  // inference
  $('image-infer-load')?.addEventListener('click', async () => {
    try {
      const base_model = $('infer-base-model').value;
      const adapter_path = $('infer-adapter').value.trim() || null;
      const res = await api.post('/image/inference/load', { base_model, adapter_path });
      toast(`ロード: ${res.status}`);
    } catch (e) {
      toast(`ロード失敗: ${e.message}`);
    }
  });

  $('image-infer-unload')?.addEventListener('click', async () => {
    try {
      const res = await api.post('/image/inference/unload', {});
      toast(`アンロード: ${res.status}`);
    } catch (e) {
      toast(`アンロード失敗: ${e.message}`);
    }
  });

  $('image-generate')?.addEventListener('click', async () => {
    try {
      const payload = {
        base_model: $('infer-base-model').value,
        adapter_path: $('infer-adapter').value.trim() || null,
        prompt: $('infer-prompt').value,
        negative_prompt: $('infer-neg').value,
        width: Number($('infer-w').value || 1024),
        height: Number($('infer-h').value || 1024),
        steps: Number($('infer-steps').value || 28),
        guidance_scale: Number($('infer-guidance').value || 5),
        seed: $('infer-seed').value === '' ? null : Number($('infer-seed').value),
      };

      const useAdvanced = !!$('infer-advanced')?.checked;
      if (useAdvanced) {
        payload.preset_id = $('infer-preset')?.value || null;
        payload.scheduler = $('infer-scheduler')?.value || '';
        payload.hires_scale = Number($('infer-hires-scale')?.value || 1.5);
        payload.hires_steps = Number($('infer-hires-steps')?.value || 15);
        payload.hires_denoise = Number($('infer-hires-denoise')?.value || 0.35);
        payload.use_refiner = !!$('infer-use-refiner')?.checked;
        // guidance_scale を cfg としても送る（バックエンド側は cfg を参照）
        payload.cfg = Number($('infer-guidance')?.value || payload.guidance_scale || 7.0);

        // ControlNet（任意）
        const cnType = String($('infer-controlnet-type')?.value || '');
        if (cnType && _controlnetDataUrl) {
          payload.controlnet_type = cnType;
          payload.control_image_base64 = _controlnetDataUrl;
          const cnModel = String($('infer-controlnet-model')?.value || '').trim();
          if (cnModel) payload.controlnet_model = cnModel;
        }

        // Img2Img / Inpaint（任意）
        const ipMode = String($('infer-inpaint-mode')?.value || '').trim();
        const hasInit = !!_initImageDataUrl;
        const hasMask = !!_maskImageDataUrl;

        if (ipMode) payload.inpaint_mode = ipMode;
        if (hasInit) payload.init_image_base64 = _initImageDataUrl;
        if (hasMask) payload.mask_image_base64 = _maskImageDataUrl;

        // クライアント側の整合性チェック（サーバーでも検証します）
        if (hasMask && !hasInit) return toast('マスク画像を指定する場合は「元画像（init）」も指定してください。');
        if ((ipMode === 'inpaint' || ipMode === 'outpaint') && (!hasInit || !hasMask)) {
          return toast('inpaint/outpaint を選んだ場合は「元画像（init）」と「マスク画像（mask）」の両方が必要です。');
        }
        if (ipMode === 'img2img' && !hasInit) return toast('img2img を選んだ場合は「元画像（init）」が必要です。');
        if (ipMode === 'img2img' && hasMask) return toast('img2img の場合、マスク画像は使用できません（inpaint を選んでください）。');
        if (cnType && hasMask) return toast('ControlNet と Inpaint（mask指定）は同時に使えません。どちらか一方を選んでください。');
      }

      const endpoint = useAdvanced ? '/image/inference/generate_advanced' : '/image/inference/generate';
      const res = await api.post(endpoint, payload);
      if (res.status !== 'ok') throw new Error(res.message || 'generate failed');

      const imgEl = $('infer-image');
      imgEl.src = res.image_base64;
      imgEl.style.display = 'block';

      if (useAdvanced) {
        setText($('infer-meta'), `seed=${res.seed} / artifact_id=${res.artifact_id || '-'} / hires=${res.hires_used} / refiner=${res.refiner_used}`);
      } else {
        setText($('infer-meta'), `seed=${res.seed} / artifact_id=${res.artifact_id || '-'}`);
      }
    } catch (e) {
      toast(`生成失敗: ${e.message}`);
    }
  });

  // initial load
  try {
  await Promise.all([
    refreshSystemInfo(),
    refreshモデルs(),
    refreshデータセットs(),
    refreshHistory(),
  ]);
} catch (e) {
    console.warn('[image_ui] initial load failed', e);
  }
}
  catch (e) {
    console.warn('[image_ui] try block failed', e);
  }

function installOpenFolderButtons() {
  // datasets/image 配下（フォルダ一覧）
  const dsSel = $('image-dataset-select');
  if (dsSel) {
    _appendOpenRow(dsSel, [
      _mkOpenBtn('データセットフォルダを開く (datasets/image)', () => openFolder({ key: 'datasets_image' })),
      _mkOpenBtn('選択データセットを開く', () => {
        const v = (dsSel.value || '').trim();
        if (!v) return toast('データセットを選択してください');
        openFolder({ path: `datasets/image/${v}` });
      }),
    ]);
  }

  // 検証（生成）: LoRA アダプタ（任意）
  const lora = $('infer-adapter');
  if (lora) {
    _appendOpenRow(lora, [
      _mkOpenBtn('LoRAフォルダを開く (lora_adapters/image)', () => openFolder({ key: 'lora_adapters_image' })),
    ]);
  }
}






async function validateDatasetAndToggleStart_image() {
  const btn = document.getElementById("image-train-start");
  const sel = document.getElementById("image-dataset-select");
  if(!btn || !sel) return;
  const warnId = "image-dataset-validate";
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
    const res = await api.post("/api/datasets/validate", { mode: "image", dataset, kind });
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



}
window.imageUI = window.imageUI || {}; if (typeof refreshModels==='function') window.imageUI.refreshModels = refreshModels;