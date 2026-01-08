/**
 * static/js/app.js
 * アプリケーションのメインエントリポイント。
 * モード切替（Text/Image/Audio）とモジュールのルーティングを管理します。
 * (v14: Modular Architecture Support)
 */

import * as TextUI from './modules/text_ui.js';
import * as ImageUI from './modules/image_ui.js';
import * as AudioUI from './modules/audio_ui.js';
import { api } from './api.js';

// モード定義（capabilities に応じて有効化します）
const MODES_BASE = {
    'text': { module: TextUI, label: 'テキスト（LLM）' },
    'image': { module: ImageUI, label: '画像（Diffusers）' },
    'audio': { module: AudioUI, label: '音声（TTS/VC）' },
};

function buildModes(capabilities) {
    const caps = capabilities || {};
    const enabled = {};
    for (const [key, def] of Object.entries(MODES_BASE)) {
        if (caps[key] === true) enabled[key] = def;
    }
    // 何らかの理由で取れない場合は text のみを安全側で有効化
    if (Object.keys(enabled).length === 0) enabled['text'] = MODES_BASE['text'];
    return enabled;
}

class App {
    constructor() {
        this.currentMode = null;
        this.navContainer = document.getElementById('dynamic-nav');
        this.contentContainer = document.getElementById('dynamic-content');
        this.modeSelect = document.getElementById('app-mode-select');
        this.modes = buildModes({text:true});
        
        // DOM要素が見つからない場合の安全策 (index.htmlが古い場合など)
        if (!this.navContainer || !this.contentContainer) {
            console.error("Critical Error: Dynamic containers not found in DOM.");
            return;
        }

        this.init();
    }


    async init() {
        console.log("LoRA Factory App Initialized");

        // 画面に初期化中表示
        this.contentContainer.innerHTML = '<div class="loading-placeholder"><div><i class="fas fa-circle-notch fa-spin"></i> サーバー状態を確認しています...</div></div>';

        // capabilities を取得（失敗しても text のみで継続）
        let caps = null;
        try {
            const res = await api.get('/capabilities');
            caps = res && res.capabilities ? res.capabilities : null;
        } catch (e) {
            console.warn('capabilities の取得に失敗しました。text モードのみで起動します。', e);
        }

        this.modes = buildModes(caps);

        // モード切替ドロップダウンを capabilities に合わせて再構築
        if (this.modeSelect) {
            const current = this.modeSelect.value || 'text';
            this.modeSelect.innerHTML = '';
            for (const [key, def] of Object.entries(this.modes)) {
                const opt = document.createElement('option');
                opt.value = key;
                opt.textContent = def.label;
                this.modeSelect.appendChild(opt);
            }
            // 現在選択が無効なら text に寄せる
            if (!this.modes[current]) {
                this.modeSelect.value = 'text';
            } else {
                this.modeSelect.value = current;
            }

            this.modeSelect.addEventListener('change', (e) => {
                this.switchMode(e.target.value);
            });
        }

        // 初期モード
        const initialMode = (this.modeSelect && this.modeSelect.value) ? this.modeSelect.value : 'text';
        this.switchMode(initialMode);
    }

    async switchMode(modeKey) {
        const mode = this.modes[modeKey];
        
        if (!mode) {
            alert(`不明なモード: ${modeKey}`);
            return;
        }

        if (!mode.module) {
            alert(`「${mode.label}」モードは現在利用できません。\n`);
            // 選択を元に戻す
            if (this.modeSelect) this.modeSelect.value = this.currentMode;
            return;
        }

        if (this.currentMode === modeKey) return;

        console.log(`Switching to mode: ${modeKey}`);
        this.currentMode = modeKey;

        // UIのリセット（フェードアウト効果などを入れるならここ）
        this.navContainer.innerHTML = '';
        this.contentContainer.innerHTML = '';
        
        // ローディング表示
        this.contentContainer.innerHTML = '<div style="padding:2rem; color:#888;">モジュールを読み込み中...</div>';

        try {
            // 各モジュールの mount メソッドを呼び出して描画を委譲
            await mode.module.mount(this.navContainer, this.contentContainer);
            
            // 成功したらプルダウンも同期（初期ロード時用）
            if (this.modeSelect) this.modeSelect.value = modeKey;
            
        } catch (e) {
            console.error(e);
            this.contentContainer.innerHTML = `<div style="padding:2rem; color:var(--danger);">
                <h3>モジュール読み込みエラー</h3>
                <p>${e.message}</p>
            </div>`;
        }
    }
}

// DOM読み込み完了後に起動
window.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
window.addEventListener('lora:job_succeeded', async (ev) => {
  try {
    const mod = ev?.detail?.modality || 'text';
    try { await api.post('/utils/refresh_models', {}); } catch (_) {}
    if (mod === 'text' && window.textUI?.refreshModels) await window.textUI.refreshModels();
    if (mod === 'image' && window.imageUI?.refreshModels) await window.imageUI.refreshModels();
    if (mod === 'audio' && window.audioUI?.refreshModels) await window.audioUI.refreshModels();

    const toast = document.createElement('div');
    toast.textContent = '学習が完了しました。モデル一覧を更新しました。';
    toast.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:9999;background:#222;color:#fff;padding:10px 12px;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,.35);font-size:13px;';
    document.body.appendChild(toast);
    setTimeout(()=>toast.remove(), 3500);
  } catch (_) {}
});
