/**
 * api.js
 * バックエンドAPIとの通信を管理するラッパーモジュール。
 * (v14: Modular Architecture Support)
 */

const API_BASE = '/api';

async function readErrorMessage(res) {
    try {
        const data = await res.clone().json();
        if (data && typeof data === 'object') {
            if (data.detail) return String(data.detail);
            return JSON.stringify(data);
        }
    } catch (_) { /* ignore */ }
    try {
        const txt = await res.clone().text();
        return txt ? txt.slice(0, 5000) : '';
    } catch (_) { /* ignore */ }
    return '';
}

export const api = {
    /**
     * GETリクエストを送信する
     * @param {string} endpoint - APIエンドポイント (例: '/system_info')
     * @returns {Promise<any>} レスポンスJSON
     */
    async get(endpoint) {
        const res = await fetch(`${API_BASE}${endpoint}`);
        if (!res.ok) {
            let detail = '';
            try {
                const j = await res.json();
                if (j && typeof j === 'object') {
                    if (j.message) detail += `
原因: ${j.message}`;
                    if (Array.isArray(j.user_messages) && j.user_messages.length) {
                        // NOTE: 文字列中に改行を含める場合はテンプレートリテラル or \n を使用する。
                        // ここはUIへ人間向けの追記事項を出すため、\n\n を明示して整形する。
                        detail += "\n\n不足/対処:";
                        for (const m of j.user_messages) {
                            detail += `
- ${m.title || m.code || '不明'}: ${m.detail || ''}`.trimEnd();
                        }
                    }
                }
            } catch (_) {}
            throw new Error(`APIエラー: ${res.status} ${res.statusText}${detail}`);
        }
        return res.json();
    },

    /**
     * POSTリクエストを送信する (JSON)
     * @param {string} endpoint - APIエンドポイント
     * @param {object} data - 送信するデータオブジェクト
     * @returns {Promise<any>} レスポンスJSON
     */
    async post(endpoint, data) {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data),
        });
        if (!res.ok) {
            let detail = '';
            try {
                const j = await res.json();
                if (j && typeof j === 'object') {
                    if (j.message) detail += `
原因: ${j.message}`;
                    if (Array.isArray(j.user_messages) && j.user_messages.length) {
                        // NOTE: 文字列中に改行を含める場合はテンプレートリテラル or \n を使用する。
                        // ここはUIへ人間向けの追記事項を出すため、\n\n を明示して整形する。
                        detail += "\n\n不足/対処:";
                        for (const m of j.user_messages) {
                            detail += `
- ${m.title || m.code || '不明'}: ${m.detail || ''}`.trimEnd();
                        }
                    }
                }
            } catch (_) {}
            throw new Error(`APIエラー: ${res.status} ${res.statusText}${detail}`);
        }
        return res.json();
    },

    /**
     * ファイルをアップロードする (Multipart)
     * @param {string} endpoint - APIエンドポイント
     * @param {File} file - アップロードするファイルオブジェクト
     * @returns {Promise<any>} レスポンスJSON
     */
    async upload(endpoint, file) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) throw new Error(`アップロード失敗: ${res.statusText}`);
        return res.json();
    },

    /**
     * 複数のファイルをアップロードする (Multipart)
     * @param {string} endpoint - APIエンドポイント
     * @param {FileList|File[]} files - アップロードするファイルのリスト
     * @returns {Promise<any>} レスポンスJSON
     */
    async uploadMany(endpoint, files) {
        const formData = new FormData();
        for (const file of files) {
            // webkitRelativePath があればそれを使い、フォルダ構造情報を維持する
            const rel = file.webkitRelativePath || file.name;
            formData.append('files', file, rel);
        }
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `フォルダアップロード失敗: ${res.statusText}`);
        }
        return res.json();
    },

    /**
     * DELETEリクエストを送信する
     * @param {string} endpoint - APIエンドポイント
     * @returns {Promise<any>} レスポンスJSON
     */
    async delete(endpoint) {
        const res = await fetch(`${API_BASE}${endpoint}`, { method: 'DELETE' });
        if (!res.ok) { const msg = await readErrorMessage(res); throw new Error(`削除失敗: ${res.status} ${msg || res.statusText}`); }
        return res.json();
    }
};