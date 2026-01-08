// plugins/heretic/frontend_plugin.js
// v5.9 - Full UI for all "The Void" features:
//   - Library layout guide + auto-move (flatten src layout)
//   - Debug Hooks UI (module picker, enable/disable, stats viewer)
//   - SVD Weight Analysis UI (pick A/B checkpoints, calc diff, export LoRA safetensors)
//
// Notes:
// - Hook stats require that Text inference model is loaded (user should select a model in normal Text tab at least once).
// - Weight diff uses checkpoints under models/text; export writes to lora_adapters/text/heretic by default.

const HERETIC_UI_VERSION = "v6.1-connect";


function codePillStyle() {
  // Theme-safe chip for code: higher contrast but works on light/dark.
  return "background:rgba(127,127,127,0.16);border:1px solid rgba(127,127,127,0.32);border-radius:6px;padding:1px 6px;";
}
function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function isHereticActive() {
  // Most robust: check the Heretic tab button state directly.
  const btn =
    document.querySelector('.nav-btn[data-target="tab-heretic"].active') ||
    document.querySelector('.nav-btn[data-target="tab-heretic"].is-active') ||
    document.querySelector('.nav-btn[data-target="tab-heretic"].selected') ||
    document.querySelector('.nav-btn[data-target="tab-heretic"][aria-selected="true"]') ||
    document.querySelector('[role="tab"][data-target="tab-heretic"][aria-selected="true"]') ||
    null;
  if (btn) return true;

  // Fallback: pane has active/show
  const pane = document.getElementById("tab-heretic");
  if (pane && (pane.classList.contains("active") || pane.classList.contains("show"))) return true;

  // Fallback: any tab button marked selected whose target equals tab-heretic
  const anySelected =
    document.querySelector('.nav-btn.active[data-target]') ||
    document.querySelector('.nav-btn[aria-selected="true"][data-target]') ||
    document.querySelector('.nav-btn.is-active[data-target]') ||
    document.querySelector('[role="tab"][aria-selected="true"][data-target]') ||
    null;
  if (anySelected?.dataset?.target === "tab-heretic") return true;

  return false;
}

function activeTargetId() {
  const btn =
    document.querySelector('.nav-btn.active[data-target]') ||
    document.querySelector('.nav-btn[aria-selected="true"][data-target]') ||
    document.querySelector('.nav-btn.is-active[data-target]') ||
    document.querySelector('[role="tab"][aria-selected="true"][data-target]') ||
    null;
  if (btn?.dataset?.target) return btn.dataset.target;

  const pane = document.querySelector('.tab-pane.active[id], .tab-pane.show[id]');
  if (pane?.id) return pane.id;

  return null;
}

function removeDuplicateHereticPanes() {
  try {
    const panes = Array.from(document.querySelectorAll('#tab-heretic'));
    if (panes.length <= 1) return;
    // Keep the first one in DOM order, remove others.
    for (let i = 1; i < panes.length; i++) {
      try { panes[i].remove(); } catch (_) {}
    }
  } catch (_) {}
}

function installVisibilityGuard() {
  // Force-hide Heretic pane unless it is the active tab.
  const tab = document.getElementById("tab-heretic");
  if (!tab) return () => {};

  // If a previous guard exists, clean it up.
  try {
    if (window.__heretic_guard_cleanup__) window.__heretic_guard_cleanup__();
  } catch (_) {}

  
const tick = () => {
  try {
    const shouldShow = isHereticActive();

    if (shouldShow) {
      tab.style.display = "";
      tab.style.visibility = "visible";
      tab.style.pointerEvents = "";
      tab.style.maxHeight = "";
      tab.style.height = "100%";
      tab.style.overflow = "hidden";
    } else {
      // Strong hide: prevents "bleed" even if host positions panes unusually.
      tab.style.display = "none";
      tab.style.visibility = "hidden";
      tab.style.pointerEvents = "none";
      tab.style.maxHeight = "0";
      tab.style.height = "0";
      tab.style.overflow = "hidden";
    }
  } catch (_) {}
};

tick();
  const id = window.setInterval(tick, 250);

  const clickHandler = () => tick();
  document.addEventListener("click", clickHandler, true);

  const cleanup = () => {
    try { window.clearInterval(id); } catch (_) {}
    try { document.removeEventListener("click", clickHandler, true); } catch (_) {}
  };

  return cleanup;
}

function badgeStyle(kind) {
  const map = {
    ok:   "background:#e8fff0;border:1px solid #7ad59a;color:#0f5132;",
    warn: "background:#fff8e1;border:1px solid #f0c36d;color:#7a5a00;",
    err:  "background:#ffe8e8;border:1px solid #e07a7a;color:#7a0a0a;",
    info: "background:#eef5ff;border:1px solid #8bb3ff;color:#113a7a;",
  };
  return map[kind] || map.info;
}

function cardStyle() {
  return "padding:10px;border-radius:12px;border:1px solid rgba(0,0,0,0.10);background:rgba(127,127,127,0.10);";
}

function debugBoxStyle() {
  // Theme-friendly debug panel
  return "margin-top:10px;padding:10px;border-radius:10px;background:rgba(127,127,127,0.10);border:1px solid rgba(127,127,127,0.28);color:inherit;";
}

function inputStyle() {
  return "padding:6px 8px;border-radius:10px;border:1px solid rgba(0,0,0,0.18);min-width:240px;";
}

function smallInputStyle() {
  return "padding:6px 8px;border-radius:10px;border:1px solid rgba(0,0,0,0.18);min-width:120px;";
}

function btnStyle(kind="") {
  const base = "padding:6px 10px;border-radius:10px;border:1px solid rgba(0,0,0,0.18);background:#f7f7f7;cursor:pointer;";
  if (kind==="primary") return base + "background:#eef5ff;border-color:#8bb3ff;";
  if (kind==="danger") return base + "background:#ffe8e8;border-color:#e07a7a;";
  return base;
}

function paneHTML() {
  return `
  <div id="tab-heretic" class="tab-pane" style="height:100%;min-height:0;flex-direction:column;overflow:hidden;color:var(--text);">
    <div id="heretic-scroll" style="flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;padding-right:6px;">
    <div class="card" style="padding:12px;">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <div style="font-weight:700;font-size:15px;">
          Heretic <span style="font-weight:600;font-size:12px;color:var(--text-sub);">(${HERETIC_UI_VERSION})</span>
        </div>
        <div id="heretic-lib-badge"
             style="padding:2px 8px;border-radius:999px;font-size:12px;${badgeStyle("info")}">
          Checking library...
        </div>
      </div>

      <div style="margin-top:10px;display:grid;grid-template-columns:1fr;gap:10px;">

        <div class="card" style="${cardStyle()}">
          <div style="font-weight:700;margin-bottom:6px;">Heretic Library (upstream) 配置ガイド</div>
          <div id="heretic-lib-summary" style="font-size:13px;line-height:1.7;">読み込み中...</div>

          <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;">
            <button id="heretic-open-folder" style="${btnStyle()}">heretic_master フォルダを開く</button>
            <button id="heretic-open-site" style="${btnStyle()}">Heretic のサイトを開く</button>
            <button id="heretic-auto-flatten" style="${btnStyle("primary")};display:none;">推奨配置へ自動移動（安全確認）</button>
          </div>

          <div id="heretic-lib-steps" style="margin-top:10px;font-size:13px;line-height:1.7;"></div>

          <div id="heretic-debug" style="${debugBoxStyle()}">
            <div style="font-weight:800;margin-bottom:6px;">Debug</div>
            <div id="heretic-debug-lines" style="font-size:12.5px;line-height:1.7;color:var(--text);">初期化中...</div>
          </div>
        </div>

        <div class="card" style="${cardStyle()}">
          <div style="font-weight:700;margin-bottom:6px;">Debug Hooks</div>
          <div style="font-size:13px;line-height:1.7;color:var(--text);margin-bottom:8px;">
            推論モデルがロードされている状態で、任意モジュールに hook を付けて統計を取得します（mean/std/norm/min/max）。
          </div>

          <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
            <input id="heretic-hook-search" placeholder="module名 検索（例: model.layers.0）" style="${inputStyle()}" />
            <button id="heretic-hook-refresh-modules" style="${btnStyle()}">候補更新</button>
            <span style="font-size:12px;color:var(--text-sub);" id="heretic-hook-count"></span>
          </div>

          <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
            <select id="heretic-hook-module" style="${inputStyle()}"></select>
            <select id="heretic-hook-type" style="${smallInputStyle()}">
              <option value="forward_hook">forward_hook</option>
              <option value="forward_pre_hook">forward_pre_hook</option>
            </select>
            <button id="heretic-hook-enable" style="${btnStyle("primary")}">Enable</button>
            <button id="heretic-hook-disable" style="${btnStyle("danger")}">Disable</button>
            <button id="heretic-hook-stats" style="${btnStyle()}">Stats更新</button>
          </div>

          <div id="heretic-hook-status" style="margin-top:8px;font-size:13px;color:var(--text);"></div>
          <div id="heretic-hook-stats-view" style="margin-top:8px;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;font-size:12.5px;line-height:1.6;"></div>
        </div>

        <div class="card" style="${cardStyle()}">
          <div style="font-weight:700;margin-bottom:6px;">SVD Weight Analysis</div>
          <div style="font-size:13px;line-height:1.7;color:var(--text);margin-bottom:8px;">
            models/text 配下のチェックポイント2つを選び、差分(A-B)を計算し、SVDで低ランク化して safetensors に保存します。
          </div>

          <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
            <button id="heretic-wd-refresh" style="${btnStyle()}">候補ファイル更新</button>
            <span style="font-size:12px;color:var(--text-sub);" id="heretic-wd-root"></span>
          </div>

          <div style="margin-top:8px;display:grid;grid-template-columns:1fr;gap:8px;">
            <div>
              <div style="font-size:12px;color:var(--text-sub);margin-bottom:4px;">Model A</div>
              <select id="heretic-wd-a" style="${inputStyle()};width:100%;"></select>
            </div>
            <div>
              <div style="font-size:12px;color:var(--text-sub);margin-bottom:4px;">Model B</div>
              <select id="heretic-wd-b" style="${inputStyle()};width:100%;"></select>
            </div>
          </div>

          <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
            <button id="heretic-wd-calc" style="${btnStyle("primary")}">Diff計算 (A-B)</button>
            <input id="heretic-wd-rank" type="number" value="8" min="1" max="256" style="${smallInputStyle()}" />
            <input id="heretic-wd-maxkeys" type="number" value="64" min="1" max="512" style="${smallInputStyle()}" />
            <input id="heretic-wd-regex" placeholder="key_regex（任意）" style="${inputStyle()}" />
            <button id="heretic-wd-export" style="${btnStyle()}">LoRA保存（SVD）</button>
          </div>

          <div id="heretic-wd-meta" style="margin-top:8px;font-size:13px;color:var(--text);"></div>
          <div id="heretic-wd-top" style="margin-top:8px;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;font-size:12.5px;line-height:1.6;"></div>
        </div>

      </div>
    </div>
      </div>
  </div>`;
}

async function fetchJson(url, opts) {
  const r = await fetch(url, opts || { method: "GET" });
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText} for ${url}`);
  return await r.json();
}

function setDebug(lines) {
  const el = document.getElementById("heretic-debug-lines");
  if (!el) return;
  el.innerHTML = `
    <ul style="margin:0 0 0 18px;">
      ${lines.map(l => `<li><span style="${codePillStyle()}"><code style="color:inherit;">${esc(l)}</code></span></li>`).join("")}
    </ul>
  `;
}

async function getLibraryInfo(ctx) {
  const api = ctx?.api;
  const tried = [];

  if (api?.get) {
    tried.push("TRY ctx.api.get('/heretic/library/info')");
    try {
      const info = await api.get("/heretic/library/info");
      tried.push("OK  ctx.api.get('/heretic/library/info')");
      return { info, tried, via: "ctx.api" };
    } catch (e) {
      tried.push(`FAIL ctx.api.get -> ${String(e)}`);
    }
  } else {
    tried.push("SKIP ctx.api.get (unavailable)");
  }

  tried.push("TRY fetch('/api/heretic/library/info')");
  try {
    const info = await fetchJson("/api/heretic/library/info");
    tried.push("OK  fetch('/api/heretic/library/info')");
    return { info, tried, via: "/api" };
  } catch (e) {
    tried.push(`FAIL fetch('/api/heretic/library/info') -> ${String(e)}`);
  }

  tried.push("TRY fetch('/heretic/library/info')");
  try {
    const info = await fetchJson("/heretic/library/info");
    tried.push("OK  fetch('/heretic/library/info')");
    return { info, tried, via: "legacy" };
  } catch (e) {
    tried.push(`FAIL fetch('/heretic/library/info') -> ${String(e)}`);
    const err = new Error("All library info probes failed");
    err.cause = e;
    throw Object.assign(err, { tried });
  }
}

async function postJson(ctx, path, payload) {
  const api = ctx?.api;
  if (api?.post) return await api.post(path, payload || {});
  return await fetchJson("/api" + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload || {}) });
}

function fmtBytes(n) {
  const x = Number(n || 0);
  if (x < 1024) return `${x} B`;
  if (x < 1024*1024) return `${(x/1024).toFixed(1)} KB`;
  if (x < 1024*1024*1024) return `${(x/1024/1024).toFixed(1)} MB`;
  return `${(x/1024/1024/1024).toFixed(2)} GB`;
}

function asciiBar(value, max, width=24) {
  const v = Number(value || 0);
  const m = Math.max(1e-9, Number(max || 1));
  const filled = Math.max(0, Math.min(width, Math.round((v/m)*width)));
  return "█".repeat(filled) + "░".repeat(width-filled);
}

async function refreshLibraryPanel(ctx) {
  const notify = ctx?.notify;
  const badge = document.getElementById("heretic-lib-badge");
  const summary = document.getElementById("heretic-lib-summary");
  const steps = document.getElementById("heretic-lib-steps");
  const btnFolder = document.getElementById("heretic-open-folder");
  const btnSite = document.getElementById("heretic-open-site");
  const btnFlatten = document.getElementById("heretic-auto-flatten");

  const GH_URL = "https://github.com/p-e-w/heretic";
  if (btnSite) btnSite.onclick = () => window.open(GH_URL, "_blank", "noopener,noreferrer");

  if (btnFolder) {
    btnFolder.onclick = async () => {
      try {
        await postJson(ctx, "/heretic/library/open_folder", {});
      } catch (e) {
        console.warn("[Heretic] open_folder failed:", e);
        if (notify) notify("フォルダを開く操作に失敗しました（APIが無い可能性）", "warn");
        else alert("フォルダを開く操作に失敗しました。");
      }
    };
  }

  try {
    const { info, tried, via } = await getLibraryInfo(ctx);
    const layout = info?.layout || "unknown";
    const p = info?.paths || {};

    setDebug([
      `UI_VERSION=${HERETIC_UI_VERSION}`,
      `VIA=${via}`,
      ...tried,
      `RESULT layout=${layout} layout_raw=${info?.layout_raw ?? "?"}`,
    ]);

    if (btnFlatten) {
      const can = !!info?.can_auto_flatten && layout === "src";
      btnFlatten.style.display = can ? "" : "none";
      btnFlatten.onclick = async () => {
        const srcDir = p?.expected_src_dir || "(unknown)";
        const dstDir = p?.expected_flat_dir || "(unknown)";
        const msg =
          "自動移動を実行します（非破壊・移動のみ）。\n\n" +
          `移動元: ${srcDir}\n` +
          `移動先: ${dstDir}\n\n` +
          "実行しますか？";
        if (!confirm(msg)) return;

        btnFlatten.disabled = true;
        btnFlatten.textContent = "移動中...";
        try {
          const res = await postJson(ctx, "/heretic/library/flatten_layout", {});
          if (res?.ok) {
            if (notify) notify("自動移動が完了しました。再判定します。", "ok");
            else alert("自動移動が完了しました。");
            await refreshLibraryPanel(ctx);
          } else {
            const m = res?.message || "自動移動に失敗しました。";
            if (notify) notify(m, "warn");
            else alert(m);
          }
        } catch (e) {
          console.warn("[Heretic] flatten_layout failed:", e);
          if (notify) notify("自動移動に失敗しました（権限/ロックなど）", "warn");
          else alert("自動移動に失敗しました。");
        } finally {
          btnFlatten.disabled = false;
          btnFlatten.textContent = "推奨配置へ自動移動（安全確認）";
        }
      };
    }

    let kind = "info";
    let title = "確認";
    let body = "";

    if (layout === "flat") {
      kind = "ok";
      title = "OK: heretic_master/heretic を検出";
      body = "ライブラリ配置は推奨レイアウトです。";
    } else if (layout === "src") {
      kind = "warn";
      title = "注意: heretic_master/src/heretic を検出";
      body = "このレイアウトでも開発は可能ですが、推奨は heretic_master/heretic 直下です。";
    } else if (layout === "hyphen") {
      kind = "err";
      title = "要対応: heretic-master（ハイフン）を検出";
      body = "フォルダ名の '-' は Python import で問題になるため、'_' に変更して下さい（例: heretic_master）。";
    } else if (layout === "hyphen_src") {
      kind = "err";
      title = "要対応: heretic-master/src/heretic を検出";
      body = "フォルダ名の '-' を '_' に変更し、必要なら src レイアウトの注意も確認してください。";
    } else if (layout === "missing") {
      kind = "warn";
      title = "未検出";
      body = "heretic_master が見つかりません。plugins/heretic/ 配下に配置してください。";
    } else {
      kind = "info";
      title = "確認";
      body = "ライブラリ配置を判定できませんでした。";
    }

    if (badge) {
      badge.textContent = title;
      badge.setAttribute("style", "padding:2px 8px;border-radius:999px;font-size:12px;" + badgeStyle(kind));
    }

    const expectedDisplay =
      (layout === "flat" && p.expected_flat_dir) ? p.expected_flat_dir :
      (layout === "src" && p.expected_src_dir) ? p.expected_src_dir :
      (p.expected_root || "");

    const foundDisplay =
      (layout === "flat" && p.expected_flat_dir) ? p.expected_flat_dir :
      (layout === "src" && p.expected_src_dir) ? p.expected_src_dir :
      (p.found_root || "");

    const expected = expectedDisplay ? `<div>推奨配置: <code style="color:inherit;">${esc(expectedDisplay)}</code></div>` : "";
    const found = foundDisplay ? `<div>検出: <code style="color:inherit;">${esc(foundDisplay)}</code></div>` : "";
    const src = p.src_root ? `<div style="color:var(--text-sub);">src候補: <code style="color:inherit;">${esc(p.src_root)}</code></div>` : "";
    const hy = p.hyphen_root ? `<div style="color:var(--text-sub);">hyphen候補: <code style="color:inherit;">${esc(p.hyphen_root)}</code></div>` : "";

    if (summary) {
      summary.innerHTML = `
        <div style="margin-bottom:6px;">${esc(body)}</div>
        ${expected}${found}${src}${hy}
      `;
    }

    const notes = Array.isArray(info?.notes) ? info.notes : [];
    const manual = Array.isArray(info?.manual_steps) ? info.manual_steps : [];

    const notesHtml = notes.length
      ? `<div style="margin-top:8px;"><div style="font-weight:700;">注釈</div><ul style="margin:6px 0 0 18px;">${notes.map(n => `<li>${esc(n)}</li>`).join("")}</ul></div>`
      : "";

    const manualColor = (kind === "err") ? "color:#7a0a0a;" : (kind === "warn") ? "color:#7a5a00;" : "";
    const manualHtml = manual.length
      ? `<div style="margin-top:8px;"><div style="font-weight:700;${manualColor}">手動手順</div><ol style="margin:6px 0 0 18px;">${manual.map(s => `<li>${esc(s)}</li>`).join("")}</ol></div>`
      : "";

    if (steps) steps.innerHTML = notesHtml + manualHtml;

  } catch (e) {
    console.warn("[Heretic] library info unavailable:", e);
    const tried = e?.tried || [];
    setDebug([
      `UI_VERSION=${HERETIC_UI_VERSION}`,
      "ERROR=All library info probes failed",
      ...tried,
      "HINT open: http://localhost:8081/api/heretic/library/info",
    ]);

    if (badge) {
      badge.textContent = "Info: backend endpoints not detected";
      badge.setAttribute("style", "padding:2px 8px;border-radius:999px;font-size:12px;" + badgeStyle("info"));
    }
    if (summary) {
      summary.innerHTML = `
        <div>バックエンド側の <code style="color:inherit;">/api/heretic/library/info</code> が利用できないため、詳細判定はできません。</div>
      `;
    }
    if (steps) steps.innerHTML = "";
    const btnFlatten = document.getElementById("heretic-auto-flatten");
    if (btnFlatten) btnFlatten.style.display = "none";
  }
}

// ----------------------
// Hooks UI
// ----------------------

async function refreshHookModules(ctx) {
  const notify = ctx?.notify;
  const search = document.getElementById("heretic-hook-search");
  const sel = document.getElementById("heretic-hook-module");
  const count = document.getElementById("heretic-hook-count");
  const q = (search?.value || "").trim();

  if (!sel) return;
  sel.innerHTML = `<option value="">(読み込み中...)</option>`;
  if (count) count.textContent = "";

  try {
    // Try ctx.api first
    const api = ctx?.api;
    let res;
    if (api?.get) {
      res = await api.get(`/heretic/hooks/modules?q=${encodeURIComponent(q)}&limit=200`);
    } else {
      res = await fetchJson(`/api/heretic/hooks/modules?q=${encodeURIComponent(q)}&limit=200`);
    }
    const mods = res?.modules || [];
    sel.innerHTML = mods.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join("") || `<option value="">(候補なし)</option>`;
    if (count) count.textContent = `候補: ${mods.length}`;
  } catch (e) {
    console.warn("[Heretic] hooks/modules failed:", e);
    sel.innerHTML = `<option value="">(取得失敗)</option>`;
    const msg = "モジュール一覧の取得に失敗しました。先に通常のText推論でモデルをロードしてください。";
    if (notify) notify(msg, "warn"); else alert(msg);
  }
}

async function hookEnable(ctx) {
  const notify = ctx?.notify;
  const sel = document.getElementById("heretic-hook-module");
  const type = document.getElementById("heretic-hook-type");
  const status = document.getElementById("heretic-hook-status");
  const moduleName = sel?.value || "";
  const hookType = type?.value || "forward_hook";
  if (!moduleName) { if (notify) notify("module を選択してください", "warn"); else alert("module を選択してください"); return; }

  try {
    const res = await postJson(ctx, "/heretic/hooks/enable", { module_name: moduleName, hook_type: hookType });
    if (status) status.innerHTML = `✅ Enabled: <code style="color:inherit;">${esc(res.module_name)}</code> (${esc(res.hook_type)})`;
    if (notify) notify("Hook enabled", "ok");
  } catch (e) {
    console.warn("[Heretic] hooks/enable failed:", e);
    const msg = `Hook有効化に失敗: ${String(e)}`;
    if (status) status.textContent = msg;
    if (notify) notify(msg, "warn"); else alert(msg);
  }
}

async function hookDisable(ctx) {
  const notify = ctx?.notify;
  const status = document.getElementById("heretic-hook-status");
  try {
    const res = await postJson(ctx, "/heretic/hooks/disable", {});
    if (status) status.innerHTML = `🧹 Disabled: ${res.disabled ? "OK" : "No-op"}`;
    if (notify) notify("Hook disabled", "ok");
  } catch (e) {
    console.warn("[Heretic] hooks/disable failed:", e);
    const msg = `Hook無効化に失敗: ${String(e)}`;
    if (status) status.textContent = msg;
    if (notify) notify(msg, "warn"); else alert(msg);
  }
}

async function hookStats(ctx) {
  const notify = ctx?.notify;
  const view = document.getElementById("heretic-hook-stats-view");
  if (view) view.textContent = "取得中...";
  try {
    const api = ctx?.api;
    let res;
    if (api?.get) res = await api.get("/heretic/hooks/stats");
    else res = await fetchJson("/api/heretic/hooks/stats");
    const s = res?.stats;
    if (!s) {
      if (view) view.innerHTML = `<div style="color:var(--text-sub);">stats: (まだ取得されていません) → Hook有効化後、推論を1回実行してからStats更新してください。    </div>
  </div>`;
      return;
    }
    const rows = [
      ["module", s.module_name],
      ["hook_type", s.hook_type],
      ["shape", s.shape],
      ["dtype", s.dtype],
      ["device", s.device],
      ["mean", String(s.mean)],
      ["std", String(s.std)],
      ["norm", String(s.norm)],
      ["min", String(s.min)],
      ["max", String(s.max)],
    ];
    const maxAbs = Math.max(Math.abs(Number(s.min||0)), Math.abs(Number(s.max||0)), 1e-9);
    const bar = asciiBar(Math.abs(Number(s.mean||0)), maxAbs, 24);
    if (view) view.innerHTML = `
      <div>Activation stats (latest):</div>
      <pre style="margin:6px 0 0 0;white-space:pre-wrap;">${rows.map(([k,v])=>`${k.padEnd(9)}: ${v}`).join("\n")}</pre>
      <div style="margin-top:6px;color:var(--text);">|mean| relative bar: <code style="color:inherit;">${bar}</code></div>
      <div style="margin-top:4px;color:var(--text-sub);font-size:12px;">※ hook は推論中に値が流れたタイミングで更新されます。</div>
    `;
  } catch (e) {
    console.warn("[Heretic] hooks/stats failed:", e);
    const msg = `Stats取得に失敗: ${String(e)}`;
    if (view) view.textContent = msg;
    if (notify) notify(msg, "warn"); else alert(msg);
  }
}

// ----------------------
// Weight diff UI
// ----------------------

async function refreshWeightCandidates(ctx) {
  const notify = ctx?.notify;
  const root = document.getElementById("heretic-wd-root");
  const a = document.getElementById("heretic-wd-a");
  const b = document.getElementById("heretic-wd-b");

  if (a) a.innerHTML = `<option>(読み込み中...)</option>`;
  if (b) b.innerHTML = `<option>(読み込み中...)</option>`;
  if (root) root.textContent = "";

  try {
    const api = ctx?.api;
    let res;
    if (api?.get) res = await api.get("/heretic/weightdiff/candidates?limit=2000");
    else res = await fetchJson("/api/heretic/weightdiff/candidates?limit=2000");

    if (root) root.textContent = `root: ${res.root} / files: ${res.count}`;
    const files = res.files || [];
    const opts = files.map(f => `<option value="${esc(f.path)}">${esc(f.name)}  (${esc(fmtBytes(f.bytes))})</option>`).join("");
    if (a) a.innerHTML = opts || `<option value="">(候補なし)</option>`;
    if (b) b.innerHTML = opts || `<option value="">(候補なし)</option>`;
  } catch (e) {
    console.warn("[Heretic] weightdiff/candidates failed:", e);
    const msg = "候補ファイル取得に失敗しました。models/text 配下に .safetensors / .bin / .pt を置いてください。";
    if (notify) notify(msg, "warn"); else alert(msg);
    if (a) a.innerHTML = `<option value="">(取得失敗)</option>`;
    if (b) b.innerHTML = `<option value="">(取得失敗)</option>`;
  }
}

function renderTopList(top) {
  if (!Array.isArray(top) || top.length === 0) return "";
  const max = Math.max(...top.map(t => Number(t[1]||0)), 1e-9);
  return `
    <div style="margin-top:6px;">Top norms (up to 50):</div>
    <pre style="margin:6px 0 0 0;white-space:pre-wrap;">${top.map(t => {
      const k = t[0]; const n = Number(t[1]||0); const shape = JSON.stringify(t[2]||"");
      const bar = asciiBar(n, max, 18);
      return `${bar}  ${n.toFixed(4).padStart(10)}  ${shape.padEnd(14)}  ${k}`;
    }).join("\n")}</pre>
  `;
}

async function calcDiff(ctx) {
  const notify = ctx?.notify;
  const a = document.getElementById("heretic-wd-a")?.value || "";
  const b = document.getElementById("heretic-wd-b")?.value || "";
  const meta = document.getElementById("heretic-wd-meta");
  const topEl = document.getElementById("heretic-wd-top");
  if (!a || !b) { if (notify) notify("Model A/B を選択してください", "warn"); else alert("Model A/B を選択してください"); return; }
  if (meta) meta.textContent = "diff計算中...";
  if (topEl) topEl.textContent = "";

  try {
    const res = await postJson(ctx, "/heretic/weightdiff/calc", { modelA_path: a, modelB_path: b });
    const m = res.meta || {};
    if (meta) meta.innerHTML = `
      ✅ diff computed<br/>
      keys: <code style="color:inherit;">${esc(m.n_keys)}</code><br/>
      A: <code style="color:inherit;">${esc(m.modelA_path)}</code><br/>
      B: <code style="color:inherit;">${esc(m.modelB_path)}</code>
    `;
    if (topEl) topEl.innerHTML = renderTopList(m.top);
    if (notify) notify("diff computed", "ok");
  } catch (e) {
    console.warn("[Heretic] weightdiff/calc failed:", e);
    const msg = `diff計算に失敗: ${String(e)}`;
    if (meta) meta.textContent = msg;
    if (notify) notify(msg, "warn"); else alert(msg);
  }
}

async function exportLora(ctx) {
  const notify = ctx?.notify;
  const rank = Number(document.getElementById("heretic-wd-rank")?.value || 8);
  const maxKeys = Number(document.getElementById("heretic-wd-maxkeys")?.value || 64);
  const regex = String(document.getElementById("heretic-wd-regex")?.value || "").trim();
  const meta = document.getElementById("heretic-wd-meta");

  if (meta) meta.textContent = "LoRA書き出し中...";

  try {
    const res = await postJson(ctx, "/heretic/weightdiff/export_lora", {
      out_path: "",
      rank: rank,
      max_keys: maxKeys,
      key_regex: regex
    });
    if (meta) meta.innerHTML = `✅ saved: <code style="color:inherit;">${esc(res.out_path)}</code><br/>rank=${esc(res.rank)} keys_used=${esc(res.n_keys_used)}`;
    if (notify) notify("LoRA saved", "ok");
  } catch (e) {
    console.warn("[Heretic] weightdiff/export_lora failed:", e);
    const msg = `LoRA書き出しに失敗: ${String(e)}`;
    if (meta) meta.textContent = msg;
    if (notify) notify(msg, "warn"); else alert(msg);
  }
}

// ----------------------
// Plugin entry
// ----------------------

export function getTextModeTabs(ctx = {}) {
  const buttonHTML = `
    <button class="nav-btn" data-target="tab-heretic">
      <i class="fas fa-flask"></i> Heretic
    </button>
  `;

  return [{
    id: "heretic",
    buttonHTML,
    paneHTML: paneHTML(),
    onMount: async () => {
      await refreshLibraryPanel(ctx);

      // wire hook UI
      const btnMods = document.getElementById("heretic-hook-refresh-modules");
      const btnEnable = document.getElementById("heretic-hook-enable");
      const btnDisable = document.getElementById("heretic-hook-disable");
      const btnStats = document.getElementById("heretic-hook-stats");
      const search = document.getElementById("heretic-hook-search");

      if (btnMods) btnMods.onclick = () => refreshHookModules(ctx);
      if (btnEnable) btnEnable.onclick = () => hookEnable(ctx);
      if (btnDisable) btnDisable.onclick = () => hookDisable(ctx);
      if (btnStats) btnStats.onclick = () => hookStats(ctx);
      if (search) search.onkeydown = (ev) => { if (ev.key === "Enter") refreshHookModules(ctx); };

      await refreshHookModules(ctx);

      // wire weight diff UI
      const btnWdRefresh = document.getElementById("heretic-wd-refresh");
      const btnCalc = document.getElementById("heretic-wd-calc");
      const btnExport = document.getElementById("heretic-wd-export");
      if (btnWdRefresh) btnWdRefresh.onclick = () => refreshWeightCandidates(ctx);
      if (btnCalc) btnCalc.onclick = () => calcDiff(ctx);
      if (btnExport) btnExport.onclick = () => exportLora(ctx);

      await refreshWeightCandidates(ctx);
    },
  }];
}

export default { getTextModeTabs };
