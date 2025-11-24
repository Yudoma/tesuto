function initializeBoard(wrapperSelector, idPrefix) {
    const wrapperElement = document.querySelector(wrapperSelector);
    if (!wrapperElement) return;

    const boardSlots = wrapperElement.querySelectorAll('.card-slot');
    boardSlots.forEach(addSlotEventListeners);

    const iconZone = document.getElementById(idPrefix + 'icon-zone');
    if(iconZone) addSlotEventListeners(iconZone);

    const drawerId = (idPrefix === 'opponent-') ? 'opponent-drawer' : 'player-drawer';
    const drawerWrapper = document.getElementById(drawerId);
    if(drawerWrapper) {
        drawerWrapper.querySelectorAll('.card-slot').forEach(addSlotEventListeners);
    }
    
    // バンク (旧C-Navi) のイベントリスナー設定 (Player初期化時に一度だけ行う)
    if (idPrefix === '') {
        const cDrawerWrapper = document.getElementById('c-drawer');
        if(cDrawerWrapper) {
            cDrawerWrapper.querySelectorAll('.card-slot').forEach(addSlotEventListeners);
        }
    }
    
    setupBoardUI(idPrefix);
    setupBoardButtons(idPrefix);
    setupCounters(idPrefix);

    const addDecorationWithErrorHandler = (path, zoneId) => {
        const container = document.getElementById(idPrefix + zoneId);
        
        // 修正: コンテナ自体がスロットの場合、または子要素にスロットがある場合の両対応
        let slot = container;
        if (container && !container.classList.contains('card-slot')) {
            const childSlot = container.querySelector('.card-slot');
            if (childSlot) {
                slot = childSlot;
            }
        }

        if (slot) {
            // 既存の装飾があれば削除（重複防止）
            const existing = slot.querySelector('.thumbnail[data-is-decoration="true"]');
            if (existing) existing.remove();

            const thumbnail = createCardThumbnail(path, slot, true, false, idPrefix);
            // アイコンの場合はデフォルトメモを設定
            if (zoneId === 'icon-zone') {
                thumbnail.dataset.memo = `[カード名:-]/#e0e0e0/#555/1.0/非表示/
[属性:-]/#e0e0e0/#555/1.0/非表示/
[マナ:-]/#e0e0e0/#555/1.0/非表示/
[BP:-]/#e0e0e0/#555/1.0/非表示/
[スペル:-]/#e0e0e0/#555/1.0/非表示/
[フレーバーテキスト:-]/#fff/#555/1.0/非表示/
[効果:-]/#e0e0e0/#555/0.7/非表示/`;
            }

            const img = thumbnail.querySelector('img');
            if (img) {
                img.onerror = () => {
                    thumbnail.remove();
                    if (zoneId !== 'icon-zone') {
                        syncMainZoneImage(zoneId, idPrefix);
                    }
                };
            }
        }
    };

    // 初期画像の適用処理（ストックまたはデフォルトを参照）
    const owner = idPrefix ? 'opponent' : 'player';
    
    // デフォルトパスの定義（フォールバック用）
    const defaultPaths = {
        'deck': './decoration/デッキ.png',
        'side-deck': './decoration/EXデッキ.png',
        'grave': './decoration/墓地エリア.png',
        'exclude': './decoration/除外エリア.png',
        'icon': idPrefix ? './decoration/サディスト.png' : './decoration/マゾヒスト.png'
    };

    const applyInitialDecoration = (targetType, zoneId) => {
        let path = defaultPaths[targetType];
        
        // customIconStocksが定義されており、該当の画像があればその先頭を使用
        if (typeof customIconStocks !== 'undefined' && 
            customIconStocks[owner] && 
            customIconStocks[owner][targetType] && 
            customIconStocks[owner][targetType].length > 0) {
            path = customIconStocks[owner][targetType][0];
        }

        addDecorationWithErrorHandler(path, zoneId);
    };

    applyInitialDecoration('deck', 'deck');
    applyInitialDecoration('side-deck', 'side-deck');
    applyInitialDecoration('grave', 'grave');
    applyInitialDecoration('exclude', 'exclude');
    applyInitialDecoration('icon', 'icon-zone');

    ['deck', 'grave', 'exclude', 'side-deck'].forEach(zone => syncMainZoneImage(zone, idPrefix));

    // 初期テーマ適用
    const smBtn = document.getElementById(idPrefix + 'sm-toggle-btn');
    let initialMode = smBtn ? smBtn.dataset.mode : null;
    if (!initialMode) {
        initialMode = idPrefix ? 'striker' : 'magickers';
    }
    updateSmTheme(idPrefix, initialMode);
}

function setupBoardUI(idPrefix) {
    const drawerId = (idPrefix === 'opponent-') ? 'opponent-drawer' : 'player-drawer';
    const drawerWrapper = document.getElementById(drawerId);
    if (!drawerWrapper) return;
    
    const drawerToggleBtn = document.getElementById(idPrefix === 'opponent-' ? 'opponent-drawer-toggle' : 'player-drawer-toggle');
    drawerToggleBtn?.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        const isOpen = drawerWrapper.classList.toggle('open');
        if (isOpen) {
            activateDrawerTab(idPrefix + 'deck-back-slots', drawerWrapper);
        }
    });

    const zoneSlotSelectors = {'deck': 'deck-back-slots', 'grave': 'grave-back-slots', 'exclude': 'exclude-back-slots', 'side-deck': 'side-deck-back-slots'};
    Object.keys(zoneSlotSelectors).forEach(zoneBaseId => {
        const zoneElement = document.getElementById(idPrefix + zoneBaseId);
        const slot = zoneElement?.querySelector('.card-slot');
        if (slot) {
            slot.addEventListener('click', (e) => {
                if (typeof isDecorationMode !== 'undefined' && isDecorationMode) {
                    return;
                }
                
                e.stopPropagation(); 
                playSe('ボタン共通.mp3');

                if (drawerWrapper) {
                    drawerWrapper.classList.add('open');
                    activateDrawerTab(idPrefix + zoneSlotSelectors[zoneBaseId], drawerWrapper);
                }
            });
        }
    });
}


function setupBoardButtons(idPrefix) {
    document.getElementById(idPrefix + 'draw-card')?.addEventListener('click', () => {
        playSe('1枚ドロー＆5枚ドロー.mp3');
        drawCardFromDeck(idPrefix);
    });
    document.getElementById(idPrefix + 'draw-5-card')?.addEventListener('click', () => {
        playSe('1枚ドロー＆5枚ドロー.mp3');
        for (let i = 0; i < 5; i++) if (!drawCardFromDeck(idPrefix)) break;
    });
    document.getElementById(idPrefix + 'shuffle-deck')?.addEventListener('click', () => {
        playSe('シャッフル.mp3');
        shuffleDeck(idPrefix);
    });
    
    document.getElementById(idPrefix + 'reset-and-draw')?.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        resetBoard(idPrefix);
    });
    document.getElementById(idPrefix + 'delete-deck-btn')?.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        deleteDeck(idPrefix);
    });
    
    // テーマ切り替えボタン
    document.getElementById(idPrefix + 'sm-toggle-btn')?.addEventListener('click', (e) => {
        playSe('ボタン共通.mp3');
        const btn = e.currentTarget;
        const currentMode = btn.dataset.mode;
        
        // 切り替え順序: マジッカーズ -> マゾヒスト -> ストライカー -> シンプル -> サディスト -> マジッカーズ...
        const themeOrder = ['magickers', 'masochist', 'striker', 'sadist', 'simple'];
        let nextIndex = themeOrder.indexOf(currentMode) + 1;
        if (nextIndex >= themeOrder.length) nextIndex = 0;
        const nextMode = themeOrder[nextIndex];

        updateSmTheme(idPrefix, nextMode);
    });

    document.getElementById(idPrefix + 'surrender-btn')?.addEventListener('click', () => {
        playSe('敗北.mp3'); // 降参SEを敗北に変更
        // 降参時の敗北表示
        if (typeof autoConfig !== 'undefined' && autoConfig.autoGameEnd) {
             const msg = idPrefix ? 'YOU WIN!' : 'YOU LOSE...';
             if (typeof window.showGameResult === 'function') {
                 window.showGameResult(msg);
             }
        }
    });
    
    // システムボタン (stopPropagation追加)
    document.getElementById(idPrefix + 'system-btn')?.addEventListener('click', (e) => {
        e.stopPropagation();
        playSe('ボタン共通.mp3');
        const commonDrawer = document.getElementById('common-drawer');
        if (commonDrawer) {
            // トグル動作はui.js側で制御するか、ここでもトグルにする
            // ここではシンプルにopenクラスを付与するだけに留め、ui.js側で競合しないようにする
            // ※ui.jsのトグル化対応に合わせて、ここでは単純な開く動作、またはui.jsのロジックに任せる
            commonDrawer.classList.add('open');
            if (typeof activateDrawerTab === 'function') {
                activateDrawerTab('common-spec-panel', commonDrawer);
            }
        }
    });
    
    document.getElementById(idPrefix + 'export-deck-btn')?.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        const defaultName = (idPrefix ? 'opponent' : 'player') + '_deck';
        const fileName = prompt("保存するファイル名を入力してください", defaultName);
        if(fileName) {
            exportDeck(idPrefix, fileName);
        }
    });
    
    document.getElementById(idPrefix + 'import-deck-btn')?.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        importDeck(idPrefix);
    });
}

function setupCounters(idPrefix) {
    let lpDecreaseTimer = null, manaDecreaseTimer = null;
    const lpCounter = document.getElementById(idPrefix + 'counter-value');
    const manaCounter = document.getElementById(idPrefix + 'mana-counter-value');
    const hyphenCounter = document.getElementById(idPrefix + 'hyphen-counter-value');

    const attachAutoDecreaseLogic = (btnId, counter, intervalInputId) => {
        const btn = document.getElementById(btnId);
        if (!btn) return;
        
        const wrapperClass = idPrefix ? '.opponent-wrapper' : '.player-wrapper';
        const wrapper = document.querySelector(wrapperClass);

        let timerId = null;
        if (!btn.dataset.originalText) btn.dataset.originalText = btn.textContent;

        btn.addEventListener('click', () => {
            if (timerId) {
                clearInterval(timerId);
                timerId = null;
                btn.textContent = btn.dataset.originalText;
                btn.style.backgroundColor = '';
                btn.style.boxShadow = '';
                if(wrapper) wrapper.classList.remove('auto-decrease-active');
                stopSe('自動減少.mp3');
                playSe('ボタン共通.mp3');
            } else {
                if (btn.textContent !== '停止') btn.dataset.originalText = btn.textContent;
                btn.textContent = '停止';
                btn.style.backgroundColor = '#cc0000';
                btn.style.boxShadow = '0 2px #800000';
                if(wrapper) wrapper.classList.add('auto-decrease-active');
                playSe('自動減少.mp3', true);
                
                let interval = 1000;
                const input = document.getElementById(intervalInputId);
                if (input) {
                    const val = parseFloat(input.value);
                    if (!isNaN(val) && val > 0) {
                        interval = val * 1000;
                    }
                }

                timerId = setInterval(() => {
                    const newVal = Math.max(0, (parseInt(counter.value) || 0) - 1);
                    counter.value = newVal;
                    
                    // LP0時の敗北判定
                    if (newVal === 0 && counter.classList.contains('lp-counter-input') && autoConfig.autoGameEnd) {
                        clearInterval(timerId);
                        timerId = null;
                        btn.textContent = btn.dataset.originalText;
                        btn.style.backgroundColor = '';
                        btn.style.boxShadow = '';
                        if(wrapper) wrapper.classList.remove('auto-decrease-active');
                        stopSe('自動減少.mp3');
                        
                        const msg = idPrefix ? 'YOU WIN!' : 'YOU LOSE...';
                        if (typeof window.showGameResult === 'function') {
                            window.showGameResult(msg);
                        }
                    }
                }, interval);
            }
        });
    };

    const intervalId = idPrefix ? 'opponent-auto-decrease-interval' : 'player-auto-decrease-interval';
    attachAutoDecreaseLogic(idPrefix + 'lp-auto-decrease-btn', lpCounter, intervalId);
    attachAutoDecreaseLogic(idPrefix + 'mana-auto-decrease-btn', manaCounter, intervalId);

    const counterWrapperId = idPrefix ? idPrefix + 'counter-wrapper' : 'player-counter-wrapper';
    const counterWrapper = document.getElementById(counterWrapperId);
    
    counterWrapper?.querySelectorAll('.counter-btn[data-value]').forEach(button => {
        const value = parseInt(button.dataset.value);
        const group = button.closest('.hand-counter-group');
        const targetCounter = group.querySelector('input[type="number"]') || group.querySelector('.hyphen-counter-input');
        
        let repeatTimer = null;
        let initialTimer = null;
        const startAction = (e) => {
            if (e.button !== undefined && e.button !== 0) return;
            playSe('ボタン共通.mp3');
            if(!targetCounter) return;
            
            const updateVal = () => {
                let currentVal;
                if (targetCounter.isContentEditable) {
                    currentVal = parseInt(targetCounter.textContent) || 0;
                    const newVal = Math.max(0, currentVal + value);
                    targetCounter.textContent = newVal;
                } else {
                    currentVal = parseInt(targetCounter.value) || 0;
                    const newVal = Math.max(0, currentVal + value);
                    targetCounter.value = newVal;
                    // LP手動操作時の敗北判定
                    if (newVal === 0 && targetCounter.classList.contains('lp-counter-input') && autoConfig.autoGameEnd) {
                        const msg = idPrefix ? 'YOU WIN!' : 'YOU LOSE...';
                        if (typeof window.showGameResult === 'function') window.showGameResult(msg);
                    }
                }
            };
            
            updateVal();
            initialTimer = setTimeout(() => {
                repeatTimer = setInterval(updateVal, 200);
            }, 300);
        };
        const stopAction = () => {
            clearTimeout(initialTimer);
            clearInterval(repeatTimer);
        };

        button.addEventListener('mousedown', startAction);
        button.addEventListener('mouseup', stopAction);
        button.addEventListener('mouseleave', stopAction);
        button.addEventListener('touchstart', startAction);
        button.addEventListener('touchend', stopAction);
    });
}

function drawCardFromDeck(idPrefix) {
    const deckSlots = document.querySelectorAll(`#${idPrefix}deck-back-slots .card-slot`);
    const cardToDraw = Array.from(deckSlots).map(s => s.querySelector('.thumbnail')).find(t => t);
    
    // デッキ切れの場合の敗北判定
    if (!cardToDraw) {
        if (typeof autoConfig !== 'undefined' && autoConfig.autoGameEnd) {
             const msg = idPrefix ? 'YOU WIN!' : 'YOU LOSE...';
             if (typeof window.showGameResult === 'function') {
                 window.showGameResult(msg);
             }
        }
        return false;
    }

    const handSlots = document.querySelectorAll(`#${idPrefix}hand-zone .card-slot`);
    const emptyHandSlot = Array.from(handSlots).find(s => !s.querySelector('.thumbnail'));
    if (!emptyHandSlot) return false;

    const sourceSlot = cardToDraw.parentNode;
    sourceSlot.removeChild(cardToDraw);
    emptyHandSlot.appendChild(cardToDraw);

    // ドロー時の反転設定 (drawFlipped)
    if (typeof autoConfig !== 'undefined' && autoConfig.drawFlipped) {
        const imgElement = cardToDraw.querySelector('.card-image');
        const deckZone = document.getElementById(idPrefix + 'deck');
        let deckImgSrc = './decoration/デッキ.png';
        
        // 現在のデッキ装飾画像を取得
        if (deckZone) {
            const decoratedThumbnail = deckZone.querySelector('.thumbnail[data-is-decoration="true"]');
            if (decoratedThumbnail) {
                const decoratedImg = decoratedThumbnail.querySelector('.card-image');
                if (decoratedImg) deckImgSrc = decoratedImg.src;
            }
        }
        
        // 既に裏側でなければ裏側にする
        if (cardToDraw.dataset.isFlipped !== 'true') {
            cardToDraw.dataset.originalSrc = imgElement.src;
            imgElement.src = deckImgSrc;
            cardToDraw.dataset.isFlipped = 'true';
        }
    } else {
        resetCardFlipState(cardToDraw);
    }

    arrangeSlots(idPrefix + 'deck-back-slots');
    syncMainZoneImage('deck', idPrefix);
    return true;
}

function shuffleDeck(idPrefix) {
    const deckContainer = document.getElementById(idPrefix + 'deck-back-slots');
    if (!deckContainer) return;
    const slots = deckContainer.querySelectorAll('.card-slot');
    let thumbnails = [];
    slots.forEach(s => {
        s.querySelectorAll('.thumbnail').forEach(t => thumbnails.push(s.removeChild(t)));
    });
    shuffleArray(thumbnails);
    thumbnails.forEach((t, i) => slots[i]?.appendChild(t));
    syncMainZoneImage('deck', idPrefix);
}

// 手札シャッフル機能
window.shuffleHand = function(idPrefix) {
    const handContainer = document.getElementById(idPrefix + 'hand-zone');
    if (!handContainer) return;
    const slots = handContainer.querySelectorAll('.card-slot');
    let thumbnails = [];
    slots.forEach(s => {
        s.querySelectorAll('.thumbnail').forEach(t => thumbnails.push(s.removeChild(t)));
    });
    
    shuffleArray(thumbnails);
    
    thumbnails.forEach((t, i) => {
        if (slots[i]) slots[i].appendChild(t);
    });
};

function resetBoard(idPrefix) {
    const wrapperSelector = idPrefix ? '.opponent-wrapper' : '.player-wrapper';
    const allSlots = document.querySelectorAll(`${wrapperSelector} .card-slot, #${idPrefix}drawer .card-slot`);
    let cardThumbnails = [];

    allSlots.forEach(slot => {
        const baseParentZoneId = getBaseId(getParentZoneId(slot));
        if (['free-space-slots', 'icon-zone', 'side-deck', 'side-deck-back-slots', 'token-zone-slots'].includes(baseParentZoneId)) return;
        slot.querySelectorAll('.thumbnail:not([data-is-decoration="true"])').forEach(t => {
            cardThumbnails.push(slot.removeChild(t));
            resetCardFlipState(t);
        });
        resetSlotToDefault(slot);
        slot.classList.remove('stacked');
    });

    document.getElementById(idPrefix + 'counter-value').value = 30; 
    document.getElementById(idPrefix + 'mana-counter-value').value = 0;

    const deckSlots = document.querySelectorAll(`#${idPrefix}deck-back-slots .card-slot`);
    shuffleArray(cardThumbnails);
    cardThumbnails.forEach((t, i) => deckSlots[i]?.appendChild(t));

    ['deck', 'grave', 'exclude'].forEach(zone => syncMainZoneImage(zone, idPrefix));
    
    // 勝敗表示を消す
    if (typeof closeGameResult === 'function') closeGameResult();
    
    // テーマリセット
    const defaultTheme = idPrefix ? 'simple' : 'simple';
    updateSmTheme(idPrefix, defaultTheme);
    
    const drawerId = idPrefix ? 'opponent-drawer' : 'player-drawer';
    document.getElementById(drawerId)?.classList.remove('open');
}

function deleteDeck(idPrefix) {
    const wrapperSelector = idPrefix ? '.opponent-wrapper' : '.player-wrapper';
    const allSlots = document.querySelectorAll(`${wrapperSelector} .card-slot, #${idPrefix}drawer .card-slot`);
    allSlots.forEach(slot => {
        const baseParentZoneId = getBaseId(getParentZoneId(slot));
        if (['free-space-slots', 'icon-zone', 'side-deck', 'side-deck-back-slots', 'token-zone-slots'].includes(baseParentZoneId)) return;
        slot.querySelectorAll('.thumbnail:not([data-is-decoration="true"])').forEach(t => slot.removeChild(t));
        resetSlotToDefault(slot);
        slot.classList.remove('stacked');
    });
     ['deck', 'grave', 'exclude'].forEach(zone => syncMainZoneImage(zone, idPrefix));
}

function clearAllBoard(idPrefix) {
    const wrapperSelector = idPrefix ? '.opponent-wrapper' : '.player-wrapper';
    const drawerId = idPrefix ? 'opponent-drawer' : 'player-drawer';
    const allSlots = document.querySelectorAll(`${wrapperSelector} .card-slot, #${drawerId} .card-slot`);
    
    allSlots.forEach(slot => {
        slot.querySelectorAll('.thumbnail').forEach(t => slot.removeChild(t));
        resetSlotToDefault(slot);
        slot.classList.remove('stacked');
    });
    ['deck', 'grave', 'exclude', 'side-deck'].forEach(zone => syncMainZoneImage(zone, idPrefix));
}


function updateSmTheme(idPrefix, nextMode = null) {
    const wrapperElement = document.querySelector(idPrefix ? '.opponent-wrapper' : '.player-wrapper');
    const smToggleBtn = document.getElementById(idPrefix + 'sm-toggle-btn');
    const iconZone = document.getElementById(idPrefix + 'icon-zone');
    
    if (!wrapperElement || !smToggleBtn) return;

    const themeOrder = ['magickers', 'masochist', 'striker', 'sadist', 'simple'];
    const themeLabels = {
        'magickers': 'マジッカーズ',
        'masochist': 'マゾヒスト',
        'striker': 'ストライカー',
        'sadist': 'サディスト',
        'simple': 'シンプル'
    };

    let currentMode = smToggleBtn.dataset.mode;
    
    if (!nextMode) {
        nextMode = currentMode || (idPrefix ? 'striker' : 'magickers');
    }
    
    themeOrder.forEach(theme => {
        const className = idPrefix ? `opponent-${theme}-mode` : `player-${theme}-mode`;
        const bodyClassName = idPrefix ? `opponent-${theme}-active` : `player-${theme}-active`;
        wrapperElement.classList.remove(className);
        document.body.classList.remove(bodyClassName);
    });

    wrapperElement.classList.add(idPrefix ? `opponent-${nextMode}-mode` : `player-${nextMode}-mode`);
    document.body.classList.add(idPrefix ? `opponent-${nextMode}-active` : `player-${nextMode}-active`);

    smToggleBtn.textContent = themeLabels[nextMode];
    smToggleBtn.dataset.mode = nextMode;

    const counterWrapperId = idPrefix ? idPrefix + 'counter-wrapper' : 'player-counter-wrapper';
    const counterWrapper = document.getElementById(counterWrapperId);
    if(counterWrapper) {
        themeOrder.forEach(theme => {
            counterWrapper.classList.remove(`${theme}-ui-active`);
        });
        counterWrapper.classList.add(`${nextMode}-ui-active`);
    }
    
    updateBodyThemeClasses();
}

function updateBodyThemeClasses() {
    const themes = ['magickers', 'masochist', 'striker', 'sadist', 'simple'];
    const playerMode = document.getElementById('sm-toggle-btn')?.dataset.mode || 'magickers';
    
    themes.forEach(theme => {
        document.body.classList.remove(`${theme}-mode-active`);
    });
    document.body.classList.remove('sm-mode-active');

    document.body.classList.add(`${playerMode}-mode-active`);
    
    if (playerMode === 'sadist' || playerMode === 'masochist') {
        document.body.classList.add('sm-mode-active');
    }
}

function exportDeck(idPrefix, fileName = 'deck') {
     try {
        const exportData = {
            deck: extractZoneData(idPrefix + 'deck-back-slots'),
            sideDeck: extractZoneData(idPrefix + 'side-deck-back-slots'),
            freeSpace: extractZoneData(idPrefix + 'free-space-slots'),
            token: extractZoneData(idPrefix + 'token-zone-slots'),
            decorations: {
                deck: extractZoneData(idPrefix + 'deck', true),
                sideDeck: extractZoneData(idPrefix + 'side-deck', true),
                grave: extractZoneData(idPrefix + 'grave', true),
                exclude: extractZoneData(idPrefix + 'exclude', true),
                icon: extractZoneData(idPrefix + 'icon-zone', true),
            },
        };
        const jsonData = JSON.stringify(exportData, null, 2);
        const blob = new Blob([jsonData], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${fileName}.json`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error("Export failed:", error);
    }
}

function importDeck(idPrefix) {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json, application/json';
    fileInput.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const importData = JSON.parse(event.target.result);
                
                clearZoneData(idPrefix + 'deck-back-slots');
                clearZoneData(idPrefix + 'side-deck-back-slots');
                clearZoneData(idPrefix + 'free-space-slots');
                clearZoneData(idPrefix + 'token-zone-slots');
                Object.keys(importData.decorations || {}).forEach(zone => clearZoneData(idPrefix + zone, true));
                
                applyDataToZone(idPrefix + 'deck-back-slots', importData.deck);
                applyDataToZone(idPrefix + 'side-deck-back-slots', importData.sideDeck);
                applyDataToZone(idPrefix + 'free-space-slots', importData.freeSpace);
                applyDataToZone(idPrefix + 'token-zone-slots', importData.token);
                Object.keys(importData.decorations || {}).forEach(zone => {
                    if (importData.decorations[zone]) applyDataToZone(idPrefix + zone, [importData.decorations[zone]]);
                });
                
                ['deck', 'side-deck', 'grave', 'exclude'].forEach(zone => syncMainZoneImage(zone, idPrefix));

            } catch (error) {
                console.error("Import failed:", error);
            }
        };
        reader.readAsText(file);
    };
    fileInput.click();
}

function exportAllBoardData(fileName = 'sm_solitaire_board') {
    try {
        const state = getAllBoardState();
        const jsonData = JSON.stringify(state, null, 2);
        const blob = new Blob([jsonData], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${fileName}.json`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error("Full Board Export failed:", error);
    }
}

function importAllBoardData() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json, application/json';
    fileInput.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const importData = JSON.parse(event.target.result);
                restoreAllBoardState(importData);
            } catch (error) {
                console.error("Full Board Import failed:", error);
                alert("盤面データの読み込みに失敗しました。");
            }
        };
        reader.readAsText(file);
    };
    fileInput.click();
}

function getAllBoardState() {
    const getSideState = (idPrefix) => {
        const zones = [
            'deck', 'grave', 'exclude', 'side-deck', 'icon-zone',
            'deck-back-slots', 'grave-back-slots', 'exclude-back-slots', 'side-deck-back-slots',
            'free-space-slots', 'token-zone-slots', 'hand-zone',
            'mana-left', 'mana-right', 'battle', 'spell', 'special1', 'special2'
        ];
        
        const state = {};
        zones.forEach(zoneId => {
            const fullId = idPrefix + zoneId;
            state[zoneId] = extractZoneData(fullId);
        });

        const smBtn = document.getElementById(idPrefix + 'sm-toggle-btn');
        const lpVal = document.getElementById(idPrefix + 'counter-value')?.value;
        const manaVal = document.getElementById(idPrefix + 'mana-counter-value')?.value;
        const nameVal = document.getElementById(idPrefix + (idPrefix ? 'player-name' : 'player-name'))?.value;
        
        const intervalId = idPrefix ? 'opponent-auto-decrease-interval' : 'player-auto-decrease-interval';
        const intervalVal = document.getElementById(intervalId)?.value;

        state.meta = {
            smMode: smBtn ? smBtn.dataset.mode : (idPrefix ? 'striker' : 'magickers'),
            lp: lpVal,
            mana: manaVal,
            name: nameVal,
            autoDecreaseInterval: intervalVal
        };

        return state;
    };

    return {
        player: getSideState(''),
        opponent: getSideState('opponent-'),
        common: {
            turnValue: document.getElementById('common-turn-value')?.value || 1,
            turnPlayer: document.getElementById('turn-player-select')?.value || 'first',
            isBoardFlipped: document.body.classList.contains('board-flipped'),
            currentStepIndex: (typeof currentStepIndex !== 'undefined') ? currentStepIndex : 0,
            cNavi: extractZoneData('c-free-space'),
            customCounterTypes: customCounterTypes || [], // カスタムカウンターの定義(画像含む)も保存
            settings: {
                bgmVolume: typeof bgmVolume !== 'undefined' ? bgmVolume : 5,
                seVolume: typeof seVolume !== 'undefined' ? seVolume : 5,
                seConfig: typeof seConfig !== 'undefined' ? seConfig : {},
                effectConfig: typeof effectConfig !== 'undefined' ? effectConfig : {},
                autoConfig: typeof autoConfig !== 'undefined' ? autoConfig : {},
                replayWaitTime: document.getElementById('replay-wait-time-input')?.value
            }
        },
        timestamp: Date.now()
    };
}

function restoreAllBoardState(state) {
    if (!state || !state.player || !state.opponent) return;

    clearAllBoard('');
    clearAllBoard('opponent-');
    clearZoneData('c-free-space');

    const restoreSide = (idPrefix, sideState) => {
        Object.keys(sideState).forEach(zoneId => {
            if (zoneId === 'meta') return;
            const fullId = idPrefix + zoneId;
            const zoneData = sideState[zoneId];
            if (zoneData && Array.isArray(zoneData)) {
                applyDataToZone(fullId, zoneData);
            }
        });

        if (sideState.meta) {
            const smBtn = document.getElementById(idPrefix + 'sm-toggle-btn');
            if (smBtn) {
                updateSmTheme(idPrefix, sideState.meta.smMode || (idPrefix ? 'striker' : 'magickers'));
            }
            
            const lpInput = document.getElementById(idPrefix + 'counter-value');
            if (lpInput) lpInput.value = sideState.meta.lp || 30;
            
            const manaInput = document.getElementById(idPrefix + 'mana-counter-value');
            if (manaInput) manaInput.value = sideState.meta.mana || 0;

            const nameInput = document.getElementById(idPrefix + (idPrefix ? 'player-name' : 'player-name'));
            if (nameInput) nameInput.value = sideState.meta.name || (idPrefix ? 'Opponent' : 'Player');
            
            const intervalId = idPrefix ? 'opponent-auto-decrease-interval' : 'player-auto-decrease-interval';
            const intervalInput = document.getElementById(intervalId);
            if(intervalInput && sideState.meta.autoDecreaseInterval) {
                intervalInput.value = sideState.meta.autoDecreaseInterval;
            }
        }
        
        ['deck', 'side-deck', 'grave', 'exclude'].forEach(zone => syncMainZoneImage(zone, idPrefix));
    };

    restoreSide('', state.player);
    restoreSide('opponent-', state.opponent);

    if (state.common) {
        const turnInput = document.getElementById('common-turn-value');
        if(turnInput) turnInput.value = state.common.turnValue || 1;
        
        const turnSelect = document.getElementById('turn-player-select');
        if(turnSelect) turnSelect.value = state.common.turnPlayer || 'first';
        
        if (state.common.isBoardFlipped) {
            document.body.classList.add('board-flipped');
        } else {
            document.body.classList.remove('board-flipped');
        }

        if (typeof state.common.currentStepIndex !== 'undefined') {
            currentStepIndex = state.common.currentStepIndex;
            if (typeof updateStepUI === 'function') updateStepUI();
        }
        
        if (state.common.cNavi) {
            applyDataToZone('c-free-space', state.common.cNavi);
        }
        
        // カスタムカウンター定義の復元
        if (state.common.customCounterTypes) {
            customCounterTypes = state.common.customCounterTypes;
        }
        
        if (state.common.settings) {
            const s = state.common.settings;
            if(typeof s.bgmVolume !== 'undefined') bgmVolume = s.bgmVolume;
            if(typeof s.seVolume !== 'undefined') seVolume = s.seVolume;
            if(s.seConfig) Object.assign(seConfig, s.seConfig);
            if(s.effectConfig) Object.assign(effectConfig, s.effectConfig);
            if(s.autoConfig) Object.assign(autoConfig, s.autoConfig);
            if(s.replayWaitTime) {
                const w = document.getElementById('replay-wait-time-input');
                if(w) w.value = s.replayWaitTime;
            }
            
            if(typeof window.updateSettingsUIFromState === 'function') {
                window.updateSettingsUIFromState();
            }
            if(typeof updateBgmVolume === 'function') updateBgmVolume();
        }
    }
}

function extractZoneData(containerId, singleSlot = false) {
    const container = document.getElementById(containerId);
    if (!container) return null;
    
    let slots = [];
    if (container.classList.contains('card-slot')) {
        slots = [container];
    } else {
        slots = Array.from(container.querySelectorAll('.card-slot'));
    }
    
    const data = slots.map(slot => {
        const thumbnails = slot.querySelectorAll('.thumbnail');
        if (thumbnails.length === 0) return null;
        return Array.from(thumbnails).map(thumb => ({
            src: thumb.querySelector('.card-image').src,
            isDecoration: thumb.dataset.isDecoration === 'true',
            isFlipped: thumb.dataset.isFlipped === 'true',
            originalSrc: thumb.dataset.originalSrc,
            counter: parseInt(thumb.querySelector('.card-counter-overlay').dataset.counter) || 0,
            memo: thumb.dataset.memo || '',
            flavor1: thumb.dataset.flavor1 || '',
            flavor2: thumb.dataset.flavor2 || '',
            rotation: parseInt(thumb.querySelector('.card-image').dataset.rotation) || 0,
            isMasturbating: thumb.dataset.isMasturbating === 'true',
            isBlocker: thumb.dataset.isBlocker === 'true',
            isPermanent: thumb.dataset.isPermanent === 'true', 
            ownerPrefix: thumb.dataset.ownerPrefix || '',
            customCounters: JSON.parse(thumb.dataset.customCounters || '[]')
        }));
    });
    
    if (singleSlot) {
        const validData = data.filter(d => d);
        return validData.length > 0 ? validData[0][0] : null; 
    }
    
    return data;
}

function clearZoneData(containerId, clearDecorations = false) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const slots = container.classList.contains('card-slot') ? [container] : container.querySelectorAll('.card-slot');
    
    slots.forEach(slot => {
        slot.querySelectorAll('.thumbnail').forEach(thumb => {
            if (clearDecorations || thumb.dataset.isDecoration !== 'true') {
                slot.removeChild(thumb);
            }
        });
        resetSlotToDefault(slot);
        updateSlotStackState(slot);
    });
}

function applyDataToZone(containerId, zoneData) {
    const container = document.getElementById(containerId);
    if (!container || !zoneData) return;
    
    const slots = container.classList.contains('card-slot') ? [container] : container.querySelectorAll('.card-slot');
    
    const dataArray = Array.isArray(zoneData) ? zoneData : [zoneData];

    dataArray.forEach((cardsInSlot, i) => {
        if (slots[i] && cardsInSlot) {
            const cards = Array.isArray(cardsInSlot) ? cardsInSlot : [cardsInSlot];
            
            cards.forEach(cardData => {
                const thumb = createCardThumbnail(cardData, slots[i], cardData.isDecoration, false, cardData.ownerPrefix);
                
                if (cardData.rotation) {
                    const img = thumb.querySelector('.card-image');
                    if (img) {
                        img.dataset.rotation = cardData.rotation;
                        if (Math.abs(cardData.rotation) % 180 !== 0) {
                            slots[i].classList.add('rotated-90');
                             const { width, height } = getCardDimensions();
                             const scaleFactor = height / width;
                             img.style.transform = `rotate(${cardData.rotation}deg) scale(${scaleFactor})`;
                        } else {
                            slots[i].classList.remove('rotated-90');
                            img.style.transform = `rotate(${cardData.rotation}deg)`;
                        }
                    }
                }
            });
            
            updateSlotStackState(slots[i]);
        }
    });
    
    if(containerId.endsWith('-back-slots') || containerId.includes('free-space') || containerId.includes('token-zone')) {
        arrangeSlots(containerId);
    }
}

window.executeBattle = function(attackerThumbnail, targetSlot) {
    if (typeof openBattleConfirmModal === 'function') {
        openBattleConfirmModal(attackerThumbnail, targetSlot);
    }
};

window.resolveBattle = function(attackerBP, targetBP) {
    if (!currentAttacker || !currentBattleTarget) {
        if(typeof closeBattleConfirmModal === 'function') closeBattleConfirmModal();
        return;
    }

    const isInvalidZone = (element) => {
        if (!element) return true;
        const zoneId = getParentZoneId(element.classList.contains('thumbnail') ? element.parentNode : element);
        const baseId = getBaseId(zoneId);
        return ['hand-zone', 'side-deck', 'side-deck-back-slots', 'grave', 'grave-back-slots', 'exclude', 'exclude-back-slots', 'deck', 'deck-back-slots', 'token-zone-slots', 'c-free-space'].includes(baseId);
    };

    if (isInvalidZone(currentAttacker) || isInvalidZone(currentBattleTarget)) {
        if(typeof closeBattleConfirmModal === 'function') closeBattleConfirmModal();
        return;
    }

    let targetThumbnail = null;
    const isPlayerIcon = currentBattleTarget.id === 'icon-zone' || currentBattleTarget.id === 'opponent-icon-zone';
    if (!isPlayerIcon) {
        targetThumbnail = currentBattleTarget.querySelector('.thumbnail');
        if (!targetThumbnail) {
             if(typeof closeBattleConfirmModal === 'function') closeBattleConfirmModal();
             return;
        }
    }

    const playHitEffect = (element) => {
        // playSe('被弾.mp3'); // 削除
        element.classList.add('target-active');
        setTimeout(() => element.classList.remove('target-active'), 1000);
    };

    const playDestructEffect = (element, callback) => {
        
        element.classList.add('attack-active');
        setTimeout(() => {
            element.classList.remove('attack-active');
            if (callback) callback();
        }, 800);
    };

    playSe('アタック.mp3');
    currentAttacker.classList.add('attack-active');
    setTimeout(() => {
        currentAttacker.classList.remove('attack-active');
    }, 1000);

    if (isPlayerIcon) {
        playHitEffect(currentBattleTarget);
    } else {
        triggerEffect(targetThumbnail, 'target');
    }

    let attackerDestructed = false;

    if (isPlayerIcon) {
        if (typeof autoConfig !== 'undefined' && autoConfig.autoBattleCalc && !isNaN(attackerBP)) {
            const damage = Math.ceil(attackerBP / 1000);
            const targetIdPrefix = currentBattleTarget.id.includes('opponent') ? 'opponent-' : '';
            const lpInput = document.getElementById(targetIdPrefix + 'counter-value');
            
            if (lpInput) {
                const currentLP = parseInt(lpInput.value) || 0;
                const newLP = Math.max(0, currentLP - damage);
                lpInput.value = newLP;
                
                if (isRecording && typeof recordAction === 'function') {
                    recordAction({
                        type: 'counterChange',
                        inputId: targetIdPrefix + 'counter-value',
                        change: -damage
                    });
                }
                
                if (newLP === 0 && autoConfig.autoGameEnd) {
                    const msg = targetIdPrefix ? 'YOU WIN!' : 'YOU LOSE...';
                    setTimeout(() => window.showGameResult(msg), 1000);
                }
            }
        }
    } else {
        if (typeof autoConfig !== 'undefined' && autoConfig.autoBattleCalc && !isNaN(attackerBP) && !isNaN(targetBP)) {
            
            if (attackerBP <= 0 && targetBP <= 0) {
            } 
            else if (attackerBP > targetBP) {
                playDestructEffect(targetThumbnail, () => {
                    if (typeof moveCardToMultiZone === 'function') moveCardToMultiZone(targetThumbnail, 'grave');
                });
            } else if (attackerBP < targetBP) {
                attackerDestructed = true;
                playDestructEffect(currentAttacker, () => {
                    if (typeof moveCardToMultiZone === 'function') moveCardToMultiZone(currentAttacker, 'grave');
                });
            } else {
                attackerDestructed = true;
                playDestructEffect(targetThumbnail, () => {
                    if (typeof moveCardToMultiZone === 'function') moveCardToMultiZone(targetThumbnail, 'grave');
                });
                playDestructEffect(currentAttacker, () => {
                    if (typeof moveCardToMultiZone === 'function') moveCardToMultiZone(currentAttacker, 'grave');
                });
            }
        } else {
            playHitEffect(targetThumbnail);
        }
    }

    if (typeof autoConfig !== 'undefined' && autoConfig.autoAttackTap) {
        if (!attackerDestructed) {
            setTimeout(() => {
                if (document.body.contains(currentAttacker) && !isInvalidZone(currentAttacker)) {
                    const imgElement = currentAttacker.querySelector('.card-image');
                    const slotElement = currentAttacker.parentNode;
                    if (imgElement && slotElement) {
                        const currentRotation = parseInt(imgElement.dataset.rotation) || 0;
                        if (currentRotation === 0) {
                            const newRotation = 90;
                            slotElement.classList.add('rotated-90');
                            const { width, height } = getCardDimensions();
                            const scaleFactor = height / width;
                            imgElement.style.transform = `rotate(${newRotation}deg) scale(${scaleFactor})`;
                            imgElement.dataset.rotation = newRotation;
                            playSe('タップ.mp3');
                            
                            if (isRecording && typeof recordAction === 'function') {
                                recordAction({
                                    type: 'rotate',
                                    zoneId: getParentZoneId(slotElement),
                                    slotIndex: Array.from(slotElement.parentNode.children).indexOf(slotElement),
                                    rotation: newRotation
                                });
                            }
                        }
                    }
                }
            }, 900); 
        }
    }

    if(typeof closeBattleConfirmModal === 'function') closeBattleConfirmModal();
};