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
    
    setupBoardUI(idPrefix);
    setupBoardButtons(idPrefix);
    setupCounters(idPrefix);

    const addDecorationWithErrorHandler = (path, zoneId) => {
        const container = document.getElementById(idPrefix + zoneId);
        const slot = container?.querySelector('.card-slot');
        if (slot) {
            const thumbnail = createCardThumbnail(path, slot, true, false, idPrefix);
            const img = thumbnail.querySelector('img');
            if (img) {
                img.onerror = () => {
                    thumbnail.remove();
                    syncMainZoneImage(zoneId, idPrefix);
                };
            }
        }
    };

    addDecorationWithErrorHandler('./decoration/デッキ.png', 'deck');
    addDecorationWithErrorHandler('./decoration/EXデッキ.png', 'side-deck');
    addDecorationWithErrorHandler('./decoration/墓地エリア.png', 'grave');
    addDecorationWithErrorHandler('./decoration/除外エリア.png', 'exclude');

    ['deck', 'grave', 'exclude', 'side-deck'].forEach(zone => syncMainZoneImage(zone, idPrefix));

    updateSmTheme(idPrefix);
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
                if (isDecorationMode) {
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
    document.getElementById(idPrefix + 'sm-toggle-btn')?.addEventListener('click', (e) => {
        playSe('ボタン共通.mp3');
        const btn = e.currentTarget;
        btn.dataset.mode = (btn.dataset.mode === 'sadist') ? 'masochist' : 'sadist';
        updateSmTheme(idPrefix);
    });
    document.getElementById(idPrefix + 'surrender-btn')?.addEventListener('click', () => {
        playSe('降参.mp3');
    });
    document.getElementById(idPrefix + 'export-deck-btn')?.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        exportDeck(idPrefix);
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

    const setupAutoDecrease = (btnId, timer, counter, interval, seName) => {
        const btn = document.getElementById(btnId);
        if(!btn) return null;
        
        if (!btn.dataset.originalText) {
            btn.dataset.originalText = btn.textContent;
        }
        
        btn.addEventListener('click', () => {
            if (timer) {
                clearInterval(timer);
                timer = null;
                btn.textContent = btn.dataset.originalText;
                btn.style.backgroundColor = '';
                btn.style.boxShadow = '';
                stopSe(seName);
                playSe('ボタン共通.mp3');
            } else {
                if (btn.textContent !== '停止') {
                     btn.dataset.originalText = btn.textContent;
                }
                btn.textContent = '停止';
                btn.style.backgroundColor = '#cc0000';
                btn.style.boxShadow = '0 2px #800000';
                playSe(seName, true);
                timer = setInterval(() => {
                    counter.value = Math.max(0, (parseInt(counter.value) || 0) - 1);
                }, interval);
            }
        });
        return timer; // Note: timer reference needs to be managed by caller or object in real app, but here we return new ID
    };

    // タイマー変数の参照管理のため、少し実装を変更（クロージャ内の変数を更新できないため）
    // ここでは簡易的にイベントリスナー内でグローバル変数を操作する形はとれないため、
    // setupAutoDecreaseを少し書き換えて、クリックイベントを直接定義します。

    const attachAutoDecreaseLogic = (btnId, counter, interval) => {
        const btn = document.getElementById(btnId);
        if (!btn) return;
        
        let timerId = null;
        if (!btn.dataset.originalText) btn.dataset.originalText = btn.textContent;

        btn.addEventListener('click', () => {
            if (timerId) {
                clearInterval(timerId);
                timerId = null;
                btn.textContent = btn.dataset.originalText;
                btn.style.backgroundColor = '';
                btn.style.boxShadow = '';
                stopSe('自動減少.mp3');
                playSe('ボタン共通.mp3');
            } else {
                if (btn.textContent !== '停止') btn.dataset.originalText = btn.textContent;
                btn.textContent = '停止';
                btn.style.backgroundColor = '#cc0000';
                btn.style.boxShadow = '0 2px #800000';
                playSe('自動減少.mp3', true);
                timerId = setInterval(() => {
                    counter.value = Math.max(0, (parseInt(counter.value) || 0) - 1);
                }, interval);
            }
        });
    };

    attachAutoDecreaseLogic(idPrefix + 'lp-auto-decrease-btn', lpCounter, 1000);
    attachAutoDecreaseLogic(idPrefix + 'mana-auto-decrease-btn', manaCounter, 1000);

    const counterWrapperId = idPrefix ? idPrefix + 'counter-wrapper' : 'player-counter-wrapper';
    const counterWrapper = document.getElementById(counterWrapperId);
    
    counterWrapper?.querySelectorAll('.counter-btn[data-value]').forEach(button => {
        const value = parseInt(button.dataset.value);
        const targetCounter = button.closest('.hand-counter-group').querySelector('input');
        
        let repeatTimer = null;
        let initialTimer = null;
        const startAction = (e) => {
            if (e.button !== undefined && e.button !== 0) return;
            playSe('ボタン共通.mp3');
            if(!targetCounter) return;
            targetCounter.value = Math.max(0, (parseInt(targetCounter.value) || 0) + value);
            initialTimer = setTimeout(() => {
                repeatTimer = setInterval(() => {
                    if(!targetCounter) return;
                    targetCounter.value = Math.max(0, (parseInt(targetCounter.value) || 0) + value);
                }, 200);
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
    if (!cardToDraw) return false;

    const handSlots = document.querySelectorAll(`#${idPrefix}hand-zone .card-slot`);
    const emptyHandSlot = Array.from(handSlots).find(s => !s.querySelector('.thumbnail'));
    if (!emptyHandSlot) return false;

    const sourceSlot = cardToDraw.parentNode;
    sourceSlot.removeChild(cardToDraw);
    emptyHandSlot.appendChild(cardToDraw);

    resetCardFlipState(cardToDraw);
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

function resetBoard(idPrefix) {
    const wrapperSelector = idPrefix ? '.opponent-wrapper' : '.player-wrapper';
    const allSlots = document.querySelectorAll(`${wrapperSelector} .card-slot, #${idPrefix}drawer .card-slot`);
    let cardThumbnails = [];

    allSlots.forEach(slot => {
        const baseParentZoneId = getBaseId(getParentZoneId(slot));
        if (['free-space-slots', 'icon-zone', 'side-deck', 'side-deck-back-slots'].includes(baseParentZoneId)) return;
        slot.querySelectorAll('.thumbnail:not([data-is-decoration="true"])').forEach(t => {
            cardThumbnails.push(slot.removeChild(t));
            resetCardFlipState(t);
        });
        resetSlotToDefault(slot);
        slot.classList.remove('stacked');
    });

    document.getElementById(idPrefix + 'counter-value').value = 20;
    document.getElementById(idPrefix + 'mana-counter-value').value = 0;

    const deckSlots = document.querySelectorAll(`#${idPrefix}deck-back-slots .card-slot`);
    shuffleArray(cardThumbnails);
    cardThumbnails.forEach((t, i) => deckSlots[i]?.appendChild(t));

    ['deck', 'grave', 'exclude'].forEach(zone => syncMainZoneImage(zone, idPrefix));
    updateSmTheme(idPrefix, true);
    
    const drawerId = idPrefix ? 'opponent-drawer' : 'player-drawer';
    document.getElementById(drawerId)?.classList.remove('open');
}

function deleteDeck(idPrefix) {
    const wrapperSelector = idPrefix ? '.opponent-wrapper' : '.player-wrapper';
    const allSlots = document.querySelectorAll(`${wrapperSelector} .card-slot, #${idPrefix}drawer .card-slot`);
    allSlots.forEach(slot => {
        const baseParentZoneId = getBaseId(getParentZoneId(slot));
        if (['free-space-slots', 'icon-zone', 'side-deck', 'side-deck-back-slots'].includes(baseParentZoneId)) return;
        slot.querySelectorAll('.thumbnail:not([data-is-decoration="true"])').forEach(t => slot.removeChild(t));
        resetSlotToDefault(slot);
        slot.classList.remove('stacked');
    });
     ['deck', 'grave', 'exclude'].forEach(zone => syncMainZoneImage(zone, idPrefix));
}


function updateSmTheme(idPrefix, forceDefault = false) {
    const wrapperElement = document.querySelector(idPrefix ? '.opponent-wrapper' : '.player-wrapper');
    const smToggleBtn = document.getElementById(idPrefix + 'sm-toggle-btn');
    if (!wrapperElement || !smToggleBtn) return;

    let mode = smToggleBtn.dataset.mode;
    if (forceDefault) {
        mode = idPrefix ? 'sadist' : 'masochist';
        smToggleBtn.dataset.mode = mode;
    }

    const isSadist = mode === 'sadist';
    const sadistClass = idPrefix ? 'opponent-sadist-mode' : 'player-sadist-mode';
    const bodySadistClass = idPrefix ? 'opponent-sadist-active' : 'player-sadist-active';

    smToggleBtn.textContent = isSadist ? 'サディスト' : 'マゾヒスト';
    wrapperElement.classList.toggle(sadistClass, isSadist);
    document.body.classList.toggle(bodySadistClass, isSadist);
    
    const counterWrapperId = idPrefix ? idPrefix + 'counter-wrapper' : 'player-counter-wrapper';
    const counterWrapper = document.getElementById(counterWrapperId);
    if(counterWrapper) {
        counterWrapper.classList.toggle('sadist-ui-active', isSadist);
        counterWrapper.classList.toggle('masochist-ui-active', !isSadist);
    }
    
    const iconSlot = document.getElementById(idPrefix + 'icon-zone');
    if (iconSlot) {
        const existingIcon = iconSlot.querySelector('.thumbnail');
        if (existingIcon) iconSlot.removeChild(existingIcon);
        
        const thumbnail = createCardThumbnail(isSadist ? './decoration/サディスト.png' : './decoration/マゾヒスト.png', iconSlot, true, false, idPrefix);
        const img = thumbnail.querySelector('img');
        if (img) {
            img.onerror = () => thumbnail.remove();
        }
    }

    const isAnySadist = document.querySelector('.player-sadist-mode, .opponent-sadist-mode');
    document.body.classList.toggle('sm-mode-active', !!isAnySadist);
}

function exportDeck(idPrefix) {
     try {
        const exportData = {
            deck: extractZoneData(idPrefix + 'deck-back-slots'),
            sideDeck: extractZoneData(idPrefix + 'side-deck-back-slots'),
            freeSpace: extractZoneData(idPrefix + 'free-space-slots'),
            decorations: {
                deck: extractZoneData(idPrefix + 'deck', true),
                sideDeck: extractZoneData(idPrefix + 'side-deck', true),
                grave: extractZoneData(idPrefix + 'grave', true),
                exclude: extractZoneData(idPrefix + 'exclude', true),
                icon: extractZoneData(idPrefix + 'icon-zone', true),
            }
        };
        const jsonData = JSON.stringify(exportData, null, 2);
        const blob = new Blob([jsonData], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${idPrefix || 'player'}_deck_export.json`;
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
                Object.keys(importData.decorations || {}).forEach(zone => clearZoneData(idPrefix + zone, true));
                
                applyDataToZone(idPrefix + 'deck-back-slots', importData.deck);
                applyDataToZone(idPrefix + 'side-deck-back-slots', importData.sideDeck);
                applyDataToZone(idPrefix + 'free-space-slots', importData.freeSpace);
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


function extractZoneData(containerId, singleSlot = false) {
    const container = document.getElementById(containerId);
    if (!container) return null;
    const slots = container.querySelectorAll('.card-slot');
    const data = Array.from(slots).map(slot => {
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
        }));
    });
    return singleSlot ? (data[0] ? data[0] : null) : data.filter(d => d);
}

function clearZoneData(containerId, clearDecorations = false) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.querySelectorAll('.card-slot').forEach(slot => {
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
    const slots = container.querySelectorAll('.card-slot');
    zoneData.forEach((cardsInSlot, i) => {
        if (slots[i] && cardsInSlot) {
            cardsInSlot.forEach(cardData => createCardThumbnail(cardData, slots[i], cardData.isDecoration, false, cardData.ownerPrefix));
        }
    });
    if(containerId.endsWith('-back-slots') || containerId.includes('free-space')) {
        arrangeSlots(containerId);
    }
}