// 汎用ユーティリティ関数

function getBaseId(prefixedId) {
    if (!prefixedId) return null;
    return prefixedId.replace('opponent-', '');
}

function getPrefixFromZoneId(zoneId) {
    return zoneId && zoneId.startsWith('opponent-') ? 'opponent-' : '';
}

function getCardDimensions() {
    const rootStyles = getComputedStyle(document.documentElement);
    const width = parseFloat(rootStyles.getPropertyValue('--card-width').replace('px', '')) || 70;
    const height = parseFloat(rootStyles.getPropertyValue('--card-height').replace('px', '')) || 124.7;
    return { width, height };
}

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

function resetSlotToDefault(slotElement) {
    if (!slotElement) return;
    slotElement.classList.remove('rotated-90');
    const imgElement = slotElement.querySelector('.thumbnail img');
    if (imgElement) {
        imgElement.style.transform = `rotate(0deg)`;
        imgElement.dataset.rotation = 0;
    }
}

function getExistingThumbnail(slotElement) {
    const thumbnails = slotElement.querySelectorAll('.thumbnail');
    return thumbnails.length > 0 ? thumbnails[thumbnails.length - 1] : null;
}

function getParentZoneId(element) {
    if (!element) return null;
    if (element.id === 'icon-zone' || element.id === 'opponent-icon-zone') {
        return element.id;
    }
    const closestZone = element.closest('.zone, .hand-zone-slots, .drawer-panel, .drawer-free-space, .player-icon-slot');
    if (closestZone && closestZone.classList.contains('player-icon-slot')) {
        const iconSlot = closestZone.querySelector('.card-slot');
        if (iconSlot) return iconSlot.id;
    }
    return closestZone ? closestZone.id : null;
}

// SE再生機能
const loopSeInstances = {};
let isSeMuted = false;

function toggleSeMute() {
    isSeMuted = !isSeMuted;
    if (isSeMuted) {
        // ミュートにしたタイミングでループ再生中のSEがあれば停止する
        Object.keys(loopSeInstances).forEach(key => stopSe(key));
    }
    return isSeMuted;
}

function playSe(filename, isLoop = false) {
    if (isSeMuted) return;

    const path = `./se/${filename}`;
    
    if (isLoop) {
        // 既に再生中なら重複して再生しない
        if (loopSeInstances[filename]) return;
        
        const audio = new Audio(path);
        audio.loop = true;
        audio.play().catch(e => console.error('SE再生エラー:', e));
        loopSeInstances[filename] = audio;
    } else {
        const audio = new Audio(path);
        audio.currentTime = 0;
        audio.play().catch(e => console.error('SE再生エラー:', e));
    }
}

function stopSe(filename) {
    const audio = loopSeInstances[filename];
    if (audio) {
        audio.pause();
        audio.currentTime = 0;
        delete loopSeInstances[filename];
    }
}

// --- リプレイ機能 ---

let replayInitialState = null;

function startReplayRecording() {
    if (isRecording) return;
    isRecording = true;
    actionLog = [];
    replayStartTime = Date.now();
    replayInitialState = getAllBoardState(); 
    
    alert("記録を開始しました。");
    const btn = document.getElementById('record-start-btn');
    if(btn) btn.style.backgroundColor = '#ffcc00';
}

function stopReplayRecording() {
    if (!isRecording) return;
    isRecording = false;
    alert("記録を終了しました。");
    const btn = document.getElementById('record-start-btn');
    if(btn) btn.style.backgroundColor = '';
}

function recordAction(actionData) {
    if (!isRecording) return;
    const timestamp = Date.now() - replayStartTime;
    actionLog.push({
        time: timestamp,
        data: actionData
    });
}

function exportReplayData() {
    if (!replayInitialState || actionLog.length === 0) {
        alert("保存するリプレイデータがありません。");
        return;
    }
    const replayData = {
        initialState: replayInitialState,
        log: actionLog
    };
    const jsonData = JSON.stringify(replayData, null, 2);
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const dateStr = new Date().toISOString().replace(/[:.]/g, '-');
    a.download = `sm_solitaire_replay_${dateStr}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

function importReplayData() {
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
                if (importData.initialState && importData.log) {
                    replayInitialState = importData.initialState;
                    actionLog = importData.log;
                    alert("リプレイデータを読み込みました。「再生」ボタンで開始します。");
                } else {
                    alert("無効なリプレイデータ形式です。");
                }
            } catch (error) {
                console.error("Replay Import failed:", error);
                alert("リプレイデータの読み込みに失敗しました。");
            }
        };
        reader.readAsText(file);
    };
    fileInput.click();
}

function playReplay() {
    if (!replayInitialState || actionLog.length === 0) {
        alert("再生するリプレイデータがありません。");
        return;
    }
    
    if (isPlaying) {
        // 既に再生中の場合は停止処理（全タイマークリア）
        replayTimerIds.forEach(id => clearTimeout(id));
        replayTimerIds = [];
        isPlaying = false;
        alert("再生を停止しました。");
        return;
    }

    if (isRecording) {
        stopReplayRecording();
    }

    // 初期状態に復元
    restoreAllBoardState(replayInitialState);
    isPlaying = true;
    replayTimerIds = [];
    
    alert("リプレイ再生を開始します。");

    actionLog.forEach(entry => {
        const timerId = setTimeout(() => {
            executeAction(entry.data);
        }, entry.time);
        replayTimerIds.push(timerId);
    });

    // 終了検知用のタイマー
    const lastTime = actionLog.length > 0 ? actionLog[actionLog.length - 1].time : 0;
    const endTimerId = setTimeout(() => {
        isPlaying = false;
        replayTimerIds = [];
        alert("リプレイ再生が終了しました。");
    }, lastTime + 500);
    replayTimerIds.push(endTimerId);
}

// アクション実行ロジック
function executeAction(action) {
    // アクション実行中は効果音を鳴らすため、playSeは各処理内で適宜呼ぶか、
    // ここでswitch文で分岐して処理する
    
    switch (action.type) {
        case 'move': {
            const fromSlot = getSlotByIndex(action.fromZone, action.fromSlotIndex);
            const toSlot = getSlotByIndex(action.toZone, action.toSlotIndex);
            
            if (fromSlot && toSlot) {
                const card = fromSlot.querySelector('.thumbnail');
                const targetCard = toSlot.querySelector('.thumbnail');
                
                if (card) {
                    if (targetCard) {
                        // Swap
                        fromSlot.appendChild(targetCard);
                        toSlot.appendChild(card);
                    } else {
                        // Move
                        toSlot.appendChild(card);
                    }
                    
                    // ゾーンごとの後処理
                    [fromSlot, toSlot].forEach(slot => {
                        resetSlotToDefault(slot);
                        updateSlotStackState(slot);
                        const zId = getParentZoneId(slot);
                        if (zId && zId.endsWith('-back-slots')) arrangeSlots(zId);
                        const baseId = getBaseId(zId);
                        if (decorationZones.includes(baseId)) syncMainZoneImage(baseId, getPrefixFromZoneId(zId));
                    });
                    
                    // SE判定
                    const toBase = getBaseId(getParentZoneId(toSlot));
                    if (toBase === 'grave' || toBase === 'grave-back-slots') playSe('墓地に送る.mp3');
                    else if (toBase === 'exclude' || toBase === 'exclude-back-slots') playSe('除外する.mp3');
                    else playSe('カードを配置する.mp3');
                }
            }
            break;
        }
        case 'newCard': {
            const slot = getSlotByIndex(action.zoneId, action.slotIndex);
            if (slot) {
                const prefix = getPrefixFromZoneId(action.zoneId);
                createCardThumbnail(action.cardData, slot, false, false, prefix);
                updateSlotStackState(slot);
                playSe('カードを配置する.mp3');
                
                const zId = getParentZoneId(slot);
                if (zId && zId.endsWith('-back-slots')) arrangeSlots(zId);
                const baseId = getBaseId(zId);
                if (decorationZones.includes(baseId)) syncMainZoneImage(baseId, prefix);
            }
            break;
        }
        case 'updateDecoration': {
            const container = document.getElementById(action.zoneId);
            const slot = container ? container.querySelector('.card-slot') : null;
            if (slot) {
                // 既存のデコレーションを探す
                let existingThumbnail = slot.querySelector('.thumbnail[data-is-decoration="true"]');
                if (existingThumbnail) {
                    const img = existingThumbnail.querySelector('img');
                    if (img) img.src = action.imageData;
                } else {
                    // 既存のカードがあれば消す
                    const anyThumb = getExistingThumbnail(slot);
                    if (anyThumb) slot.removeChild(anyThumb);
                    const prefix = getPrefixFromZoneId(action.zoneId);
                    createCardThumbnail(action.imageData, slot, true, false, prefix);
                }
                playSe('カードを配置する.mp3');
                syncMainZoneImage(getBaseId(action.zoneId), getPrefixFromZoneId(action.zoneId));
            }
            break;
        }
        case 'rotate': {
            const slot = getSlotByIndex(action.zoneId, action.slotIndex);
            if (slot) {
                const card = slot.querySelector('.thumbnail');
                if (card) {
                    const img = card.querySelector('.card-image');
                    if (img) {
                        const currentRotation = action.rotation;
                        img.dataset.rotation = currentRotation;
                        
                        if (Math.abs(currentRotation) % 180 !== 0) {
                            slot.classList.add('rotated-90');
                            const { width, height } = getCardDimensions();
                            const scaleFactor = height / width;
                            img.style.transform = `rotate(${currentRotation}deg) scale(${scaleFactor})`;
                            if(getBaseId(action.zoneId).startsWith('mana')) playSe('マナ増加.mp3');
                            else playSe('タップ.mp3');
                        } else {
                            slot.classList.remove('rotated-90');
                            img.style.transform = `rotate(${currentRotation}deg)`;
                            if(!getBaseId(action.zoneId).startsWith('mana')) playSe('タップ.mp3');
                        }
                    }
                }
            }
            break;
        }
        case 'flip': {
            const slot = getSlotByIndex(action.zoneId, action.slotIndex);
            if (slot) {
                const card = slot.querySelector('.thumbnail');
                if (card) {
                    const prefix = getPrefixFromZoneId(action.zoneId);
                    // 現在の状態を確認して、必要なら反転実行
                    const currentFlipped = card.dataset.isFlipped === 'true';
                    if (currentFlipped !== action.isFlipped) {
                         flipCard(card, prefix); // card.jsの関数を使用
                         playSe('カードを反転させる.wav');
                    }
                }
            }
            break;
        }
        case 'delete': {
            const slot = getSlotByIndex(action.zoneId, action.slotIndex);
            if (slot) {
                const card = slot.querySelector('.thumbnail');
                if (card) {
                    deleteCard(card); // card.jsの関数
                    playSe('ボタン共通.mp3');
                }
            }
            break;
        }
        case 'cardCounter': {
            const slot = getSlotByIndex(action.zoneId, action.slotIndex);
            if (slot) {
                const card = slot.querySelector('.thumbnail');
                if (card) {
                    const overlay = card.querySelector('.card-counter-overlay');
                    if (overlay) {
                        overlay.dataset.counter = action.counter;
                        overlay.textContent = action.counter;
                        overlay.style.display = action.counter > 0 ? 'flex' : 'none';
                        if (action.counter > 0) playSe('カウンターを置く.mp3');
                        else playSe('カウンターを取り除く.mp3');
                    }
                }
            }
            break;
        }
        case 'masturbate': {
            const slot = getSlotByIndex(action.zoneId, action.slotIndex);
            if (slot) {
                const card = slot.querySelector('.thumbnail');
                if (card) {
                    card.dataset.isMasturbating = action.isMasturbating;
                    if(action.isMasturbating) playSe('O.mp3', true);
                    else stopSe('O.mp3');
                }
            }
            break;
        }
        case 'counterChange': {
            const input = document.getElementById(action.inputId);
            if (input) {
                const newVal = parseInt(input.value) + action.change;
                input.value = Math.max(0, newVal);
                playSe('ボタン共通.mp3');
            }
            break;
        }
        case 'turnChange': {
            const input = document.getElementById('common-turn-value');
            if (input) {
                input.value = action.value;
                playSe('ボタン共通.mp3');
            }
            break;
        }
        case 'turnPlayerChange': {
            const select = document.getElementById('turn-player-select');
            if (select) select.value = action.value;
            break;
        }
        case 'turnAutoUpdate': {
             const input = document.getElementById('common-turn-value');
             if (input) input.value = action.turnValue;
             const select = document.getElementById('turn-player-select');
             if (select) select.value = action.turnPlayer;
             break;
        }
        case 'stepChange': {
            currentStepIndex = action.index;
            updateStepUI();
            playSe('ボタン共通.mp3');
            break;
        }
        case 'dice': {
            const display = document.getElementById('random-result');
            if (display) display.textContent = `ダイス: ${action.result}`;
            playSe('サイコロ.mp3');
            break;
        }
        case 'coin': {
            const display = document.getElementById('random-result');
            if (display) display.textContent = `コイン: ${action.result}`;
            playSe('コイントス.mp3');
            break;
        }
        case 'boardFlip': {
            if (action.isFlipped) document.body.classList.add('board-flipped');
            else document.body.classList.remove('board-flipped');
            playSe('ボタン共通.mp3');
            break;
        }
        case 'autoDecreaseToggle': {
            const btn = document.getElementById(action.id);
            if (btn) btn.click(); // 既存のロジックをトリガー
            break;
        }
        case 'memoChange': {
            const slot = getSlotByIndex(action.zoneId, action.cardIndex); // cardIndexとslotIndexの混同に注意。記録側はslotIndex相当の意図
            if (slot) {
                const card = slot.querySelector('.thumbnail');
                if (card) {
                    card.dataset.memo = action.memo;
                    playSe('ボタン共通.mp3');
                }
            }
            break;
        }
        case 'flavorUpdate': {
            const slot = getSlotByIndex(action.zoneId, action.cardIndex);
            if (slot) {
                const card = slot.querySelector('.thumbnail');
                if (card) {
                    if (action.slotNumber === 1) card.dataset.flavor1 = action.imgSrc;
                    else if (action.slotNumber === 2) card.dataset.flavor2 = action.imgSrc;
                    playSe('ボタン共通.mp3');
                }
            }
            break;
        }
        case 'flavorDelete': {
            const slot = getSlotByIndex(action.zoneId, action.cardIndex);
            if (slot) {
                const card = slot.querySelector('.thumbnail');
                if (card) {
                    if (action.slotNumber === 1) delete card.dataset.flavor1;
                    else if (action.slotNumber === 2) delete card.dataset.flavor2;
                    playSe('ボタン共通.mp3');
                }
            }
            break;
        }
        case 'effectAction':
            console.log('Replay: Effect Action Triggered');
            playSe('効果発動.mp3');
            break;
        case 'target':
            console.log('Replay: Target Action Triggered');
            playSe('対象に取る.mp3');
            break;
    }
}

function getSlotByIndex(zoneId, index) {
    const zone = document.getElementById(zoneId);
    if (!zone) return null;
    
    // zoneId自体がカードスロットの場合（例: icon-zone）
    if (zone.classList.contains('card-slot')) return zone;

    // コンテナを探す
    const container = zone.querySelector('.slot-container, .deck-back-slot-container, .free-space-slot-container, .hand-zone-slots') || zone;
    const slots = container.querySelectorAll('.card-slot');
    return slots[index] || null;
}