let contextMenu, deleteMenuItem, toGraveMenuItem, toExcludeMenuItem, toHandMenuItem, toDeckMenuItem, toSideDeckMenuItem, flipMenuItem, memoMenuItem, addCounterMenuItem, removeCounterMenuItem, masturbateMenuItem, blockerMenuItem, permanentMenuItem, attackMenuItem;
let actionMenuItem, targetMenuItem, addFlavorMenuItem;
let customCounterMenuItem, changeStyleMenuItem, duplicateMenuItem; 
let exportCardMenuItem, importCardMenuItem, setAsTopMenuItem;
let exportPreviewMenuItem; // プレビューエクスポート用
let memoEditorModal, memoTextarea, memoSaveBtn, memoCancelBtn, memoTooltip, memoEditorHeader;
let lightboxOverlay, lightboxContent;
let commonDrawer, commonDrawerToggle;
let cDrawer, cDrawerToggle;
let commonFlipBoardBtn, commonDecorationSettingsBtn, commonToggleSeBtn;
let diceRollBtn, coinTossBtn, randomResultDisplay;
let commonToggleNavBtn;
let flavorEditorModal, flavorEditorHeader, flavorPreview1, flavorPreview2;
let flavorDelete1, flavorDelete2, flavorCancelBtn, flavorSaveBtn; // 保存ボタン追加
let flavorUpload1, flavorUpload2;
let flavorDropZone1, flavorDropZone2;

// カスタムカウンターモーダル用
let customCounterModal, customCounterCloseBtn, customCounterSaveBtn, createCounterBtn, newCounterNameInput, newCounterImageDrop, customCounterListContainer; // 保存ボタン追加
let currentCustomCounterTarget = null; 
let newCounterImageSrc = null; 

// ファイルダイアログ用のグローバルフラグ
let isFileDialogOpen = false;

// 装飾＆アイコン設定モーダル用
let decorationSettingsModal, decorationSettingsHeader, decorationSettingsCloseBtn;

// デフォルト装飾の定義
const defaultDecorations = {
    'deck': ['./decoration/デッキ.png'],
    'side-deck': ['./decoration/EXデッキ.png'],
    'grave': ['./decoration/墓地エリア.png'],
    'exclude': ['./decoration/除外エリア.png']
};

const defaultIconsPlayer = [
    './decoration/マゾヒスト.png',
    './decoration/マジッカーズ.png',
    './decoration/ストライカー.png',
    './decoration/サディスト.png',
    './decoration/シンプル.png'
];

const defaultIconsOpponent = [
    './decoration/サディスト.png',
    './decoration/マジッカーズ.png',
    './decoration/マゾヒスト.png',
    './decoration/ストライカー.png',
    './decoration/シンプル.png'
];

// ストックの初期化（デフォルト値をセット）
let customIconStocks = {
    player: {
        deck: [...defaultDecorations['deck']],
        'side-deck': [...defaultDecorations['side-deck']],
        grave: [...defaultDecorations['grave']],
        exclude: [...defaultDecorations['exclude']],
        icon: [...defaultIconsPlayer]
    }, 
    opponent: {
        deck: [...defaultDecorations['deck']],
        'side-deck': [...defaultDecorations['side-deck']],
        grave: [...defaultDecorations['grave']],
        exclude: [...defaultDecorations['exclude']],
        icon: [...defaultIconsOpponent]
    }
};

let draggedStockItem = null; 

// バトル確認モーダル用
let battleConfirmModal, battleConfirmHeader, battleConfirmAttackerImg, battleConfirmTargetImg;
let battleConfirmAttackerBpInput, battleConfirmTargetBpInput;
let battleConfirmExecuteBtn, battleConfirmCancelBtn;

// 勝敗表示用
let gameResultOverlay, gameResultMessage, gameResultCloseBtn;

let commonExportBoardBtn, commonImportBoardBtn;
let recordStartBtn, recordStopBtn, replayPlayBtn, replayPauseBtn, replayStopBtn, loadReplayBtn;
let replayFileNameDisplay, replayFileNameText, replayWaitTimeInput;

let bgmSelect, bgmPlayBtn, bgmPauseBtn, bgmStopBtn;
let bgmVolumeSlider, bgmVolumeVal, seVolumeSlider, seVolumeVal;

let seCheckAllBtn, seUncheckAllBtn, effectCheckAllBtn, effectUncheckAllBtn;
let autoCheckAllBtn, autoUncheckAllBtn, autoConfigContainer;
let battleTargetOverlay, battleCancelBtn;

let playerAutoDecreaseInput, opponentAutoDecreaseInput;

// シャッフルボタン
let shuffleDeckBtn, shuffleHandBtn;
let opponentShuffleDeckBtn, opponentShuffleHandBtn;

// システムボタン
let systemBtn, opponentSystemBtn;

let cSearchFilter;

let isResizingDrawer = false;
let lastHoveredElement = null;
let lastRightClickedElement = null;

let stepButtons = [];
const stepOrder = ['step-start', 'step-draw', 'step-mana', 'step-main', 'step-attack', 'step-end'];
let currentStepIndex = 0;

// メモ編集キャンセル用の一時保存変数
let currentMemoOriginalText = '';

// 重複回避のため削除済み (state.jsで定義)
// let currentPreviewExportHandler = null; 

function closeLightbox() {
    if (lightboxOverlay) {
        lightboxOverlay.classList.remove('show');
    }
    if (lightboxContent) {
        lightboxContent.innerHTML = '';
    }
}

function closeContextMenu() {
    if (contextMenu) {
        contextMenu.style.display = 'none';
        // サブメニューの展開状態をリセット
        const submenus = contextMenu.querySelectorAll('.submenu');
        submenus.forEach(sub => {
            sub.classList.remove('open-left', 'open-top');
        });
        
        // 非表示にされたメニュー項目をリセット
        const allItems = contextMenu.querySelectorAll('li');
        allItems.forEach(li => {
            li.style.display = ''; 
        });
        
        const hasSubmenus = contextMenu.querySelectorAll('.has-submenu');
        hasSubmenus.forEach(li => {
            li.style.display = ''; 
        });

        // プレビューエクスポート項目を確実に非表示にする
        if (exportPreviewMenuItem) {
            exportPreviewMenuItem.style.display = 'none';
        }
    }
    currentDeleteHandler = null;
    currentMoveToGraveHandler = null;
    currentMoveToExcludeHandler = null;
    currentMoveToHandHandler = null;
    currentMoveToDeckHandler = null;
    currentMoveToSideDeckHandler = null;
    currentFlipHandler = null;
    currentMemoHandler = null;
    currentAddCounterHandler = null;
    currentRemoveCounterHandler = null;
    currentActionHandler = null;
    currentAttackHandler = null;
    currentTargetHandler = null;
    currentPermanentHandler = null;
    currentAddFlavorHandler = null;
    currentBlockerHandler = null;
    currentMasturbateHandler = null;
    currentExportCardHandler = null;
    currentImportCardHandler = null;
    currentStockItemTarget = null;
    currentPreviewExportHandler = null;
}

// --- Custom Counter UI Logic ---

function openCustomCounterModal(targetCard) {
    currentCustomCounterTarget = targetCard;
    renderCustomCounterList();
    
    newCounterNameInput.value = '';
    newCounterImageSrc = null;
    newCounterImageDrop.innerHTML = '画像D&Dまたはクリック';
    newCounterImageDrop.style.backgroundImage = '';
    
    customCounterModal.style.display = 'block';
}

function closeCustomCounterModal() {
    customCounterModal.style.display = 'none';
    currentCustomCounterTarget = null;
}

function handleNewCounterImageFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        newCounterImageSrc = e.target.result;
        newCounterImageDrop.innerHTML = '';
        newCounterImageDrop.style.backgroundImage = `url(${newCounterImageSrc})`;
        newCounterImageDrop.style.backgroundSize = 'contain';
        newCounterImageDrop.style.backgroundRepeat = 'no-repeat';
        newCounterImageDrop.style.backgroundPosition = 'center';
    };
    reader.readAsDataURL(file);
}

function createNewCustomCounterType() {
    const name = newCounterNameInput.value.trim();
    if (!name) {
        alert('カウンター名を入力してください');
        return;
    }
    if (!newCounterImageSrc) {
        alert('画像を設定してください');
        return;
    }

    const id = 'cnt_' + Date.now();
    const newCounter = { id, name, icon: newCounterImageSrc };
    customCounterTypes.push(newCounter);
    
    playSe('ボタン共通.mp3');
    renderCustomCounterList();
    
    newCounterNameInput.value = '';
    newCounterImageSrc = null;
    newCounterImageDrop.innerHTML = '画像D&Dまたはクリック';
    newCounterImageDrop.style.backgroundImage = '';
}

function deleteCustomCounterType(id) {
    if (!confirm('このカウンター種類を削除しますか？（使用中のカードからは消えません）')) return;
    customCounterTypes = customCounterTypes.filter(c => c.id !== id);
    playSe('ボタン共通.mp3');
    renderCustomCounterList();
}

function renderCustomCounterList() {
    customCounterListContainer.innerHTML = '';
    
    customCounterTypes.forEach(counter => {
        const item = document.createElement('div');
        item.className = 'counter-list-item';
        item.title = counter.name; 
        
        const img = document.createElement('img');
        img.src = counter.icon;
        
        const span = document.createElement('span');
        span.textContent = counter.name;
        
        const deleteBtn = document.createElement('div');
        deleteBtn.className = 'counter-delete-btn';
        deleteBtn.textContent = '×';
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteCustomCounterType(counter.id);
        });
        
        item.appendChild(img);
        item.appendChild(span);
        item.appendChild(deleteBtn);
        
        item.addEventListener('click', () => {
            if (currentCustomCounterTarget) {
                if (typeof addCustomCounter === 'function') {
                    addCustomCounter(currentCustomCounterTarget, counter.id);
                    playSe('カウンターを置く.mp3');
                    closeCustomCounterModal();
                }
            }
        });
        
        customCounterListContainer.appendChild(item);
    });
}

// --- Decoration Settings UI Logic ---

function openDecorationSettingsModal() {
    if (!decorationSettingsModal) return;
    decorationSettingsModal.style.display = 'flex';
    setupDecorationDropZones();
    
    // URL比較用ヘルパー関数（デコード対応）
    const isSameUrl = (url1, url2) => {
        if (!url1 || !url2) return false;
        // Data URIの場合は完全一致
        if (url1.startsWith('data:') || url2.startsWith('data:')) {
            return url1 === url2;
        }
        // 相対パスやURLエンコードされたパスの比較のためデコードしてファイル名で比較
        const normalize = (u) => decodeURIComponent(u).split('/').pop().split('?')[0];
        return normalize(url1) === normalize(url2);
    };

    ['player', 'opponent'].forEach(owner => {
        const idPrefix = (owner === 'opponent') ? 'opponent-' : '';
        
        ['deck', 'side-deck', 'grave', 'exclude', 'icon'].forEach(targetType => {
            const column = decorationSettingsModal.querySelector(`.decoration-column[data-target-type="${targetType}"][data-owner="${owner}"]`);
            if (column) {
                const container = column.querySelector('.decoration-stock-container');
                
                // 現在の画像を取得
                let currentImgSrc = null;
                
                if (targetType === 'icon') {
                    const iconZone = document.getElementById(idPrefix + 'icon-zone');
                    if (iconZone) {
                        const thumb = iconZone.querySelector('.thumbnail');
                        if (thumb) {
                            const img = thumb.querySelector('.card-image');
                            if (img) currentImgSrc = img.src;
                        }
                    }
                } else {
                    const zoneId = idPrefix + targetType;
                    const zone = document.getElementById(zoneId);
                    if (zone) {
                        const slot = zone.querySelector('.card-slot');
                        if (slot) {
                            const decoThumb = slot.querySelector('.thumbnail[data-is-decoration="true"]');
                            if (decoThumb) {
                                const img = decoThumb.querySelector('.card-image');
                                if (img) currentImgSrc = img.src;
                            }
                        }
                    }
                }

                // ストックの取得（初期化済みのグローバル変数を使用）
                if (!customIconStocks[owner]) customIconStocks[owner] = {};
                if (!customIconStocks[owner][targetType]) customIconStocks[owner][targetType] = [];
                
                let stock = [...customIconStocks[owner][targetType]];
                
                // 現在の画像をストックの先頭に移動・追加
                if (currentImgSrc) {
                    const existingIndex = stock.findIndex(src => isSameUrl(src, currentImgSrc));
                    if (existingIndex > -1) {
                        stock.splice(existingIndex, 1);
                    }
                    stock.unshift(currentImgSrc);
                }
                
                // デフォルト画像の追加（重複チェックを強化）
                let defaults = [];
                if (targetType === 'icon') {
                    defaults = (owner === 'player') ? defaultIconsPlayer : defaultIconsOpponent;
                } else {
                    defaults = defaultDecorations[targetType] || [];
                }

                defaults.forEach(defSrc => {
                    // ストック内に同じ画像が無ければ追加
                    if (!stock.some(s => isSameUrl(s, defSrc))) {
                        stock.push(defSrc);
                    }
                });

                // コンテナをクリアして再描画
                container.innerHTML = '';
                stock.forEach(imgSrc => {
                    addDecorationToStock(imgSrc, targetType, owner, container, false, isSameUrl(imgSrc, currentImgSrc));
                });
                
                // 更新されたストックを保存
                customIconStocks[owner][targetType] = stock;
            }
        });
    });
}

function closeDecorationSettingsModal() {
    decorationSettingsModal.style.display = 'none';
}

function setupDecorationDropZones() {
    const dropZones = decorationSettingsModal.querySelectorAll('.decoration-drop-zone');
    dropZones.forEach(zone => {
        if (zone.dataset.listenerAttached) return;
        zone.dataset.listenerAttached = true;

        const column = zone.closest('.decoration-column');
        const targetType = column.dataset.targetType;
        const owner = column.dataset.owner; 

        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.style.backgroundColor = '#5a5a7e';
        });

        zone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.style.backgroundColor = '';
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.style.backgroundColor = '';
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (evt) => {
                    addDecorationToStock(evt.target.result, targetType, owner, column.querySelector('.decoration-stock-container'));
                };
                reader.readAsDataURL(files[0]);
            }
        });

        zone.addEventListener('click', (e) => {
            e.stopPropagation(); 
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.accept = 'image/*';
            fileInput.style.display = 'none';
            fileInput.onchange = (e) => {
                if (e.target.files.length > 0) {
                    const reader = new FileReader();
                    reader.onload = (evt) => {
                        addDecorationToStock(evt.target.result, targetType, owner, column.querySelector('.decoration-stock-container'));
                    };
                    reader.readAsDataURL(e.target.files[0]);
                }
            };
            document.body.appendChild(fileInput);
            isFileDialogOpen = true;
            fileInput.click();
            document.body.removeChild(fileInput);
            setTimeout(() => { isFileDialogOpen = false; }, 100);
        });
    });
}

function addDecorationToStock(imageSrc, targetType, owner, container, isNew = true, isCurrent = false) {
    // DOM上での簡易重複チェック（新規追加時など）
    if (isNew) {
        const existingImgs = Array.from(container.querySelectorAll('img'));
        const exists = existingImgs.some(img => img.src === imageSrc);
        if (exists) return;
    }

    const item = document.createElement('div');
    item.className = 'decoration-stock-item';
    item.draggable = true;
    item.dataset.targetType = targetType;
    item.dataset.owner = owner;
    
    if (isCurrent) {
        item.style.borderColor = '#ffcc00';
        item.style.boxShadow = '0 0 5px #ffcc00';
    }
    
    // デフォルトメモ定義
    const defaultMemo = `[カード名:-]/#e0e0e0/#555/1.0/非表示/
[属性:-]/#e0e0e0/#555/1.0/非表示/
[マナ:-]/#e0e0e0/#555/1.0/非表示/
[BP:-]/#e0e0e0/#555/1.0/非表示/
[スペル:-]/#e0e0e0/#555/1.0/非表示/
[フレーバーテキスト:-]/#fff/#555/1.0/非表示/
[効果:-]/#e0e0e0/#555/0.7/非表示/`;

    item.dataset.memo = defaultMemo;

    const img = document.createElement('img');
    img.src = imageSrc;
    img.onerror = () => {
        item.remove();
    };
    item.appendChild(img);

    const deleteBtn = document.createElement('div');
    deleteBtn.className = 'decoration-delete-btn';
    deleteBtn.textContent = '×';
    deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        item.remove();
        playSe('ボタン共通.mp3');
        updateZoneDecorationFromStock(targetType, owner, container);
        saveIconStock(targetType, owner, container);
    });
    item.appendChild(deleteBtn);

    item.addEventListener('dragstart', handleStockItemDragStart);
    item.addEventListener('dragover', handleStockItemDragOver);
    item.addEventListener('drop', handleStockItemDrop);
    
    item.addEventListener('contextmenu', (e) => {
        handleStockItemContextMenu(e, item, targetType, owner, container);
    });

    if (isNew) {
        container.insertBefore(item, container.firstChild);
        playSe('ボタン共通.mp3');
        updateZoneDecorationFromStock(targetType, owner, container);
        saveIconStock(targetType, owner, container);
    } else {
        container.appendChild(item);
    }
}

function handleStockItemContextMenu(e, item, targetType, owner, container) {
    e.preventDefault();
    e.stopPropagation();
    
    if (!contextMenu) return;

    const allItems = contextMenu.querySelectorAll('li');
    allItems.forEach(li => li.style.display = 'none');
    
    const topItems = contextMenu.querySelectorAll('#custom-context-menu > ul > li');
    topItems.forEach(li => li.style.display = 'none');

    if (memoMenuItem) memoMenuItem.style.display = 'block'; 
    if (setAsTopMenuItem) setAsTopMenuItem.style.display = 'block';
    
    if (deleteMenuItem) deleteMenuItem.style.display = 'none';
    
    currentMemoHandler = () => {
        currentMemoTarget = item;
        memoTextarea.value = item.dataset.memo || '';
        openMemoEditor(); 
    };
    
    currentStockItemTarget = () => setDecorationAsTop(item, targetType, owner, container);

    contextMenu.style.display = 'block';
    contextMenu.style.visibility = 'hidden';
    contextMenu.style.zIndex = '10005'; 
    
    const menuWidth = contextMenu.offsetWidth;
    const menuHeight = contextMenu.offsetHeight;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;

    let left = e.pageX;
    let top = e.pageY;

    if (left + menuWidth > windowWidth) left -= menuWidth;
    if (top + menuHeight > windowHeight) top -= menuHeight;

    contextMenu.style.top = `${top}px`;
    contextMenu.style.left = `${left}px`;
    contextMenu.style.visibility = 'visible';
}

function setDecorationAsTop(item, targetType, owner, container) {
    if (container.firstChild !== item) {
        container.insertBefore(item, container.firstChild);
    }
    updateZoneDecorationFromStock(targetType, owner, container);
    saveIconStock(targetType, owner, container);
    playSe('ボタン共通.mp3');
    
    const allItems = container.querySelectorAll('.decoration-stock-item');
    allItems.forEach(i => {
        i.style.borderColor = '';
        i.style.boxShadow = '';
    });
    item.style.borderColor = '#ffcc00';
    item.style.boxShadow = '0 0 5px #ffcc00';
}

function handleStockItemDragStart(e) {
    draggedStockItem = e.currentTarget;
    e.dataTransfer.effectAllowed = 'move';
}

function handleStockItemDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
}

function handleStockItemDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    const targetItem = e.currentTarget;
    
    if (draggedStockItem && draggedStockItem !== targetItem && draggedStockItem.parentNode === targetItem.parentNode) {
        const container = targetItem.parentNode;
        const items = Array.from(container.children);
        const fromIndex = items.indexOf(draggedStockItem);
        const toIndex = items.indexOf(targetItem);
        
        if (fromIndex < toIndex) {
            container.insertBefore(draggedStockItem, targetItem.nextSibling);
        } else {
            container.insertBefore(draggedStockItem, targetItem);
        }
        
        const targetType = draggedStockItem.dataset.targetType;
        const owner = draggedStockItem.dataset.owner;
        
        updateZoneDecorationFromStock(targetType, owner, container);
        saveIconStock(targetType, owner, container);
    }
    draggedStockItem = null;
}

function saveIconStock(targetType, owner, container) {
    const items = Array.from(container.querySelectorAll('.decoration-stock-item img'));
    // 所有者・タイプ別に保存
    if (!customIconStocks[owner]) customIconStocks[owner] = {};
    customIconStocks[owner][targetType] = items.map(img => img.src);
}

function updateZoneDecorationFromStock(targetType, owner, container) {
    const firstItem = container.querySelector('.decoration-stock-item');
    let imgSrc = null;
    let memo = null;
    
    if (firstItem) {
        imgSrc = firstItem.querySelector('img').src;
        memo = firstItem.dataset.memo;
    } else {
        // ストックが空の場合のデフォルトフォールバック
        const defaults = {
            'deck': './decoration/デッキ.png',
            'side-deck': './decoration/EXデッキ.png',
            'grave': './decoration/墓地エリア.png',
            'exclude': './decoration/除外エリア.png',
            'icon': '' 
        };
        imgSrc = defaults[targetType];
    }
    
    const idPrefix = (owner === 'opponent') ? 'opponent-' : '';
    let zoneId = '';
    
    if (targetType === 'icon') {
        zoneId = idPrefix + 'icon-zone';
    } else {
        zoneId = idPrefix + targetType;
    }
    
    const zoneElement = document.getElementById(zoneId);
    if (!zoneElement) return;
    
    if (targetType === 'icon') {
        zoneElement.innerHTML = ''; 
        if (imgSrc) {
             createCardThumbnail({
                src: imgSrc,
                isDecoration: true,
                memo: memo || '',
                ownerPrefix: idPrefix
            }, zoneElement, true, false, idPrefix);
        }
    } else {
        const slot = zoneElement.querySelector('.card-slot');
        if (slot) {
            let decorationThumb = slot.querySelector('.thumbnail[data-is-decoration="true"]');
            
            if (imgSrc) {
                if (decorationThumb) {
                    const img = decorationThumb.querySelector('img');
                    if (img) img.src = imgSrc;
                } else {
                    createCardThumbnail(imgSrc, slot, true, false, idPrefix);
                }
            } else {
                if (decorationThumb) decorationThumb.remove();
            }
            
            if (typeof syncMainZoneImage === 'function') {
                syncMainZoneImage(targetType, idPrefix);
            }
        }
    }
}


// --- Game Result UI Logic ---

window.showGameResult = function(message) {
    if (!gameResultOverlay || !gameResultMessage) return;
    
    if (typeof autoConfig !== 'undefined' && !autoConfig.autoGameEnd) return;

    gameResultMessage.textContent = message;
    gameResultOverlay.style.display = 'flex';
    
    if (message.includes('WIN')) {
        playSe('勝利.mp3');
    } else if (message.includes('LOSE')) {
        playSe('敗北.mp3');
    } else {
        playSe('Theme.mp3');
    }
};

function closeGameResult() {
    if (gameResultOverlay) {
        gameResultOverlay.style.display = 'none';
    }
}

// --- Battle Confirm Modal Logic ---

window.openBattleConfirmModal = function(attacker, target) {
    if (!battleConfirmModal) return;
    
    document.body.classList.remove('battle-target-mode');
    if (battleTargetOverlay) battleTargetOverlay.style.display = 'none';
    isBattleTargetMode = false;
    const candidates = document.querySelectorAll('.battle-target-candidate');
    candidates.forEach(el => el.classList.remove('battle-target-candidate'));

    const attackerImg = attacker.querySelector('.card-image');
    if (attackerImg && battleConfirmAttackerImg) {
        battleConfirmAttackerImg.style.backgroundImage = `url(${attackerImg.src})`;
    }
    
    if (battleConfirmTargetImg) {
        battleConfirmTargetImg.style.backgroundImage = 'none'; 
        if (target.id === 'icon-zone' || target.id === 'opponent-icon-zone') {
            const iconImg = target.closest('.player-icon-slot')?.querySelector('.thumbnail .card-image');
            if (iconImg) {
                battleConfirmTargetImg.style.backgroundImage = `url(${iconImg.src})`;
            } else {
                battleConfirmTargetImg.innerHTML = '<span style="color:#ccc; font-size:0.8em; display: flex; justify-content: center; align-items: center; height: 100%;">Player</span>';
            }
        } else {
            const targetThumb = target.querySelector('.thumbnail');
            const targetImg = targetThumb ? targetThumb.querySelector('.card-image') : null;
            if (targetImg) {
                battleConfirmTargetImg.style.backgroundImage = `url(${targetImg.src})`;
            }
        }
    }

    battleConfirmModal.style.display = 'flex';
    isBattleConfirmMode = true;
    currentAttacker = attacker;
    currentBattleTarget = target;

    window.updateBattleConfirmModal();
};

window.updateBattleConfirmModal = function() {
    if (!battleConfirmModal || battleConfirmModal.style.display === 'none') return;
    
    const getBP = (element) => {
        if (!element) return 0;
        const memo = element.dataset.memo || '';
        const match = memo.match(/\[BP:(.*?)\]/i);
        let bp = 0;
        if (match) {
            bp = parseInt(match[1]);
            if (isNaN(bp)) bp = 0;
        }
        return bp;
    };

    if (currentAttacker && battleConfirmAttackerBpInput) {
        battleConfirmAttackerBpInput.value = getBP(currentAttacker);
    }

    if (currentBattleTarget && battleConfirmTargetBpInput) {
        let targetEl = currentBattleTarget;
        if (targetEl.id !== 'icon-zone' && targetEl.id !== 'opponent-icon-zone') {
            targetEl = targetEl.querySelector('.thumbnail') || targetEl;
        }
        battleConfirmTargetBpInput.value = getBP(targetEl);
    }
};

window.closeBattleConfirmModal = function() {
    if (battleConfirmModal) {
        battleConfirmModal.style.display = 'none';
    }
    isBattleConfirmMode = false;
    currentBattleTarget = null;
    if (currentAttacker) {
        currentAttacker.classList.remove('battle-attacker');
    }
};

// --- Existing Logic ---

function performMemoSave() {
    if (currentMemoTarget) {
        const newMemo = memoTextarea.value;
        if (newMemo) {
            currentMemoTarget.dataset.memo = newMemo;
        } else {
            delete currentMemoTarget.dataset.memo;
        }
        
        if (!currentMemoTarget.classList.contains('decoration-stock-item')) {
            if (isRecording && typeof recordAction === 'function') {
                recordAction({
                    type: 'memoChange',
                    zoneId: getParentZoneId(currentMemoTarget.parentNode),
                    cardIndex: Array.from(currentMemoTarget.parentNode.parentNode.children).indexOf(currentMemoTarget.parentNode),
                    memo: newMemo
                });
            }
            
            if (typeof window.updateCardPreview === 'function') {
                window.updateCardPreview(currentMemoTarget);
            }

            if (isBattleConfirmMode) {
                window.updateBattleConfirmModal();
            }
        } else {
            const container = currentMemoTarget.parentNode;
            const firstItem = container.querySelector('.decoration-stock-item');
            if (currentMemoTarget === firstItem) {
                const targetType = currentMemoTarget.dataset.targetType;
                const owner = currentMemoTarget.dataset.owner;
                updateZoneDecorationFromStock(targetType, owner, container);
            }
        }
    }
    memoEditorModal.style.display = 'none';
    currentMemoTarget = null;
    currentMemoOriginalText = '';
}

function performMemoCancel() {
    if (currentMemoTarget && currentMemoOriginalText !== undefined) {
        currentMemoTarget.dataset.memo = currentMemoOriginalText;
        if (typeof window.updateCardPreview === 'function') {
            window.updateCardPreview(currentMemoTarget);
        }
    }
    memoEditorModal.style.display = 'none';
    currentMemoTarget = null;
    currentMemoOriginalText = '';
}

function openMemoEditor() {
    memoEditorModal.style.display = 'flex';
    memoTextarea.style.height = 'auto'; 
    if (memoTextarea.scrollHeight > 0) {
        memoTextarea.style.height = (memoTextarea.scrollHeight + 10) + 'px';
    }
    memoTextarea.focus();
    
    if (currentMemoTarget) {
        currentMemoOriginalText = currentMemoTarget.dataset.memo || '';
    }
}

function openFlavorEditor(targetThumbnail) {
    if (!targetThumbnail) return;
    currentFlavorTarget = targetThumbnail;
    flavorEditorHeader.textContent = `フレーバーイラスト設定`;
    updateFlavorPreview(1, currentFlavorTarget.dataset.flavor1);
    updateFlavorPreview(2, currentFlavorTarget.dataset.flavor2);
    flavorEditorModal.style.display = 'block';
}

function closeFlavorEditor() {
    flavorEditorModal.style.display = 'none';
    currentFlavorTarget = null;
}

function updateFlavorPreview(slotNumber, imgSrc) {
    const previewEl = (slotNumber === 1) ? flavorPreview1 : flavorPreview2;
    if (!previewEl) return;
    previewEl.innerHTML = '';
    if (imgSrc) {
        const img = document.createElement('img');
        img.src = imgSrc;
        previewEl.appendChild(img);
    }
}

function deleteFlavorImage(slotNumber) {
    if (!currentFlavorTarget) return;
    if (slotNumber === 1) {
        delete currentFlavorTarget.dataset.flavor1;
        updateFlavorPreview(1, null);
    } else if (slotNumber === 2) {
        delete currentFlavorTarget.dataset.flavor2;
        updateFlavorPreview(2, null);
    }
    if (isRecording && typeof recordAction === 'function') {
        recordAction({
            type: 'flavorDelete',
            zoneId: getParentZoneId(currentFlavorTarget.parentNode),
            cardIndex: Array.from(currentFlavorTarget.parentNode.parentNode.children).indexOf(currentFlavorTarget.parentNode),
            slotNumber: slotNumber
        });
    }
    if (typeof window.updateCardPreview === 'function') {
        window.updateCardPreview(currentFlavorTarget);
    }
}

function handleFlavorFile(file, slotNumber) {
    if (!currentFlavorTarget) return;
    if (!file || !file.type.startsWith('image/')) {
        console.warn("画像ファイルを選択してください。");
        return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
        const imgSrc = event.target.result;
        if (slotNumber === 1) {
            currentFlavorTarget.dataset.flavor1 = imgSrc;
            updateFlavorPreview(1, imgSrc);
        } else if (slotNumber === 2) {
            currentFlavorTarget.dataset.flavor2 = imgSrc;
            updateFlavorPreview(2, imgSrc);
        }
        
        if (isRecording && typeof recordAction === 'function') {
            recordAction({
                type: 'flavorUpdate',
                zoneId: getParentZoneId(currentFlavorTarget.parentNode),
                cardIndex: Array.from(currentFlavorTarget.parentNode.parentNode.children).indexOf(currentFlavorTarget.parentNode),
                slotNumber: slotNumber,
                imgSrc: imgSrc
            });
        }
        if (typeof window.updateCardPreview === 'function') {
            window.updateCardPreview(currentFlavorTarget);
        }
    };
    reader.readAsDataURL(file);
}

function openFlavorFileInput(slotNumber) {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.style.display = 'none';
    fileInput.addEventListener('change', (event) => {
        try {
            const file = event.target.files[0];
            if (file) {
                handleFlavorFile(file, slotNumber);
            }
        } catch (error) {
            console.error("フレーバー画像の読み込みに失敗:", error);
        } finally {
            if (document.body.contains(fileInput)) {
                document.body.removeChild(fileInput);
            }
        }
    });
    fileInput.addEventListener('cancel', () => {
         if (document.body.contains(fileInput)) {
            document.body.removeChild(fileInput);
         }
    });
    document.body.appendChild(fileInput);
    isFileDialogOpen = true;
    fileInput.click();
    setTimeout(() => { isFileDialogOpen = false; }, 100);
}

function updateStepUI() {
    stepButtons.forEach((btn, index) => {
        if (!btn) return;
        btn.classList.toggle('active', index === currentStepIndex);
        const nextStepIndex = (currentStepIndex + 1) % stepOrder.length;
        btn.disabled = (index !== nextStepIndex);
    });
}

function setupStepButtons() {
    stepButtons = stepOrder.map(id => document.getElementById(id));

    stepButtons.forEach((button, index) => {
        if (button) {
            button.addEventListener('click', () => {
                if (!button.disabled) {
                    if (button.id === 'step-start') {
                        playSe('ターン開始.mp3');
                    } else {
                        playSe('ボタン共通.mp3');
                    }
                    
                    currentStepIndex = index;
                    updateStepUI();

                    if (isRecording && typeof recordAction === 'function') {
                        recordAction({ type: 'stepChange', index: index });
                    }

                    if (button.id === 'step-start' && currentStepIndex === 0) {
                         const turnPlayerSelect = document.getElementById('turn-player-select');
                         const turnInput = document.getElementById('common-turn-value');
                         if (turnPlayerSelect.value === 'first') {
                             turnPlayerSelect.value = 'second';
                         } else {
                             let currentValue = parseInt(turnInput.value) || 1;
                             turnInput.value = currentValue + 1;
                             turnPlayerSelect.value = 'first';
                         }
                         
                         if (isRecording && typeof recordAction === 'function') {
                             recordAction({ 
                                 type: 'turnAutoUpdate', 
                                 turnValue: turnInput.value, 
                                 turnPlayer: turnPlayerSelect.value 
                             });
                         }
                    }
                }
            });
        }
    });

    currentStepIndex = 0;
    updateStepUI();
}

function initializeTokens() {
    const initToken = (zoneId, slotIndex, imgSrc, memo) => {
        const zone = document.getElementById(zoneId);
        if (!zone) return;
        const container = zone.querySelector('.token-slot-container');
        if (!container) return;
        const slots = container.querySelectorAll('.card-slot');
        const slot = slots[slotIndex];
        if (slot && !slot.querySelector('.thumbnail')) {
            const prefix = zoneId.startsWith('opponent-') ? 'opponent-' : '';
            createCardThumbnail({
                src: imgSrc,
                memo: memo,
                ownerPrefix: prefix
            }, slot, false, false, prefix);
        }
    };

    const token1Memo = '[属性:S]/#e0e0e0/#555/1.0/表示/\n[マナ:1]/#e0e0e0/#555/1.0/表示/\n[BP:1000]/#e0e0e0/#555/1.0/表示/\n[カード名:トークンカード（S）]/#e0e0e0/#555/1.0/表示/\n[フレーバーテキスト:女王]/#fff/#555/1.0/非表示/\n[効果:なし]/#e0e0e0/#555/1.0/表示/\n';
    const token2Memo = '[属性:M]/#e0e0e0/#555/1.0/表示/\n[マナ:1]/#e0e0e0/#555/1.0/表示/\n[BP:1000]/#e0e0e0/#555/1.0/表示/\n[カード名:トークンカード（M）]/#e0e0e0/#555/1.0/表示/\n[フレーバーテキスト:奴隷]/#fff/#555/1.0/非表示/\n[効果:なし]/#e0e0e0/#555/1.0/表示/\n';

    initToken('token-zone-slots', 0, './decoration/トークン1.png', token1Memo);
    initToken('token-zone-slots', 1, './decoration/トークン2.png', token2Memo);

    initToken('opponent-token-zone-slots', 0, './decoration/トークン1.png', token1Memo);
    initToken('opponent-token-zone-slots', 1, './decoration/トークン2.png', token2Memo);
}

function makeDraggable(headerElement, containerElement) {
    if (!headerElement || !containerElement) return;
    let isDragging = false;
    let currentX = 0;
    let currentY = 0;
    let initialX;
    let initialY;
    let xOffset = 0;
    let yOffset = 0;

    headerElement.addEventListener("mousedown", dragStart);
    document.addEventListener("mouseup", dragEnd);
    document.addEventListener("mousemove", drag);

    function dragStart(e) {
        initialX = e.clientX - xOffset;
        initialY = e.clientY - yOffset;

        if (e.target === headerElement || headerElement.contains(e.target)) {
            isDragging = true;
        }
    }

    function dragEnd(e) {
        initialX = currentX;
        initialY = currentY;
        isDragging = false;
    }

    function drag(e) {
        if (isDragging) {
            e.preventDefault();
            currentX = e.clientX - initialX;
            currentY = e.clientY - initialY;

            containerElement.style.transform = `translate(calc(-50% + ${currentX}px), calc(-50% + ${currentY}px))`;
            xOffset = currentX;
            yOffset = currentY;
        }
    }
}

function exportCardPreviewAsImage() {
    const previewArea = document.getElementById('common-card-preview');
    const previewImg = previewArea.querySelector('img');

    if (!previewImg || !previewImg.src || previewImg.src === window.location.href) {
        alert('エクスポートするカード画像がありません。');
        return;
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.src = previewImg.src;

    img.onload = () => {
        // キャンバスサイズを元画像のサイズに設定
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;

        // 元画像を描画
        ctx.drawImage(img, 0, 0);

        // プレビューエリアの表示サイズと元画像のスケール比率を計算
        const previewRect = previewArea.getBoundingClientRect();
        const scaleX = canvas.width / previewRect.width;
        const scaleY = canvas.height / previewRect.height;
        
        // フォントサイズ用のスケールは、縦横の小さい方に合わせる（文字が極端に変形するのを防ぐ）
        const fontScale = Math.min(scaleX, scaleY);

        // 描画対象の要素IDリスト
        const elementIds = [
            'preview-attribute',
            'preview-cost',
            'preview-top-right-stat',
            'preview-flavor-text',
            'preview-effect-text',
            'preview-card-name'
        ];

        elementIds.forEach(id => {
            const el = document.getElementById(id);
            if (!el || el.style.display === 'none' || !el.textContent.trim()) return;

            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();

            // 相対位置計算（位置はスケール通りに配置）
            const x = (rect.left - previewRect.left) * scaleX;
            const y = (rect.top - previewRect.top) * scaleY;
            const w = rect.width * scaleX;
            const h = rect.height * scaleY;

            // 背景描画 (不透明度、色を反映) - 枠のサイズはスケール通り
            if (style.backgroundColor !== 'rgba(0, 0, 0, 0)' && style.backgroundColor !== 'transparent') {
                ctx.save();
                ctx.fillStyle = style.backgroundColor;
                if (el.style.opacity) {
                    ctx.globalAlpha = parseFloat(el.style.opacity);
                }
                
                if (ctx.roundRect) {
                    ctx.beginPath();
                    ctx.roundRect(x, y, w, h, parseFloat(style.borderRadius) * fontScale || 0);
                    ctx.fill();
                } else {
                    ctx.fillRect(x, y, w, h);
                }
                ctx.restore();
            }

            // テキスト描画
            ctx.save();
            ctx.fillStyle = style.color;
            ctx.textAlign = style.textAlign === 'center' ? 'center' : (style.textAlign === 'right' ? 'right' : 'left');
            ctx.textBaseline = 'middle';

            const fontSize = parseFloat(style.fontSize) * fontScale; // 修正: scaleY -> fontScale
            const fontWeight = style.fontWeight;
            const fontFamily = style.fontFamily;
            ctx.font = `${fontWeight} ${fontSize}px ${fontFamily}`;

            const paddingLeft = parseFloat(style.paddingLeft) * scaleX;
            const paddingRight = parseFloat(style.paddingRight) * scaleX;
            const paddingTop = parseFloat(style.paddingTop) * scaleY;
            
            let textX = x;
            if (style.textAlign === 'center') {
                textX = x + w / 2;
            } else if (style.textAlign === 'right') {
                textX = x + w - paddingRight;
            } else {
                textX = x + paddingLeft;
            }
            
            const textContent = el.textContent;
            const lineHeight = fontSize * 1.2; 
            
            // maxWidthを設定して描画（枠内に収めるため）
            const maxTextWidth = w - (paddingLeft + paddingRight);

            if (id === 'preview-flavor-text' || id === 'preview-effect-text') {
                ctx.textBaseline = 'top';
                wrapText(ctx, textContent, textX, y + paddingTop, maxTextWidth, lineHeight);
            } else {
                ctx.textBaseline = 'middle';
                ctx.fillText(textContent, textX, y + h / 2, maxTextWidth);
            }

            ctx.restore();
        });

        const link = document.createElement('a');
        link.download = `card_export_${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    };

    img.onerror = () => {
        alert('画像の読み込みに失敗しました。');
    };
}

// テキスト折り返しヘルパー
function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
    const lines = text.split('\n');
    
    for (let i = 0; i < lines.length; i++) {
        let line = '';
        const words = lines[i].split(''); 
        
        for (let n = 0; n < words.length; n++) {
            const testLine = line + words[n];
            const metrics = ctx.measureText(testLine);
            const testWidth = metrics.width;
            
            if (testWidth > maxWidth && n > 0) {
                ctx.fillText(line, x, y);
                line = words[n];
                y += lineHeight;
            } else {
                line = testLine;
            }
        }
        ctx.fillText(line, x, y);
        y += lineHeight;
    }
}

function setupUI() {
    document.addEventListener('contextmenu', (e) => {
        lastRightClickedElement = e.target.closest('.thumbnail') || e.target.closest('.card-slot');
        if (!e.target.closest('.decoration-stock-item') && !e.target.closest('#common-card-preview')) {
            e.preventDefault();
        }
    });
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());

    contextMenu = document.getElementById('custom-context-menu');
    actionMenuItem = document.getElementById('context-menu-action');
    permanentMenuItem = document.getElementById('context-menu-permanent'); 
    attackMenuItem = document.getElementById('context-menu-attack'); 
    targetMenuItem = document.getElementById('context-menu-target');
    deleteMenuItem = document.getElementById('context-menu-delete');
    toGraveMenuItem = document.getElementById('context-menu-to-grave');
    toExcludeMenuItem = document.getElementById('context-menu-to-exclude');
    toHandMenuItem = document.getElementById('context-menu-to-hand');
    toDeckMenuItem = document.getElementById('context-menu-to-deck');
    toSideDeckMenuItem = document.getElementById('context-menu-to-side-deck');
    flipMenuItem = document.getElementById('context-menu-flip');
    changeStyleMenuItem = document.getElementById('context-menu-change-style');
    memoMenuItem = document.getElementById('context-menu-memo');
    addCounterMenuItem = document.getElementById('context-menu-add-counter');
    removeCounterMenuItem = document.getElementById('context-menu-remove-counter');
    addFlavorMenuItem = document.getElementById('context-menu-add-flavor');
    masturbateMenuItem = document.getElementById('context-menu-masturbate');
    blockerMenuItem = document.getElementById('context-menu-blocker');
    exportCardMenuItem = document.getElementById('context-menu-export');
    importCardMenuItem = document.getElementById('context-menu-import');
    
    customCounterMenuItem = document.getElementById('context-menu-custom-counter');
    duplicateMenuItem = document.getElementById('context-menu-duplicate');
    setAsTopMenuItem = document.getElementById('context-menu-set-as-top');
    
    // HTML側で定義された要素を取得
    exportPreviewMenuItem = document.getElementById('context-menu-export-preview');
    // 万が一HTMLに無い場合のみ作成（整合性のため）
    if (!exportPreviewMenuItem && contextMenu) {
        exportPreviewMenuItem = document.createElement('li');
        exportPreviewMenuItem.id = 'context-menu-export-preview';
        exportPreviewMenuItem.textContent = '画像としてエクスポート';
        exportPreviewMenuItem.style.display = 'none';
        contextMenu.querySelector('ul').appendChild(exportPreviewMenuItem);
    }

    memoEditorModal = document.getElementById('memo-editor');
    memoEditorHeader = document.getElementById('memo-editor-header');
    memoTextarea = document.getElementById('memo-editor-textarea');
    memoSaveBtn = document.getElementById('memo-editor-save');
    memoCancelBtn = document.getElementById('memo-editor-cancel');
    memoTooltip = document.getElementById('memo-tooltip');

    lightboxOverlay = document.getElementById('lightbox-overlay');
    lightboxContent = document.getElementById('lightbox-content');

    const commonPreviewArea = document.getElementById('common-card-preview');
    
    // プレビューエリアの右クリック処理
    if (commonPreviewArea) {
        commonPreviewArea.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            // メニュー項目をリセット（全て非表示）
            const allItems = contextMenu.querySelectorAll('li');
            allItems.forEach(li => li.style.display = 'none');
            const topItems = contextMenu.querySelectorAll('#custom-context-menu > ul > li');
            topItems.forEach(li => li.style.display = 'none');
            
            // エクスポートのみ表示
            if (exportPreviewMenuItem) {
                exportPreviewMenuItem.style.display = 'block';
            }
            
            currentPreviewExportHandler = () => exportCardPreviewAsImage();
            
            // メニュー表示位置調整
            contextMenu.style.display = 'block';
            contextMenu.style.visibility = 'hidden';
            contextMenu.style.zIndex = '10005'; 
            
            const menuWidth = contextMenu.offsetWidth;
            const menuHeight = contextMenu.offsetHeight;
            const windowWidth = window.innerWidth;
            const windowHeight = window.innerHeight;

            let left = e.pageX;
            let top = e.pageY;

            if (left + menuWidth > windowWidth) left -= menuWidth;
            if (top + menuHeight > windowHeight) top -= menuHeight;

            contextMenu.style.top = `${top}px`;
            contextMenu.style.left = `${left}px`;
            contextMenu.style.visibility = 'visible';
        });
    }

    commonDrawer = document.getElementById('common-drawer');
    commonDrawerToggle = document.getElementById('common-drawer-toggle');
    cDrawer = document.getElementById('c-drawer');
    cDrawerToggle = document.getElementById('c-drawer-toggle');

    commonFlipBoardBtn = document.getElementById('common-flip-board-btn');
    commonDecorationSettingsBtn = document.getElementById('common-decoration-settings-btn');
    commonToggleNavBtn = document.getElementById('common-toggle-nav-btn');
    commonToggleSeBtn = document.getElementById('common-toggle-se-btn');

    commonExportBoardBtn = document.getElementById('common-export-board-btn');
    commonImportBoardBtn = document.getElementById('common-import-board-btn');
    
    recordStartBtn = document.getElementById('record-start-btn');
    recordStopBtn = document.getElementById('record-stop-btn');
    replayPlayBtn = document.getElementById('replay-play-btn');
    replayPauseBtn = document.getElementById('replay-pause-btn');
    replayStopBtn = document.getElementById('replay-stop-btn');
    loadReplayBtn = document.getElementById('load-replay-btn');
    replayFileNameDisplay = document.getElementById('replay-file-name-display');
    replayFileNameText = document.getElementById('replay-file-name-text');
    replayWaitTimeInput = document.getElementById('replay-wait-time-input');

    diceRollBtn = document.getElementById('dice-roll-btn');
    coinTossBtn = document.getElementById('coin-toss-btn');
    randomResultDisplay = document.getElementById('random-result');

    flavorEditorModal = document.getElementById('flavor-editor');
    flavorEditorHeader = document.getElementById('flavor-editor-header');
    flavorPreview1 = document.getElementById('flavor-preview-1');
    flavorPreview2 = document.getElementById('flavor-preview-2');
    flavorDelete1 = document.getElementById('flavor-delete-1');
    flavorDelete2 = document.getElementById('flavor-delete-2');
    flavorDropZone1 = document.getElementById('flavor-drop-zone-1');
    flavorDropZone2 = document.getElementById('flavor-drop-zone-2');
    flavorUpload1 = document.getElementById('flavor-upload-1');
    flavorUpload2 = document.getElementById('flavor-upload-2');
    flavorCancelBtn = document.getElementById('flavor-editor-cancel');
    flavorSaveBtn = document.getElementById('flavor-editor-save-btn'); // 保存ボタン
    
    customCounterModal = document.getElementById('custom-counter-modal');
    // ヘッダー要素の取得 (HTMLのクラス構成に依存)
    const customCounterHeader = customCounterModal.querySelector('.custom-counter-header');
    customCounterCloseBtn = document.getElementById('custom-counter-close-btn');
    customCounterSaveBtn = document.getElementById('custom-counter-save-btn'); // 保存ボタン
    createCounterBtn = document.getElementById('create-counter-btn');
    newCounterNameInput = document.getElementById('new-counter-name');
    newCounterImageDrop = document.getElementById('new-counter-image-drop');
    customCounterListContainer = document.getElementById('custom-counter-list');

    decorationSettingsModal = document.getElementById('decoration-settings-modal');
    decorationSettingsHeader = document.getElementById('decoration-settings-header');
    decorationSettingsCloseBtn = document.getElementById('decoration-settings-close-btn');

    battleConfirmModal = document.getElementById('battle-confirm-modal');
    battleConfirmHeader = document.getElementById('battle-confirm-header');
    battleConfirmAttackerImg = document.getElementById('battle-confirm-attacker-img');
    battleConfirmTargetImg = document.getElementById('battle-confirm-target-img');
    battleConfirmAttackerBpInput = document.getElementById('battle-confirm-attacker-bp');
    battleConfirmTargetBpInput = document.getElementById('battle-confirm-target-bp');
    battleConfirmExecuteBtn = document.getElementById('battle-confirm-execute-btn');
    battleConfirmCancelBtn = document.getElementById('battle-confirm-cancel-btn');

    gameResultOverlay = document.getElementById('game-result-overlay');
    gameResultMessage = document.getElementById('game-result-message');
    gameResultCloseBtn = document.getElementById('game-result-close-btn');

    bgmSelect = document.getElementById('bgm-select');
    bgmPlayBtn = document.getElementById('bgm-play-btn');
    bgmPauseBtn = document.getElementById('bgm-pause-btn');
    bgmStopBtn = document.getElementById('bgm-stop-btn');
    bgmVolumeSlider = document.getElementById('bgm-volume-slider');
    bgmVolumeVal = document.getElementById('bgm-volume-val');
    seVolumeSlider = document.getElementById('se-volume-slider');
    seVolumeVal = document.getElementById('se-volume-val');

    seCheckAllBtn = document.getElementById('se-check-all');
    seUncheckAllBtn = document.getElementById('se-uncheck-all');
    effectCheckAllBtn = document.getElementById('effect-check-all');
    effectUncheckAllBtn = document.getElementById('effect-uncheck-all');
    autoCheckAllBtn = document.getElementById('auto-check-all');
    autoUncheckAllBtn = document.getElementById('auto-uncheck-all');
    autoConfigContainer = document.getElementById('auto-config-container');
    battleTargetOverlay = document.getElementById('battle-target-overlay');
    battleCancelBtn = document.getElementById('battle-cancel-btn');
    
    playerAutoDecreaseInput = document.getElementById('player-auto-decrease-interval');
    opponentAutoDecreaseInput = document.getElementById('opponent-auto-decrease-interval');
    
    shuffleDeckBtn = document.getElementById('shuffle-deck');
    opponentShuffleDeckBtn = document.getElementById('opponent-shuffle-deck');
    shuffleHandBtn = document.getElementById('shuffle-hand');
    opponentShuffleHandBtn = document.getElementById('opponent-shuffle-hand');
    
    systemBtn = document.getElementById('system-btn');
    opponentSystemBtn = document.getElementById('opponent-system-btn');

    cSearchFilter = document.getElementById('c-search-filter');

    if (!contextMenu) {
        console.error("必須UI要素が見つかりません。");
        return;
    }

    makeDraggable(battleConfirmHeader, battleConfirmModal);
    makeDraggable(memoEditorHeader, memoEditorModal);
    makeDraggable(decorationSettingsHeader, decorationSettingsModal);
    
    // カスタムカウンターとフレーバーイラスト設定画面のドラッグ化
    if(customCounterHeader) makeDraggable(customCounterHeader, customCounterModal);
    makeDraggable(flavorEditorHeader, flavorEditorModal);

    const seSettingsContainer = document.getElementById('se-settings-container');
    if (seSettingsContainer && typeof seConfig !== 'undefined') {
        seSettingsContainer.innerHTML = '';
        Object.keys(seConfig).forEach(seName => {
            const label = document.createElement('label');
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = seConfig[seName];
            input.dataset.seName = seName;
            input.addEventListener('change', (e) => {
                seConfig[seName] = e.target.checked;
            });
            
            const span = document.createElement('span');
            span.textContent = seName.replace('.mp3', '').replace('.wav', '');
            
            label.appendChild(input);
            label.appendChild(span);
            seSettingsContainer.appendChild(label);
        });
    }

    if (seCheckAllBtn) {
        seCheckAllBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            Object.keys(seConfig).forEach(key => seConfig[key] = true);
            const boxes = seSettingsContainer.querySelectorAll('input[type="checkbox"]');
            boxes.forEach(box => box.checked = true);
        });
    }
    if (seUncheckAllBtn) {
        seUncheckAllBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            Object.keys(seConfig).forEach(key => seConfig[key] = false);
            const boxes = seSettingsContainer.querySelectorAll('input[type="checkbox"]');
            boxes.forEach(box => box.checked = false);
        });
    }

    const effectSettingsContainer = document.getElementById('effect-settings-container');
    if (effectSettingsContainer && typeof effectConfig !== 'undefined') {
        effectSettingsContainer.innerHTML = '';
        const effectNames = {
            'masturbate': 'オナニー',
            'permanent': '常時発動',
            'attack': 'アタック',
            'effect': '効果発動',
            'target': '対象選択',
            'autoDecrease': '自動減少',
            'blocker': 'ブロッカー表示',
            'bpChange': 'BP変動演出'
        };
        Object.keys(effectConfig).forEach(key => {
            const label = document.createElement('label');
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = effectConfig[key];
            input.dataset.effectKey = key;
            input.addEventListener('change', (e) => {
                effectConfig[key] = e.target.checked;
            });
            
            const span = document.createElement('span');
            span.textContent = effectNames[key] || key;
            
            label.appendChild(input);
            label.appendChild(span);
            effectSettingsContainer.appendChild(label);
        });
    }

    if (effectCheckAllBtn) {
        effectCheckAllBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            Object.keys(effectConfig).forEach(key => effectConfig[key] = true);
            const boxes = effectSettingsContainer.querySelectorAll('input[type="checkbox"]');
            boxes.forEach(box => box.checked = true);
        });
    }
    if (effectUncheckAllBtn) {
        effectUncheckAllBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            Object.keys(effectConfig).forEach(key => effectConfig[key] = false);
            const boxes = effectSettingsContainer.querySelectorAll('input[type="checkbox"]');
            boxes.forEach(box => box.checked = false);
        });
    }

    if (autoConfigContainer && typeof autoConfig !== 'undefined') {
        autoConfigContainer.innerHTML = '';
        const autoConfigLabels = {
            'autoManaTap': 'マナタップ時自動+1',
            'autoManaPlacement': 'マナ配置時自動+1',
            'autoBattleCalc': 'バトル自動計算処理',
            'autoManaTapInZone': 'マナ配置時タップ状態',
            'autoAttackTap': 'アタック後タップ',
            'autoManaCost': 'カード配置時マナ消費',
            'autoGameEnd': '勝敗判定の自動表示',
            'drawFlipped': 'ドロー時裏側表示',
            'autoBpDestruction': 'BP0以下で自動破壊',
            'autoMasturbateDrain': 'オナニー中BP減少'
        };
        Object.keys(autoConfig).forEach(key => {
            const label = document.createElement('label');
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = autoConfig[key];
            input.dataset.autoKey = key;
            input.addEventListener('change', (e) => {
                autoConfig[key] = e.target.checked;
            });
            
            const span = document.createElement('span');
            span.textContent = autoConfigLabels[key] || key;
            
            label.appendChild(input);
            label.appendChild(span);
            autoConfigContainer.appendChild(label);
        });
    }

    if (autoCheckAllBtn) {
        autoCheckAllBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            Object.keys(autoConfig).forEach(key => autoConfig[key] = true);
            const boxes = autoConfigContainer.querySelectorAll('input[type="checkbox"]');
            boxes.forEach(box => box.checked = true);
        });
    }
    if (autoUncheckAllBtn) {
        autoUncheckAllBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            Object.keys(autoConfig).forEach(key => autoConfig[key] = false);
            const boxes = autoConfigContainer.querySelectorAll('input[type="checkbox"]');
            boxes.forEach(box => box.checked = false);
        });
    }

    lightboxOverlay.addEventListener('click', (e) => closeLightbox());
    lightboxContent.addEventListener('click', (e) => {
        if (e.target === lightboxContent) {
            closeLightbox();
        }
    });

    document.addEventListener('click', (e) => {
        if (isFileDialogOpen) {
            return;
        }
        if (isResizingDrawer) {
            return;
        }

        if (contextMenu.style.display === 'block' && !e.target.closest('#custom-context-menu')) {
            closeContextMenu();
        }

        const isInteractionTarget = 
            e.target.closest('#custom-context-menu') ||
            (memoEditorModal.style.display === 'flex' && e.target.closest('.memo-editor-modal')) || 
            (flavorEditorModal.style.display === 'block' && e.target.closest('.flavor-editor-modal')) ||
            (customCounterModal.style.display === 'block' && e.target.closest('.custom-counter-modal')) ||
            (decorationSettingsModal.style.display === 'flex' && e.target.closest('.custom-counter-modal')) || 
            (battleConfirmModal && battleConfirmModal.style.display === 'flex' && e.target.closest('.custom-counter-modal')) ||
            (gameResultOverlay.style.display === 'flex' && e.target.closest('.game-result-content')) ||
            e.target.closest('.drawer-toggle');

        if (isInteractionTarget) {
            return;
        }

        closeContextMenu();

        const clickedInsideCommon = e.target.closest('#common-drawer');

        if (commonDrawer.classList.contains('open') && !clickedInsideCommon) {
            commonDrawer.classList.remove('open');
        }

        if (decorationSettingsModal.style.display === 'flex' && e.target === decorationSettingsModal) {
            closeDecorationSettingsModal();
        }

        // Pカード、Oカード、バンク（c-drawer）は、それぞれの領域内または他のドロワーの領域内をクリックした場合は閉じない
        // つまり、「全てのドロワーの外側（盤面など）」をクリックした場合のみ閉じる
        const clickedInsideSideDrawers = e.target.closest('#player-drawer') || 
                                         e.target.closest('#opponent-drawer') || 
                                         e.target.closest('#c-drawer');

        if (!clickedInsideCommon) { 
            
            const playerDrawer = document.getElementById('player-drawer');
            if (playerDrawer && playerDrawer.classList.contains('open')) {
                // Pカードが開いている時、ドロワー群のいずれもクリックしていなければ閉じる
                if (!clickedInsideSideDrawers) {
                    playerDrawer.classList.remove('open');
                }
            }

            const opponentDrawer = document.getElementById('opponent-drawer');
            if (opponentDrawer && opponentDrawer.classList.contains('open')) {
                if (!clickedInsideSideDrawers) {
                    opponentDrawer.classList.remove('open');
                }
            }
            
            const cDrawer = document.getElementById('c-drawer');
            if (cDrawer && cDrawer.classList.contains('open')) {
                if (!clickedInsideSideDrawers) {
                    cDrawer.classList.remove('open');
                }
            }
        }
    });

    contextMenu.addEventListener('contextmenu', (e) => e.preventDefault());
    
    actionMenuItem.addEventListener('click', () => { 
        let isMana = false;
        let isSpell = false;
        
        if (lastRightClickedElement) {
            const zoneId = getParentZoneId(lastRightClickedElement);
            if (zoneId) {
                const baseId = getBaseId(zoneId);
                if (zoneId.includes('mana') || baseId.startsWith('mana')) {
                    isMana = true;
                } else if (baseId === 'spell') {
                    isSpell = true;
                }
            }
        }
        
        if (isMana) {
            // マナエリアならSEなし
        } else if (isSpell) {
            playSe('スペル効果発動.mp3');
        } else {
            playSe('効果発動.mp3');
        }

        if (typeof currentActionHandler === 'function') currentActionHandler(); 
        closeContextMenu(); 
    });
    targetMenuItem.addEventListener('click', () => { 
        playSe('対象に取る.mp3');
        if (typeof currentTargetHandler === 'function') currentTargetHandler(); 
        closeContextMenu(); 
    });
    
    attackMenuItem.addEventListener('click', () => {
        // playSe('ボタン共通.mp3'); // 削除: 他のSEとの重複回避
        startBattleTargetSelection(lastRightClickedElement);
        closeContextMenu();
    });

    permanentMenuItem.addEventListener('click', () => { 
        // playSe('ボタン共通.mp3'); // 削除: card.js側でON時のみ再生、OFF時はボタン共通音無しで統一
        if (typeof currentPermanentHandler === 'function') currentPermanentHandler(); 
        closeContextMenu(); 
    });
    blockerMenuItem.addEventListener('click', () => {
        playSe('ブロッカー.wav');
        if (typeof currentBlockerHandler === 'function') currentBlockerHandler();
        closeContextMenu();
    });

    deleteMenuItem.addEventListener('click', () => { 
        playSe('ボタン共通.mp3');
        if (typeof currentDeleteHandler === 'function') currentDeleteHandler(); 
        closeContextMenu(); 
    });
    toGraveMenuItem.addEventListener('click', () => { 
        playSe('墓地に送る.mp3');
        if (typeof currentMoveToGraveHandler === 'function') currentMoveToGraveHandler(); 
        closeContextMenu(); 
    });
    toExcludeMenuItem.addEventListener('click', () => { 
        playSe('除外する.mp3');
        if (typeof currentMoveToExcludeHandler === 'function') currentMoveToExcludeHandler(); 
        closeContextMenu(); 
    });
    toHandMenuItem.addEventListener('click', () => { 
        playSe('手札に戻す.mp3');
        if (typeof currentMoveToHandHandler === 'function') currentMoveToHandHandler(); 
        closeContextMenu(); 
    });
    toDeckMenuItem.addEventListener('click', () => { 
        playSe('ボタン共通.mp3');
        if (typeof currentMoveToDeckHandler === 'function') currentMoveToDeckHandler(); 
        closeContextMenu(); 
    });
    toSideDeckMenuItem.addEventListener('click', () => { 
        playSe('ボタン共通.mp3');
        if (typeof currentMoveToSideDeckHandler === 'function') currentMoveToSideDeckHandler(); 
        closeContextMenu(); 
    });
    flipMenuItem.addEventListener('click', () => { 
        playSe('カードを反転させる.wav');
        if (typeof currentFlipHandler === 'function') currentFlipHandler(); 
        closeContextMenu(); 
    });
    memoMenuItem.addEventListener('click', () => { 
        playSe('ボタン共通.mp3');
        if (typeof currentMemoHandler === 'function') currentMemoHandler(); 
        closeContextMenu(); 
    });
    
    if (setAsTopMenuItem) {
        setAsTopMenuItem.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            if (typeof currentStockItemTarget === 'function') currentStockItemTarget();
            closeContextMenu();
        });
    }

    addCounterMenuItem.addEventListener('click', () => { 
        playSe('カウンターを置く.mp3');
        if (typeof currentAddCounterHandler === 'function') currentAddCounterHandler(); 
        closeContextMenu(); 
    });
    removeCounterMenuItem.addEventListener('click', () => { 
        playSe('カウンターを取り除く.mp3');
        if (typeof currentRemoveCounterHandler === 'function') currentRemoveCounterHandler(); 
        closeContextMenu(); 
    });
    
    customCounterMenuItem.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        if (lastRightClickedElement) {
            openCustomCounterModal(lastRightClickedElement);
        }
        closeContextMenu();
    });

    if (duplicateMenuItem) {
        duplicateMenuItem.addEventListener('click', () => {
            // playSe('ボタン共通.mp3'); // 削除: 複製先で配置SEが鳴るため重複回避
            if (lastRightClickedElement) {
                duplicateCardToFreeSpace(lastRightClickedElement);
            }
            closeContextMenu();
        });
    }

    addFlavorMenuItem.addEventListener('click', () => { 
        playSe('ボタン共通.mp3');
        if (typeof currentAddFlavorHandler === 'function') currentAddFlavorHandler(); 
        closeContextMenu(); 
    });
    masturbateMenuItem.addEventListener('click', () => { 
        if (masturbateMenuItem.textContent === 'オナニーする') {
            playSe('オナニー.wav', true);
        } else {
            stopSe('オナニー.wav');
        }
        if (typeof currentMasturbateHandler === 'function') currentMasturbateHandler(); 
        closeContextMenu(); 
    });
    
    exportCardMenuItem.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        if (typeof currentExportCardHandler === 'function') currentExportCardHandler();
        closeContextMenu();
    });
    
    importCardMenuItem.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        if (typeof currentImportCardHandler === 'function') currentImportCardHandler();
        closeContextMenu();
    });

    changeStyleMenuItem.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        openDecorationSettingsModal();
        closeContextMenu();
    });
    
    // プレビューエクスポート
    if (exportPreviewMenuItem) {
        exportPreviewMenuItem.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            if (typeof currentPreviewExportHandler === 'function') currentPreviewExportHandler();
            closeContextMenu();
        });
    }

    const bpModifyBtns = document.querySelectorAll('.bp-modify-btn');
    bpModifyBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation(); 
            // playSe('ボタン共通.mp3'); // 削除: BP増減SEと重複するため
            const val = parseInt(btn.dataset.value);
            if (lastRightClickedElement && typeof modifyCardBP === 'function') {
                modifyCardBP(lastRightClickedElement, val);
                if (isBattleConfirmMode) {
                    updateBattleConfirmModal();
                }
            }
            closeContextMenu();
        });
    });

    if (battleConfirmExecuteBtn) {
        battleConfirmExecuteBtn.addEventListener('click', () => {
            // playSe('ボタン共通.mp3'); // 削除: resolveBattle内のアタックSEと重複するため
            if (typeof resolveBattle === 'function') {
                const aBp = parseInt(battleConfirmAttackerBpInput.value);
                const tBp = parseInt(battleConfirmTargetBpInput.value);
                resolveBattle(aBp, tBp);
            }
        });
    }
    if (battleConfirmCancelBtn) {
        battleConfirmCancelBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            closeBattleConfirmModal();
        });
    }

    memoTextarea.addEventListener('input', () => {
        if (currentMemoTarget) {
            currentMemoTarget.dataset.memo = memoTextarea.value;
            if (typeof window.updateCardPreview === 'function') {
                window.updateCardPreview(currentMemoTarget);
            }
        }
    });

    memoSaveBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        performMemoSave();
    });
    memoCancelBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        performMemoCancel();
    });

    document.addEventListener('mousemove', (e) => {
        if (memoTooltip.style.display === 'block') {
            memoTooltip.style.left = (e.pageX + 10) + 'px';
            memoTooltip.style.top = (e.pageY + 10) + 'px';
            const rect = memoTooltip.getBoundingClientRect();
            if (rect.right > window.innerWidth) memoTooltip.style.left = (e.pageX - rect.width - 10) + 'px';
            if (rect.bottom > window.innerHeight) memoTooltip.style.top = (e.pageY - rect.height - 10) + 'px';
        }
    });

    flavorCancelBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        closeFlavorEditor();
    });
    if(flavorSaveBtn) {
        flavorSaveBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            closeFlavorEditor();
        });
    }
    
    flavorDelete1.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        deleteFlavorImage(1);
    });
    flavorDelete2.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        deleteFlavorImage(2);
    });
    flavorUpload1.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        openFlavorFileInput(1);
    });
    flavorUpload2.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        openFlavorFileInput(2);
    });
    flavorDropZone1.addEventListener('drop', (e) => { e.preventDefault(); e.stopPropagation(); flavorDropZone1.classList.remove('drag-over'); const files = e.dataTransfer.files; if (files.length > 0) handleFlavorFile(files[0], 1); });
    flavorDropZone2.addEventListener('drop', (e) => { e.preventDefault(); e.stopPropagation(); flavorDropZone2.classList.remove('drag-over'); const files = e.dataTransfer.files; if (files.length > 0) handleFlavorFile(files[0], 2); });
    flavorDropZone1.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); flavorDropZone1.classList.add('drag-over'); });
    flavorDropZone2.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); flavorDropZone2.classList.add('drag-over'); });
    flavorDropZone1.addEventListener('dragleave', (e) => { e.preventDefault(); e.stopPropagation(); flavorDropZone1.classList.remove('drag-over'); });
    flavorDropZone2.addEventListener('dragleave', (e) => { e.preventDefault(); e.stopPropagation(); flavorDropZone2.classList.remove('drag-over'); });
    flavorDropZone1.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        openFlavorFileInput(1);
    });
    flavorDropZone2.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        openFlavorFileInput(2);
    });
    
    customCounterCloseBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        closeCustomCounterModal();
    });
    if(customCounterSaveBtn) {
        customCounterSaveBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            closeCustomCounterModal();
        });
    }
    
    createCounterBtn.addEventListener('click', () => {
        createNewCustomCounterType();
    });
    newCounterImageDrop.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); newCounterImageDrop.classList.add('drag-over'); });
    newCounterImageDrop.addEventListener('dragleave', (e) => { e.preventDefault(); e.stopPropagation(); newCounterImageDrop.classList.remove('drag-over'); });
    newCounterImageDrop.addEventListener('drop', (e) => {
        e.preventDefault(); e.stopPropagation();
        newCounterImageDrop.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleNewCounterImageFile(files[0]);
    });
    newCounterImageDrop.addEventListener('click', () => {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*';
        fileInput.style.display = 'none';
        fileInput.onchange = (e) => {
            if (e.target.files.length > 0) handleNewCounterImageFile(e.target.files[0]);
        };
        document.body.appendChild(fileInput);
        isFileDialogOpen = true;
        fileInput.click();
        document.body.removeChild(fileInput);
        setTimeout(() => { isFileDialogOpen = false; }, 100);
    });

    decorationSettingsCloseBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // 修正: 親要素（Common Menu）への伝播を防止
        playSe('ボタン共通.mp3');
        closeDecorationSettingsModal();
    });
    
    if (gameResultCloseBtn) {
        gameResultCloseBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            closeGameResult();
        });
    }
    
    if (shuffleHandBtn) {
        shuffleHandBtn.addEventListener('click', () => {
            playSe('シャッフル.mp3');
            if (typeof shuffleHand === 'function') shuffleHand('');
        });
    }
    if (opponentShuffleHandBtn) {
        opponentShuffleHandBtn.addEventListener('click', () => {
            playSe('シャッフル.mp3');
            if (typeof shuffleHand === 'function') shuffleHand('opponent-');
        });
    }
    
    // システムボタンのハンドラ (トグル化)
    if (systemBtn) {
        systemBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            // playSe('ボタン共通.mp3'); // 重複回避のため削除 (commonDrawerの開閉SEに任せるか、ここで鳴らすならtoggleのロジック次第)
            if (commonDrawer.classList.contains('open')) {
                commonDrawer.classList.remove('open');
            } else {
                playSe('ボタン共通.mp3'); // 開く時だけ鳴らす
                commonDrawer.classList.add('open');
                activateDrawerTab('common-spec-panel', commonDrawer);
            }
        });
    }
    if (opponentSystemBtn) {
        opponentSystemBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (commonDrawer.classList.contains('open')) {
                commonDrawer.classList.remove('open');
            } else {
                playSe('ボタン共通.mp3');
                commonDrawer.classList.add('open');
                activateDrawerTab('common-spec-panel', commonDrawer);
            }
        });
    }

    if (bgmSelect && typeof bgmFileList !== 'undefined') {
        bgmFileList.forEach(filename => {
            const option = document.createElement('option');
            option.value = filename;
            option.textContent = filename;
            bgmSelect.appendChild(option);
        });
    }

    if (bgmPlayBtn) {
        bgmPlayBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            if (bgmSelect) {
                const filename = bgmSelect.value;
                if (filename && typeof playBgm === 'function') {
                    playBgm(filename);
                }
            }
        });
    }
    if (bgmPauseBtn) {
        bgmPauseBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            if (typeof pauseBgm === 'function') pauseBgm();
        });
    }
    if (bgmStopBtn) {
        bgmStopBtn.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            if (typeof stopBgm === 'function') stopBgm();
        });
    }

    if (bgmVolumeSlider) {
        bgmVolumeSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            bgmVolumeVal.textContent = val;
            bgmVolume = val;
            if (typeof updateBgmVolume === 'function') updateBgmVolume();
        });
    }
    if (seVolumeSlider) {
        seVolumeSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            seVolumeVal.textContent = val;
            seVolume = val;
        });
    }

    const turnInput = document.getElementById('common-turn-value');
    const turnPrevBtn = document.getElementById('common-turn-prev');
    const turnNextBtn = document.getElementById('common-turn-next');
    
    if (turnInput && turnPrevBtn && turnNextBtn) {
        const updateTurnValue = (change) => {
            playSe('ボタン共通.mp3');
            let currentValue = parseInt(turnInput.value) || 1;
            currentValue = Math.max(1, currentValue + change);
            turnInput.value = currentValue;
            
            if (isRecording && typeof recordAction === 'function') {
                recordAction({ type: 'turnChange', value: currentValue });
            }
        };
        turnPrevBtn.addEventListener('click', () => updateTurnValue(-1));
        turnNextBtn.addEventListener('click', () => updateTurnValue(1));
    }
    
    const turnPlayerSelect = document.getElementById('turn-player-select');
    if (turnPlayerSelect) {
        turnPlayerSelect.addEventListener('change', () => {
             if (isRecording && typeof recordAction === 'function') {
                recordAction({ type: 'turnPlayerChange', value: turnPlayerSelect.value });
            }
        });
    }

    commonDrawerToggle.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        commonDrawer.classList.toggle('open');
    });
    
    if (cDrawerToggle) {
        cDrawerToggle.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            cDrawer.classList.toggle('open');
        });
    }

    commonFlipBoardBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        document.body.classList.toggle('board-flipped');
        document.getElementById('player-drawer')?.classList.remove('open');
        document.getElementById('opponent-drawer')?.classList.remove('open');
    });

    commonDecorationSettingsBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        openDecorationSettingsModal();
    });
    
    const openPlayerDrawerBtn = document.getElementById('common-open-player-drawer');
    const openOpponentDrawerBtn = document.getElementById('common-open-opponent-drawer');
    const openBankDrawerBtn = document.getElementById('common-open-bank-drawer');

    // Drawer Toggle Logic Helper
    const toggleDrawerWithTab = (drawerId, tabId) => {
        const drawer = document.getElementById(drawerId);
        if (drawer) {
            // 既に開いていて、かつ指定のタブが表示されている場合は閉じる
            const activePanel = drawer.querySelector('.drawer-panel.active');
            if (drawer.classList.contains('open') && activePanel && activePanel.id === tabId) {
                drawer.classList.remove('open');
            } else {
                // それ以外（閉じている、または別のタブが開いている）は開いてタブをアクティブ化
                drawer.classList.add('open');
                activateDrawerTab(tabId, drawer);
            }
        }
    };

    if (openPlayerDrawerBtn) {
        openPlayerDrawerBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            playSe('ボタン共通.mp3');
            toggleDrawerWithTab('player-drawer', 'deck-back-slots');
        });
    }
    if (openOpponentDrawerBtn) {
        openOpponentDrawerBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            playSe('ボタン共通.mp3');
            toggleDrawerWithTab('opponent-drawer', 'opponent-deck-back-slots');
        });
    }
    if (openBankDrawerBtn) {
        openBankDrawerBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            playSe('ボタン共通.mp3');
            toggleDrawerWithTab('c-drawer', 'c-free-space');
        });
    }

    commonExportBoardBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        const defaultName = "sm_solitaire_board";
        const fileName = prompt("保存するファイル名を入力してください", defaultName);
        if (fileName) {
             if (typeof exportAllBoardData === 'function') exportAllBoardData(fileName);
        }
    });
    commonImportBoardBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        if (typeof importAllBoardData === 'function') importAllBoardData();
    });

    recordStartBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        if (typeof startReplayRecording === 'function') startReplayRecording();
    });
    recordStopBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        if (typeof stopReplayRecording === 'function') stopReplayRecording();
    });
    replayPlayBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        if (isReplayPaused) {
            if (typeof resumeReplay === 'function') resumeReplay();
        } else {
            if (typeof playReplay === 'function') playReplay();
        }
    });
    replayPauseBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        if (typeof pauseReplay === 'function') pauseReplay();
    });
    replayStopBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        if (typeof stopReplay === 'function') stopReplay();
    });
    
    loadReplayBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        if (typeof importReplayData === 'function') importReplayData();
    });

    diceRollBtn.addEventListener('click', () => {
        playSe('サイコロ.mp3');
        const result = Math.floor(Math.random() * 6) + 1;
        randomResultDisplay.textContent = `ダイス: ${result}`;
        if (isRecording && typeof recordAction === 'function') {
            recordAction({ type: 'dice', result: result });
        }
    });

    coinTossBtn.addEventListener('click', () => {
        playSe('コイントス.mp3');
        const result = Math.random() < 0.5 ? 'ウラ' : 'オモテ';
        randomResultDisplay.textContent = `コイン: ${result}`;
        if (isRecording && typeof recordAction === 'function') {
            recordAction({ type: 'coin', result: result });
        }
    });

    commonToggleNavBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        const isHidden = document.body.classList.toggle('nav-hidden');
        commonToggleNavBtn.textContent = isHidden ? 'ナビ再表示' : 'ナビ非表示';
    });
    
    if (battleCancelBtn) {
        battleCancelBtn.addEventListener('click', () => {
            // playSe('ボタン共通.mp3'); // 削除: 他のSEと重なる可能性があるため（バトル中断時の混乱を防ぐ）
            if (typeof cancelBattleTargetSelection === 'function') cancelBattleTargetSelection();
        });
    }

    setupStepButtons();

    setupHorizontalScroll();

    const allDrawerTabs = document.querySelectorAll('.drawer-tab-btn');
    allDrawerTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            playSe('ボタン共通.mp3');
            const targetId = tab.dataset.target;
            const drawer = tab.closest('.drawer-wrapper');
            activateDrawerTab(targetId, drawer);
        });
    });

    const playerDrawer = document.getElementById('player-drawer');
    if (playerDrawer) activateDrawerTab('deck-back-slots', playerDrawer);
    const opponentDrawer = document.getElementById('opponent-drawer');
    if (opponentDrawer) activateDrawerTab('opponent-deck-back-slots', opponentDrawer);

    if (commonDrawer) activateDrawerTab('common-general-panel', commonDrawer);
    if (cDrawer) activateDrawerTab('c-free-space', cDrawer);

    setupDrawerResize();

    const commonDrawerHeader = document.getElementById('common-drawer-header');
    if (commonDrawer && commonDrawerHeader) {
        let isDragging = false;
        let currentX;
        let currentY;
        let initialX;
        let initialY;
        let xOffset = 0;
        let yOffset = 0;

        commonDrawerHeader.addEventListener("mousedown", dragStart);
        document.addEventListener("mouseup", dragEnd);
        document.addEventListener("mousemove", drag);

        function dragStart(e) {
            if (e.target.classList.contains('resize-handle') || isResizingDrawer) return;
            
            initialX = e.clientX - xOffset;
            initialY = e.clientY - yOffset;

            if (e.target === commonDrawerHeader || e.target.parentNode === commonDrawerHeader) {
                isDragging = true;
                e.preventDefault(); 
            }
        }

        function dragEnd(e) {
            initialX = currentX;
            initialY = currentY;
            isDragging = false;
        }

        function drag(e) {
            if (isDragging) {
                e.preventDefault();
                
                let newX = e.clientX - initialX;
                let newY = e.clientY - initialY;

                const drawerWidth = commonDrawer.offsetWidth;
                const drawerHeight = commonDrawer.offsetHeight;
                const vw = window.innerWidth;
                const vh = window.innerHeight;

                const limitX = Math.max(0, (vw - drawerWidth) / 2);
                const limitY = Math.max(0, (vh - drawerHeight) / 2);

                if (newX < -limitX) newX = -limitX;
                if (newX > limitX) newX = limitX;
                
                if (newY < -limitY) newY = -limitY;
                if (newY > limitY) newY = limitY;

                currentX = newX;
                currentY = newY;

                xOffset = currentX;
                yOffset = currentY;

                commonDrawer.style.transform = `translate(calc(-50% + ${currentX}px), calc(-50% + ${currentY}px))`;
            }
        }
        
        const closeHint = commonDrawerHeader.querySelector('.drawer-close-hint');
        if(closeHint) {
            closeHint.addEventListener('click', (e) => {
                e.stopPropagation();
                playSe('ボタン共通.mp3');
                commonDrawer.classList.remove('open');
            });
        }
    }
    
    document.body.addEventListener('click', (e) => {
        if (!isRecording) return;
        const btn = e.target.closest('.counter-btn');
        if (!btn) return;
        
        if (btn.id === 'dice-roll-btn' || btn.id === 'coin-toss-btn') return; 
        
        if (btn.id && btn.id.includes('auto-decrease')) {
             if (typeof recordAction === 'function') {
                 recordAction({ type: 'autoDecreaseToggle', id: btn.id });
             }
             return;
        }

        if (btn.dataset.value) {
            const group = btn.closest('.hand-counter-group');
            if (group) {
                const input = group.querySelector('input');
                if (input) {
                     if (typeof recordAction === 'function') {
                         recordAction({ 
                             type: 'counterChange', 
                             inputId: input.id, 
                             change: parseInt(btn.dataset.value) 
                         });
                     }
                }
            }
        }
    });

    initializeTokens();
    
    if (cSearchFilter) {
        cSearchFilter.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            const container = document.getElementById('c-free-space');
            if (!container) return;
            const thumbnails = container.querySelectorAll('.thumbnail');
            thumbnails.forEach(thumb => {
                const memo = (thumb.dataset.memo || '').toLowerCase();
                const slot = thumb.closest('.card-slot');
                if (slot) {
                    if (memo.includes(query)) {
                        slot.style.display = ''; 
                    } else {
                        slot.style.display = 'none';
                    }
                }
            });
            
            if (!query) {
                 const allSlots = container.querySelectorAll('.card-slot');
                 allSlots.forEach(s => s.style.display = '');
            }
        });
    }
}

function startBattleTargetSelection(attackerThumbnail) {
    if (!attackerThumbnail) return;
    isBattleTargetMode = true;
    currentAttacker = attackerThumbnail;
    document.body.classList.add('battle-target-mode');
    battleTargetOverlay.style.display = 'flex';
    attackerThumbnail.classList.add('battle-attacker');

    const allThumbnails = document.querySelectorAll('.card-slot .thumbnail');
    allThumbnails.forEach(thumb => {
        const zone = getParentZoneId(thumb.parentNode);
        const base = getBaseId(zone);
        
        if (['deck', 'grave', 'exclude', 'side-deck', 'hand-zone', 'token-zone-slots', 'c-free-space'].includes(base)) return;
        
        thumb.classList.add('battle-target-candidate');
    });

    const iconZone = document.getElementById('icon-zone');
    const oppIconZone = document.getElementById('opponent-icon-zone');
    
    if(iconZone) {
        const playerIconSlot = iconZone.closest('.player-icon-slot');
        if(playerIconSlot) playerIconSlot.classList.add('battle-target-candidate');
    }
    if(oppIconZone) {
        const oppIconSlot = oppIconZone.closest('.player-icon-slot');
        if(oppIconSlot) oppIconSlot.classList.add('battle-target-candidate');
    }
}

window.cancelBattleTargetSelection = function() {
    isBattleTargetMode = false;
    document.body.classList.remove('battle-target-mode');
    battleTargetOverlay.style.display = 'none';
    if (currentAttacker) {
        currentAttacker.classList.remove('battle-attacker');
        currentAttacker = null;
    }

    const candidates = document.querySelectorAll('.battle-target-candidate');
    candidates.forEach(el => el.classList.remove('battle-target-candidate'));
}

function activateDrawerTab(targetId, drawerElement) {
    if (!drawerElement) return;
    const drawerPanels = drawerElement.querySelectorAll('.drawer-panel');
    const drawerTabs = drawerElement.querySelectorAll('.drawer-tab-btn');
    
    drawerPanels.forEach(p => p.classList.toggle('active', p.id === targetId));
    drawerTabs.forEach(t => t.classList.toggle('active', t.dataset.target === targetId));

    if (targetId === 'common-spec-panel') {
        loadTextContent('txt/機能説明.txt', 'spec-text-content');
    } else if (targetId === 'common-about-panel') {
        loadTextContent('txt/S＆Mとは.txt', 'about-text-content');
    } else if (targetId === 'common-credit-panel') {
        loadTextContent('txt/クレジット.txt', 'credit-text-content');
    }
}

async function loadTextContent(filePath, elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    if (!element.textContent.includes('読み込み中...')) return;

    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const text = await response.text();
        element.textContent = text;
    } catch (error) {
        element.textContent = `読み込みに失敗しました:\n${error.message}\n\n(ローカル環境の場合、ブラウザのセキュリティ制限によりテキストファイルを読み込めない場合があります。Webサーバー経由で実行してください)`;
        console.error("Text load failed:", error);
    }
}

function setupHorizontalScroll() {
    const scrollableContainers = document.querySelectorAll('.deck-back-slot-container, .free-space-slot-container, .token-slot-container');

    scrollableContainers.forEach(container => {
        container.addEventListener('wheel', (e) => {
            if (container.scrollWidth <= container.clientWidth) {
                return;
            }
            e.preventDefault();
            container.scrollLeft += e.deltaY;
        });
    });
}

function setupDrawerResize() {
    const drawer = document.getElementById('common-drawer');
    const handle = drawer ? drawer.querySelector('.resize-handle') : null;
    
    if (!drawer || !handle) return;

    let isResizing = false;
    let startW, startH, startX, startY;

    handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        e.stopPropagation();
        isResizing = true;
        isResizingDrawer = true;
        startW = drawer.offsetWidth;
        startH = drawer.offsetHeight;
        startX = e.clientX;
        startY = e.clientY;
        
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    });

    function handleMouseMove(e) {
        if (!isResizing) return;
        
        let newW = startW + (e.clientX - startX);
        let newH = startH + (e.clientY - startY);

        const rect = drawer.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        
        const maxWidth = 2 * Math.min(centerX, vw - centerX);
        const maxHeight = 2 * Math.min(centerY, vh - centerY);
        
        newW = Math.max(500, Math.min(newW, maxWidth));
        newH = Math.max(400, Math.min(newH, maxHeight));
        
        drawer.style.width = `${newW}px`;
        drawer.style.height = `${newH}px`;
    }

    function handleMouseUp(e) {
        isResizing = false;
        setTimeout(() => {
            isResizingDrawer = false;
        }, 100);

        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
    }
}

window.updateSettingsUIFromState = function() {
    if (bgmVolumeSlider && bgmVolumeVal) {
        bgmVolumeSlider.value = bgmVolume;
        bgmVolumeVal.textContent = bgmVolume;
    }
    if (seVolumeSlider && seVolumeVal) {
        seVolumeSlider.value = seVolume;
        seVolumeVal.textContent = seVolume;
    }
    
    const seContainer = document.getElementById('se-settings-container');
    if (seContainer && typeof seConfig !== 'undefined') {
        const checkboxes = seContainer.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(box => {
            const seName = box.dataset.seName;
            if (seName && typeof seConfig[seName] !== 'undefined') {
                box.checked = seConfig[seName];
            }
        });
    }
    
    const effectContainer = document.getElementById('effect-settings-container');
    if (effectContainer && typeof effectConfig !== 'undefined') {
        const checkboxes = effectContainer.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(box => {
            const key = box.dataset.effectKey;
            if (key && typeof effectConfig[key] !== 'undefined') {
                box.checked = effectConfig[key];
            }
        });
    }

    const autoContainer = document.getElementById('auto-config-container');
    if (autoContainer && typeof autoConfig !== 'undefined') {
        const checkboxes = autoContainer.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(box => {
            const key = box.dataset.autoKey;
            if (key && typeof autoConfig[key] !== 'undefined') {
                box.checked = autoConfig[key];
            }
        });
    }
};

function duplicateCardToFreeSpace(sourceCard) {
    const freeSpaceContainer = document.getElementById('free-space-slots');
    if (!freeSpaceContainer) return;
    
    const slotsContainer = freeSpaceContainer.querySelector('.free-space-slot-container');
    if (!slotsContainer) return;

    const emptySlot = Array.from(slotsContainer.querySelectorAll('.card-slot')).find(s => !s.querySelector('.thumbnail'));
    
    if (!emptySlot) {
        alert("フリースペースに空きがありません。");
        return;
    }

    const imgElement = sourceCard.querySelector('.card-image');
    const cardData = {
        src: imgElement ? imgElement.src : '',
        memo: sourceCard.dataset.memo || '',
        flavor1: sourceCard.dataset.flavor1 || '',
        flavor2: sourceCard.dataset.flavor2 || '',
        ownerPrefix: '', 
        customCounters: JSON.parse(sourceCard.dataset.customCounters || '[]'),
        isFlipped: false, 
        rotation: 0
    };

    if (typeof createCardThumbnail === 'function') {
        createCardThumbnail(cardData, emptySlot, false, false, '');
        if (typeof updateSlotStackState === 'function') updateSlotStackState(emptySlot);
        playSe('カードを配置する.mp3');
        
        // 自動オープン処理 (トグルではないので直接操作)
        const drawer = document.getElementById('player-drawer');
        if (drawer) {
            drawer.classList.add('open');
            activateDrawerTab('free-space-slots', drawer);
        }
    }
}