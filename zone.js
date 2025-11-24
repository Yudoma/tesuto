const DEFAULT_CARD_MEMO = `[カード名:-]/#e0e0e0/#555/1.0/非表示/
[属性:-]/#e0e0e0/#555/1.0/非表示/
[マナ:-]/#e0e0e0/#555/1.0/非表示/
[BP:-]/#e0e0e0/#555/1.0/非表示/
[スペル:-]/#e0e0e0/#555/1.0/非表示/
[フレーバーテキスト:-]/#fff/#555/1.0/非表示/
[効果:-]/#e0e0e0/#555/0.7/非表示/`;

// 配置SE再生ヘルパー
function playPlacementSe(baseZoneId) {
    if (baseZoneId === 'spell') {
        playSe('スペル配置.mp3');
    } else if (baseZoneId === 'battle') {
        playSe('バトル配置.mp3');
    } else if (baseZoneId === 'special1' || baseZoneId === 'special2') {
        playSe('特殊配置.mp3');
    } else if (baseZoneId && baseZoneId.startsWith('mana')) {
        playSe('マナ配置.mp3');
    } else {
        playSe('カードを配置する.mp3');
    }
}

// マナコスト自動消費ヘルパー
function tryAutoManaCost(memo, idPrefix) {
    if (typeof autoConfig === 'undefined' || !autoConfig.autoManaCost) return;
    if (!memo) return;
    
    // [マナ:数字] または [コスト:数字] を検索
    const match = memo.match(/\[(?:マナ|コスト):([0-9]+)\]/);
    if (match) {
        const cost = parseInt(match[1]);
        if (cost > 0) {
            const manaInput = document.getElementById(idPrefix + 'mana-counter-value');
            if (manaInput) {
                const current = parseInt(manaInput.value) || 0;
                // マイナスにはしない
                const change = -Math.min(current, cost);
                
                if (change !== 0) {
                    manaInput.value = current + change;
                    if (isRecording && typeof recordAction === 'function') {
                        recordAction({
                            type: 'counterChange',
                            inputId: idPrefix + 'mana-counter-value',
                            change: change
                        });
                    }
                }
            }
        }
    }
}

// マナ配置時自動タップヘルパー
function tryAutoManaTapIn(slotElement, idPrefix, zoneId) {
    if (typeof autoConfig === 'undefined' || !autoConfig.autoManaTapInZone) return;
    
    const card = slotElement.querySelector('.thumbnail');
    if (!card) return;
    
    const img = card.querySelector('.card-image');
    if (!img) return;

    // 既に回転していなければ回転させる
    const currentRotation = parseInt(img.dataset.rotation) || 0;
    if (currentRotation === 0) {
        const newRotation = 90;
        slotElement.classList.add('rotated-90');
        const { width, height } = getCardDimensions();
        const scaleFactor = height / width;
        img.style.transform = `rotate(${newRotation}deg) scale(${scaleFactor})`;
        img.dataset.rotation = newRotation;
        
        if (isRecording && typeof recordAction === 'function') {
            recordAction({
                type: 'rotate',
                zoneId: zoneId,
                slotIndex: Array.from(slotElement.parentNode.children).indexOf(slotElement),
                rotation: newRotation
            });
        }
    }
}

function addSlotEventListeners(slot) {
    slot.addEventListener('dragover', handleDragOver);
    slot.addEventListener('dragleave', handleDragLeave);
    slot.addEventListener('drop', handleDropOnSlot);
    slot.addEventListener('click', handleSlotClick);
    slot.addEventListener('contextmenu', handleSlotContextMenu);
}

function handleSlotContextMenu(e) {
    // バトルターゲット選択モード中はメニューを出さない
    if (typeof isBattleTargetMode !== 'undefined' && isBattleTargetMode) {
        e.preventDefault();
        e.stopPropagation();
        return;
    }
    
    const slot = e.currentTarget;
    // カードがある場合はカード側のコンテキストメニューが出るので何もしない
    if (slot.querySelector('.thumbnail')) return;

    // 装飾対象ゾーン（デッキ、墓地、除外、EX、アイコン）かどうか判定
    const zoneId = getParentZoneId(slot);
    const baseZoneId = getBaseId(zoneId);
    const isDecorationZone = ['deck', 'grave', 'exclude', 'side-deck', 'icon-zone'].includes(baseZoneId);

    e.preventDefault();
    e.stopPropagation();

    if (typeof contextMenu === 'undefined') return;

    // メニュー項目の表示制御: 装飾対象なら「スタイルの変更」、それ以外は「インポート」を表示
    Array.from(contextMenu.querySelectorAll('li')).forEach(li => li.style.display = 'none');
    
    // 親メニュー（has-submenu）も非表示にする
    const topItems = contextMenu.querySelectorAll('#custom-context-menu > ul > li');
    topItems.forEach(li => li.style.display = 'none');

    if (isDecorationZone) {
        const changeStyleItem = document.getElementById('context-menu-change-style');
        if (changeStyleItem) changeStyleItem.style.display = 'block';
    } else {
        const importItem = document.getElementById('context-menu-import');
        if (importItem) {
            importItem.style.display = 'block';
            // ui.js の currentImportCardHandler を設定
            currentImportCardHandler = () => importCardToSlot(slot);
        }
    }

    // メニュー表示と位置計算
    contextMenu.style.display = 'block';
    contextMenu.style.visibility = 'hidden';
    
    // サブメニュー展開クラスのリセット
    const submenus = contextMenu.querySelectorAll('.submenu');
    submenus.forEach(sub => sub.classList.remove('open-left', 'open-top'));

    const menuWidth = contextMenu.offsetWidth;
    const menuHeight = contextMenu.offsetHeight;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;

    let left = e.pageX;
    let top = e.pageY;

    // 右にはみ出す場合
    if (left + menuWidth > windowWidth) {
        left -= menuWidth;
    }

    // 下にはみ出す場合
    if (top + menuHeight > windowHeight) {
        top -= menuHeight;
    }

    contextMenu.style.top = `${top}px`;
    contextMenu.style.left = `${left}px`;
    contextMenu.style.visibility = 'visible';
    
    lastRightClickedElement = slot;
}

function importCardToSlot(slot) {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json, application/json';
    fileInput.style.display = 'none';
    
    fileInput.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const cardData = JSON.parse(event.target.result);
                const zoneId = getParentZoneId(slot);
                const idPrefix = getPrefixFromZoneId(zoneId);
                const baseZoneId = getBaseId(zoneId);
                const owner = (baseZoneId === 'c-free-space') ? '' : idPrefix;
                
                // インポート先の所有権を適用
                cardData.ownerPrefix = owner;
                
                createCardThumbnail(cardData, slot, false, false, owner);
                updateSlotStackState(slot);
                
                const isMana = baseZoneId.startsWith('mana');
                
                // 自動マナ消費 (インポート時も一応適用)
                const targetZonesForCost = ['special1', 'battle', 'special2', 'spell', 'mana', 'mana-left', 'mana-right'];
                const isTargetZone = targetZonesForCost.some(z => baseZoneId.includes(z));
                if (isTargetZone) {
                    tryAutoManaCost(cardData.memo, idPrefix);
                }

                if (isMana) {
                    playSe('マナ配置.mp3');
                    // 自動処理設定チェック: マナ配置時+1
                    if (autoConfig.autoManaPlacement) {
                        const manaInput = document.getElementById(idPrefix + 'mana-counter-value');
                        if (manaInput) {
                            manaInput.value = parseInt(manaInput.value || 0) + 1;
                            if (isRecording && typeof recordAction === 'function') {
                                recordAction({
                                    type: 'counterChange',
                                    inputId: idPrefix + 'mana-counter-value',
                                    change: 1
                                });
                            }
                        }
                    }
                    // 自動処理設定チェック: マナ配置時タップイン
                    tryAutoManaTapIn(slot, idPrefix, zoneId);
                } else {
                    playPlacementSe(baseZoneId);
                }
                
                if (isRecording && typeof recordAction === 'function') {
                    recordAction({
                        type: 'newCard',
                        zoneId: zoneId,
                        slotIndex: Array.from(slot.parentNode.children).indexOf(slot),
                        cardData: cardData
                    });
                }
                
                if (zoneId.endsWith('-back-slots') || baseZoneId === 'c-free-space') {
                    arrangeSlots(zoneId);
                }
                
            } catch (err) {
                console.error("Import failed:", err);
                alert("カードの読み込みに失敗しました。");
            }
        };
        reader.readAsText(file);
    };
    
    document.body.appendChild(fileInput);
    isFileDialogOpen = true;
    fileInput.click();
    document.body.removeChild(fileInput);
    setTimeout(() => { isFileDialogOpen = false; }, 100);
}

function handleSlotClick(e) {
    const slot = e.currentTarget;
    
    // --- バトルターゲット選択モード ---
    if (typeof isBattleTargetMode !== 'undefined' && isBattleTargetMode) {
        e.stopPropagation();
        
        // ターゲットが有効かチェック（カードがある、またはプレイヤーアイコンである）
        const isPlayerIcon = slot.id === 'icon-zone' || slot.id === 'opponent-icon-zone';
        const hasCard = slot.querySelector('.thumbnail');

        if (hasCard || isPlayerIcon) {
            // バトル確認画面を開く
            if (typeof openBattleConfirmModal === 'function') {
                openBattleConfirmModal(currentAttacker, slot);
            }
        }
        return;
    }
    // ----------------------------------

    const parentZoneId = getParentZoneId(slot);
    const baseParentZoneId = getBaseId(parentZoneId);

    const drawerOpeningZones = ['deck', 'grave', 'exclude', 'side-deck'];
    
    if (drawerOpeningZones.includes(baseParentZoneId)) {
        // 通常のクリックでは何もしない（board.js側でドロワーを開く処理がある）
        return;
    }

    if (memoEditorModal.style.display === 'flex' || contextMenu.style.display === 'block' || flavorEditorModal.style.display === 'block') {
        return;
    }

    // カードがある場合は、カード側のクリックイベントに任せる
    if (slot.querySelector('.thumbnail')) {
        return;
    }

    const allowedZonesForNormalModeFileDrop = [
        'hand-zone', 'battle', 'spell', 'special1', 'special2', 'free-space-slots',
        'deck-back-slots', 'grave-back-slots', 'exclude-back-slots', 'side-deck-back-slots', 'token-zone-slots',
        'c-free-space'
    ];

    if (allowedZonesForNormalModeFileDrop.includes(baseParentZoneId) || (baseParentZoneId && baseParentZoneId.startsWith('mana'))) {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*';
        fileInput.style.display = 'none';
        fileInput.onchange = (event) => {
            try {
                const file = event.target.files[0];
                if (!file) {
                    document.body.removeChild(fileInput);
                    return;
                }
                const reader = new FileReader();
                reader.onload = (readEvent) => {
                    const idPrefix = getPrefixFromZoneId(getParentZoneId(slot));
                    const imageData = readEvent.target.result;
                    const owner = (baseParentZoneId === 'c-free-space') ? '' : idPrefix;
                    
                    // デフォルトメモを適用してカード作成
                    createCardThumbnail({
                        src: imageData,
                        memo: DEFAULT_CARD_MEMO,
                        ownerPrefix: owner
                    }, slot, false, false, owner);
                    
                    if (baseParentZoneId === 'c-free-space') {
                        const thumb = slot.querySelector('.thumbnail');
                        if (thumb) delete thumb.dataset.ownerPrefix;
                    }

                    updateSlotStackState(slot);
                    
                    // 自動マナ消費 (ファイルからの新規作成時)
                    const targetZonesForCost = ['special1', 'battle', 'special2', 'spell', 'mana', 'mana-left', 'mana-right'];
                    const isTargetZone = targetZonesForCost.some(z => baseParentZoneId.includes(z));
                    if (isTargetZone) {
                        tryAutoManaCost(DEFAULT_CARD_MEMO, idPrefix);
                    }

                    const isMana = baseParentZoneId.startsWith('mana');
                    if (isMana) {
                        playSe('マナ配置.mp3');
                        // 自動処理設定チェック
                        if (autoConfig.autoManaPlacement) {
                            const manaInput = document.getElementById(idPrefix + 'mana-counter-value');
                            if (manaInput) {
                                manaInput.value = parseInt(manaInput.value || 0) + 1;
                                if (isRecording && typeof recordAction === 'function') {
                                    recordAction({
                                        type: 'counterChange',
                                        inputId: idPrefix + 'mana-counter-value',
                                        change: 1
                                    });
                                }
                            }
                        }
                        // 自動処理設定チェック: マナ配置時タップイン
                        tryAutoManaTapIn(slot, idPrefix, parentZoneId);
                    } else {
                        playPlacementSe(baseParentZoneId);
                    }

                    if (isRecording && typeof recordAction === 'function') {
                        recordAction({
                            type: 'newCard',
                            zoneId: getParentZoneId(slot),
                            slotIndex: Array.from(slot.parentNode.children).indexOf(slot),
                            cardData: {
                                src: imageData,
                                memo: DEFAULT_CARD_MEMO
                            }
                        });
                    }
                };
                reader.readAsDataURL(file);
            } catch (error) {
                console.error("File read failed:", error);
            } finally {
                 if (document.body.contains(fileInput)) {
                    document.body.removeChild(fileInput);
                }
            }
        };
        fileInput.oncancel = () => {
            if (document.body.contains(fileInput)) {
                document.body.removeChild(fileInput);
            }
        };
        document.body.appendChild(fileInput);
        
        isFileDialogOpen = true;
        fileInput.click();
        setTimeout(() => { isFileDialogOpen = false; }, 100);

        e.stopPropagation();
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.types.includes('Files')) {
        e.dataTransfer.dropEffect = 'copy';
    } else {
        e.dataTransfer.dropEffect = 'move';
    }
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleDropOnSlot(e) {
    e.preventDefault();
    e.stopPropagation();
    const slot = e.currentTarget;
    slot.classList.remove('drag-over');

    const idPrefix = getPrefixFromZoneId(getParentZoneId(slot));

    // ストック装飾からのドロップ判定
    if (typeof draggedItem !== 'undefined' && draggedItem && draggedItem.dataset.isStockDecoration === 'true') {
        handleStockDecorationDrop(draggedItem, slot, idPrefix);
        return;
    }

    if (e.dataTransfer.files.length > 0) {
        handleFileDrop(e, slot, idPrefix);
        return;
    }

    if (typeof draggedItem !== 'undefined' && draggedItem) {
        handleCardDrop(draggedItem, slot, idPrefix);
    }
}

function handleStockDecorationDrop(stockItem, targetSlot, idPrefix) {
    const img = stockItem.querySelector('img');
    if (!img) return;
    const imageData = img.src;
    
    const targetParentZoneId = getParentZoneId(targetSlot);
    const targetParentBaseId = getBaseId(targetParentZoneId);
    
    // アイコンゾーン、またはデッキ等の装飾対象ゾーンのみ許可
    const validTargets = ['icon-zone', 'deck', 'grave', 'exclude', 'side-deck'];
    if (!validTargets.includes(targetParentBaseId)) return;

    // アイコンゾーンならデフォルトメモを設定
    let memoToSet = '';
    if (targetParentBaseId === 'icon-zone') {
        memoToSet = DEFAULT_CARD_MEMO;
    }

    const existingThumbnail = targetSlot.querySelector('.thumbnail[data-is-decoration="true"]');
    if (existingThumbnail) {
        const existingImg = existingThumbnail.querySelector('img');
        if (existingImg) existingImg.src = imageData;
        if (memoToSet) existingThumbnail.dataset.memo = memoToSet;
    } else {
        // 既存の通常サムネイルがあれば削除 (装飾モードなので置き換え)
        const anyExistingThumbnail = getExistingThumbnail(targetSlot);
        if (anyExistingThumbnail) targetSlot.removeChild(anyExistingThumbnail);
        
        createCardThumbnail({
            src: imageData,
            isDecoration: true,
            memo: memoToSet,
            ownerPrefix: idPrefix
        }, targetSlot, true, false, idPrefix);
    }

    playSe('カードを配置する.mp3');
    
    if (targetParentBaseId !== 'icon-zone') {
        syncMainZoneImage(targetParentBaseId, idPrefix);
    }

    if (isRecording && typeof recordAction === 'function') {
        recordAction({
            type: 'updateDecoration',
            zoneId: targetParentZoneId,
            imageData: imageData
        });
        // アイコンの場合はメモも記録
        if (memoToSet) {
             const slotIndex = Array.from(targetSlot.parentNode.children).indexOf(targetSlot);
             recordAction({
                type: 'memoChange',
                zoneId: targetParentZoneId,
                cardIndex: slotIndex,
                memo: memoToSet
            });
        }
    }
}

function handleFileDrop(e, targetSlot, idPrefix) {
    const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith('image/'));
    if (files.length === 0) return;

    const targetParentZoneId = getParentZoneId(targetSlot);
    const targetParentBaseId = getBaseId(targetParentZoneId);

    // アイコンゾーンへのファイルドロップは装飾として扱い、デフォルトメモを付与する
    if (targetParentBaseId === 'icon-zone') {
        const file = files[0];
        const reader = new FileReader();
        reader.onload = (event) => {
            const imageData = event.target.result;
            const existingThumbnail = targetSlot.querySelector('.thumbnail[data-is-decoration="true"]');
            if (existingThumbnail) {
                const existingImg = existingThumbnail.querySelector('img');
                if (existingImg) existingImg.src = imageData;
                existingThumbnail.dataset.memo = DEFAULT_CARD_MEMO;
            } else {
                const anyExistingThumbnail = getExistingThumbnail(targetSlot);
                if (anyExistingThumbnail) targetSlot.removeChild(anyExistingThumbnail);
                createCardThumbnail({
                    src: imageData,
                    isDecoration: true,
                    memo: DEFAULT_CARD_MEMO,
                    ownerPrefix: idPrefix
                }, targetSlot, true, false, idPrefix);
            }
            playSe('カードを配置する.mp3');

            if (isRecording && typeof recordAction === 'function') {
                recordAction({
                    type: 'updateDecoration',
                    zoneId: targetParentZoneId,
                    imageData: imageData
                });
                recordAction({
                    type: 'memoChange',
                    zoneId: targetParentZoneId,
                    cardIndex: 0,
                    memo: DEFAULT_CARD_MEMO
                });
            }
        };
        reader.readAsDataURL(file);
        return;
    }

    const pileZones = ['deck', 'grave', 'exclude', 'side-deck'];
    const isPileZone = pileZones.includes(targetParentBaseId);

    const isDirectBoardSlot = !targetParentBaseId.endsWith('-back-slots') 
                              && targetParentBaseId !== 'hand-zone' 
                              && targetParentBaseId !== 'free-space-slots'
                              && targetParentBaseId !== 'token-zone-slots'
                              && targetParentBaseId !== 'c-free-space'
                              && !isPileZone;

    if (isDirectBoardSlot) {
        const file = files[0]; 
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const isTargetStackable = stackableZones.includes(targetParentBaseId);
                const existingThumbnail = getExistingThumbnail(targetSlot);
                
                if (!isTargetStackable && existingThumbnail) {
                    targetSlot.removeChild(existingThumbnail);
                    resetSlotToDefault(targetSlot);
                }
                
                const imageData = event.target.result;
                // デフォルトメモ適用
                createCardThumbnail({
                    src: imageData,
                    memo: DEFAULT_CARD_MEMO,
                    ownerPrefix: idPrefix
                }, targetSlot, false, false, idPrefix);
                
                if (isTargetStackable) {
                    updateSlotStackState(targetSlot);
                }
                
                // 自動マナ消費
                const targetZonesForCost = ['special1', 'battle', 'special2', 'spell', 'mana', 'mana-left', 'mana-right'];
                const isTargetZone = targetZonesForCost.some(z => targetParentBaseId.includes(z));
                if (isTargetZone) {
                    tryAutoManaCost(DEFAULT_CARD_MEMO, idPrefix);
                }

                const isMana = targetParentBaseId.startsWith('mana');
                if (isMana) {
                    playSe('マナ配置.mp3');
                    const manaInput = document.getElementById(idPrefix + 'mana-counter-value');
                    if (manaInput) {
                        manaInput.value = parseInt(manaInput.value || 0) + 1;
                        if (isRecording && typeof recordAction === 'function') {
                            recordAction({
                                type: 'counterChange',
                                inputId: idPrefix + 'mana-counter-value',
                                change: 1
                            });
                        }
                    }
                    // マナ配置時タップイン
                    tryAutoManaTapIn(targetSlot, idPrefix, targetParentZoneId);
                } else {
                    playPlacementSe(targetParentBaseId);
                }

                if (isRecording && typeof recordAction === 'function') {
                    recordAction({
                        type: 'newCard',
                        zoneId: targetParentZoneId,
                        slotIndex: Array.from(targetSlot.parentNode.children).indexOf(targetSlot),
                        cardData: {
                            src: imageData,
                            memo: DEFAULT_CARD_MEMO
                        }
                    });
                }
            };
            reader.readAsDataURL(file);
        }
    } else {
        
        const addFilesToContainer = (fileList, containerId) => {
            const container = document.getElementById(containerId);
            if (!container) return;
            
            let slotsContainer = container.querySelector('.deck-back-slot-container, .free-space-slot-container, .token-slot-container') || container;
            const allSlots = Array.from(slotsContainer.querySelectorAll('.card-slot'));
            const availableSlots = allSlots.filter(s => !s.querySelector('.thumbnail'));
            
            fileList.forEach((file, index) => {
                if (availableSlots[index]) {
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        const imageData = event.target.result;
                        // バンクならPrefixなし
                        const owner = (containerId === 'c-free-space') ? '' : idPrefix;
                        // デフォルトメモ適用
                        createCardThumbnail({
                            src: imageData,
                            memo: DEFAULT_CARD_MEMO,
                            ownerPrefix: owner
                        }, availableSlots[index], false, false, owner);
                        
                        if (containerId === 'c-free-space') {
                            const thumb = availableSlots[index].querySelector('.thumbnail');
                            if (thumb) delete thumb.dataset.ownerPrefix;
                        }
                        
                        if (isRecording && typeof recordAction === 'function') {
                            const slotIndex = allSlots.indexOf(availableSlots[index]);
                            recordAction({
                                type: 'newCard',
                                zoneId: containerId,
                                slotIndex: slotIndex,
                                cardData: {
                                    src: imageData,
                                    memo: DEFAULT_CARD_MEMO
                                }
                            });
                        }
                    };
                    reader.readAsDataURL(file);
                }
            });

            if (fileList.length > 0) {
                const baseId = getBaseId(containerId).replace('-back-slots', '');
                playPlacementSe(baseId);
            }

            setTimeout(() => {
                arrangeSlots(containerId);
                const baseId = getBaseId(containerId).replace('-back-slots', '');
                if (['deck', 'grave', 'exclude', 'side-deck'].includes(baseId)) {
                    syncMainZoneImage(baseId, idPrefix);
                }
            }, 100);
        };

        if (targetParentBaseId === 'hand-zone') {
            const handContainerId = idPrefix + 'hand-zone';
            const handContainer = document.getElementById(handContainerId);
            const availableHandSlots = Array.from(handContainer.querySelectorAll('.card-slot')).filter(s => !s.querySelector('.thumbnail'));
            
            const filesForHand = files.slice(0, availableHandSlots.length);
            const filesForDeck = files.slice(availableHandSlots.length);
            
            if (filesForHand.length > 0) {
                addFilesToContainer(filesForHand, handContainerId);
            }
            
            if (filesForDeck.length > 0) {
                addFilesToContainer(filesForDeck, idPrefix + 'deck-back-slots');
            }
            
        } else {
            let destinationId;
            
            if (isPileZone) {
                destinationId = idPrefix + targetParentBaseId + '-back-slots';
            } else if (targetParentBaseId.endsWith('-back-slots') || targetParentBaseId === 'free-space-slots' || targetParentBaseId === 'token-zone-slots' || targetParentBaseId === 'c-free-space') {
                destinationId = targetParentZoneId;
            } else {
                destinationId = idPrefix + 'deck-back-slots'; 
            }
            
            addFilesToContainer(files, destinationId);
        }
    }
}

function handleCardDrop(draggedItem, targetSlot, idPrefix) {
    if (draggedItem.dataset.isDecoration === 'true') return; // 通常のカード移動では装飾カードは動かさない（ストックからのドロップは別処理）

    const sourceSlot = draggedItem.parentNode;
    const sourceZoneId = getParentZoneId(sourceSlot);
    const sourceBaseZoneId = getBaseId(sourceZoneId);

    const targetZoneId = getParentZoneId(targetSlot);
    const targetBaseZoneId = getBaseId(targetZoneId);

    if (sourceSlot === targetSlot) return;

    // ★バトル確認中のカードが移動したらバトル強制終了
    if (typeof isBattleConfirmMode !== 'undefined' && isBattleConfirmMode) {
        if (draggedItem === currentAttacker || draggedItem === currentBattleTarget) {
            if (typeof closeBattleConfirmModal === 'function') {
                closeBattleConfirmModal();
                alert("バトル中のカードが移動したため、バトルを中断しました。");
            }
        }
    }

    const fromZoneId = sourceZoneId;
    const fromSlotIndex = Array.from(sourceSlot.parentNode.children).indexOf(sourceSlot);

    // トークンエリアからのドロップのみ「複製」
    if (sourceBaseZoneId === 'token-zone-slots') {
        const imgElement = draggedItem.querySelector('.card-image');
        const src = imgElement ? imgElement.src : '';
        const memo = draggedItem.dataset.memo || '';
        const flavor1 = draggedItem.dataset.flavor1 || '';
        const flavor2 = draggedItem.dataset.flavor2 || '';
        const customCounters = JSON.parse(draggedItem.dataset.customCounters || '[]');
        
        // 複製時の所有者はターゲット先の盤面に合わせる
        const newOwnerPrefix = (targetBaseZoneId === 'c-free-space') ? '' : idPrefix;

        const cardData = {
            src: src,
            memo: memo,
            flavor1: flavor1,
            flavor2: flavor2,
            ownerPrefix: newOwnerPrefix,
            customCounters: customCounters
        };

        let destSlot = targetSlot;
        let destZoneId = targetZoneId;
        
        const isPileZone = ['deck', 'grave', 'exclude', 'side-deck'].includes(targetBaseZoneId);
        const isBackSlots = targetBaseZoneId.endsWith('-back-slots');
        const isHand = targetBaseZoneId === 'hand-zone';

        if (isPileZone || isBackSlots || isHand) {
             let containerId = targetZoneId;
             if (isPileZone) {
                 containerId = idPrefix + targetBaseZoneId + '-back-slots';
             } else if (isHand) {
                 containerId = idPrefix + 'hand-zone';
             }
             
             const container = document.getElementById(containerId);
             if (!container) return;
             
             const slotsContainer = container.querySelector('.deck-back-slot-container, .free-space-slot-container, .hand-zone-slots') || container;
             const emptySlot = Array.from(slotsContainer.querySelectorAll('.card-slot')).find(s => !s.querySelector('.thumbnail'));
             
             if (!emptySlot) {
                 console.warn('空きスロットがありません');
                 return;
             }
             destSlot = emptySlot;
             destZoneId = getParentZoneId(destSlot);
        } else {
            const isTargetStackable = stackableZones.includes(targetBaseZoneId);
            const existing = getExistingThumbnail(targetSlot);
            if (!isTargetStackable && existing) {
                return; 
            }
            destSlot = targetSlot;
        }

        createCardThumbnail(cardData, destSlot, false, false, newOwnerPrefix);
        
        // バンクへの複製ならownerPrefixを消す
        if (targetBaseZoneId === 'c-free-space') {
            const thumb = destSlot.querySelector('.thumbnail:last-child');
            if(thumb) delete thumb.dataset.ownerPrefix;
        }
        
        // 自動マナ消費 (トークン生成時もコストがかかるとみなす場合)
        const targetZonesForCost = ['special1', 'battle', 'special2', 'spell', 'mana', 'mana-left', 'mana-right'];
        const isTargetZone = targetZonesForCost.some(z => targetBaseZoneId.includes(z));
        if (isTargetZone) {
            tryAutoManaCost(memo, idPrefix);
        }

        // 配置SE分岐
        if (targetBaseZoneId.startsWith('mana')) {
             if (autoConfig.autoManaPlacement) {
                const manaInput = document.getElementById(idPrefix + 'mana-counter-value');
                if (manaInput) {
                    manaInput.value = parseInt(manaInput.value || 0) + 1;
                    if (isRecording && typeof recordAction === 'function') {
                        recordAction({
                            type: 'counterChange',
                            inputId: idPrefix + 'mana-counter-value',
                            change: 1
                        });
                    }
                }
            }
            tryAutoManaTapIn(destSlot, idPrefix, destZoneId);
            playSe('マナ配置.mp3');
        } else {
            playPlacementSe(targetBaseZoneId);
        }
        
        updateSlotStackState(destSlot);
        
        if (isRecording && typeof recordAction === 'function') {
            recordAction({
                type: 'newCard',
                zoneId: destZoneId,
                slotIndex: Array.from(destSlot.parentNode.children).indexOf(destSlot),
                cardData: cardData
            });
        }
        
        const destBaseId = getBaseId(destZoneId);
        if (destBaseId.endsWith('-back-slots') || destBaseId === 'hand-zone' || destBaseId === 'free-space-slots' || destBaseId === 'c-free-space') {
            arrangeSlots(destZoneId);
        }
        // 複製先のカウント更新
        const mainBaseId = destBaseId.replace('-back-slots', '');
        if (['deck', 'grave', 'exclude', 'side-deck'].includes(mainBaseId)) {
            syncMainZoneImage(mainBaseId, idPrefix);
        }

        return;
    }

    // 通常移動（バンク含む）
    const isGrave = targetBaseZoneId === 'grave' || targetBaseZoneId === 'grave-back-slots';
    const isExclude = targetBaseZoneId === 'exclude' || targetBaseZoneId === 'exclude-back-slots';
    const isMana = targetBaseZoneId.startsWith('mana');
    
    // 自動マナ消費判定
    const costSourceZones = ['hand-zone', 'deck', 'deck-back-slots', 'grave', 'grave-back-slots', 'exclude', 'exclude-back-slots', 'side-deck', 'side-deck-back-slots'];
    const targetZonesForCost = ['special1', 'battle', 'special2', 'spell', 'mana', 'mana-left', 'mana-right'];
    
    if (costSourceZones.includes(sourceBaseZoneId)) {
        // ターゲットがコスト消費対象のゾーンか確認
        const isTargetCostZone = targetZonesForCost.some(z => targetBaseZoneId.includes(z));
        if (isTargetCostZone) {
            tryAutoManaCost(draggedItem.dataset.memo, idPrefix);
        }
    }
    
    if (isGrave) {
        playSe('墓地に送る.mp3');
    } else if (isExclude) {
        playSe('除外する.mp3');
    } else if (isMana) {
        playSe('マナ配置.mp3');
    } else {
        playPlacementSe(targetBaseZoneId);
    }

    if (isMana && !sourceBaseZoneId.startsWith('mana')) {
        if (autoConfig.autoManaPlacement) {
            const targetPrefix = getPrefixFromZoneId(targetZoneId);
            const manaInputId = targetPrefix + 'mana-counter-value';
            const manaInput = document.getElementById(manaInputId);
            if (manaInput) {
                manaInput.value = parseInt(manaInput.value || 0) + 1;
                if (isRecording && typeof recordAction === 'function') {
                    recordAction({
                        type: 'counterChange',
                        inputId: manaInputId,
                        change: 1
                    });
                }
            }
        }
    }

    const isTargetStackable = stackableZones.includes(targetBaseZoneId);
    const existingThumbnail = getExistingThumbnail(targetSlot);

    if (targetBaseZoneId.endsWith('-back-slots') || ['deck', 'grave', 'exclude', 'side-deck'].includes(targetBaseZoneId) || targetBaseZoneId === 'c-free-space') {
        let multiZoneId = targetZoneId;
        if(['deck', 'grave', 'exclude', 'side-deck'].includes(targetBaseZoneId)) {
            multiZoneId = idPrefix + targetBaseZoneId + '-back-slots';
        }
        moveCardToMultiZone(draggedItem, getBaseId(multiZoneId).replace('-back-slots',''));
        return;
    }

    if (isTargetStackable) { // 装飾モードチェックは不要（通常のカード移動のみここに来る）
        sourceSlot.removeChild(draggedItem);
        targetSlot.insertBefore(draggedItem, targetSlot.firstChild);
        
        if (isRecording && typeof recordAction === 'function') {
            recordAction({
                type: 'move',
                fromZone: fromZoneId,
                fromSlotIndex: fromSlotIndex,
                toZone: targetZoneId,
                toSlotIndex: Array.from(targetSlot.parentNode.children).indexOf(targetSlot)
            });
        }
    }
    else if (existingThumbnail && sourceSlot !== targetSlot) {
        sourceSlot.appendChild(existingThumbnail);
        targetSlot.appendChild(draggedItem);
        
        if (isRecording && typeof recordAction === 'function') {
            recordAction({
                type: 'move',
                fromZone: fromZoneId,
                fromSlotIndex: fromSlotIndex,
                toZone: targetZoneId,
                toSlotIndex: Array.from(targetSlot.parentNode.children).indexOf(targetSlot)
            });
        }
    }
    else if (!existingThumbnail) {
        sourceSlot.removeChild(draggedItem);
        targetSlot.appendChild(draggedItem);
        
        if (isRecording && typeof recordAction === 'function') {
            recordAction({
                type: 'move',
                fromZone: fromZoneId,
                fromSlotIndex: fromSlotIndex,
                toZone: targetZoneId,
                toSlotIndex: Array.from(targetSlot.parentNode.children).indexOf(targetSlot)
            });
        }
    } else {
        return; 
    }

    [sourceSlot, targetSlot].forEach(slot => {
        resetSlotToDefault(slot);
        updateSlotStackState(slot);
        
        const zoneId = getParentZoneId(slot);
        const baseId = getBaseId(zoneId);

        if(zoneId.endsWith('-back-slots')) arrangeSlots(zoneId);
        // 装飾ゾーン（アイコン含む）の場合の更新は不要（通常カード移動なので）
        // ただし、バックヤードのカウント更新が必要
        const realBaseId = baseId.replace('-back-slots', '');
        if (['deck', 'grave', 'exclude', 'side-deck'].includes(realBaseId)) {
             syncMainZoneImage(realBaseId, getPrefixFromZoneId(zoneId));
        }
        
        // バンク（c-free-space）に移動した場合は所有権を消す
        if (baseId === 'c-free-space') {
            const thumb = slot.querySelector('.thumbnail');
            if(thumb) delete thumb.dataset.ownerPrefix;
        }
    });

    // ★修正: マナゾーンへの移動で、かつ移動元がマナゾーンでない場合のみタップイン
    if (targetBaseZoneId.startsWith('mana') && !sourceBaseZoneId.startsWith('mana')) {
        tryAutoManaTapIn(targetSlot, idPrefix, targetZoneId);
    }
}


function updateSlotStackState(slotElement) {
    if (!slotElement) return;
    const thumbnailCount = slotElement.querySelectorAll('.thumbnail:not([data-is-decoration="true"])').length;
    slotElement.classList.toggle('stacked', thumbnailCount > 1);
}

function arrangeSlots(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    let slotsContainer = container.querySelector('.deck-back-slot-container, .free-space-slot-container, .token-slot-container') || container;
    if (!slotsContainer) return;

    const slots = Array.from(slotsContainer.querySelectorAll('.card-slot'));
    let cardThumbnails = [];
    slots.forEach(slot => {
        slot.querySelectorAll('.thumbnail').forEach(thumbnail => {
            cardThumbnails.push(thumbnail);
            slot.removeChild(thumbnail);
        });
        resetSlotToDefault(slot);
    });

    cardThumbnails.forEach((thumbnail, i) => {
        if (slots[i]) {
            slots[i].appendChild(thumbnail);
            resetSlotToDefault(slots[i]);
            updateSlotStackState(slots[i]);
        }
    });
}

function syncMainZoneImage(baseZoneId, idPrefix) {
    const mainZone = document.getElementById(idPrefix + baseZoneId);
    if (!mainZone) return;
    const mainSlot = mainZone.querySelector('.card-slot');
    if (!mainSlot) return;

    const backSlotsId = `${idPrefix}${baseZoneId}-back-slots`;
    const backSlotsContainer = document.getElementById(backSlotsId);
    const backSlots = backSlotsContainer ? backSlotsContainer.querySelector('.deck-back-slot-container') : null;
    const occupiedThumbnails = backSlots ? Array.from(backSlots.querySelectorAll('.thumbnail')) : [];
    const cardCount = occupiedThumbnails.length;

    let countOverlay = mainSlot.querySelector('.count-overlay');
    if (!countOverlay) {
        countOverlay = document.createElement('div');
        countOverlay.classList.add('count-overlay');
        mainSlot.appendChild(countOverlay);
    }
    countOverlay.textContent = cardCount;
    countOverlay.style.display = cardCount > 0 ? 'block' : 'none';

    const decoratedThumbnail = mainSlot.querySelector('.thumbnail[data-is-decoration="true"]');
    let targetCardThumbnail = null;
    if (cardCount > 0) {
        if (baseZoneId === 'deck' || baseZoneId === 'side-deck') {
            targetCardThumbnail = occupiedThumbnails[0];
        } else if (baseZoneId === 'grave' || baseZoneId === 'exclude') {
            targetCardThumbnail = occupiedThumbnails[occupiedThumbnails.length - 1];
        }
    }

    let mainSlotImg = mainSlot.querySelector('img.zone-image');
    if (!mainSlotImg) {
        mainSlotImg = document.createElement('img');
        mainSlotImg.classList.add('zone-image');
        mainSlotImg.draggable = false;
        mainSlot.insertBefore(mainSlotImg, countOverlay);
    }

    if (decoratedThumbnail) {
        mainSlotImg.style.display = 'none';
        decoratedThumbnail.style.display = 'block';
    } else if (targetCardThumbnail) {
        const cardImg = targetCardThumbnail.querySelector('.card-image');
        mainSlotImg.src = targetCardThumbnail.dataset.isFlipped === 'true' ? targetCardThumbnail.dataset.originalSrc : cardImg.src;
        mainSlotImg.style.display = 'block';
    } else {
        mainSlotImg.style.display = 'none';
    }
    mainSlot.dataset.hasCard = !!(decoratedThumbnail || targetCardThumbnail);
}

function moveCardToMultiZone(thumbnailElement, targetBaseZoneId) {
    const sourceSlot = thumbnailElement.parentNode;
    if (!sourceSlot) return;

    const idPrefix = thumbnailElement.dataset.ownerPrefix || '';

    // バンクへの移動か判定
    const isCNavi = (targetBaseZoneId === 'c-free-space' || targetBaseZoneId === 'c-free-space-slots');
    const isTargetHand = (targetBaseZoneId === 'hand');
    
    let destinationMultiZoneId;
    if (isCNavi) {
        destinationMultiZoneId = 'c-free-space';
    } else if (isTargetHand) {
        destinationMultiZoneId = idPrefix + 'hand-zone';
    } else {
        destinationMultiZoneId = idPrefix + targetBaseZoneId + '-back-slots';
    }

    if (sourceSlot === destinationMultiZoneId || sourceSlot.id === destinationMultiZoneId) return;

    const fromZoneId = getParentZoneId(sourceSlot);
    const fromSlotIndex = Array.from(sourceSlot.parentNode.children).indexOf(sourceSlot);

    const destinationContainer = document.getElementById(destinationMultiZoneId);
    if (!destinationContainer) return;

    const slotsContainer = destinationContainer.querySelector('.deck-back-slot-container, .free-space-slot-container') || destinationContainer;
    const emptySlot = Array.from(slotsContainer.querySelectorAll('.card-slot')).find(s => !s.querySelector('.thumbnail'));

    if (!emptySlot) {
        console.warn(`「${targetBaseZoneId}」がいっぱいです。`);
        return;
    }

    sourceSlot.removeChild(thumbnailElement);
    emptySlot.appendChild(thumbnailElement);
    
    // 回転状態のリセットを明示的に行う
    const img = thumbnailElement.querySelector('.card-image');
    if (img) {
        img.dataset.rotation = 0;
        img.style.transform = 'rotate(0deg)';
    }
    emptySlot.classList.remove('rotated-90');
    
    // バンクへ移動した場合は所有権を消す
    if (isCNavi) {
        delete thumbnailElement.dataset.ownerPrefix;
    }

    if (isRecording && typeof recordAction === 'function') {
        recordAction({
            type: 'move',
            fromZone: fromZoneId,
            fromSlotIndex: fromSlotIndex,
            toZone: destinationMultiZoneId,
            toSlotIndex: Array.from(emptySlot.parentNode.children).indexOf(emptySlot)
        });
    }

    resetCardFlipState(thumbnailElement);
    resetSlotToDefault(emptySlot);
    updateSlotStackState(emptySlot);

    const sourceParentZoneId = getParentZoneId(sourceSlot);
    const sourceParentBaseId = getBaseId(sourceParentZoneId);

    if (sourceParentZoneId.endsWith('-back-slots') || sourceParentBaseId === 'hand-zone' || sourceParentBaseId === 'free-space-slots' || sourceParentBaseId === 'token-zone-slots' || sourceParentBaseId === 'c-free-space') {
        arrangeSlots(sourceParentZoneId);
    }
    resetSlotToDefault(sourceSlot);
    updateSlotStackState(sourceSlot);

    // 移動元のカウント更新（バックヤードからの移動であれば）
    const realSourceBaseId = sourceParentBaseId.replace('-back-slots', '');
    if (['deck', 'grave', 'exclude', 'side-deck'].includes(realSourceBaseId)) {
         syncMainZoneImage(realSourceBaseId, getPrefixFromZoneId(sourceParentZoneId));
    }

    arrangeSlots(destinationMultiZoneId);
    if (!isTargetHand && !isCNavi) {
        syncMainZoneImage(targetBaseZoneId, idPrefix);
    }
}