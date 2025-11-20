function createCardThumbnail(cardData, slotElement, isDecoration = false, insertAtBottom = false, ownerPrefix = '') {
    let imageSrc, isFlipped, originalSrc, counter, memo, flavor1, flavor2, rotation, isMasturbating, isBlocker, isPermanent;

    if (typeof cardData === 'string') {
        imageSrc = cardData;
        isFlipped = false;
        originalSrc = null;
        counter = 0;
        memo = '';
        flavor1 = '';
        flavor2 = '';
        rotation = 0;
        isMasturbating = false;
        isBlocker = false;
        isPermanent = false;
    } else { 
        imageSrc = cardData.src;
        isDecoration = cardData.isDecoration || isDecoration; 
        isFlipped = cardData.isFlipped || false;
        originalSrc = cardData.originalSrc || null;
        counter = cardData.counter || 0;
        memo = cardData.memo || '';
        flavor1 = cardData.flavor1 || '';
        flavor2 = cardData.flavor2 || '';
        ownerPrefix = cardData.ownerPrefix || ownerPrefix;
        rotation = cardData.rotation || 0;
        isMasturbating = cardData.isMasturbating || false;
        isBlocker = cardData.isBlocker || false;
        isPermanent = cardData.isPermanent || false;
    }

    const thumbnailElement = document.createElement('div');
    thumbnailElement.classList.add('thumbnail');
    thumbnailElement.setAttribute('draggable', true);

    if (isDecoration) {
        thumbnailElement.dataset.isDecoration = 'true';
    }
    
    if (isMasturbating) {
        thumbnailElement.dataset.isMasturbating = 'true';
    }
    
    if (isBlocker) {
        thumbnailElement.dataset.isBlocker = 'true';
    }

    if (isPermanent) {
        thumbnailElement.dataset.isPermanent = 'true';
    }

    const imgElement = document.createElement('img');
    imgElement.classList.add('card-image');
    imgElement.dataset.rotation = rotation; 
    
    if (isFlipped && originalSrc) {
        thumbnailElement.dataset.isFlipped = 'true';
        thumbnailElement.dataset.originalSrc = originalSrc;
        imgElement.src = imageSrc;
    } else {
        thumbnailElement.dataset.isFlipped = 'false';
        imgElement.src = imageSrc;
    }

    thumbnailElement.appendChild(imgElement);

    // ブロッカーオーバーレイ
    if (isBlocker) {
        addBlockerOverlay(thumbnailElement);
    }

    const counterOverlay = document.createElement('div');
    counterOverlay.classList.add('card-counter-overlay');
    counterOverlay.dataset.counter = counter;
    counterOverlay.textContent = counter;
    counterOverlay.style.display = counter > 0 ? 'flex' : 'none';
    thumbnailElement.appendChild(counterOverlay);

    if (memo) thumbnailElement.dataset.memo = memo;
    if (flavor1) thumbnailElement.dataset.flavor1 = flavor1;
    if (flavor2) thumbnailElement.dataset.flavor2 = flavor2;
    if (ownerPrefix) thumbnailElement.dataset.ownerPrefix = ownerPrefix;

    if (insertAtBottom) {
        const firstCard = slotElement.querySelector('.thumbnail');
        if (firstCard) {
            slotElement.insertBefore(thumbnailElement, firstCard);
        } else {
            slotElement.appendChild(thumbnailElement);
        }
    } else {
        slotElement.appendChild(thumbnailElement);
    }

    addCardEventListeners(thumbnailElement);

    return thumbnailElement;
}

function addCardEventListeners(thumbnailElement) {
    thumbnailElement.addEventListener('dragstart', handleDragStart);
    thumbnailElement.addEventListener('dragend', handleDragEnd);
    thumbnailElement.addEventListener('click', handleCardClick);
    thumbnailElement.addEventListener('contextmenu', handleCardContextMenu);
    thumbnailElement.addEventListener('mouseover', handleCardMouseOver);
    thumbnailElement.addEventListener('mouseout', handleCardMouseOut);
}

function handleCardClick(e) {
    const thumbnailElement = e.target.closest('.thumbnail');
    if (!thumbnailElement) return;
    const slotElement = thumbnailElement.parentNode;
    const parentZoneId = getParentZoneId(slotElement);
    const baseParentZoneId = getBaseId(parentZoneId);

    if (isDecorationMode && decorationZones.includes(baseParentZoneId)) {
        openDecorationImageDialog(slotElement);
        e.stopPropagation();
        return;
    }

    if (thumbnailElement.dataset.isDecoration === 'true') {
        return;
    }
    
    if (contextMenu.style.display === 'block' || memoEditorModal.style.display === 'block' || flavorEditorModal.style.display === 'block' || draggedItem) {
        return;
    }

    if (getExistingThumbnail(slotElement) !== thumbnailElement) {
        e.stopPropagation();
        return;
    }

    const imgElement = thumbnailElement.querySelector('.card-image');
    if (!imgElement) return;

    if (nonRotatableZones.includes(baseParentZoneId) || baseParentZoneId === 'free-space-slots') {
        e.stopPropagation();
        return;
    }

    let currentRotation = parseInt(imgElement.dataset.rotation) || 0;
    const idPrefix = getPrefixFromZoneId(parentZoneId);
    const manaCounterValueElement = document.getElementById(idPrefix + 'mana-counter-value');

    if (currentRotation === 0) {
        currentRotation = 90;
        slotElement.classList.add('rotated-90');
        const { width, height } = getCardDimensions();
        const scaleFactor = height / width;
        imgElement.style.transform = `rotate(${currentRotation}deg) scale(${scaleFactor})`;

        if (baseParentZoneId.startsWith('mana')) {
            if (manaCounterValueElement) {
                const currentValue = parseInt(manaCounterValueElement.value) || 0;
                manaCounterValueElement.value = currentValue + 1;
                
                if (isRecording && typeof recordAction === 'function') {
                    recordAction({
                        type: 'counterChange',
                        inputId: idPrefix + 'mana-counter-value',
                        change: 1
                    });
                }
            }
            playSe('マナ増加.mp3');
        } else {
            // マナエリア以外で横向き（タップ）にした場合
            playSe('タップ.mp3');
        }
    } else {
        currentRotation = 0;
        slotElement.classList.remove('rotated-90');
        imgElement.style.transform = `rotate(${currentRotation}deg)`;
    }
    imgElement.dataset.rotation = currentRotation;
    
    if (isRecording && typeof recordAction === 'function') {
        recordAction({
            type: 'rotate',
            zoneId: parentZoneId,
            slotIndex: Array.from(slotElement.parentNode.children).indexOf(slotElement),
            rotation: currentRotation
        });
    }
    
    e.stopPropagation();
}

function handleDragStart(e) {
    const thumbnailElement = e.target;
    if (thumbnailElement.dataset.isDecoration === 'true' && !isDecorationMode) {
        e.preventDefault();
        return;
    }
    draggedItem = thumbnailElement;
    setTimeout(() => {
        thumbnailElement.style.visibility = 'hidden';
    }, 0);
    e.dataTransfer.setData('text/plain', '');
}

function handleDragEnd(e) {
    const thumbnailElement = e.target;
    thumbnailElement.style.visibility = 'visible';
    draggedItem = null;
}

function handleCardContextMenu(e) {
    if (isDecorationMode) {
        e.preventDefault();
        e.stopPropagation();
        return;
    }
    e.preventDefault();
    e.stopPropagation();
    const thumbnailElement = e.target.closest('.thumbnail');

    if (memoEditorModal.style.display === 'block' || flavorEditorModal.style.display === 'block') {
        return;
    }

    const sourceZoneId = getParentZoneId(thumbnailElement.parentNode);
    const sourceBaseId = getBaseId(sourceZoneId);
    const isIconZone = (sourceBaseId === 'icon-zone');
    const idPrefix = getPrefixFromZoneId(sourceZoneId);

    if (!isIconZone && thumbnailElement.dataset.isDecoration === 'true' && !isDecorationMode) {
        return;
    }
    
    // スロットのインデックスを取得（記録用）
    const slotIndex = Array.from(thumbnailElement.parentNode.parentNode.children).indexOf(thumbnailElement.parentNode);

    // アタック
    if (attackMenuItem) {
        currentAttackHandler = () => {
            triggerEffect(thumbnailElement, 'attack');
        };
    }

    // 効果発動
    currentActionHandler = () => {
        triggerEffect(thumbnailElement, 'effect');
    };
    
    // 対象に取る
    currentTargetHandler = () => {
        triggerEffect(thumbnailElement, 'target');
    };

    // 常時発動
    if (permanentMenuItem) {
        const isPermanent = thumbnailElement.dataset.isPermanent === 'true';
        // 修正: 属性がtrueなら「発動停止」と表示
        permanentMenuItem.textContent = isPermanent ? '発動停止' : '常時発動';
        
        currentPermanentHandler = () => {
            const newState = !isPermanent;
            thumbnailElement.dataset.isPermanent = newState;
            
            if (newState) {
                playSe('常時発動.mp3');
            } else {
                playSe('ボタン共通.mp3');
            }
            
            if (isRecording && typeof recordAction === 'function') {
                recordAction({ 
                    type: 'permanent', 
                    zoneId: sourceZoneId, 
                    slotIndex: slotIndex, 
                    isPermanent: newState 
                });
            }
        };
    }

    currentAddFlavorHandler = () => openFlavorEditor(thumbnailElement);
    currentDeleteHandler = () => deleteCard(thumbnailElement);
    
    // moveCardToMultiZone内ですでに記録処理が入っているため、ここでは呼び出しのみ
    currentMoveToGraveHandler = () => moveCardToMultiZone(thumbnailElement, 'grave');
    currentMoveToExcludeHandler = () => moveCardToMultiZone(thumbnailElement, 'exclude');
    currentMoveToHandHandler = () => moveCardToMultiZone(thumbnailElement, 'hand');
    currentMoveToDeckHandler = () => moveCardToMultiZone(thumbnailElement, 'deck');
    currentMoveToSideDeckHandler = () => moveCardToMultiZone(thumbnailElement, 'side-deck');
    
    currentAddCounterHandler = () => addCounterToCard(thumbnailElement);
    currentRemoveCounterHandler = () => removeCounterFromCard(thumbnailElement);
    currentMemoHandler = () => {
        currentMemoTarget = thumbnailElement;
        memoTextarea.value = thumbnailElement.dataset.memo || '';
        memoEditorModal.style.display = 'block';
        memoTextarea.focus();
    };
    currentFlipHandler = () => flipCard(thumbnailElement, idPrefix);

    // オナニーメニュー
    if (masturbateMenuItem) {
        masturbateMenuItem.style.display = 'block';
        const isMasturbating = thumbnailElement.dataset.isMasturbating === 'true';
        // 修正: 属性がtrueなら「オナニーを止める」と表示
        masturbateMenuItem.textContent = isMasturbating ? 'オナニーを止める' : 'オナニーする';

        currentMasturbateHandler = () => {
            const newState = !isMasturbating;
            thumbnailElement.dataset.isMasturbating = newState;
            
            if (isRecording && typeof recordAction === 'function') {
                recordAction({ 
                    type: 'masturbate', 
                    zoneId: sourceZoneId, 
                    slotIndex: slotIndex, 
                    isMasturbating: newState 
                });
            }
        };
    }
    
    // ブロッカーメニュー
    if (blockerMenuItem) {
        blockerMenuItem.style.display = 'block';
        const isBlocker = thumbnailElement.dataset.isBlocker === 'true';
        blockerMenuItem.textContent = isBlocker ? 'ブロッカー効果を削除' : 'ブロッカー効果を付与';
        currentBlockerHandler = () => {
            toggleBlocker(thumbnailElement);
        };
    }

    const hideMoveFlip = isIconZone;
    // アイコンの場合は移動・反転系を隠す
    toGraveMenuItem.style.display = hideMoveFlip ? 'none' : 'block';
    toExcludeMenuItem.style.display = hideMoveFlip ? 'none' : 'block';
    toHandMenuItem.style.display = hideMoveFlip ? 'none' : 'block';
    toDeckMenuItem.style.display = hideMoveFlip ? 'none' : 'block';
    toSideDeckMenuItem.style.display = hideMoveFlip ? 'none' : 'block';
    flipMenuItem.style.display = hideMoveFlip ? 'none' : 'block';

    // アイコンでも表示するメニュー
    if (attackMenuItem) attackMenuItem.style.display = 'block';
    actionMenuItem.style.display = 'block';
    if (permanentMenuItem) permanentMenuItem.style.display = 'block'; 
    targetMenuItem.style.display = 'block';
    blockerMenuItem.style.display = 'block';

    // メモ・フレーバーは共通
    memoMenuItem.style.display = 'block';
    addFlavorMenuItem.style.display = 'block';
    
    // カウンターは特定のゾーンのみ（アイコンは stackableZones に含まれないため非表示）
    if (!stackableZones.includes(sourceBaseId)) {
        addCounterMenuItem.style.display = 'none';
        removeCounterMenuItem.style.display = 'none';
    } else {
        addCounterMenuItem.style.display = 'block';
        removeCounterMenuItem.style.display = 'block';
    }

    deleteMenuItem.style.display = 'block';

    // デコレーションカードの特殊制御
    if (!isIconZone && thumbnailElement.dataset.isDecoration === 'true') {
        if (attackMenuItem) attackMenuItem.style.display = 'none';
        actionMenuItem.style.display = 'none';
        if (permanentMenuItem) permanentMenuItem.style.display = 'none';
        targetMenuItem.style.display = 'none';
        blockerMenuItem.style.display = 'none'; 
        addCounterMenuItem.style.display = 'none';
        removeCounterMenuItem.style.display = 'none';
        memoMenuItem.style.display = 'none';
        addFlavorMenuItem.style.display = 'none';
        flipMenuItem.style.display = 'none';
        deleteMenuItem.style.display = isDecorationMode ? 'block' : 'none';
    }

    contextMenu.style.visibility = 'hidden';
    contextMenu.style.display = 'block';
    const { offsetWidth: menuWidth, offsetHeight: menuHeight } = contextMenu;
    contextMenu.style.display = 'none';
    contextMenu.style.visibility = 'visible';

    let left = e.pageX;
    let top = e.pageY - (menuHeight / 2);

    contextMenu.style.top = `${top}px`;
    contextMenu.style.left = `${left}px`;
    contextMenu.style.display = 'block';
}


// プレビュー更新関数を外部公開
window.updateCardPreview = function(thumbnailElement) {
    if (!thumbnailElement) return;
    const imgElement = thumbnailElement.querySelector('.card-image');
    if (!imgElement) return;

    const commonPreviewArea = document.getElementById('common-card-preview');
    const previewImageContainer = commonPreviewArea.querySelector('#preview-image-container');
    previewImageContainer.innerHTML = '';
    const previewImg = document.createElement('img');
    previewImg.src = thumbnailElement.dataset.isFlipped === 'true' ? (thumbnailElement.dataset.originalSrc || imgElement.src) : imgElement.src;
    
    previewImg.onerror = () => previewImg.remove();

    previewImageContainer.appendChild(previewImg);

    commonPreviewArea.dataset.flavor1 = thumbnailElement.dataset.flavor1 || '';
    commonPreviewArea.dataset.flavor2 = thumbnailElement.dataset.flavor2 || '';

    const memo = thumbnailElement.dataset.memo || '';
    const extractValue = (key) => {
        const regex = new RegExp(`\\[${key}:([\\s\\S]*?)\\]`, 'i');
        const match = memo.match(regex);
        if (match && match[1]) {
            const value = match[1].trim();
            return (value === '' || value === '-') ? null : value;
        }
        return null;
    };

    const cardInfo = {
        attribute: extractValue('属性'),
        cost: extractValue('コスト'),
        bp: extractValue('BP'),
        spell: extractValue('スペル'),
        cardName: extractValue('カード名'),
        flavor: extractValue('フレーバーテキスト'),
        effect: extractValue('効果'),
    };

    const previewElements = {
        'preview-attribute': cardInfo.attribute ? `属性: ${cardInfo.attribute}` : null,
        'preview-cost': cardInfo.cost ? `コスト: ${cardInfo.cost}` : null,
        'preview-card-name': cardInfo.cardName,
        'preview-top-right-stat': cardInfo.spell ? `スペル: ${cardInfo.spell}` : (cardInfo.bp ? `BP: ${cardInfo.bp}` : null),
        'preview-flavor-text': cardInfo.flavor,
        'preview-effect-text': cardInfo.effect,
    };

    Object.keys(previewElements).forEach(id => {
        const el = commonPreviewArea.querySelector(`#${id}`);
        if (el) {
            const content = previewElements[id];
            el.textContent = content;
            el.style.display = content ? 'block' : 'none';
        }
    });

    const memoTooltip = document.getElementById('memo-tooltip');
    if (memo && memoTooltip) {
        memoTooltip.textContent = memo;
        memoTooltip.style.display = 'block';
    } else if (memoTooltip) {
        memoTooltip.style.display = 'none';
    }
};

function handleCardMouseOver(e) {
    const thumbnailElement = e.target.closest('.thumbnail');
    if (thumbnailElement) {
        window.updateCardPreview(thumbnailElement);
    }
    e.stopPropagation();
}

function handleCardMouseOut(e) {
    const memoTooltip = document.getElementById('memo-tooltip');
    if(memoTooltip) memoTooltip.style.display = 'none';
    e.stopPropagation();
}

function deleteCard(thumbnailElement) {
    const slotElement = thumbnailElement.parentNode;
    if (!slotElement) return;

    const parentZoneId = getParentZoneId(slotElement);
    const baseParentZoneId = getBaseId(parentZoneId);
    
    if (isRecording && typeof recordAction === 'function') {
        recordAction({
            type: 'delete',
            zoneId: parentZoneId,
            slotIndex: Array.from(slotElement.parentNode.children).indexOf(slotElement)
        });
    }

    slotElement.removeChild(thumbnailElement);
    
    const commonPreviewArea = document.getElementById('common-card-preview');
    const previewImageContainer = commonPreviewArea.querySelector('#preview-image-container');
    previewImageContainer.innerHTML = '<p>カードにカーソルを合わせてください</p>';
    commonPreviewArea.querySelector('#preview-attribute').style.display = 'none';
    commonPreviewArea.querySelector('#preview-cost').style.display = 'none';
    commonPreviewArea.querySelector('#preview-top-right-stat').style.display = 'none';
    commonPreviewArea.querySelector('#preview-card-name').style.display = 'none';
    commonPreviewArea.querySelector('#preview-flavor-text').style.display = 'none';
    commonPreviewArea.querySelector('#preview-effect-text').style.display = 'none';
    delete commonPreviewArea.dataset.flavor1;
    delete commonPreviewArea.dataset.flavor2;

    resetSlotToDefault(slotElement);
    updateSlotStackState(slotElement);
    draggedItem = null;

    if (baseParentZoneId.endsWith('-back-slots')) {
        arrangeSlots(parentZoneId);
        syncMainZoneImage(baseParentZoneId.replace('-back-slots', ''));
    } else if (baseParentZoneId === 'hand-zone' || baseParentZoneId === 'free-space-slots') {
        arrangeSlots(parentZoneId);
    } else if (thumbnailElement.dataset.isDecoration === 'true') {
        syncMainZoneImage(baseParentZoneId);
    }
}

function addCounterToCard(thumbnailElement) {
    const counterOverlay = thumbnailElement.querySelector('.card-counter-overlay');
    if (!counterOverlay) return;
    let count = parseInt(counterOverlay.dataset.counter) || 0;
    count++;
    counterOverlay.dataset.counter = count;
    counterOverlay.textContent = count;
    counterOverlay.style.display = 'flex';
    
    if (isRecording && typeof recordAction === 'function') {
        recordAction({
            type: 'cardCounter',
            zoneId: getParentZoneId(thumbnailElement.parentNode),
            slotIndex: Array.from(thumbnailElement.parentNode.parentNode.children).indexOf(thumbnailElement.parentNode),
            counter: count
        });
    }
}

function removeCounterFromCard(thumbnailElement) {
    const counterOverlay = thumbnailElement.querySelector('.card-counter-overlay');
    if (!counterOverlay) return;
    let count = parseInt(counterOverlay.dataset.counter) || 0;
    if (count > 0) count--;
    counterOverlay.dataset.counter = count;
    counterOverlay.textContent = count;
    if (count === 0) counterOverlay.style.display = 'none';
    
    if (isRecording && typeof recordAction === 'function') {
        recordAction({
            type: 'cardCounter',
            zoneId: getParentZoneId(thumbnailElement.parentNode),
            slotIndex: Array.from(thumbnailElement.parentNode.parentNode.children).indexOf(thumbnailElement.parentNode),
            counter: count
        });
    }
}

function flipCard(thumbnailElement, idPrefix) {
    const imgElement = thumbnailElement.querySelector('.card-image');
    if (!imgElement) return;

    const isFlipped = thumbnailElement.dataset.isFlipped === 'true';

    if (isFlipped) {
        resetCardFlipState(thumbnailElement);
    } else {
        const deckZone = document.getElementById(idPrefix + 'deck');
        let deckImgSrc = './decoration/デッキ.png';
        if (deckZone) {
            const decoratedThumbnail = deckZone.querySelector('.thumbnail[data-is-decoration="true"]');
            if (decoratedThumbnail) {
                const decoratedImg = decoratedThumbnail.querySelector('.card-image');
                if (decoratedImg) deckImgSrc = decoratedImg.src;
            }
        }
        thumbnailElement.dataset.originalSrc = imgElement.src;
        imgElement.src = deckImgSrc;
        thumbnailElement.dataset.isFlipped = 'true';
    }
    
    if (isRecording && typeof recordAction === 'function') {
        recordAction({
            type: 'flip',
            zoneId: getParentZoneId(thumbnailElement.parentNode),
            slotIndex: Array.from(thumbnailElement.parentNode.parentNode.children).indexOf(thumbnailElement.parentNode),
            isFlipped: !isFlipped
        });
    }

    const slotElement = thumbnailElement.parentNode;
    const parentZoneId = getParentZoneId(slotElement);
    const baseParentZoneId = getBaseId(parentZoneId);

    if (baseParentZoneId.endsWith('-back-slots')) {
        syncMainZoneImage(baseParentZoneId.replace('-back-slots', ''));
    }
}

function resetCardFlipState(thumbnailElement) {
    if (!thumbnailElement || thumbnailElement.dataset.isFlipped !== 'true') {
        return;
    }
    const originalSrc = thumbnailElement.dataset.originalSrc;
    const imgElement = thumbnailElement.querySelector('.card-image');
    if (imgElement && originalSrc) {
        imgElement.src = originalSrc;
        thumbnailElement.dataset.isFlipped = 'false';
        delete thumbnailElement.dataset.originalSrc;
    }
}

// --- ブロッカー / エフェクト関連ヘルパー関数 ---

function toggleBlocker(thumbnailElement) {
    const isBlocker = thumbnailElement.dataset.isBlocker === 'true';
    const newState = !isBlocker;
    thumbnailElement.dataset.isBlocker = newState;
    
    if (newState) {
        addBlockerOverlay(thumbnailElement);
    } else {
        removeBlockerOverlay(thumbnailElement);
    }
    
    if (isRecording && typeof recordAction === 'function') {
        recordAction({
            type: 'blocker',
            zoneId: getParentZoneId(thumbnailElement.parentNode),
            slotIndex: Array.from(thumbnailElement.parentNode.parentNode.children).indexOf(thumbnailElement.parentNode),
            isBlocker: newState
        });
    }
}

function addBlockerOverlay(thumbnailElement) {
    if (thumbnailElement.querySelector('.blocker-overlay')) return;
    const overlay = document.createElement('div');
    overlay.classList.add('blocker-overlay');
    const img = document.createElement('img');
    img.src = './decoration/ブロッカー.png';
    img.onerror = () => { overlay.remove(); console.warn('ブロッカー画像が見つかりません: ./decoration/ブロッカー.png'); };
    overlay.appendChild(img);
    
    // カウンターオーバーレイより手前、カード画像より後
    const counter = thumbnailElement.querySelector('.card-counter-overlay');
    if(counter) {
        thumbnailElement.insertBefore(overlay, counter);
    } else {
        thumbnailElement.appendChild(overlay);
    }
}

function removeBlockerOverlay(thumbnailElement) {
    const overlay = thumbnailElement.querySelector('.blocker-overlay');
    if (overlay) thumbnailElement.removeChild(overlay);
}

function triggerEffect(thumbnailElement, type) {
    // type: 'effect' or 'target' or 'attack'
    
    // エフェクト個別設定のチェック (無効なら表示しない)
    if (typeof effectConfig !== 'undefined' && effectConfig[type] === false) return;

    // 永続エフェクトの状態を一時的に保存
    const wasPermanentActive = thumbnailElement.dataset.isPermanent === 'true';
    const wasMasturbatingActive = thumbnailElement.dataset.isMasturbating === 'true';

    // 一時停止
    if (wasPermanentActive) thumbnailElement.dataset.isPermanent = 'false';
    if (wasMasturbatingActive) thumbnailElement.dataset.isMasturbating = 'false';
    
    let className = 'effect-active';
    if (type === 'target') className = 'target-active';
    else if (type === 'attack') className = 'attack-active';

    thumbnailElement.classList.add(className);
    
    // CSSアニメーション終了後にクラス削除 & 継続エフェクト復帰
    setTimeout(() => {
        thumbnailElement.classList.remove(className);
        
        // 永続エフェクトの再開（設定がtrueであれば復帰させる）
        if (wasPermanentActive && effectConfig.permanent !== false) {
             thumbnailElement.dataset.isPermanent = 'true';
        }
        if (wasMasturbatingActive && effectConfig.masturbate !== false) {
             thumbnailElement.dataset.isMasturbating = 'true';
        }

    }, 1000); 
    
    if (isRecording && typeof recordAction === 'function') {
        recordAction({
            type: 'effect',
            subType: type,
            zoneId: getParentZoneId(thumbnailElement.parentNode),
            slotIndex: Array.from(thumbnailElement.parentNode.parentNode.children).indexOf(thumbnailElement.parentNode)
        });
    }
}