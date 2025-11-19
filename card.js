function createCardThumbnail(cardData, slotElement, isDecoration = false, insertAtBottom = false, ownerPrefix = '') {
    let imageSrc, isFlipped, originalSrc, counter, memo, flavor1, flavor2;

    if (typeof cardData === 'string') {
        imageSrc = cardData;
        isFlipped = false;
        originalSrc = null;
        counter = 0;
        memo = '';
        flavor1 = '';
        flavor2 = '';
    } else { // cardData is an object
        imageSrc = cardData.src;
        isDecoration = cardData.isDecoration || isDecoration; // 引数とcardDataの情報を統合
        isFlipped = cardData.isFlipped || false;
        originalSrc = cardData.originalSrc || null;
        counter = cardData.counter || 0;
        memo = cardData.memo || '';
        flavor1 = cardData.flavor1 || '';
        flavor2 = cardData.flavor2 || '';
        ownerPrefix = cardData.ownerPrefix || ownerPrefix; // 引数とcardDataの情報を統合
    }

    const thumbnailElement = document.createElement('div');
    thumbnailElement.classList.add('thumbnail');
    thumbnailElement.setAttribute('draggable', true);

    if (isDecoration) {
        thumbnailElement.dataset.isDecoration = 'true';
    }

    const imgElement = document.createElement('img');
    imgElement.classList.add('card-image');
    imgElement.dataset.rotation = 0;

    if (isFlipped && originalSrc) {
        thumbnailElement.dataset.isFlipped = 'true';
        thumbnailElement.dataset.originalSrc = originalSrc;
        imgElement.src = imageSrc;
    } else {
        thumbnailElement.dataset.isFlipped = 'false';
        imgElement.src = imageSrc;
    }

    thumbnailElement.appendChild(imgElement);

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

    // --- Event Listeners ---
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

    // Handle clicks during decoration mode for decoration zones
    if (isDecorationMode && decorationZones.includes(baseParentZoneId)) {
        openDecorationImageDialog(slotElement);
        e.stopPropagation();
        return;
    }

    // In normal mode, if a decoration card is clicked, let the parent slot's listener handle it
    if (thumbnailElement.dataset.isDecoration === 'true') {
        return;
    }
    
    // Stop if any modal is open
    if (contextMenu.style.display === 'block' || memoEditorModal.style.display === 'block' || flavorEditorModal.style.display === 'block' || draggedItem) {
        return;
    }

    // --- Normal card click logic (rotation) ---

    // Only rotate the top card in a stack
    if (getExistingThumbnail(slotElement) !== thumbnailElement) {
        e.stopPropagation();
        return;
    }

    const imgElement = thumbnailElement.querySelector('.card-image');
    if (!imgElement) return;

    // Do not rotate in non-rotatable zones
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
            }
            playSe('マナ増加.mp3');
        }
    } else {
        currentRotation = 0;
        slotElement.classList.remove('rotated-90');
        imgElement.style.transform = `rotate(${currentRotation}deg)`;
    }
    imgElement.dataset.rotation = currentRotation;
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

    // --- Set up context menu handlers ---
    currentActionHandler = () => console.log(`[${idPrefix}] 効果発動（未実装）`);
    currentTargetHandler = () => console.log(`[${idPrefix}] 対象に取る（未実装）`);
    currentAddFlavorHandler = () => openFlavorEditor(thumbnailElement);
    currentDeleteHandler = () => deleteCard(thumbnailElement);
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

    // --- Show/Hide menu items ---
    // Hide all move/flip actions for icon zone
    const hideForIcon = isIconZone;
    toGraveMenuItem.style.display = hideForIcon ? 'none' : 'block';
    toExcludeMenuItem.style.display = hideForIcon ? 'none' : 'block';
    toHandMenuItem.style.display = hideForIcon ? 'none' : 'block';
    toDeckMenuItem.style.display = hideForIcon ? 'none' : 'block';
    toSideDeckMenuItem.style.display = hideForIcon ? 'none' : 'block';
    flipMenuItem.style.display = hideForIcon ? 'none' : 'block';

    // Show other items for icon zone as well
    actionMenuItem.style.display = 'block';
    targetMenuItem.style.display = 'block';
    addCounterMenuItem.style.display = 'block';
    removeCounterMenuItem.style.display = 'block';
    memoMenuItem.style.display = 'block';
    addFlavorMenuItem.style.display = 'block';
    deleteMenuItem.style.display = 'block'; // Always show delete, except for maybe specific logic below

    // Specific logic for non-icon decoration cards
    if (!isIconZone && thumbnailElement.dataset.isDecoration === 'true') {
        // Hide most actions for decoration cards
        actionMenuItem.style.display = 'none';
        targetMenuItem.style.display = 'none';
        addCounterMenuItem.style.display = 'none';
        removeCounterMenuItem.style.display = 'none';
        memoMenuItem.style.display = 'none';
        addFlavorMenuItem.style.display = 'none';
        flipMenuItem.style.display = 'none';
        // Only show delete if in decoration mode
        deleteMenuItem.style.display = isDecorationMode ? 'block' : 'none';
    } else if (isIconZone) {
        // For icon zone, never show move/flip, but allow delete.
        // This is handled by the initial hideForIcon check.
        // We might want to prevent deleting the icon though. For now, we allow it.
    } else {
        // Normal card, check for stackable zones for counters
        if (!stackableZones.includes(sourceBaseId)) {
            addCounterMenuItem.style.display = 'none';
            removeCounterMenuItem.style.display = 'none';
        }
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


function handleCardMouseOver(e) {
    const thumbnailElement = e.target.closest('.thumbnail');
    const imgElement = thumbnailElement.querySelector('.card-image');
    if (!imgElement) return;

    const commonPreviewArea = document.getElementById('common-card-preview');
    const previewImageContainer = commonPreviewArea.querySelector('#preview-image-container');
    previewImageContainer.innerHTML = '';
    const previewImg = document.createElement('img');
    previewImg.src = thumbnailElement.dataset.isFlipped === 'true' ? (thumbnailElement.dataset.originalSrc || imgElement.src) : imgElement.src;
    
    // 修正: プレビュー画像の読み込み失敗時に画像を削除する処理を追加
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

    if (memo) {
        memoTooltip.textContent = memo;
        memoTooltip.style.display = 'block';
    }
    e.stopPropagation();
}

function handleCardMouseOut(e) {
    memoTooltip.style.display = 'none';
    e.stopPropagation();
}

function deleteCard(thumbnailElement) {
    const slotElement = thumbnailElement.parentNode;
    if (!slotElement) return;

    const parentZoneId = getParentZoneId(slotElement);
    const baseParentZoneId = getBaseId(parentZoneId);

    slotElement.removeChild(thumbnailElement);
    
    // Clear preview
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
}

function removeCounterFromCard(thumbnailElement) {
    const counterOverlay = thumbnailElement.querySelector('.card-counter-overlay');
    if (!counterOverlay) return;
    let count = parseInt(counterOverlay.dataset.counter) || 0;
    if (count > 0) count--;
    counterOverlay.dataset.counter = count;
    counterOverlay.textContent = count;
    if (count === 0) counterOverlay.style.display = 'none';
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