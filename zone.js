function addSlotEventListeners(slot) {
    slot.addEventListener('dragover', handleDragOver);
    slot.addEventListener('dragleave', handleDragLeave);
    slot.addEventListener('drop', handleDropOnSlot);
    slot.addEventListener('click', handleSlotClick);
}

function handleSlotClick(e) {
    const slot = e.currentTarget;
    const parentZoneId = getParentZoneId(slot);
    const baseParentZoneId = getBaseId(parentZoneId);

    const drawerOpeningZones = ['deck', 'grave', 'exclude', 'side-deck'];
    // 装飾モードでない場合のみ、ドローワー開閉ゾーンでのクリックを無視する
    if (drawerOpeningZones.includes(baseParentZoneId) && !isDecorationMode) {
        return;
    }

    if (memoEditorModal.style.display === 'block' || contextMenu.style.display === 'block' || flavorEditorModal.style.display === 'block') {
        return;
    }

    if (isDecorationMode) {
        if (e.target.closest('.thumbnail')) {
            return;
        }
        if (decorationZones.includes(baseParentZoneId)) {
            openDecorationImageDialog(slot);
            e.stopPropagation();
        }
        return;
    }

    if (slot.querySelector('.thumbnail')) {
        return;
    }

    const allowedZonesForNormalModeFileDrop = [
        'hand-zone', 'battle', 'spell', 'special1', 'special2', 'free-space-slots',
        'deck-back-slots', 'grave-back-slots', 'exclude-back-slots', 'side-deck-back-slots'
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
                    createCardThumbnail(imageData, slot, false, false, idPrefix);
                    updateSlotStackState(slot);
                    playSe('カードを配置する.mp3');

                    if (isRecording && typeof recordAction === 'function') {
                        recordAction({
                            type: 'newCard',
                            zoneId: getParentZoneId(slot),
                            slotIndex: Array.from(slot.parentNode.children).indexOf(slot),
                            cardData: imageData
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
        fileInput.click();
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

    // Handle file drop
    if (e.dataTransfer.files.length > 0) {
        handleFileDrop(e, slot, idPrefix);
        return;
    }

    // Handle card drop
    if (draggedItem) {
        handleCardDrop(draggedItem, slot, idPrefix);
    }
}

function handleFileDrop(e, targetSlot, idPrefix) {
    const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith('image/'));
    if (files.length === 0) return;

    const targetParentZoneId = getParentZoneId(targetSlot);
    const targetParentBaseId = getBaseId(targetParentZoneId);

    // Decoration mode logic
    if (isDecorationMode && decorationZones.includes(targetParentBaseId)) {
        const file = files[0];
        const reader = new FileReader();
        reader.onload = (event) => {
            const imageData = event.target.result;
            const existingThumbnail = targetSlot.querySelector('.thumbnail[data-is-decoration="true"]');
            if (existingThumbnail) {
                const existingImg = existingThumbnail.querySelector('img');
                if (existingImg) existingImg.src = imageData;
            } else {
                const anyExistingThumbnail = getExistingThumbnail(targetSlot);
                if (anyExistingThumbnail) targetSlot.removeChild(anyExistingThumbnail);
                createCardThumbnail(imageData, targetSlot, true, false, idPrefix);
            }
            syncMainZoneImage(targetParentBaseId, idPrefix);
            playSe('カードを配置する.mp3');

            if (isRecording && typeof recordAction === 'function') {
                recordAction({
                    type: 'updateDecoration',
                    zoneId: targetParentZoneId,
                    imageData: imageData
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
                createCardThumbnail(imageData, targetSlot, false, false, idPrefix);
                
                if (isTargetStackable) {
                    updateSlotStackState(targetSlot);
                }
                playSe('カードを配置する.mp3');

                if (isRecording && typeof recordAction === 'function') {
                    recordAction({
                        type: 'newCard',
                        zoneId: targetParentZoneId,
                        slotIndex: Array.from(targetSlot.parentNode.children).indexOf(targetSlot),
                        cardData: imageData
                    });
                }
            };
            reader.readAsDataURL(file);
        }
    } else {
        // Multi-file handling
        
        const addFilesToContainer = (fileList, containerId) => {
            const container = document.getElementById(containerId);
            if (!container) return;
            
            let slotsContainer = container.querySelector('.deck-back-slot-container, .free-space-slot-container') || container;
            const allSlots = Array.from(slotsContainer.querySelectorAll('.card-slot'));
            const availableSlots = allSlots.filter(s => !s.querySelector('.thumbnail'));
            
            fileList.forEach((file, index) => {
                if (availableSlots[index]) {
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        const imageData = event.target.result;
                        createCardThumbnail(imageData, availableSlots[index], false, false, idPrefix);
                        
                        if (isRecording && typeof recordAction === 'function') {
                            // Find the current index of the slot we are dropping into
                            const slotIndex = allSlots.indexOf(availableSlots[index]);
                            recordAction({
                                type: 'newCard',
                                zoneId: containerId,
                                slotIndex: slotIndex,
                                cardData: imageData
                            });
                        }
                    };
                    reader.readAsDataURL(file);
                }
            });

            if (fileList.length > 0) {
                playSe('カードを配置する.mp3');
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
            } else if (targetParentBaseId.endsWith('-back-slots') || targetParentBaseId === 'free-space-slots') {
                destinationId = targetParentZoneId;
            } else {
                destinationId = idPrefix + 'deck-back-slots'; 
            }
            
            addFilesToContainer(files, destinationId);
        }
    }
}

function handleCardDrop(draggedItem, targetSlot, idPrefix) {
    if (draggedItem.dataset.isDecoration === 'true' && !isDecorationMode) return;

    const sourceSlot = draggedItem.parentNode;
    const sourceZoneId = getParentZoneId(sourceSlot);
    const sourceBaseZoneId = getBaseId(sourceZoneId);

    const targetZoneId = getParentZoneId(targetSlot);
    const targetBaseZoneId = getBaseId(targetZoneId);

    if (sourceSlot === targetSlot) return;

    // 記録用に移動前の情報を取得
    const fromZoneId = sourceZoneId;
    const fromSlotIndex = Array.from(sourceSlot.parentNode.children).indexOf(sourceSlot);

    // --- SE Playback Logic ---
    const isGrave = targetBaseZoneId === 'grave' || targetBaseZoneId === 'grave-back-slots';
    const isExclude = targetBaseZoneId === 'exclude' || targetBaseZoneId === 'exclude-back-slots';
    
    if (isGrave) {
        playSe('墓地に送る.mp3');
    } else if (isExclude) {
        playSe('除外する.mp3');
    } else {
        playSe('カードを配置する.mp3');
    }

    // --- Logic for moving between different zones ---
    const isTargetStackable = stackableZones.includes(targetBaseZoneId);
    const existingThumbnail = getExistingThumbnail(targetSlot);

    // Moving to a multi-card zone (deck, grave, etc.)
    if (targetBaseZoneId.endsWith('-back-slots') || ['deck', 'grave', 'exclude', 'side-deck'].includes(targetBaseZoneId)) {
        let multiZoneId = targetZoneId;
        if(['deck', 'grave', 'exclude', 'side-deck'].includes(targetBaseZoneId)) {
            multiZoneId = idPrefix + targetBaseZoneId + '-back-slots';
        }
        // moveCardToMultiZone内で記録を行うため、ここでは呼び出すだけ
        moveCardToMultiZone(draggedItem, getBaseId(multiZoneId).replace('-back-slots',''));
        return;
    }

    // Moving to a stackable zone
    if (isTargetStackable && !isDecorationMode) {
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
    // Swapping cards
    else if (existingThumbnail && sourceSlot !== targetSlot) {
        sourceSlot.appendChild(existingThumbnail);
        targetSlot.appendChild(draggedItem);
        
        if (isRecording && typeof recordAction === 'function') {
            recordAction({
                type: 'move', // Swapもmoveとして記録し、再生側ロジックで「移動先にカードがあればSwap」として扱う
                fromZone: fromZoneId,
                fromSlotIndex: fromSlotIndex,
                toZone: targetZoneId,
                toSlotIndex: Array.from(targetSlot.parentNode.children).indexOf(targetSlot)
            });
        }
    }
    // Moving to an empty slot
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

    // --- Cleanup and update ---
    [sourceSlot, targetSlot].forEach(slot => {
        resetSlotToDefault(slot);
        updateSlotStackState(slot);
        const zoneId = getParentZoneId(slot);
        if(zoneId.endsWith('-back-slots')) arrangeSlots(zoneId);
        if(decorationZones.includes(getBaseId(zoneId))) syncMainZoneImage(getBaseId(zoneId), getPrefixFromZoneId(zoneId));
    });
}


function updateSlotStackState(slotElement) {
    if (!slotElement) return;
    const thumbnailCount = slotElement.querySelectorAll('.thumbnail:not([data-is-decoration="true"])').length;
    slotElement.classList.toggle('stacked', thumbnailCount > 1);
}

function arrangeSlots(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    let slotsContainer = container.querySelector('.deck-back-slot-container, .free-space-slot-container') || container;
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

    const isTargetHand = (targetBaseZoneId === 'hand');
    const destinationMultiZoneId = isTargetHand ? idPrefix + 'hand-zone' : idPrefix + targetBaseZoneId + '-back-slots';

    if (sourceSlot === destinationMultiZoneId || sourceSlot.id === destinationMultiZoneId) return;

    // 記録用：移動前の情報
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

    // 記録用：移動後の情報
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

    if (sourceParentZoneId.endsWith('-back-slots') || sourceParentBaseId === 'hand-zone' || sourceParentBaseId === 'free-space-slots') {
        arrangeSlots(sourceParentZoneId);
    }
    resetSlotToDefault(sourceSlot);
    updateSlotStackState(sourceSlot);

    if (decorationZones.includes(sourceParentBaseId)) syncMainZoneImage(sourceParentBaseId, getPrefixFromZoneId(sourceParentZoneId));

    arrangeSlots(destinationMultiZoneId);
    if (!isTargetHand) {
        syncMainZoneImage(targetBaseZoneId, idPrefix);
    }
}