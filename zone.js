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
                    createCardThumbnail(readEvent.target.result, slot, false, false, idPrefix);
                    updateSlotStackState(slot);
                    playSe('カードを配置する.mp3');
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

    // Guard clause: Only allow drops on decoration zones if in decoration mode.
    // Note: If not in decoration mode, we allow dropping on pile zones (deck, grave etc) to add cards.
    
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
        };
        reader.readAsDataURL(file);
        return;
    }

    const pileZones = ['deck', 'grave', 'exclude', 'side-deck'];
    const isPileZone = pileZones.includes(targetParentBaseId);

    // Determine if it's a direct single-card slot (battle, spell, mana, special, etc.)
    // Pile zones (even the top slot) should be treated as "add to list" in normal mode, not "replace single card".
    const isDirectBoardSlot = !targetParentBaseId.endsWith('-back-slots') 
                              && targetParentBaseId !== 'hand-zone' 
                              && targetParentBaseId !== 'free-space-slots'
                              && !isPileZone;

    if (isDirectBoardSlot) {
        const file = files[0]; // Take only the first file
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const isTargetStackable = stackableZones.includes(targetParentBaseId);
                const existingThumbnail = getExistingThumbnail(targetSlot);
                
                if (!isTargetStackable && existingThumbnail) {
                    targetSlot.removeChild(existingThumbnail);
                    resetSlotToDefault(targetSlot);
                }
                
                createCardThumbnail(event.target.result, targetSlot, false, false, idPrefix);
                
                if (isTargetStackable) {
                    updateSlotStackState(targetSlot);
                }
                playSe('カードを配置する.mp3');
            };
            reader.readAsDataURL(file);
        }
    } else {
        // Multi-file handling for Hand, Deck, Grave, Exclude, Side-Deck, Free Space
        
        // Helper to process files into a specific container
        const addFilesToContainer = (fileList, containerId) => {
            const container = document.getElementById(containerId);
            if (!container) return;
            
            let slotsContainer = container.querySelector('.deck-back-slot-container, .free-space-slot-container') || container;
            const availableSlots = Array.from(slotsContainer.querySelectorAll('.card-slot')).filter(s => !s.querySelector('.thumbnail'));
            
            fileList.forEach((file, index) => {
                if (availableSlots[index]) {
                    const reader = new FileReader();
                    reader.onload = (event) => createCardThumbnail(event.target.result, availableSlots[index], false, false, idPrefix);
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
            // Target is Deck, Grave, Exclude, Side-Deck, Free Space, or their back-slots
            let destinationId;
            
            if (isPileZone) {
                destinationId = idPrefix + targetParentBaseId + '-back-slots';
            } else if (targetParentBaseId.endsWith('-back-slots') || targetParentBaseId === 'free-space-slots') {
                destinationId = targetParentZoneId;
            } else {
                destinationId = idPrefix + 'deck-back-slots'; // Fallback
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

    // --- Logic for moving between different zones ---
    const isTargetStackable = stackableZones.includes(targetBaseZoneId);
    const existingThumbnail = getExistingThumbnail(targetSlot);

    if (sourceSlot === targetSlot) return;

    // Moving to a multi-card zone (deck, grave, etc.)
    if (targetBaseZoneId.endsWith('-back-slots') || ['deck', 'grave', 'exclude', 'side-deck'].includes(targetBaseZoneId)) {
        if (targetBaseZoneId === 'grave' || targetBaseZoneId === 'grave-back-slots') {
            playSe('墓地に送る.mp3');
        } else if (targetBaseZoneId === 'exclude' || targetBaseZoneId === 'exclude-back-slots') {
            playSe('除外する.mp3');
        }

        let multiZoneId = targetZoneId;
        if(['deck', 'grave', 'exclude', 'side-deck'].includes(targetBaseZoneId)) {
            multiZoneId = idPrefix + targetBaseZoneId + '-back-slots';
        }
        moveCardToMultiZone(draggedItem, getBaseId(multiZoneId).replace('-back-slots',''));
        return;
    }


    // Moving to a stackable zone
    if (isTargetStackable && !isDecorationMode) {
        sourceSlot.removeChild(draggedItem);
        targetSlot.insertBefore(draggedItem, targetSlot.firstChild);
    }
    // Swapping cards
    else if (existingThumbnail && sourceSlot !== targetSlot) {
        sourceSlot.appendChild(existingThumbnail);
        targetSlot.appendChild(draggedItem);
    }
    // Moving to an empty slot
    else if (!existingThumbnail) {
        sourceSlot.removeChild(draggedItem);
        targetSlot.appendChild(draggedItem);
    } else {
        return; // Do nothing if dropping on a non-stackable, occupied slot
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