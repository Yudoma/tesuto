let contextMenu, deleteMenuItem, toGraveMenuItem, toExcludeMenuItem, toHandMenuItem, toDeckMenuItem, toSideDeckMenuItem, flipMenuItem, memoMenuItem, addCounterMenuItem, removeCounterMenuItem;
let actionMenuItem, targetMenuItem, addFlavorMenuItem;
let memoEditorModal, memoTextarea, memoSaveBtn, memoCancelBtn, memoTooltip;
let lightboxOverlay, lightboxContent;
let commonDrawer, commonDrawerToggle;
let commonFlipBoardBtn, commonDecorationModeBtn;
let diceRollBtn, coinTossBtn, randomResultDisplay;
let commonToggleNavBtn;
let flavorEditorModal, flavorEditorHeader, flavorPreview1, flavorPreview2;
let flavorDelete1, flavorDelete2, flavorCancelBtn;
let flavorUpload1, flavorUpload2;
let flavorDropZone1, flavorDropZone2;

// リサイズ中判定用のフラグを追加
let isResizingDrawer = false;

let stepButtons = [];
const stepOrder = ['step-start', 'step-draw', 'step-mana', 'step-main', 'step-attack', 'step-end'];
let currentStepIndex = 0;

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
    currentTargetHandler = null;
    currentAddFlavorHandler = null;
}

function performMemoSave() {
    if (currentMemoTarget) {
        const newMemo = memoTextarea.value;
        if (newMemo) {
            currentMemoTarget.dataset.memo = newMemo;
        } else {
            delete currentMemoTarget.dataset.memo;
        }
    }
    memoEditorModal.style.display = 'none';
    currentMemoTarget = null;
}

function performMemoCancel() {
    memoEditorModal.style.display = 'none';
    currentMemoTarget = null;
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
    fileInput.click();
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
                    playSe('ボタン共通.mp3');
                    currentStepIndex = index;
                    updateStepUI();

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
                    }
                }
            });
        }
    });

    currentStepIndex = 0;
    updateStepUI();
}


function setupUI() {
    document.addEventListener('contextmenu', (e) => e.preventDefault());
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());

    contextMenu = document.getElementById('custom-context-menu');
    actionMenuItem = document.getElementById('context-menu-action');
    targetMenuItem = document.getElementById('context-menu-target');
    deleteMenuItem = document.getElementById('context-menu-delete');
    toGraveMenuItem = document.getElementById('context-menu-to-grave');
    toExcludeMenuItem = document.getElementById('context-menu-to-exclude');
    toHandMenuItem = document.getElementById('context-menu-to-hand');
    toDeckMenuItem = document.getElementById('context-menu-to-deck');
    toSideDeckMenuItem = document.getElementById('context-menu-to-side-deck');
    flipMenuItem = document.getElementById('context-menu-flip');
    memoMenuItem = document.getElementById('context-menu-memo');
    addCounterMenuItem = document.getElementById('context-menu-add-counter');
    removeCounterMenuItem = document.getElementById('context-menu-remove-counter');
    addFlavorMenuItem = document.getElementById('context-menu-add-flavor');

    memoEditorModal = document.getElementById('memo-editor');
    memoTextarea = document.getElementById('memo-editor-textarea');
    memoSaveBtn = document.getElementById('memo-editor-save');
    memoCancelBtn = document.getElementById('memo-editor-cancel');
    memoTooltip = document.getElementById('memo-tooltip');

    lightboxOverlay = document.getElementById('lightbox-overlay');
    lightboxContent = document.getElementById('lightbox-content');

    const commonPreviewArea = document.getElementById('common-card-preview');

    commonDrawer = document.getElementById('common-drawer');
    commonDrawerToggle = document.getElementById('common-drawer-toggle');
    commonFlipBoardBtn = document.getElementById('common-flip-board-btn');
    commonDecorationModeBtn = document.getElementById('common-decoration-mode-btn');
    commonToggleNavBtn = document.getElementById('common-toggle-nav-btn');

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

    if (!contextMenu || !deleteMenuItem || !toGraveMenuItem || !toExcludeMenuItem || !toHandMenuItem || !toDeckMenuItem || !toSideDeckMenuItem || !flipMenuItem || !addCounterMenuItem || !removeCounterMenuItem
        || !actionMenuItem || !targetMenuItem || !addFlavorMenuItem
        || !memoMenuItem || !memoEditorModal || !memoTextarea || !memoSaveBtn || !memoCancelBtn || !memoTooltip
        || !lightboxOverlay || !lightboxContent || !commonPreviewArea
        || !commonDrawer || !commonDrawerToggle || !commonFlipBoardBtn || !commonDecorationModeBtn || !commonToggleNavBtn
        || !diceRollBtn || !coinTossBtn || !randomResultDisplay
        || !flavorEditorModal || !flavorEditorHeader || !flavorPreview1 || !flavorPreview2 || !flavorDelete1 || !flavorDelete2
        || !flavorDropZone1 || !flavorDropZone2 || !flavorUpload1 || !flavorUpload2
        || !flavorCancelBtn
    ) {
        console.error("必須UI要素が見つかりません。");
        return;
    }

    lightboxOverlay.addEventListener('click', (e) => closeLightbox());
    lightboxContent.addEventListener('click', (e) => {
        if (e.target === lightboxContent) {
            closeLightbox();
        }
    });

    document.addEventListener('click', (e) => {
        // リサイズ操作直後のクリックイベントなら無視する
        if (isResizingDrawer) {
            return;
        }

        const shouldNotClose = 
            (contextMenu.style.display === 'block' && e.target.closest('#custom-context-menu')) ||
            (memoEditorModal.style.display === 'block' && e.target.closest('#memo-editor')) ||
            (flavorEditorModal.style.display === 'block' && e.target.closest('#flavor-editor')) ||
            e.target.closest('#common-drawer-toggle');

        if (shouldNotClose) {
            return;
        }

        closeContextMenu();

        if (commonDrawer.classList.contains('open') && !e.target.closest('#common-drawer')) {
            if (!isDecorationMode) {
                commonDrawer.classList.remove('open');
            }
        }

        const playerDrawer = document.getElementById('player-drawer');
        if (playerDrawer && playerDrawer.classList.contains('open')) {
            if (!e.target.closest('#player-drawer') && !e.target.closest('#player-drawer-toggle')) {
                playerDrawer.classList.remove('open');
            }
        }

        const opponentDrawer = document.getElementById('opponent-drawer');
        if (opponentDrawer && opponentDrawer.classList.contains('open')) {
            if (!e.target.closest('#opponent-drawer') && !e.target.closest('#opponent-drawer-toggle')) {
                opponentDrawer.classList.remove('open');
            }
        }
    });

    contextMenu.addEventListener('contextmenu', (e) => e.preventDefault());
    
    // Context Menu Actions
    actionMenuItem.addEventListener('click', () => { 
        playSe('ボタン共通.mp3');
        if (typeof currentActionHandler === 'function') currentActionHandler(); 
        closeContextMenu(); 
    });
    targetMenuItem.addEventListener('click', () => { 
        playSe('ボタン共通.mp3');
        if (typeof currentTargetHandler === 'function') currentTargetHandler(); 
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
    addFlavorMenuItem.addEventListener('click', () => { 
        playSe('ボタン共通.mp3');
        if (typeof currentAddFlavorHandler === 'function') currentAddFlavorHandler(); 
        closeContextMenu(); 
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

    const turnInput = document.getElementById('common-turn-value');
    const turnPrevBtn = document.getElementById('common-turn-prev');
    const turnNextBtn = document.getElementById('common-turn-next');
    
    if (turnInput && turnPrevBtn && turnNextBtn) {
        const updateTurnValue = (change) => {
            playSe('ボタン共通.mp3');
            let currentValue = parseInt(turnInput.value) || 1;
            currentValue = Math.max(1, currentValue + change);
            turnInput.value = currentValue;
        };
        turnPrevBtn.addEventListener('click', () => updateTurnValue(-1));
        turnNextBtn.addEventListener('click', () => updateTurnValue(1));
    }

    commonDrawerToggle.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        commonDrawer.classList.toggle('open');
    });
    commonFlipBoardBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        document.body.classList.toggle('board-flipped');
        document.getElementById('player-drawer')?.classList.remove('open');
        document.getElementById('opponent-drawer')?.classList.remove('open');
    });

    commonDecorationModeBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        isDecorationMode = !isDecorationMode;
        commonDecorationModeBtn.textContent = isDecorationMode ? 'キャンセル' : '装飾モード';
        commonDecorationModeBtn.dataset.active = isDecorationMode;
        document.body.classList.toggle('common-decoration-mode-active', isDecorationMode);

        const highlightTargets = document.querySelectorAll(
            '.exclude-zone .card-slot, .side-deck-zone .card-slot, .grave-zone .card-slot, .deck-zone .card-slot, #icon-zone, #opponent-icon-zone'
        );

        highlightTargets.forEach(target => {
            if(target) target.classList.toggle('decoration-highlight', isDecorationMode);
        });
    });

    diceRollBtn.addEventListener('click', () => {
        playSe('サイコロ.mp3');
        const result = Math.floor(Math.random() * 6) + 1;
        randomResultDisplay.textContent = `ダイス: ${result}`;
    });

    coinTossBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        const result = Math.random() < 0.5 ? 'ウラ' : 'オモテ';
        randomResultDisplay.textContent = `コイン: ${result}`;
    });

    commonToggleNavBtn.addEventListener('click', () => {
        playSe('ボタン共通.mp3');
        const isHidden = document.body.classList.toggle('nav-hidden');
        commonToggleNavBtn.textContent = isHidden ? 'ナビ再表示' : 'ナビ非表示';
    });

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

    // ドロワーのリサイズ機能を初期化
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
            initialX = e.clientX - xOffset;
            initialY = e.clientY - yOffset;

            if (e.target === commonDrawerHeader || e.target.parentNode === commonDrawerHeader) {
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
                
                let newX = e.clientX - initialX;
                let newY = e.clientY - initialY;

                // 画面外にはみ出ないように制限
                const drawerWidth = commonDrawer.offsetWidth;
                const drawerHeight = commonDrawer.offsetHeight;
                const vw = window.innerWidth;
                const vh = window.innerHeight;

                // 中心(50%, 50%)からのオフセットの限界値を計算
                // ドロワーが画面より大きい場合は、はみ出しを許容しつつ中央に寄せる（制限値0以上）
                const limitX = Math.max(0, (vw - drawerWidth) / 2);
                const limitY = Math.max(0, (vh - drawerHeight) / 2);

                // 制約を適用
                if (newX < -limitX) newX = -limitX;
                if (newX > limitX) newX = limitX;
                
                if (newY < -limitY) newY = -limitY;
                if (newY > limitY) newY = limitY;

                currentX = newX;
                currentY = newY;

                xOffset = currentX;
                yOffset = currentY;

                setTranslate(currentX, currentY, commonDrawer);
            }
        }

        function setTranslate(xPos, yPos, el) {
            el.style.transform = `translate(calc(-50% + ${xPos}px), calc(-50% + ${yPos}px))`;
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
}

function activateDrawerTab(targetId, drawerElement) {
    if (!drawerElement) return;
    const drawerPanels = drawerElement.querySelectorAll('.drawer-panel');
    const drawerTabs = drawerElement.querySelectorAll('.drawer-tab-btn');
    
    drawerPanels.forEach(p => p.classList.toggle('active', p.id === targetId));
    drawerTabs.forEach(t => t.classList.toggle('active', t.dataset.target === targetId));

    // テキストファイルの読み込みトリガー
    if (targetId === 'common-spec-panel') {
        loadTextContent('txt/仕様説明.txt', 'spec-text-content');
    } else if (targetId === 'common-about-panel') {
        loadTextContent('txt/S＆Mとは.txt', 'about-text-content');
    }
}

async function loadTextContent(filePath, elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    // 既に読み込み済み（デフォルトの読み込み中...ではない）なら再取得しない
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
    const scrollableContainers = document.querySelectorAll('.deck-back-slot-container, .free-space-slot-container');

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
        isResizingDrawer = true; // グローバルフラグON
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

        // 画面外にはみ出ないように制限を追加
        const rect = drawer.getBoundingClientRect();
        // 現在の中心座標
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        
        // 中心から画面端までの距離の2倍が、その中心位置で許容される最大サイズ
        // これを超えると画面からはみ出すことになる
        const maxWidth = 2 * Math.min(centerX, vw - centerX);
        const maxHeight = 2 * Math.min(centerY, vh - centerY);
        
        // 最小サイズ(300px)と計算した最大サイズでクリップ
        newW = Math.max(300, Math.min(newW, maxWidth));
        newH = Math.max(300, Math.min(newH, maxHeight));
        
        drawer.style.width = `${newW}px`;
        drawer.style.height = `${newH}px`;
    }

    function handleMouseUp(e) {
        isResizing = false;
        
        // クリックイベントが発火するのを待ってからフラグをOFFにする
        setTimeout(() => {
            isResizingDrawer = false;
        }, 100);

        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
    }
}

function openDecorationImageDialog(targetSlot) {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.style.display = 'none';

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) {
            document.body.removeChild(fileInput);
            return;
        }

        const reader = new FileReader();
        reader.onload = (readEvent) => {
            const imageData = readEvent.target.result;
            const idPrefix = getPrefixFromZoneId(targetSlot.id);
            const baseId = getBaseId(targetSlot.id);

            let existingThumbnail = targetSlot.querySelector('.thumbnail[data-is-decoration="true"]');
            if (existingThumbnail) {
                const img = existingThumbnail.querySelector('img');
                if (img) img.src = imageData;
            } else {
                const anyThumbnail = getExistingThumbnail(targetSlot);
                if (anyThumbnail) targetSlot.removeChild(anyThumbnail);
                createCardThumbnail(imageData, targetSlot, true, false, idPrefix);
            }

            if (baseId !== 'icon-zone') {
                 syncMainZoneImage(baseId, idPrefix);
            }
        };
        reader.readAsDataURL(file);
        document.body.removeChild(fileInput);
    });

    fileInput.addEventListener('cancel', () => {
        if (document.body.contains(fileInput)) {
            document.body.removeChild(fileInput);
        }
    });

    document.body.appendChild(fileInput);
    fileInput.click();
}