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

function playSe(filename, isLoop = false) {
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