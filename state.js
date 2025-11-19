// グローバルな状態管理
let draggedItem = null;
let isDecorationMode = false;

// UIハンドラと状態
let currentDeleteHandler = null;
let currentMoveToGraveHandler = null;
let currentMoveToExcludeHandler = null;
let currentMoveToHandHandler = null;
let currentMoveToDeckHandler = null;
let currentMoveToSideDeckHandler = null;
let currentFlipHandler = null;
let currentMemoHandler = null;
let currentAddCounterHandler = null;
let currentRemoveCounterHandler = null;
let currentActionHandler = null;
let currentTargetHandler = null;
let currentAddFlavorHandler = null;
let currentMemoTarget = null;
let currentFlavorTarget = null;

// 定数
const nonRotatableZones = ['deck', 'grave', 'exclude', 'hand-zone', 'deck-back-slots', 'side-deck', 'grave-back-slots', 'exclude-back-slots', 'side-deck-back-slots', 'icon-zone'];
const decorationZones = ['exclude', 'side-deck', 'grave', 'deck', 'icon-zone'];
const stackableZones = ['battle', 'spell', 'mana', 'special1', 'special2'];
