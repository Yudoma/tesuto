let draggedItem = null;
let isDecorationMode = false;

let isRecording = false;
let isPlaying = false;
let actionLog = [];
let replayStartTime = 0;
let replayTimerIds = [];

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
let currentMasturbateHandler = null;
let currentMemoTarget = null;
let currentFlavorTarget = null;

const nonRotatableZones = ['deck', 'grave', 'exclude', 'hand-zone', 'deck-back-slots', 'side-deck', 'grave-back-slots', 'exclude-back-slots', 'side-deck-back-slots', 'icon-zone'];
const decorationZones = ['exclude', 'side-deck', 'grave', 'deck', 'icon-zone'];
const stackableZones = ['battle', 'spell', 'mana', 'special1', 'special2'];