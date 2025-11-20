let draggedItem = null;
let isDecorationMode = false;

let isRecording = false;
let isPlaying = false;
let isReplayPaused = false;
let currentReplayIndex = 0;
let currentReplayFileName = '';
let actionLog = [];
let replayStartTime = 0;
let replayTimerIds = [];

// Audio State
let bgmVolume = 5;
let seVolume = 5;
let currentBgmAudio = null;

// BGMリスト: bgmフォルダに入れているファイル名をここに記述してください
const bgmFileList = [
    'Battle.mp3',
    'Theme.mp3',
    'Ending.wav'
];

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
let currentAttackHandler = null; // 追加: アタック
let currentTargetHandler = null;
let currentPermanentHandler = null; // 追加: 常時発動
let currentAddFlavorHandler = null;
let currentMasturbateHandler = null;
let currentBlockerHandler = null;
let currentMemoTarget = null;
let currentFlavorTarget = null;

const nonRotatableZones = ['deck', 'grave', 'exclude', 'hand-zone', 'deck-back-slots', 'side-deck', 'grave-back-slots', 'exclude-back-slots', 'side-deck-back-slots', 'icon-zone'];
const decorationZones = ['exclude', 'side-deck', 'grave', 'deck', 'icon-zone'];
const stackableZones = ['battle', 'spell', 'mana', 'special1', 'special2'];