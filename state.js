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

// SE個別設定 (初期値は全てtrue)
const seConfig = {
    'ボタン共通.mp3': true,
    'カードを配置する.mp3': true,
    'タップ.mp3': true,
    'マナ増加.mp3': true,
    'シャッフル.mp3': true,
    '1枚ドロー＆5枚ドロー.mp3': true,
    '降参.mp3': true,
    '墓地に送る.mp3': true,
    '除外する.mp3': true,
    '手札に戻す.mp3': true,
    'カードを反転させる.wav': true,
    'カウンターを置く.mp3': true,
    'カウンターを取り除く.mp3': true,
    'サイコロ.mp3': true,
    'コイントス.mp3': true,
    '自動減少.mp3': true,
    '効果発動.mp3': true,
    '対象に取る.mp3': true,
    'アタック.mp3': true,
    '常時発動.mp3': true,
    'ブロッカー.mp3': true,
    'O.mp3': true,
    'ターン開始.mp3': true
};

// エフェクト個別設定 (初期値は全てtrue)
const effectConfig = {
    'masturbate': true,   // オナニー
    'permanent': true,    // 常時発動
    'attack': true,       // アタック
    'effect': true,       // 効果発動
    'target': true,       // 対象に取る
    'autoDecrease': true  // 自動減少(盤面全体)
};

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
let currentAttackHandler = null; 
let currentTargetHandler = null;
let currentPermanentHandler = null;
let currentAddFlavorHandler = null;
let currentMasturbateHandler = null;
let currentBlockerHandler = null;
let currentMemoTarget = null;
let currentFlavorTarget = null;

const nonRotatableZones = ['deck', 'grave', 'exclude', 'hand-zone', 'deck-back-slots', 'side-deck', 'grave-back-slots', 'exclude-back-slots', 'side-deck-back-slots', 'icon-zone'];
const decorationZones = ['exclude', 'side-deck', 'grave', 'deck', 'icon-zone'];
const stackableZones = ['battle', 'spell', 'mana', 'special1', 'special2'];