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
    '【通常戦闘】For The Glorious Achievement.mp3',
    '6月の雨傘.mp3',
    'イベント31.mp3',
    'カジノ01.mp3',
    '困惑.mp3',
    '戦闘曲19.mp3',
    '夢の終わり.mp3'
];

// SE個別設定 (初期値は全てtrue)
const seConfig = {
    'ボタン共通.mp3': true,
    'カードを配置する.mp3': true,
    'スペル配置.mp3': true,
    'バトル配置.mp3': true,
    '特殊配置.mp3': true,
    'マナ配置.mp3': true,
    'タップ.mp3': true,
    'マナ増加.mp3': true,
    'シャッフル.mp3': true,
    '1枚ドロー＆5枚ドロー.mp3': true,
    '勝利.mp3': true,
    '敗北.mp3': true,
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
    'スペル効果発動.mp3': true,
    '対象に取る.mp3': true,
    'アタック.mp3': true,
    '常時発動.mp3': true,
    'ブロッカー.wav': true,
    'オナニー.wav': true,
    'ターン開始.mp3': true,
    'BPプラス.mp3': true,
    'BPマイナス.mp3': true
};

// エフェクト個別設定 (初期値は全てtrue)
const effectConfig = {
    'masturbate': true,   // オナニー
    'permanent': true,    // 常時発動
    'attack': true,       // アタック
    'effect': true,       // 効果発動
    'target': true,       // 対象選択
    'autoDecrease': true, // 自動減少
    'blocker': true,      // ブロッカー表示
    'bpChange': true      // BP変動エフェクト
};

// 自動処理設定
const autoConfig = {
    'autoManaTap': true,       // マナエリアでタップ時にマナ+1
    'autoManaPlacement': true, // マナエリアにカード配置時マナ+1
    'autoBattleCalc': true,    // 「アタック/攻撃」時のBP計算
    'autoManaTapInZone': true, // マナゾーンに置いた際にタップ状態で置く
    'autoAttackTap': true,     // 「アタック/攻撃」後タップ状態にする
    'autoManaCost': true,      // カードを出す際、メモ内の「マナ:」に応じてマナ消費
    'autoGameEnd': true,       // 降参/LP0/ドロー不可時の勝敗判定表示
    'drawFlipped': false,      // 1ドローする際、カードが反転された状態で手札に加わる
    'autoBpDestruction': true, // BP0以下で自動破壊
    'autoMasturbateDrain': true // オナニー中のBP自動減少
};

// カスタムカウンター設定
let customCounterTypes = []; // { id, name, icon }

// バトル処理状態
let isBattleTargetMode = false;
let isBattleConfirmMode = false; // バトル確認画面の状態
let currentAttacker = null;
let currentBattleTarget = null; // バトル確認中のターゲット

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
let currentExportCardHandler = null;
let currentImportCardHandler = null;
let currentMemoTarget = null;
let currentFlavorTarget = null;
let currentStockItemTarget = null;
let currentPreviewExportHandler = null; // プレビューエクスポート用

const nonRotatableZones = ['deck', 'grave', 'exclude', 'hand-zone', 'deck-back-slots', 'side-deck', 'grave-back-slots', 'exclude-back-slots', 'side-deck-back-slots', 'icon-zone', 'token-zone-slots'];
const decorationZones = ['exclude', 'side-deck', 'grave', 'deck', 'icon-zone'];
const stackableZones = ['battle', 'spell', 'mana', 'special1', 'special2'];