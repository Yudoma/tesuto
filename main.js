window.onload = function() {
    const canvas = document.getElementById('efk_canvas');
    if (!canvas) {
        console.error('Canvas element not found!');
        return;
    }

    // Effekseerの初期化 (WebGLコンテキストを取得)
    try {
        effekseer.init(canvas);
    } catch (e) {
        console.error('Failed to initialize Effekseer:', e);
        alert('Effekseerの初期化に失敗しました。WebGLがサポートされているか確認してください。');
        return;
    }

    // エフェクトファイルのパス
    // (index.html と同じ階層にある前提)
    const effectUrl = './pipoya-saceffect_001.efkefc';

    // エフェクトの読み込み
    [cite_start]// .efkefc ファイル内で参照されているテクスチャパス [cite: 1, 3] が
    // 'Texture/pipoya-saceffect_001_192.png' のため、
    // 基準パス(第二引数)は './' (カレントディレクトリ) を指定します。
    const effect = effekseer.loadEffect(effectUrl, './', () => {
        
        // 読み込み完了後にエフェクトを再生
        // (0, 0, 0) の座標で再生開始
        effekseer.play(effect, 0, 0, 0);

    }, (err) => {
        // ロード失敗時の処理
        console.error('Failed to load effect:', err);
        alert('エフェクトファイルの読み込みに失敗しました。\n' + effectUrl + '\nファイルが正しい場所にあるか確認してください。');
    });

    // 描画ループを開始
    function loop() {
        // 次のフレームで loop を再度実行
        requestAnimationFrame(loop);

        // Effekseerの内部状態を更新
        effekseer.update();
        
        // 3D空間のカメラを設定
        // ビュー（視点）マトリックス
        effekseer.setViewerMatrix(
            effekseer.createMatrix().lookAt(
                effekseer.createVector3(0, 5, 20),  // カメラの位置 (Z軸手前)
                effekseer.createVector3(0, 0, 0),   // 注視点 (原点)
                effekseer.createVector3(0, 1, 0)    // カメラの上方向 (Y軸)
            )
        );
        
        // プロジェクション（投影）マトリックス
        effekseer.setProjectionMatrix(
            effekseer.createMatrix().perspective(
                60 * Math.PI / 180,             // 視野角 (60度)
                canvas.width / canvas.height,   // アスペクト比
                1.0,                            // ニアクリップ
                100.0                           // ファークリップ
            )
        );

        // Effekseerの描画処理
        effekseer.draw();
    }

    loop(); // 最初のループを開始

    // ウィンドウサイズが変更された時にキャンバスサイズを追従させる
    window.addEventListener('resize', () => {
        effekseer.resize(canvas.width, canvas.height);
    });
};