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

    // --- ▽ 修正箇所 (ここから) ▽ ---

    // キャンバスの解像度をウインドウサイズに合わせる関数
    function handleResize() {
        // 実際のウインドウサイズを取得
        const width = window.innerWidth;
        const height = window.innerHeight;

        // キャンバスの描画バッファサイズを設定
        canvas.width = width;
        canvas.height = height;

        // Effekseerにリサイズを通知
        effekseer.resize(width, height);
    }

    // 最初に一度、サイズを合わせる
    handleResize();

    // ウィンドウサイズが変更された時にもサイズを合わせる
    window.addEventListener('resize', handleResize);

    // --- △ 修正箇所 (ここまで) △ ---


    // エフェクトファイルのパス
    const effectUrl = './pipoya-saceffect_001.efkefc';

    // エフェクトの読み込み
    const effect = effekseer.loadEffect(effectUrl, './', () => {
        
        // 読み込み完了後にエフェクトを再生
        effekseer.play(effect, 0, 0, 0);

    }, (err) => {
        // ロード失敗時の処理
        console.error('Failed to load effect:', err);
        alert('エフェクトファイルの読み込みに失敗しました。\n' + effectUrl);
    });

    // 描画ループを開始
    function loop() {
        requestAnimationFrame(loop);
        effekseer.update();
        
        // 3D空間のカメラを設定
        // (アスペクト比を canvas.width / canvas.height から取得する)
        effekseer.setViewerMatrix(
            effekseer.createMatrix().lookAt(
                effekseer.createVector3(0, 5, 20),
                effekseer.createVector3(0, 0, 0),
                effekseer.createVector3(0, 1, 0)
            )
        );
        
        effekseer.setProjectionMatrix(
            effekseer.createMatrix().perspective(
                60 * Math.PI / 180,
                canvas.width / canvas.height, // 修正後の解像度でアスペクト比を計算
                1.0,
                100.0
            )
        );

        effekseer.draw();
    }

    loop();
};