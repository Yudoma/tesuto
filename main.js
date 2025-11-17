window.onload = function() {
    const canvas = document.getElementById('efk_canvas');
    if (!canvas) {
        console.error('Canvas element not found!');
        return;
    }

    // Effekseerの初期化
    try {
        effekseer.init(canvas);
    } catch (e) {
        console.error('Failed to initialize Effekseer:', e);
        alert('Effekseerの初期化に失敗しました。');
        return;
    }

    // キャンバスの解像度と2Dカメラをウインドウサイズに合わせる関数
    function handleResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
        effekseer.resize(width, height);

        effekseer.setProjectionMatrix(
            effekseer.createMatrix().ortho(0, width, height, 0, -1000, 1000)
        );
        effekseer.setViewerMatrix(
            effekseer.createMatrix().lookAt(
                effekseer.createVector3(0, 0, 1),
                effekseer.createVector3(0, 0, 0),
                effekseer.createVector3(0, 1, 0)
            )
        );
    }

    // 最初に一度、サイズを合わせる
    handleResize();

    // ウィンドウサイズが変更された時にもサイズを合わせる
    window.addEventListener('resize', handleResize);

    // エフェクトファイルのパス
    const effectUrl = './pipoya-saceffect_001.efkefc';

    // エフェクトの読み込み
    const effect = effekseer.loadEffect(effectUrl, './', () => {
        
        console.log('Effect load complete. Playing effect at center.');

        // --- ▽ 修正箇所 (ここから) ▽ ---
        // スケール（拡大率）を10倍にして再生してみる
        const scale = 10.0;
        console.log(`Playing with scale: ${scale}`);

        // 読み込み完了後、画面の「中央」に再生
        effekseer.play(effect, canvas.width / 2, canvas.height / 2, 0, scale);

        // 3秒ごとにもう一度再生する (確認のため)
        setInterval(() => {
            console.log(`Re-playing effect at center with scale: ${scale}`);
            // 座標とスケールを再指定して再生
            effekseer.play(effect, canvas.width / 2, canvas.height / 2, 0, scale);
        }, 3000);
        // --- △ 修正箇所 (ここまで) △ ---

    }, (err) => {
        console.error('Failed to load effect:', err);
        alert('エフェクトファイルの読み込みに失敗しました。\n' + effectUrl);
    });

    // 描画ループを開始
    function loop() {
        requestAnimationFrame(loop);
        effekseer.update();
        effekseer.draw();
    }

    loop();
};