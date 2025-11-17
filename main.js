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

        // --- ▽ 2D用のカメラ設定 (ここから) ▽ ---
        // (0, 0) を左上隅、(width, height) を右下隅とする
        // Z深度（描画範囲）を -1000 ～ 1000 に拡大
        effekseer.setProjectionMatrix(
            effekseer.createMatrix().ortho(0, width, height, 0, -1000, 1000) /* <-- 修正箇所 */
        );
        effekseer.setViewerMatrix(
            effekseer.createMatrix().lookAt(
                effekseer.createVector3(0, 0, 1), // 2D用の標準ビュー
                effekseer.createVector3(0, 0, 0),
                effekseer.createVector3(0, 1, 0)
            )
        );
        // --- △ 2D用のカメラ設定 (ここまで) △ ---
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

        // 読み込み完了後、画面の「中央」に再生
        effekseer.play(effect, canvas.width / 2, canvas.height / 2, 0);

        // 3秒ごとにもう一度再生する (確認のため)
        setInterval(() => {
            console.log('Re-playing effect at center.');
            // 座標を再指定して再生
            effekseer.play(effect, canvas.width / 2, canvas.height / 2, 0);
        }, 3000);

    }, (err) => {
        console.error('Failed to load effect:', err);
        alert('エフェクトファイルの読み込みに失敗しました。\n' + effectUrl);
    });

    // 描画ループを開始
    function loop() {
        requestAnimationFrame(loop);
        effekseer.update();
        
        // (カメラ設定は resize 時に行うので、ここでは不要)

        effekseer.draw();
    }

    loop();
};