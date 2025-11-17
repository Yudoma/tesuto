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

    // キャンバスの解像度をウインドウサイズに合わせる関数
    function handleResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
        effekseer.resize(width, height);
    }

    // 最初に一度、サイズを合わせる
    handleResize();

    // ウィンドウサイズが変更された時にもサイズを合わせる
    window.addEventListener('resize', handleResize);

    // エフェクトファイルのパス
    const effectUrl = './pipoya-saceffect_001.efkefc';

    // エフェクトの読み込み
    const effect = effekseer.loadEffect(effectUrl, './', () => {
        
        // --- ▽ 修正箇所 ▽ ---
        console.log('Effect load complete. Playing effect.'); // 読み込み完了をコンソールで確認

        // 読み込み完了後にエフェクトを再生
        effekseer.play(effect, 0, 0, 0);

        // 3秒ごとにもう一度再生する (確認のため)
        setInterval(() => {
            console.log('Re-playing effect.');
            effekseer.play(effect, 0, 0, 0);
        }, 3000);
        // --- △ 修正箇所 △ ---

    }, (err) => {
        console.error('Failed to load effect:', err);
        alert('エフェクトファイルの読み込みに失敗しました。\n' + effectUrl);
    });

    // 描画ループを開始
    function loop() {
        requestAnimationFrame(loop);
        effekseer.update();
        
        // 3D空間のカメラを設定
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
                canvas.width / canvas.height, // アスペクト比を動的に取得
                1.0,
                100.0
            )
        );

        effekseer.draw();
    }

    loop();
};