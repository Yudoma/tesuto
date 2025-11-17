window.onload = () => {
    // HTMLのcanvas要素を取得
    const canvas = document.getElementById('efk_canvas');

    // canvasからWebGLコンテキストを取得
    const gl = canvas.getContext('webgl');

    // Effekseerの初期化
    // effekseer.wasm は自動的に同じ階層から読み込まれます
    // ★★★ 修正点：effekseer.init には 'canvas' ではなく 'gl' を渡します ★★★
    effekseer.init(gl).then(() => {
        console.log("Effekseer initialized.");

        // エフェクトファイルの読み込み
        // (pipoya-saceffect_001_192.png も同じ階層にある前提)
        const effectUrl = 'pipoya-saceffect_001.efkefc';
        const effect = effekseer.loadEffect(effectUrl, 1.0, () => {
            // ロード完了時の処理
            console.log("Effect loaded:", effectUrl);

            // エフェクトの再生
            const handle = effekseer.play(effect);

            // 再生位置をキャンバスの中央に設定
            handle.setLocation(0, 0, 0);

        }, (error) => {
            // ロード失敗時の処理
            console.error("Failed to load effect:", error);
        });

        // 描画ループの開始
        const loop = () => {
            // Effekseerの内部状態を更新
            effekseer.update();

            // 描画（ビュー・プロジェクション行列を簡易的に設定）
            effekseer.setProjectionMatrix(effekseer.createProjectionMatrix(60, canvas.width / canvas.height, 1, 50));
            effekseer.setCameraMatrix(effekseer.createCameraMatrix(
                [0, 5, 20], // カメラの位置
                [0, 0, 0],  // カメラの注視点
                [0, 1, 0]   // カメラの上方向
            ));
            
            // 描画実行
            effekseer.draw();

            // 次のフレームを要求
            requestAnimationFrame(loop);
        };
        loop();

    }).catch((e) => {
        // 初期化失敗
        console.error("Failed to initialize Effekseer:", e);
        alert('Effekseerの初期化に失敗しました。effekseer.wasm が見つからない可能性があります。');
    });
};
