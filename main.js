window.onload = () => {
    // HTMLのcanvas要素を取得
    const canvas = document.getElementById('efk_canvas');

    // 1. Three.js のセットアップ
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, canvas.width / canvas.height, 1, 50);
    // 元のEffekseerのカメラ設定に合わせておく
    camera.position.set(0, 5, 20);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ canvas: canvas });
    renderer.setSize(canvas.width, canvas.height);

    // (お好みで) Three.jsのシーンに目印となるものを追加
    const gridHelper = new THREE.GridHelper(50, 10);
    scene.add(gridHelper);

    // 2. Effekseerの初期化
    // ★★★ 修正点：Three.jsのレンダラーからglコンテキストを取得 ★★★
    const gl = renderer.getContext();
    
    // effekseer.wasm は自動的に同じ階層から読み込まれます
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

        // 3. 描画ループの開始
        const loop = () => {
            // Effekseerの内部状態を更新
            effekseer.update();

            // ★★★ 修正点：EffekseerのカメラをThree.jsのカメラに同期 ★★★
            // Three.jsのカメラ行列を更新 (必須)
            camera.updateMatrixWorld();

            // EffekseerにThree.jsのカメラ行列を設定
            effekseer.setProjectionMatrix(camera.projectionMatrix.elements);
            effekseer.setCameraMatrix(camera.matrixWorldInverse.elements);
            
            // ★★★ 修正点：描画順序 ★★★
            
            // 1. Three.jsのシーンを描画
            renderer.render(scene, camera);

            // 2. Effekseerのエフェクトを描画
            // (Three.jsがWebGLの状態を変更するため、リセットが必要)
            renderer.resetState(); 
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