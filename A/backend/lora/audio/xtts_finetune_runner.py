import argparse
import json
import os
import sys
from pathlib import Path

def die(msg: str, code: int = 1):
    print(msg, flush=True)
    raise SystemExit(code)

def ensure_repo(xtts_root: Path):
    if not xtts_root.exists():
        die(f"[XTTS] リポジトリが見つかりません: {xtts_root}")
    # coqui-ai/TTS layout: should have TTS/ and recipes/
    if not (xtts_root / "TTS").exists():
        die(f"[XTTS] 期待する構成が見つかりません（TTS/ が無い）: {xtts_root}")
    return True

def validate_dataset_ljspeech(dataset_dir: Path):
    # Expect dataset_dir/wavs/*.wav and dataset_dir/metadata.csv
    meta = dataset_dir / "metadata.csv"
    wavs = dataset_dir / "wavs"
    if not meta.exists():
        die(f"[XTTS] metadata.csv が見つかりません: {meta}\n"
            f"  期待: LJSpeech形式 (wavファイル名|テキスト|正規化テキスト の3列)")
    if not wavs.exists():
        die(f"[XTTS] wavs/ フォルダが見つかりません: {wavs}")
    # quick sanity check: first non-empty line should have >=3 columns by '|'
    for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
        line=line.strip()
        if not line:
            continue
        cols=line.split("|")
        if len(cols) < 3:
            die("[XTTS] metadata.csv の列数が不足しています。\n"
                "  期待: wavファイル名|テキスト|正規化テキスト (3列)\n"
                f"  例: 000001|こんにちは|こんにちは")
        break
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, help="json payload file")
    args = ap.parse_args()

    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))

    xtts_root = Path(payload["xtts_root"]).resolve()
    dataset_dir = Path(payload["dataset_dir"]).resolve()
    out_dir = Path(payload["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_repo(xtts_root)
    validate_dataset_ljspeech(dataset_dir)

    # Put repo on sys.path
    sys.path.insert(0, str(xtts_root))

    # Imports from coqui-ai/TTS recipe style
    try:
        from trainer import Trainer, TrainerArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.layers.xtts.trainer.gpt_trainer import (
            GPTArgs,
            GPTTrainer,
            GPTTrainerConfig,
            XttsAudioConfig,
        )
        from TTS.utils.manage import ModelManager
    except Exception as e:
        die(f"[XTTS] 依存 import に失敗しました: {e}\n"
            "  対処: venv に必要依存が入っているか確認してください（setup_lora_env.py を実行）。")

    # ---- Parameters (safe defaults; can be overridden by payload) ----
    run_name = payload.get("run_name") or f"XTTS_v2_FT_{int(os.environ.get('JOB_TS','0') or 0)}"
    project_name = "XTTS_trainer"
    dashboard_logger = "tensorboard"
    logger_uri = None

    batch_size = int(payload.get("batch_size", 3))
    grad_accum = int(payload.get("grad_accum_steps", 84))
    lr = float(payload.get("lr", 5e-6))
    max_wav_length = int(payload.get("max_wav_length", 255995))  # ~11.6s @ 22.05k

    language = payload.get("language", "ja")
    dataset_name = payload.get("dataset_name", "custom")

    # output paths
    out_path = out_dir
    checkpoints_out_path = out_path / "XTTS_v2_base_files"
    checkpoints_out_path.mkdir(parents=True, exist_ok=True)

    # Dataset config (LJSpeech formatter)
    config_dataset = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name=dataset_name,
        path=str(dataset_dir),
        meta_file_train=str(dataset_dir / "metadata.csv"),
        language=language,
    )
    datasets_list = [config_dataset]

    # Model files (download if missing) - based on official recipe links
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

    dvae_ckpt = checkpoints_out_path / Path(DVAE_CHECKPOINT_LINK).name
    mel_norm = checkpoints_out_path / Path(MEL_NORM_LINK).name
    tokenizer_file = checkpoints_out_path / Path(TOKENIZER_FILE_LINK).name
    xtts_ckpt = checkpoints_out_path / Path(XTTS_CHECKPOINT_LINK).name

    def _download_if_missing(urls):
        ModelManager._download_model_files(urls, str(checkpoints_out_path), progress_bar=True)

    if (not dvae_ckpt.exists()) or (not mel_norm.exists()):
        print("[XTTS] Downloading DVAE / mel stats ...", flush=True)
        _download_if_missing([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK])

    if (not tokenizer_file.exists()) or (not xtts_ckpt.exists()):
        print("[XTTS] Downloading XTTS-v2 base checkpoint ...", flush=True)
        _download_if_missing([TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK])

    # Model args (close to official recipe)
    model_args = GPTArgs(
        max_conditioning_length=int(payload.get("max_conditioning_length", 132300)),  # ~6s
        min_conditioning_length=int(payload.get("min_conditioning_length", 66150)),   # ~3s
        debug_loading_failures=bool(payload.get("debug_loading_failures", False)),
        max_wav_length=max_wav_length,
        max_text_length=int(payload.get("max_text_length", 200)),
        mel_norm_file=str(mel_norm),
        dvae_checkpoint=str(dvae_ckpt),
        xtts_checkpoint=str(xtts_ckpt),
        tokenizer_file=str(tokenizer_file),
        gpt_num_audio_tokens=int(payload.get("gpt_num_audio_tokens", 1026)),
        gpt_start_audio_token=int(payload.get("gpt_start_audio_token", 1024)),
        gpt_stop_audio_token=int(payload.get("gpt_stop_audio_token", 1025)),
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        output_path=str(out_path),
        model_args=model_args,
        run_name=run_name,
        project_name=project_name,
        dashboard_logger=dashboard_logger,
        logger_uri=logger_uri,
        audio=audio_config,
        batch_size=batch_size,
        batch_group_size=int(payload.get("batch_group_size", 48)),
        eval_batch_size=batch_size,
        num_loader_workers=int(payload.get("num_loader_workers", 4)),
        eval_split_max_size=int(payload.get("eval_split_max_size", 256)),
        print_step=int(payload.get("print_step", 50)),
        eval_split_size=float(payload.get("eval_split_size", 0.125)),
        plot_step=int(payload.get("plot_step", 100)),
        log_model_step=int(payload.get("log_model_step", 1000)),
        save_step=int(payload.get("save_step", 10000)),
        save_n_checkpoints=int(payload.get("save_n_checkpoints", 1)),
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 100000, 200000], "gamma": 0.5},
        test_sentences=[],
    )

    # Load samples
    train_samples, eval_samples = load_tts_samples(
        datasets_list,
        eval_split=True,
        eval_split_size=config.eval_split_size,
        eval_split_max_size=config.eval_split_max_size,
    )

    # Trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=str(xtts_ckpt),
            skip_train_epoch=False,
            start_with_eval=True,
            grad_accum_steps=grad_accum,
        ),
        config,
        output_path=str(out_path),
        model=GPTTrainer.init_from_config(config),
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("[XTTS] Training start ...", flush=True)
    trainer.fit()
    print("[XTTS] Training done.", flush=True)

if __name__ == "__main__":
    main()
