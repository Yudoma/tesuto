# -*- coding: utf-8 -*-
""" 
backend/workers/train_image.py

Diffusers + Accelerate を用いた Image LoRA 学習ワーカー。

要件:
- SDXL / SD1.5 の LoRA 学習
- Aspect Ratio Bucketing（画像をクロップせず、バケツ解像度へ letterbox/pad で合わせる）
- VRAM節約（Gradient Checkpointing / Mixed Precision / 8bit AdamW など）
- 画像ファイルと同名の .txt キャプション（タグシャッフル対応）

注意:
- ここでは「実運用で破綻しにくい」ことを優先し、
  SDXLの text encoder LoRA はデフォルトで OFF（UNet LoRA のみ）です。
  （必要なら --train_text_encoder を追加実装しやすい構造にしています）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _json_log(payload: Dict):
    """JobManager がログから JSON を拾えるように、1行JSONで出力する。"""
    try:
        print(json.dumps(payload, ensure_ascii=False))
    except Exception:
        # JSON化に失敗しても学習を止めない
        print(str(payload))


def _seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ============================================================
# Aspect Ratio Bucketing
# ============================================================


def _round_to_multiple(x: float, m: int) -> int:
    return int(max(m, int(round(x / m) * m)))


def build_buckets(base_res: int, ratios: Optional[Sequence[float]] = None) -> List[Tuple[int, int]]:
    """面積を base_res^2 に近づけつつ、アスペクト比ごとの (W,H) を作る。

    - SD1.5 なら base_res=512、SDXL なら base_res=1024 を推奨。
    - 64の倍数に丸める（UNet の制約）
    """
    if ratios is None:
        ratios = [
            1.0,
            4 / 3,
            3 / 4,
            16 / 9,
            9 / 16,
            3 / 2,
            2 / 3,
            5 / 4,
            4 / 5,
        ]

    area = float(base_res * base_res)
    buckets: List[Tuple[int, int]] = []
    for r in ratios:
        w = math.sqrt(area * r)
        h = area / w
        w = _round_to_multiple(w, 64)
        h = _round_to_multiple(h, 64)
        buckets.append((int(w), int(h)))

    # 重複排除
    buckets = sorted(set(buckets), key=lambda x: (x[0] * x[1], x[0]))
    return buckets


def choose_bucket(w: int, h: int, buckets: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    """画像のアスペクト比に最も近い bucket を選ぶ。"""
    if w <= 0 or h <= 0:
        return buckets[0]
    r = w / h
    best = None
    best_score = 1e9
    for bw, bh in buckets:
        br = bw / bh
        score = abs(math.log(r) - math.log(br))
        if score < best_score:
            best_score = score
            best = (bw, bh)
    return best or buckets[0]


# ============================================================
# Dataset
# ============================================================


@dataclass
class Sample:
    image_path: Path
    caption_path: Path
    bucket: Tuple[int, int]


class AspectBucketDataset:
    """画像 + キャプション（同名 .txt）のデータセット。

    - 画像はクロップせず、bucket解像度へ letterbox（縮小 + パディング）
    - キャプションは CSV タグをシャッフル可能
    """

    def __init__(
        self,
        dataset_dir: Path,
        buckets: Sequence[Tuple[int, int]],
        tokenizer_1,
        tokenizer_2=None,
        shuffle_tags: bool = True,
        tag_delim: str = ",",
        max_length: int = 77,
        caption_dropout: float = 0.0,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.buckets = list(buckets)
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.shuffle_tags = shuffle_tags
        self.tag_delim = tag_delim
        self.max_length = max_length
        self.caption_dropout = max(0.0, min(1.0, caption_dropout))

        self.samples: List[Sample] = []
        self._index()

    def _index(self):
        from PIL import Image

        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        for p in sorted(self.dataset_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            cap = p.with_suffix(".txt")
            if not cap.exists():
                # キャプションが無い場合はスキップ（要件: 対になる同名 .txt）
                continue
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            bucket = choose_bucket(w, h, self.buckets)
            self.samples.append(Sample(image_path=p, caption_path=cap, bucket=bucket))

        if not self.samples:
            raise RuntimeError(
                f"No training samples found in {self.dataset_dir} (images with paired .txt captions)."
            )

    def __len__(self):
        return len(self.samples)

    def _load_caption(self, p: Path) -> str:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            txt = ""

        if not txt:
            return ""

        # caption dropout（空プロンプトでの学習を少し混ぜる）
        if self.caption_dropout > 0 and random.random() < self.caption_dropout:
            return ""

        if self.shuffle_tags and self.tag_delim in txt:
            tags = [t.strip() for t in txt.split(self.tag_delim) if t.strip()]
            if len(tags) >= 2:
                random.shuffle(tags)
                txt = (self.tag_delim + " ").join(tags)
        return txt

    def _letterbox(self, pil_image, target_w: int, target_h: int):
        from PIL import Image

        w, h = pil_image.size
        if w <= 0 or h <= 0:
            return Image.new("RGB", (target_w, target_h), (0, 0, 0))
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = pil_image.resize((new_w, new_h), resample=Image.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        left = (target_w - new_w) // 2
        top = (target_h - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas

    def __getitem__(self, idx: int):
        from PIL import Image
        import torch
        from torchvision import transforms

        s = self.samples[idx]
        bw, bh = s.bucket
        with Image.open(s.image_path) as im:
            im = im.convert("RGB")
            im = self._letterbox(im, bw, bh)

        # 0..1 -> -1..1
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        pixel_values = tfm(im)

        caption = self._load_caption(s.caption_path)
        tok1 = self.tokenizer_1(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids_1 = tok1.input_ids[0]

        input_ids_2 = None
        if self.tokenizer_2 is not None:
            tok2 = self.tokenizer_2(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids_2 = tok2.input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids_1": input_ids_1,
            "input_ids_2": input_ids_2,
            "bucket_w": bw,
            "bucket_h": bh,
        }

class BucketBatchSampler:
    """bucket（解像度）ごとにバッチを作り、バッチ内の tensor サイズ不一致を防ぐ Sampler。

    - dataset.samples[i].bucket が (w, h) を持つ前提
    - 各 bucket 内で index をシャッフルし、batch_size ごとに切り出す
    - bucket 間のバッチ順もシャッフルする
    """

    def __init__(self, dataset: "AspectBucketDataset", batch_size: int, drop_last: bool = False, seed: int = 0):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        self._bucket_to_indices = {}
        for i, s in enumerate(dataset.samples):
            self._bucket_to_indices.setdefault(tuple(s.bucket), []).append(i)

        # 空 bucket は除外
        self._buckets = [b for b, idxs in self._bucket_to_indices.items() if idxs]

    def __iter__(self):
        rng = random.Random(self.seed)

        # 各 bucket のインデックスをシャッフルしてバッチ化
        all_batches = []
        for b in self._buckets:
            idxs = list(self._bucket_to_indices[b])
            rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)

        rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        n = 0
        for b in self._buckets:
            sz = len(self._bucket_to_indices[b])
            if self.drop_last:
                n += sz // self.batch_size
            else:
                n += (sz + self.batch_size - 1) // self.batch_size
        return n



def collate_fn(examples: List[Dict]):
    import torch

    pixel_values = torch.stack([e["pixel_values"] for e in examples]).contiguous()
    input_ids_1 = torch.stack([e["input_ids_1"] for e in examples])
    if examples[0]["input_ids_2"] is not None:
        input_ids_2 = torch.stack([e["input_ids_2"] for e in examples])
    else:
        input_ids_2 = None

    # bucket の混在を避けるには、DataLoader を bucketごとに分ける必要がある。
    # ただし Web UI の初期実装では単一 DataLoader を許容し、
    # ここでは「同一bucket混在時は最大サイズに pad」せず、
    # dataset 側で既に bucket に合わせている前提で統一サイズを想定する。
    bw = examples[0]["bucket_w"]
    bh = examples[0]["bucket_h"]

    return {
        "pixel_values": pixel_values,
        "input_ids_1": input_ids_1,
        "input_ids_2": input_ids_2,
        "bucket_w": bw,
        "bucket_h": bh,
    }


# ============================================================
# Training
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(description="Train Image LoRA (Diffusers) with Aspect Ratio Bucketing")

    # 必須
    p.add_argument("--base_model_path", type=str, required=True, help="Diffusers model path or HF repo id")
    p.add_argument("--dataset_path", type=str, required=True, help="Dataset directory path")
    p.add_argument("--dataset_dir", type=str, required=False, dest="dataset_path", help="Dataset directory path (互換: --dataset_path と同義)")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for LoRA weights")

    # 学習パラメータ
    p.add_argument("--resolution", type=int, default=1024, help="Base resolution (512 for SD1.5, 1024 for SDXL)")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_steps", type=int, default=0, help="If >0, overrides epochs")
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["constant", "cosine", "linear"])
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--warmup_ratio", type=float, default=0.0, help="If >0 and warmup_steps==0, warmup_steps = int(total_steps * warmup_ratio)")
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    # メモリ節約
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--xformers", action="store_true", help="Enable xformers memory efficient attention if available")

    # キャプション
    p.add_argument("--shuffle_tags", action="store_true", help="Shuffle comma-separated tags in captions")
    p.add_argument("--tag_delim", type=str, default=",")
    p.add_argument("--caption_dropout", type=float, default=0.0)
    p.add_argument("--max_token_length", type=int, default=77)

    # 互換/補助
    p.add_argument("--snr_gamma", type=float, default=0.0, help="If >0, apply SNR weighting (experimental)")
    p.add_argument("--save_every_n_steps", type=int, default=0)
    p.add_argument("--sample_prompt", type=str, default="", help="(任意) チェックポイント保存時に生成するサンプル画像用プロンプト")

    return p.parse_args()


def _get_weight_dtype(mixed_precision: str):
    import torch

    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _maybe_enable_xformers(unet):
    try:
        unet.enable_xformers_memory_efficient_attention()
        return True
    except Exception:
        return False


def _apply_lora_to_unet(unet, rank: int, alpha: int, dropout: float):
    """UNet の attention processor に LoRA を挿入する。"""
    from diffusers.models.attention_processor import (
        LoRAAttnProcessor,
        LoRAAttnProcessor2_0,
    )

    lora_attn_procs = {}
    for name, attn_proc in unet.attn_processors.items():
        # name 例: "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor"
        # 対象の hidden_size を推定
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[0]

        # cross attention か self attention か
        if name.endswith("attn2.processor"):
            cross_attention_dim = unet.config.cross_attention_dim
        else:
            cross_attention_dim = None

        # torch2.0 の fused attention を使っている場合は 2_0 を選択
        if hasattr(attn_proc, "__class__") and "2_0" in attn_proc.__class__.__name__:
            lora_cls = LoRAAttnProcessor2_0
        else:
            lora_cls = LoRAAttnProcessor

        lora_attn_procs[name] = lora_cls(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
        )

    unet.set_attn_processor(lora_attn_procs)

    # trainable params を返す
    trainable_params = []
    for _, p in unet.named_parameters():
        if p.requires_grad:
            trainable_params.append(p)
    # LoRA を挿した場合、attn_processors の内部だけ requires_grad=True になる
    return trainable_params


def _compute_snr(noise_scheduler, timesteps):
    # diffusers 公式のSNR weighting実装を参考にした簡易版
    import torch

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device)
    alpha = alphas_cumprod[timesteps]
    sigma = 1 - alpha
    snr = alpha / sigma
    return snr



def _maybe_generate_training_sample(
    *,
    is_sdxl: bool,
    unet,
    vae,
    text_encoder_1,
    tokenizer_1,
    text_encoder_2,
    tokenizer_2,
    infer_scheduler,
    device,
    weight_dtype,
    prompt: str,
    out_dir: Path,
    step: int,
    lora_dir: Path,
    base_resolution: int,
):
    """学習中チェックポイントのタイミングでサンプル画像を生成して保存する。

    VRAM枯渇（OOM）回避を最優先とし、以下を徹底する:
    - 生成前に gc.collect / empty_cache
    - 生成用パイプラインを都度構築して破棄
    - main process のみ実行
    """
    if not prompt:
        return

    import gc
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    # 生成先
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"sample_{step}.png"
    latest_path = samples_dir / "sample_latest.png"

    # 生成前に一度掃除
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 推論用 scheduler は training と分離して作る（設定のみ引き継ぐ）
    try:
        scheduler = type(infer_scheduler).from_config(infer_scheduler.config)
    except Exception:
        scheduler = infer_scheduler

    # train/eval の状態を退避
    prev_modes = {
        "unet": unet.training,
        "vae": vae.training,
        "te1": getattr(text_encoder_1, "training", False),
        "te2": getattr(text_encoder_2, "training", False) if text_encoder_2 is not None else False,
    }

    try:
        unet.eval()
        vae.eval()
        if text_encoder_1 is not None:
            text_encoder_1.eval()
        if text_encoder_2 is not None:
            text_encoder_2.eval()

        # パイプライン構築（同一モジュール参照）
        if is_sdxl:
            pipe = StableDiffusionXLPipeline(
                vae=vae,
                text_encoder=text_encoder_1,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer_1,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
            )
        else:
            pipe = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder_1,
                tokenizer=tokenizer_1,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
            )

        pipe.to(device=device, dtype=weight_dtype)

        # LoRA を適用（保存した checkpoint から読み込み）
        try:
            pipe.load_lora_weights(str(lora_dir))
        except Exception:
            # diffusersのバージョン差異や保存形式の差異を吸収する
            try:
                pipe.unet.load_attn_procs(str(lora_dir))
            except Exception:
                pass

        # 生成設定（解像度は base_resolution を優先。SD1.5 でも正方形で固定）
        w = int(base_resolution)
        h = int(base_resolution)

        gen = torch.Generator(device=device)
        gen.manual_seed(0)

        with torch.inference_mode():
            try:
                # SDXL は negative_prompt を省略しても動く
                out = pipe(
                    prompt=prompt,
                    width=w,
                    height=h,
                    num_inference_steps=20,
                    guidance_scale=5.0,
                    generator=gen,
                )
            except TypeError:
                # diffusers の一部バージョンで引数名が異なる場合の保険
                out = pipe(prompt)

        img = out.images[0]
        img.save(out_path)

        # latest を更新
        try:
            img.save(latest_path)
        except Exception:
            pass

    except Exception as e:
        _eprint(f"[train_image] sample generation failed at step={step}: {e}")
    finally:
        # 破棄して掃除
        try:
            del pipe
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # train/eval を戻す
        try:
            unet.train(prev_modes["unet"])
        except Exception:
            pass
        try:
            vae.train(prev_modes["vae"])
        except Exception:
            pass
        try:
            if text_encoder_1 is not None:
                text_encoder_1.train(prev_modes["te1"])
        except Exception:
            pass
        try:
            if text_encoder_2 is not None:
                text_encoder_2.train(prev_modes["te2"])
        except Exception:
            pass
def main():
    args = parse_args()

    _seed_everything(args.seed)

    base_model_path = args.base_model_path
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 依存チェック
    try:
        import torch
        from torch.utils.data import DataLoader
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from diffusers import (
            AutoencoderKL,
            DDPMScheduler,
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            UNet2DConditionModel,
        )
        from transformers import CLIPTextModel, CLIPTokenizer
        try:
            from transformers import CLIPTextModelWithProjection
        except Exception:
            CLIPTextModelWithProjection = None
    except Exception as e:
        _eprint("[train_image] Missing dependencies. Please install diffusers/accelerate/transformers/torch.")
        raise

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    set_seed(args.seed)

    device = accelerator.device
    weight_dtype = _get_weight_dtype(args.mixed_precision)

        # --------------------------------------------------------
    # パイプラインロード（SDXL / SD1.5 自動判定）
    # - Diffusersフォルダ / 単一モデルファイル（.safetensors/.ckpt）を両対応
    # - まず SDXL を試し、失敗したら SD1.5 系にフォールバック
    # --------------------------------------------------------
    is_sdxl = False
    pipe = None
    load_err = None

    base_path = Path(base_model_path)
    is_single_file = base_path.is_file() and base_path.suffix.lower() in [".safetensors", ".ckpt"]

    def _load_sdxl():
        if is_single_file:
            return StableDiffusionXLPipeline.from_single_file(
                str(base_path),
                torch_dtype=weight_dtype,
            )
        return StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=weight_dtype,
            variant=None,
        )

    def _load_sd15():
        if is_single_file:
            return StableDiffusionPipeline.from_single_file(
                str(base_path),
                torch_dtype=weight_dtype,
                safety_checker=None,
            )
        return StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=weight_dtype,
            safety_checker=None,
        )

    try:
        pipe = _load_sdxl()
        is_sdxl = True
    except Exception as e:
        load_err = e

    if pipe is None:
        try:
            pipe = _load_sd15()
            is_sdxl = False
        except Exception as e:
            _eprint(f"[train_image] Failed to load base model: {base_model_path}")
            _eprint(f"- SDXL load error: {load_err}")
            _eprint(f"- SD load error: {e}")
            raise

    # 推論スケジューラ（config抽出用）
    infer_scheduler = pipe.scheduler
# 必要部品
    unet: UNet2DConditionModel = pipe.unet
    vae: AutoencoderKL = pipe.vae
    text_encoder_1 = pipe.text_encoder
    tokenizer_1 = pipe.tokenizer
    text_encoder_2 = getattr(pipe, "text_encoder_2", None)
    tokenizer_2 = getattr(pipe, "tokenizer_2", None)

    # pipe 自体は以降使わないため解放（VRAM節約）
    try:
        del pipe
    except Exception:
        pass
    import gc as _gc
    _gc.collect()
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass

    # --------------------------------------------------------
    # モデル設定（freeze / lora / mem）
    # --------------------------------------------------------
    unet.train()
    vae.eval()
    text_encoder_1.eval()
    if text_encoder_2 is not None:
        text_encoder_2.eval()

    for p in vae.parameters():
        p.requires_grad = False
    for p in text_encoder_1.parameters():
        p.requires_grad = False
    if text_encoder_2 is not None:
        for p in text_encoder_2.parameters():
            p.requires_grad = False
    for p in unet.parameters():
        p.requires_grad = False

    # LoRA挿入
    _apply_lora_to_unet(unet, args.lora_r, args.lora_alpha, args.lora_dropout)

    if args.gradient_checkpointing:
        try:
            unet.enable_gradient_checkpointing()
        except Exception:
            pass

    if args.xformers:
        enabled = _maybe_enable_xformers(unet)
        _json_log({"event": "xformers", "enabled": bool(enabled)})

    # trainable params
    trainable_params = [p for p in unet.parameters() if p.requires_grad]

    # optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
        except Exception as e:
            _eprint("[train_image] bitsandbytes not available; falling back to torch AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(trainable_params, lr=args.learning_rate)

    # noise scheduler
    noise_scheduler = DDPMScheduler.from_config(infer_scheduler.config)

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------
    buckets = build_buckets(args.resolution)

    dataset = AspectBucketDataset(
        dataset_dir=dataset_path,
        buckets=buckets,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        shuffle_tags=args.shuffle_tags,
        tag_delim=args.tag_delim,
        max_length=args.max_token_length,
        caption_dropout=args.caption_dropout,
    )

    # シャッフルするが、bucket混在を避けたい場合は bucket単位のSamplerへ拡張可能。
    batch_sampler = BucketBatchSampler(dataset, batch_size=args.train_batch_size, drop_last=False, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # steps
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps and args.max_train_steps > 0:
        max_train_steps = args.max_train_steps
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    else:
        num_train_epochs = args.epochs
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # lr scheduler
    def lr_lambda(current_step: int):
        if args.warmup_steps > 0 and current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, max_train_steps - args.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if args.lr_scheduler == "constant":
            return 1.0
        if args.lr_scheduler == "linear":
            return 1.0 - progress
        # cosine
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # prepare with accelerate
    unet, optimizer, dataloader, scheduler = accelerator.prepare(unet, optimizer, dataloader, scheduler)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    global_step = 0
    first_log = True
    start_ts = time.time()

    _json_log(
        {
            "event": "start",
            "base_model": base_model_path,
            "sdxl": bool(is_sdxl),
            "samples": len(dataset),
            "buckets": buckets,
            "max_train_steps": max_train_steps,
        }
    )

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # image -> latents
                pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # timesteps
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.int64,
                )
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # text embeddings
                input_ids_1 = batch["input_ids_1"].to(device)
                with torch.no_grad():
                    enc1 = text_encoder_1(input_ids_1, output_hidden_states=True)
                    if is_sdxl:
                        # SDXL: prompt_embeds は最終層 hidden_states[-2]
                        prompt_embeds_1 = enc1.hidden_states[-2]
                        pooled_1 = enc1[0][:, 0]
                        if text_encoder_2 is not None and batch["input_ids_2"] is not None:
                            input_ids_2 = batch["input_ids_2"].to(device)
                            enc2 = text_encoder_2(input_ids_2, output_hidden_states=True)
                            prompt_embeds_2 = enc2.hidden_states[-2]
                            pooled_2 = enc2[0][:, 0]
                        else:
                            prompt_embeds_2 = prompt_embeds_1
                            pooled_2 = pooled_1

                        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
                        pooled_prompt_embeds = torch.cat([pooled_1, pooled_2], dim=-1)

                        # SDXL では追加の time_ids が必要
                        # ここでは簡易的に (orig_w, orig_h, crop_x, crop_y, target_w, target_h) を bucket 解像度で埋める
                        target_w = batch["bucket_w"]
                        target_h = batch["bucket_h"]
                        add_time_ids = torch.tensor(
                            [[target_w, target_h, 0, 0, target_w, target_h]] * bsz,
                            device=device,
                            dtype=prompt_embeds.dtype,
                        )
                        added_cond_kwargs = {
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": add_time_ids,
                        }
                    else:
                        prompt_embeds = enc1[0]
                        added_cond_kwargs = None

                # unet forward
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                target = noise
                loss = (model_pred.float() - target.float()) ** 2
                loss = loss.mean(dim=list(range(1, loss.ndim)))

                if args.snr_gamma and args.snr_gamma > 0:
                    # SNR weighting
                    snr = _compute_snr(noise_scheduler, timesteps)
                    # weight = min(snr, gamma) / snr
                    weights = torch.minimum(snr, torch.full_like(snr, args.snr_gamma)) / snr
                    loss = loss * weights

                loss = loss.mean()

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # logging
            if accelerator.sync_gradients:
                global_step += 1

                if first_log:
                    first_log = False
                    _json_log(
                        {
                            "event": "config",
                            "mixed_precision": args.mixed_precision,
                            "train_batch_size": args.train_batch_size,
                            "grad_accum": args.gradient_accumulation_steps,
                            "lr": args.learning_rate,
                            "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
                        }
                    )

                if global_step % 1 == 0:
                    elapsed = time.time() - start_ts
                    _json_log(
                        {
                            "event": "step",
                            "step": global_step,
                            "loss": float(loss.detach().item()),
                            "lr": float(scheduler.get_last_lr()[0]),
                            "epoch": int(epoch),
                            "elapsed_sec": round(elapsed, 2),
                        }
                    )

                # checkpoint
                if args.save_every_n_steps and args.save_every_n_steps > 0:
                    if global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            save_dir = output_dir / f"checkpoint-{global_step}"
                            save_dir.mkdir(parents=True, exist_ok=True)
                            unet_ = accelerator.unwrap_model(unet)
                            # LoRAだけ保存
                            unet_.save_attn_procs(str(save_dir))
                            _json_log({"event": "save", "step": global_step, "path": str(save_dir)})
                            # Training Preview（任意）
                            if getattr(args, "sample_prompt", ""):
                                _maybe_generate_training_sample(
                                    is_sdxl=is_sdxl,
                                    unet=accelerator.unwrap_model(unet),
                                    vae=vae,
                                    text_encoder_1=text_encoder_1,
                                    tokenizer_1=tokenizer_1,
                                    text_encoder_2=text_encoder_2,
                                    tokenizer_2=tokenizer_2,
                                    infer_scheduler=infer_scheduler,
                                    device=device,
                                    weight_dtype=weight_dtype,
                                    prompt=str(args.sample_prompt),
                                    out_dir=output_dir,
                                    step=int(global_step),
                                    lora_dir=save_dir,
                                    base_resolution=int(args.resolution),
                                )


                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    # final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_ = accelerator.unwrap_model(unet)
        unet_.save_attn_procs(str(output_dir))
        # メタ情報
        meta = {
            "base_model": base_model_path,
            "sdxl": bool(is_sdxl),
            "resolution": args.resolution,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "epochs": args.epochs,
            "max_train_steps": max_train_steps,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "shuffle_tags": bool(args.shuffle_tags),
            "caption_dropout": args.caption_dropout,
            "buckets": buckets,
        }
        (output_dir / "training_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    _json_log({"event": "done", "steps": global_step})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _json_log({"event": "stopped"})
        raise
    except Exception as e:
        _json_log({"event": "error", "message": str(e)})
        raise
