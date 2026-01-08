# -*- coding: utf-8 -*-
"""
backend/core/system_info.py
システムリソース（CPU, Memory, GPU）情報の取得や、
Flash Attention 2 などの環境適合性チェックを行うモジュール。
"""
import sys
import os
import platform
import subprocess
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import importlib.metadata

# Flash Attention チェック用に torch をインポート（重いので関数内でインポートも検討可だが、サーバー起動時にロード済みと想定）
import torch


def _installed_packages_map() -> Dict[str, str]:
    """インストール済みパッケージ一覧（package -> version）。

    目的:
      - UIの「関連ライブラリ一覧（未導入含む）」で、導入済/未導入を判定する。
    注意:
      - 量が多くなるため、キーは小文字正規化して返す。
      - 例外は握りつぶし、UIを落とさない。
    """
    mp: Dict[str, str] = {}
    try:
        for dist in importlib.metadata.distributions():
            name = (dist.metadata.get('Name') or '').strip()
            ver = (dist.version or '').strip()
            if not name:
                continue
            mp[name.lower()] = ver if ver else 'unknown'
    except Exception:
        return {}
    return mp


def _safe_version(module_name: str) -> Optional[str]:
    """モジュールのバージョンを安全に取得する（無ければ None）。"""
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    """コマンドを実行し (returncode, stdout, stderr) を返す。"""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        return res.returncode, res.stdout or "", res.stderr or ""
    except FileNotFoundError:
        return 127, "", "not found"
    except Exception as e:
        return 1, "", str(e)


def _torch_cuda_info() -> Dict[str, Any]:
    """PyTorch/CUDA周辺の情報（画像/音声共通）"""
    info: Dict[str, Any] = {
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "bf16_supported": False,
        "device": None,
        "gpu_name": None,
        "compute_capability": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
    }

    try:
        if torch.cuda.is_available():
            info["bf16_supported"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            idx = int(torch.cuda.current_device())
            props = torch.cuda.get_device_properties(idx)
            info["device"] = idx
            info["gpu_name"] = props.name
            info["compute_capability"] = f"{props.major}.{props.minor}"
            info["vram_total_gb"] = round(float(props.total_memory) / (1024**3), 2)

            # mem_get_info は新しめのPyTorchで利用可能
            if hasattr(torch.cuda, "mem_get_info"):
                free_b, total_b = torch.cuda.mem_get_info(idx)
                info["vram_free_gb"] = round(float(free_b) / (1024**3), 2)
            else:
                info["vram_free_gb"] = None
    except Exception:
        # 取得に失敗してもサーバーを落とさない
        pass

    return info


def _disk_usage(path: Path) -> Dict[str, Any]:
    """指定パスのディスク使用量（存在しない場合でも安全に）"""
    try:
        p = path
        if not p.exists():
            # 近い既存パスにフォールバック
            p = p.parent
        usage = psutil.disk_usage(str(p))
        return {
            "path": str(path),
            "total_gb": round(usage.total / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent": usage.percent,
        }
    except Exception as e:
        return {"path": str(path), "error": str(e)}

def get_system_info() -> Dict[str, Any]:
    """
    OS, Pythonバージョン, CPU/メモリ使用状況, GPU情報(nvidia-smi)を取得する。
    """
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "cpu": {
            "model": platform.processor(),
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=None)
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "percent": psutil.virtual_memory().percent
        },
        "gpu_info": "N/A",
        # 旧UI互換: "nvidia_smi" を参照する実装が存在するためエイリアスも提供する
        "nvidia_smi": "N/A",
        # インストール済みパッケージ一覧（UIの関連ライブラリ判定用）
        "packages": {},
    }

    # インストール済みパッケージ（重すぎない範囲で）
    try:
        info["packages"] = _installed_packages_map()
    except Exception:
        info["packages"] = {}

    # nvidia-smi の取得試行
    try:
        # Windows環境では .exe が必要ない場合もあるが、パスが通っている前提
        code, out, _err = _run_cmd(["nvidia-smi"])
        if code == 0:
            info["gpu_info"] = out
            info["nvidia_smi"] = out
        else:
            info["gpu_info"] = "nvidia-smi returned error."
            info["nvidia_smi"] = info["gpu_info"]
    except FileNotFoundError:
        info["gpu_info"] = "nvidia-smi not found."
        info["nvidia_smi"] = info["gpu_info"]
    except Exception as e:
        info["gpu_info"] = f"Error: {str(e)}"
        info["nvidia_smi"] = info["gpu_info"]

    # 追加: PyTorch/CUDA 情報（軽量）
    info["torch_cuda"] = _torch_cuda_info()


    # 追加: ディスク空き（運用/再現性のためテキスト系も含めて取得）
    try:
        root = Path.cwd()
        disk = {
            "project_root": _disk_usage(root),
            "logs_dir": _disk_usage(root / "logs"),
            "runs_dir": _disk_usage(root / "runs"),
            "models_text": _disk_usage(root / "models" / "text"),
            "datasets_text": _disk_usage(root / "datasets" / "text"),
            "lora_adapters_text": _disk_usage(root / "outputs" / "text"),
        }
        info["disk"] = disk
    except Exception:
        # 失敗してもUIは落とさない
        info["disk"] = {}

    return info


def get_image_system_info(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """画像（Diffusers）生成/学習に関係するシステム情報を返す。"""
    root = project_root or Path.cwd()
    base = get_system_info()

    # LoRA Factory の標準ディレクトリ（あれば優先）
    # ※ core 側から参照するため循環依存回避で try/except
    models_dir = root / "models" / "image"
    datasets_dir = root / "datasets" / "image"
    output_dir = root / "lora_adapters" / "image"
    try:
        from lora_config import settings as _settings  # type: ignore
        models_dir = Path(_settings.dirs.get("image", {}).get("models", models_dir))
        datasets_dir = Path(_settings.dirs.get("image", {}).get("datasets", datasets_dir))
        output_dir = Path(_settings.dirs.get("image", {}).get("output", output_dir))
    except Exception:
        pass

    libs = {
        "diffusers": _safe_version("diffusers"),
        "transformers": _safe_version("transformers"),
        "accelerate": _safe_version("accelerate"),
        "safetensors": _safe_version("safetensors"),
        "xformers": _safe_version("xformers"),
        "bitsandbytes": _safe_version("bitsandbytes"),
        "torchvision": _safe_version("torchvision"),
        "triton": _safe_version("triton"),
    }

    # FlashAttention2 事前チェック（ある場合のみ）
    fa2 = flash_attention_2_preflight()

    # 代表的なキャッシュ/環境変数
    env = {
        "HF_HOME": os.environ.get("HF_HOME"),
        "HF_HUB_DISABLE_TELEMETRY": os.environ.get("HF_HUB_DISABLE_TELEMETRY"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        "DIFFUSERS_CACHE": os.environ.get("DIFFUSERS_CACHE"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    # 主要リポジトリ/環境（存在チェック）
    repos: Dict[str, Any] = {}
    try:
        for k in ["COMFYUI_DIR", "SD_WEBUI_DIR", "INVOKEAI_DIR", "DIFFUSERS_REPO_DIR"]:
            v = (os.environ.get(k, "") or "").strip()
            if v:
                p = Path(v)
                repos[k] = {"path": v, "exists": bool(p.exists())}
    except Exception:
        pass

    # ユーザー向けメッセージ（日本語・具体的）
    user_messages: List[Dict[str, str]] = []
    try:
        # ライブラリ不足
        for k in ["diffusers", "transformers", "accelerate", "safetensors"]:
            v = libs.get(k)
            if not v:
                user_messages.append({
                    "level": "error",
                    "code": f"PYLIB_MISSING_{k.upper()}",
                    "title": f"Pythonライブラリが不足しています: {k}",
                    "detail": "画像（Diffusers）機能を使うには必須です。requirements を確認してインストールしてください。",
                })

        # 速度最適化（任意）
        if not libs.get("xformers") and not (libs.get("triton") or "").strip():
            user_messages.append({
                "level": "info",
                "code": "OPTIMIZATION_HINT",
                "title": "高速化オプションが未導入の可能性があります",
                "detail": "xformers / triton / FlashAttention2 は Windows では導入が難しい/非推奨なことが多い任意機能です。未導入でも動作します（速度やVRAM効率が変わる場合があります）。",
            })

        # CUDA
        if not base.get("torch_cuda", {}).get("cuda_available"):
            user_messages.append({
                "level": "warn",
                "code": "CUDA_NOT_AVAILABLE",
                "title": "CUDA が利用できません",
                "detail": "GPU推論/学習ができない可能性があります。CUDA対応のPyTorchを入れているか、GPUドライバが正しく入っているか確認してください。",
            })
    except Exception:
        pass

    # ディスク空き（学習/生成で効く場所）
    disk = {
        "project_root": _disk_usage(root),
        "models_image": _disk_usage(models_dir),
        "datasets_image": _disk_usage(datasets_dir),
        "outputs": _disk_usage(root / "outputs"),
        "lora_adapters_image": _disk_usage(output_dir),
    }

    return {
        "category": "image",
        "base": base,
        "torch_cuda": base.get("torch_cuda"),
        "libs": libs,
        "flash_attention2": fa2,
        "env": env,
        "repos": repos,
        "disk": disk,
        "user_messages": user_messages,
        "notes": [
            "画像の学習/生成ではVRAM・PyTorch/CUDAの組み合わせ・xformers/bitsandbytesの有無が重要です。",
            "nvidia-smi が利用できない環境では GPU の詳細取得が一部制限されます。",
        ],
    }


def get_audio_system_info(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """音声（Voice / GPT-SoVITS）学習/生成に関係するシステム情報を返す。"""
    root = project_root or Path.cwd()
    base = get_system_info()

    libs = {
        "torchaudio": _safe_version("torchaudio"),
        "librosa": _safe_version("librosa"),
        "soundfile": _safe_version("soundfile"),
        "faster_whisper": _safe_version("faster_whisper"),
        "pydub": _safe_version("pydub"),
        "pyloudnorm": _safe_version("pyloudnorm"),
        "numpy": _safe_version("numpy"),
    }

    # ffmpeg の有無（mp3 等のデコードに影響）
    code, out, err = _run_cmd(["ffmpeg", "-version"])
    ffmpeg = {
        "available": code == 0,
        "summary": (out.splitlines()[0] if out else None),
        "error": (err if code != 0 else None),
    }

    env = {
        "GPT_SOVITS_DIR": os.environ.get("GPT_SOVITS_DIR"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    disk = {
        "project_root": _disk_usage(root),
        "models_audio": _disk_usage(root / "models" / "audio"),
        "datasets_audio": _disk_usage(root / "datasets" / "audio"),
        "outputs": _disk_usage(root / "outputs"),
        "lora_adapters_audio": _disk_usage(root / "lora_adapters" / "audio"),
    }

    # 主要リポジトリ/実行環境の存在チェック（環境変数ベース）
    repos = {}
    try:
        for k in ["XTTS_DIR", "GPT_SOVITS_DIR", "GPT_SOVITS_VC_DIR", "RVC_DIR"]:
            v = (os.environ.get(k, "") or "").strip()
            if v:
                p = Path(v)
                repos[k] = {"path": v, "exists": bool(p.exists())}
    except Exception:
        pass

    
    # ユーザー向けメッセージ（日本語・具体的な解決策）
    user_messages: List[Dict[str, str]] = []
    try:
        if not ffmpeg.get("available"):
            user_messages.append({
                "level": "error",
                "code": "FFMPEG_MISSING",
                "title": "ffmpeg が見つかりません",
                "detail": "MP3の読み込みや一部の音声処理に ffmpeg が必要です。Windowsでは『ffmpeg.exe をPATHに追加』してください。例: Chocolatey を使う場合は `choco install ffmpeg`。",
            })
        # Pythonライブラリ
        for k in ["torchaudio", "librosa", "soundfile", "faster_whisper", "pydub", "pyloudnorm"]:
            v = libs.get(k)
            if not v:
                user_messages.append({
                    "level": "warn",
                    "code": f"PYLIB_MISSING_{k.upper()}",
                    "title": f"Pythonライブラリが不足しています: {k}",
                    "detail": f"`pip install {k}` を試してください（仮想環境を有効化した状態で実行）。",
                })
    except Exception:
        pass
    return {
        "category": "audio",
        "base": base,
        "torch_cuda": base.get("torch_cuda"),
        "libs": libs,
        "ffmpeg": ffmpeg,
        "env": env,
        "disk": disk,
        "notes": [
            "音声の前処理（mp3読み込み/変換）では ffmpeg が必要になるケースがあります。",
            "ASR（faster-whisper）はモデルサイズによりVRAM/速度が大きく変わります。",
        ],
    }

def flash_attention_2_preflight() -> Dict[str, Any]:
    """
    Flash Attention 2 が現在の環境（ハードウェア/ライブラリ）で利用可能かチェックする。
    """
    res: Dict[str, Any] = {
        "available": False,
        "import_ok": False,
        "smoke_test_ok": False,
        "error": None,
        "details": {},
    }
    try:
        if not torch.cuda.is_available():
            res["error"] = "CUDAが利用できません。"
            return res
            
        # 1. Import Test
        try:
            import flash_attn  # type: ignore
            res["import_ok"] = True
            res["details"]["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
        except ImportError:
            res["error"] = "flash_attn ライブラリがインストールされていません。"
            return res
        except Exception as e:
            res["error"] = f"flash_attn import失敗: {e}"
            return res

        # 2. Smoke Test (実際に計算してみる)
        try:
            from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
            
            device = torch.device("cuda")
            dtype = torch.float16
            
            # ダミーデータ作成: (Batch, Seq, Heads, Dim)
            q = torch.randn((1, 4, 16, 64), device=device, dtype=dtype)
            k = torch.randn((1, 4, 16, 64), device=device, dtype=dtype)
            v = torch.randn((1, 4, 16, 64), device=device, dtype=dtype)
            
            # causal=False, dropout=0.0
            _ = flash_attn_func(q, k, v, 0.0, False)
            res["smoke_test_ok"] = True
            
        except Exception as e:
            res["error"] = f"FlashAttention2 smoke test失敗: {e}"
            return res

        # 全てクリア
        res["available"] = True
        return res

    except Exception as e:
        res["error"] = str(e)
        return res
