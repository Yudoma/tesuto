# -*- coding: utf-8 -*-
"""backend/core/env_snapshot.py
実行環境スナップショット（再現性A向け）。
Windows/GPU前提で、pip freeze ではなく importlib.metadata から確実に取得する。
"""

from __future__ import annotations

import platform
import sys
from typing import Dict, Any

def _get_packages() -> Dict[str, str]:
    pkgs: Dict[str, str] = {}
    try:
        import importlib.metadata as md  # py3.8+
        for d in md.distributions():
            name = (d.metadata.get('Name') or '').strip()
            if not name:
                continue
            ver = (d.version or '').strip()
            pkgs[name.lower()] = ver
    except Exception:
        # 最悪でも空で返す
        return {}
    return pkgs

def snapshot_env() -> Dict[str, Any]:
    pkgs = _get_packages()
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": pkgs,
    }
    # optional: torch/cuda
    try:
        import torch  # type: ignore
        info["torch_version"] = getattr(torch, "__version__", None)
        info["cuda_available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_version"] = getattr(torch.version, "cuda", None)
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info
