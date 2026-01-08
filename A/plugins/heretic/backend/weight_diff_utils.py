# -*- coding: utf-8 -*-
"""plugins/heretic/backend/weight_diff_utils.py

Generic model arithmetic utilities:
- calculate_weight_diff(A, B) for matching keys
- convert_diff_to_lora(diff, rank) using SVD
- save as safetensors in a simple LoRA-like convention

This is content-agnostic math infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import re

import torch

try:
    from safetensors.torch import save_file as safetensors_save_file
except Exception:
    safetensors_save_file = None


def calculate_weight_diff(state_dict_A: Dict[str, torch.Tensor],
                          state_dict_B: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = set(state_dict_A.keys()) & set(state_dict_B.keys())
    for k in sorted(keys):
        a = state_dict_A[k]
        b = state_dict_B[k]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            continue
        if a.shape != b.shape:
            continue
        # keep on CPU for portability
        out[k] = (a.detach().cpu() - b.detach().cpu())
    return out


def convert_diff_to_lora(diff_tensor: torch.Tensor, rank: int = 8):
    """Low-rank approx of diff_tensor via SVD.

    For a 2D matrix W (out, in):
      W ≈ U_r S_r V_r^T
    We choose LoRA convention:
      down: (r, in) = sqrt(S) * V^T
      up:   (out, r) = U * sqrt(S)
      alpha = r
    So that up @ down ≈ W
    """
    if diff_tensor.ndim != 2:
        raise ValueError("diff_tensor must be 2D (out, in)")
    W = diff_tensor.float()
    # full SVD; callers should select keys to keep compute reasonable
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = max(1, min(rank, S.numel()))
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    s_sqrt = torch.sqrt(S_r)
    up = U_r * s_sqrt.unsqueeze(0)            # (out, r)
    down = s_sqrt.unsqueeze(1) * Vh_r         # (r, in)
    alpha = float(r)
    return up.contiguous(), down.contiguous(), alpha


def _sanitize_key(k: str) -> str:
    # avoid slashes in tensor names
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", k)


def save_diff_as_lora_safetensors(out_path: str,
                                 lora_tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]],
                                 metadata: Optional[Dict[str, str]] = None) -> None:
    if safetensors_save_file is None:
        raise RuntimeError("safetensors is not available in this environment")
    tensors: Dict[str, torch.Tensor] = {}
    meta = dict(metadata or {})
    for key, (up, down, alpha) in lora_tensors.items():
        sk = _sanitize_key(key)
        tensors[f"lora_up.{sk}"] = up.cpu()
        tensors[f"lora_down.{sk}"] = down.cpu()
        tensors[f"alpha.{sk}"] = torch.tensor([alpha], dtype=torch.float32)
    safetensors_save_file(tensors, out_path, metadata=meta)
