# -*- coding: utf-8 -*-
"""plugins/heretic/backend/stream_interceptor.py

Generic PyTorch hook manager for activation inspection.

This module is content-agnostic: it provides an infrastructure to attach hooks and
optionally route tensors through a user-defined callback (for trusted local scripts).

UI path in this tool uses *inspection only* (stats capture), not arbitrary modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import time

import torch


TensorLike = Union[torch.Tensor, Tuple[Any, ...], Any]


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x:
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


@dataclass
class CaptureStats:
    ts: float
    module_name: str
    hook_type: str
    shape: str
    dtype: str
    device: str
    mean: float
    std: float
    norm: float
    min: float
    max: float


class StreamInterceptor:
    """Manage forward hooks in a safe and generic way."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._handles = []
        self._module_name: Optional[str] = None
        self._hook_type: Optional[str] = None
        self.user_defined_callback: Optional[Callable[[Any], Any]] = None
        self._latest_stats: Optional[CaptureStats] = None

    def set_user_callback(self, fn: Optional[Callable[[Any], Any]]) -> None:
        self.user_defined_callback = fn

    def clear_user_callback(self) -> None:
        self.user_defined_callback = None

    def detach_all(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        self._module_name = None
        self._hook_type = None

    def attach_by_name(self, module_name: str, hook_type: str = "forward_hook") -> None:
        """Attach a hook to a module found by name."""
        self.detach_all()
        mod = dict(self.model.named_modules()).get(module_name)
        if mod is None:
            raise ValueError(f"Module not found: {module_name}")
        self._module_name = module_name
        self._hook_type = hook_type

        if hook_type == "forward_pre_hook":
            handle = mod.register_forward_pre_hook(self._pre_hook, with_kwargs=False)
        elif hook_type == "forward_hook":
            handle = mod.register_forward_hook(self._fwd_hook, with_kwargs=False)
        else:
            raise ValueError("hook_type must be 'forward_hook' or 'forward_pre_hook'")
        self._handles.append(handle)

    def _pre_hook(self, module, inputs):
        # pre-hook can't see output; capture from inputs[0] for stats
        t = _first_tensor(inputs)
        if t is not None:
            self._capture_stats(t)
        # Infrastructure: allow callback to transform inputs (advanced use)
        if self.user_defined_callback:
            try:
                out = self.user_defined_callback(inputs)
                return out
            except Exception:
                return inputs
        return inputs

    def _fwd_hook(self, module, inputs, output):
        t = _first_tensor(output)
        if t is not None:
            self._capture_stats(t)
        # Infrastructure: allow callback to transform output (advanced use)
        if self.user_defined_callback:
            try:
                return self.user_defined_callback(output)
            except Exception:
                return output
        return output

    def _capture_stats(self, t: torch.Tensor) -> None:
        with torch.no_grad():
            x = t.detach()
            if x.is_floating_point():
                xf = x.float()
            else:
                xf = x.to(torch.float32)
            # reduce to avoid huge overhead
            mean = float(xf.mean().item())
            std = float(xf.std(unbiased=False).item())
            norm = float(xf.norm().item())
            minv = float(xf.min().item())
            maxv = float(xf.max().item())
            self._latest_stats = CaptureStats(
                ts=time.time(),
                module_name=self._module_name or "",
                hook_type=self._hook_type or "",
                shape=str(tuple(x.shape)),
                dtype=str(x.dtype),
                device=str(x.device),
                mean=mean,
                std=std,
                norm=norm,
                min=minv,
                max=maxv,
            )

    def get_latest_stats(self) -> Optional[Dict[str, Any]]:
        s = self._latest_stats
        if not s:
            return None
        return {
            "ts": s.ts,
            "module_name": s.module_name,
            "hook_type": s.hook_type,
            "shape": s.shape,
            "dtype": s.dtype,
            "device": s.device,
            "mean": s.mean,
            "std": s.std,
            "norm": s.norm,
            "min": s.min,
            "max": s.max,
        }
