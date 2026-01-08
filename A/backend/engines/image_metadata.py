# -*- coding: utf-8 -*-
"""backend/engines/image_metadata.py"""
from __future__ import annotations
from typing import Any, Dict, Optional

def model_identity(base_model: str, adapter_path: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {"base_model": base_model}
    if adapter_path:
        info["adapter_path"] = adapter_path
    if extra:
        info.update(extra)
    return info
