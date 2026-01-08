# -*- coding: utf-8 -*-
"""backend/core/regression_tests.py
回帰検知のための最小ユーティリティ。
"""
from __future__ import annotations
from typing import Dict, Any, List

def basic_meta_checks(meta: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not meta.get("artifact_id"):
        errors.append("artifact_id missing")
    if not meta.get("modality"):
        errors.append("modality missing")
    if not isinstance(meta.get("outputs"), dict):
        errors.append("outputs missing")
    return errors
