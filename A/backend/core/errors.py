# -*- coding: utf-8 -*-
"""backend/core/errors.py

設計A（実運用向け基盤）におけるドメインエラー階層。

BK33 は LoRA 学習ツールとして始まっていますが、画像（Diffusers）/音声（Voice）の
「生成ジョブ基盤」を追加していくにあたり、HTTP 層から独立したエラー表現が必要です。

ここでは最小限のエラー階層のみ提供します。
"""

from __future__ import annotations


class DomainError(Exception):
    """ドメイン（仕様）に起因するエラー。"""


class ValidationError(DomainError):
    """入力やJobSpecの整合性が取れない場合。"""


class PolicyViolation(DomainError):
    """ポリシー（安全/許可制御）により拒否された場合。"""


class NotFound(DomainError):
    """要求されたリソース（モデル/成果物/ジョブ等）が存在しない場合。"""


class Conflict(DomainError):
    """同時実行や状態遷移の競合など。"""
