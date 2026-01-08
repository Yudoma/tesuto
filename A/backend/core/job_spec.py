# -*- coding: utf-8 -*-
"""backend/core/job_spec.py

設計Aの中核である JobSpec（再現性の仕様）を BK33 に導入します。

BK33 は従来「即時生成（同期）」を中心に実装されていますが、実運用では
 - 再現性（seed/モデル/前後処理/バージョン）
 - キャッシュ（同一条件の再生成回避）
 - ジョブキュー/ワーカー分離
が重要になります。

ここでは設計Aドキュメントに合わせ、JobSpec を安定ハッシュ化できるようにします。

ポイント:
- JSONとしてシリアライズ可能
- hash() は内容から安定キーを生成（キャッシュ/履歴/比較用）
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _stable_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now_iso() -> str:
    # Windows/日本語運用でも扱いやすい ISO 8601（秒まで）
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass(frozen=True)
class ModelRef:
    """モデル参照（id + revision 等）"""

    model_id: str
    revision: str = ""
    backend: str = ""  # diffusers / gpt_sovits 等
    dtype: str = ""    # fp16/bf16 等

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "backend": self.backend,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class AdapterRef:
    """LoRA/ControlNet/Voice preset 等の参照"""

    adapter_id: str
    version: str = ""
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"adapter_id": self.adapter_id, "version": self.version, "weight": self.weight}


@dataclass
class JobSpec:
    """生成のための「不変仕様」。同一specなら同一結果（seed固定）を目指す。"""

    job_type: str  # image_generate / voice_tts 等
    request_id: str = ""

    prompt_source: Dict[str, Any] = field(default_factory=dict)
    compiled_prompt: Dict[str, Any] = field(default_factory=dict)

    model_ref: Optional[ModelRef] = None
    adapter_refs: List[AdapterRef] = field(default_factory=list)
    generation_params: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    preprocess_steps: List[Dict[str, Any]] = field(default_factory=list)
    postprocess_steps: List[Dict[str, Any]] = field(default_factory=list)

    policy_context: Dict[str, Any] = field(default_factory=dict)
    runtime_hints: Dict[str, Any] = field(default_factory=dict)

    app_version: str = ""
    pipeline_version: str = ""
    created_at: str = field(default_factory=_now_iso)

    def ensure_request_id(self) -> "JobSpec":
        if self.request_id:
            return self
        self.request_id = uuid.uuid4().hex
        return self

    def ensure_seed(self) -> "JobSpec":
        if self.seed is not None:
            return self
        # 32-bit seed
        self.seed = int(uuid.uuid4().hex[:8], 16)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_type": self.job_type,
            "request_id": self.request_id,
            "prompt_source": self.prompt_source,
            "compiled_prompt": self.compiled_prompt,
            "model_ref": self.model_ref.to_dict() if self.model_ref else None,
            "adapter_refs": [a.to_dict() for a in (self.adapter_refs or [])],
            "generation_params": self.generation_params,
            "seed": self.seed,
            "preprocess_steps": self.preprocess_steps,
            "postprocess_steps": self.postprocess_steps,
            "policy_context": self.policy_context,
            "runtime_hints": self.runtime_hints,
            "app_version": self.app_version,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobSpec":
        mr = d.get("model_ref")
        model_ref = ModelRef(**mr) if isinstance(mr, dict) and mr.get("model_id") else None
        ars = []
        for a in d.get("adapter_refs") or []:
            if isinstance(a, dict) and a.get("adapter_id"):
                ars.append(AdapterRef(**a))
        return cls(
            job_type=d.get("job_type") or "",
            request_id=d.get("request_id") or "",
            prompt_source=d.get("prompt_source") or {},
            compiled_prompt=d.get("compiled_prompt") or {},
            model_ref=model_ref,
            adapter_refs=ars,
            generation_params=d.get("generation_params") or {},
            seed=d.get("seed"),
            preprocess_steps=d.get("preprocess_steps") or [],
            postprocess_steps=d.get("postprocess_steps") or [],
            policy_context=d.get("policy_context") or {},
            runtime_hints=d.get("runtime_hints") or {},
            app_version=d.get("app_version") or "",
            pipeline_version=d.get("pipeline_version") or "",
            created_at=d.get("created_at") or _now_iso(),
        )

    def hash(self) -> str:
        """JobSpec内容から安定ハッシュ（spec_hash）を生成。"""
        s = _stable_dumps(self.to_dict())
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        return h
