# -*- coding: utf-8 -*-
"""backend/core/artifact_store.py
生成物（テキスト/画像/音声）を保存し、再現性と運用性を担保するストア。

設計A:
- modality 横断で共通の meta schema を定義し、再生成に必要な情報を可能な限り保存する。
- ただし既存コードから渡される meta は自由形式のため、後方互換を壊さないように包み込む。
"""
from __future__ import annotations

import json
import time
import uuid
import hashlib
import platform
from pathlib import Path
from typing import Any, Dict, Optional

from lora_config import settings

ARTIFACT_META_SCHEMA_VERSION = 1

# schema_version 運用ルール（BK43）
# - meta.json の schema_version は「破壊的変更」のみインクリメントする（後方互換の追加は据え置き）。
# - 例: フィールド追加/説明追加 -> 変更不要 / 型変更・必須化・構造変更 -> version を上げる
# - UI/比較ツールは未知フィールドを無視できる設計とし、version mismatch は warn とする。

def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def _now_iso() -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")
    except Exception:
        return str(int(time.time()))

def _safe_pkg_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "") or ""
    except Exception:
        return ""

def _collect_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch  # type: ignore
        env["torch"] = getattr(torch, "__version__", "") or ""
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                env["cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
            except Exception:
                pass
    except Exception:
        env["torch"] = ""
        env["cuda_available"] = False

    # diffusers / transformers など（存在すれば）
    env["diffusers"] = _safe_pkg_version("diffusers")
    env["transformers"] = _safe_pkg_version("transformers")
    env["torchaudio"] = _safe_pkg_version("torchaudio")
    return env


def _ensure_meta_schema(modality: str, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """meta を共通 schema に寄せる（後方互換）。"""
    m: Dict[str, Any] = dict(meta or {})
    if m.get("schema_version") == ARTIFACT_META_SCHEMA_VERSION and m.get("modality"):
        return m

    wrapped: Dict[str, Any] = {
        "schema_version": ARTIFACT_META_SCHEMA_VERSION,
        "created_at": m.get("created_at") or _now_iso(),
        "modality": modality,
        "model": m.get("model") or {},
        "params": m.get("params") or {},
        "inputs": m.get("inputs") or {},
        "outputs": m.get("outputs") or {},
        "metrics": m.get("metrics") or {},
        "environment": m.get("environment") or _collect_environment(),
        # 既存の自由形式 meta を保持
        "extra": {k: v for k, v in m.items() if k not in {
            "schema_version","created_at","modality","model","params","inputs","outputs","metrics","environment","extra"
        }},
    }
    return wrapped

class ArtifactStore:
    def __init__(self, root_dir: Optional[Path] = None):
        if root_dir is None:
            root_dir = settings.dirs.get("artifacts") if hasattr(settings, "dirs") else None
        self.root_dir = Path(root_dir) if root_dir else Path("artifacts")
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def artifact_dir(self, modality: str, artifact_id: str) -> Path:
        d = self.root_dir / str(modality) / str(artifact_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, modality: str, data: bytes, ext: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成物を保存します。
        返り値: {"artifact_id":..., "path":..., "meta_path":...}

        設計A:
        - 失敗時も meta を残し、運用で原因追跡できるようにする（fail-soft）。
        """
        artifact_id = str(uuid.uuid4())
        d = self.artifact_dir(modality, artifact_id)
        file_name = f"output.{ext.lstrip('.')}" if ext else "output.bin"
        out_path = d / file_name
        meta_path = d / "meta.json"

        m = _ensure_meta_schema(modality, meta)
        m.setdefault("outputs", {})

        write_ok = False
        err_msg = ""

        try:
            out_path.write_bytes(data or b"")
            write_ok = True
            m["outputs"].update({
                "file": file_name,
                "sha256": _sha256_bytes(data or b""),
                "bytes": int(len(data or b"")),
            })
        except Exception as e:
            err_msg = str(e)
            # ファイル保存に失敗しても meta は残す
            m["outputs"].update({
                "file": file_name,
                "sha256": "",
                "bytes": int(len(data or b"")),
                "write_ok": False,
                "error": err_msg,
            })

        try:
            meta_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # 最後の砦: 何もできない場合は空を返す
            return {}

        return {
            "artifact_id": artifact_id,
            "path": str(out_path) if write_ok else "",
            "meta_path": str(meta_path),
        }


    def get_meta(self, modality: str, artifact_id: str) -> Dict[str, Any]:
        try:
            d = self.artifact_dir(modality, artifact_id)
            meta_path = d / "meta.json"
            if not meta_path.exists():
                return {}
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get_output_path(self, modality: str, artifact_id: str) -> Optional[Path]:
        meta = self.get_meta(modality, artifact_id)
        file_name = (((meta or {}).get("outputs") or {}).get("file"))
        if not file_name:
            # fallback: output.* を探す
            d = self.artifact_dir(modality, artifact_id)
            cands = list(d.glob("output.*"))
            return cands[0] if cands else None
        p = self.artifact_dir(modality, artifact_id) / str(file_name)
        return p if p.exists() else None


    # ===========================================================
    # BK43: Helper APIs (A/B比較・回帰比較のためのユーティリティ)
    # ===========================================================
    # 目的:
    # - 生成結果そのものだけでなく「どういう条件で生成されたか（meta）」を比較できるようにする
    # - UI側の比較画面や、回帰テストの差分解析に使用する

    def load_meta(self, artifact_id: str) -> Dict[str, Any]:
        """artifact_id から meta.json を読み込みます（存在しない場合は空dict）。"""
        try:
            mpath = self._artifact_dir / str(artifact_id) / "meta.json"
            if not mpath.exists():
                return {}
            return json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def compare_meta(self, a_id: str, b_id: str) -> Dict[str, Any]:
        """2つの artifact の meta を比較し、差分の概要を返します。"""
        a = self.load_meta(a_id)
        b = self.load_meta(b_id)

        def _get_flags_image(m: Dict[str, Any]) -> Dict[str, Any]:
            try:
                rp = (m.get("request_params") or {})
                cn = (rp.get("controlnet") or {})
                ip = (rp.get("inpaint") or {})
                return {
                    "controlnet_used": bool(cn.get("used")),
                    "controlnet_type": str(cn.get("type") or ""),
                    "inpaint_mode": str(ip.get("mode") or ""),
                    "inpaint_used": bool(str(ip.get("mode") or "")),
                }
            except Exception:
                return {}

        def _get_flags_audio(m: Dict[str, Any]) -> Dict[str, Any]:
            try:
                rp = (m.get("request_params") or {})
                vc = (m.get("vc") or rp.get("vc") or {})
                post = (m.get("post") or rp.get("post") or {})
                return {
                    "vc_used": bool(vc.get("used")) if isinstance(vc, dict) else False,
                    "target_lufs": (post.get("target_lufs") if isinstance(post, dict) else None),
                }
            except Exception:
                return {}

        diff = {
            "a_id": a_id,
            "b_id": b_id,
            "a_meta": a,
            "b_meta": b,
            "image_flags": {"a": _get_flags_image(a), "b": _get_flags_image(b)},
            "audio_flags": {"a": _get_flags_audio(a), "b": _get_flags_audio(b)},
        }

        # 指標（簡易）: Inpaint/ControlNet/VC/target_lufs の差分を human-friendly に
        try:
            ia = diff["image_flags"]["a"]
            ib = diff["image_flags"]["b"]
            diff["image_metrics"] = {
                "controlnet_changed": (ia.get("controlnet_used") != ib.get("controlnet_used")) or (ia.get("controlnet_type") != ib.get("controlnet_type")),
                "inpaint_changed": (ia.get("inpaint_mode") != ib.get("inpaint_mode")),
            }
        except Exception:
            diff["image_metrics"] = {}

        try:
            aa = diff["audio_flags"]["a"]
            ab = diff["audio_flags"]["b"]
            diff["audio_metrics"] = {
                "vc_changed": (aa.get("vc_used") != ab.get("vc_used")),
                "target_lufs_changed": (aa.get("target_lufs") != ab.get("target_lufs")),
            }
        except Exception:
            diff["audio_metrics"] = {}

        return diff

artifact_store = ArtifactStore()