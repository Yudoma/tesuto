from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def _write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

@dataclass
class ModelEntry:
    id: str
    modality: str  # text|image|audio
    kind: str      # lora|checkpoint|adapter
    name: str
    path: str
    created_at: int
    meta: Dict[str, Any]

def registry_path(base_dir: Path) -> Path:
    return (base_dir / "models" / "registry.json").resolve()

def load_registry(base_dir: Path) -> Dict[str, Any]:
    p = registry_path(base_dir)
    if not p.exists():
        _safe_mkdir(p.parent)
        _write_json(p, {"version": 1, "updated_at": int(time.time()), "entries": []})
    return _read_json(p)

def save_registry(base_dir: Path, reg: Dict[str, Any]) -> None:
    reg["updated_at"] = int(time.time())
    _write_json(registry_path(base_dir), reg)

def make_entry_id(job_id: str) -> str:
    return f"job:{job_id}"

def upsert_entry(base_dir: Path, entry: ModelEntry) -> None:
    reg = load_registry(base_dir)
    entries = reg.get("entries", [])
    entries = [e for e in entries if e.get("id") != entry.id]
    entries.insert(0, asdict(entry))
    reg["entries"] = entries
    save_registry(base_dir, reg)

def register_job_output(
    base_dir: Path,
    job_id: str,
    modality: str,
    output_dir: Path,
    kind: str = "lora",
    name: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> ModelEntry:
    output_dir = output_dir.resolve()
    entry = ModelEntry(
        id=make_entry_id(job_id),
        modality=modality,
        kind=kind,
        name=name or output_dir.name,
        path=str(output_dir),
        created_at=int(time.time()),
        meta=meta or {},
    )
    upsert_entry(base_dir, entry)
    # Drop a manifest for portability
    try:
        _write_json(output_dir / "manifest.json", asdict(entry))
    except Exception:
        pass
    return entry
