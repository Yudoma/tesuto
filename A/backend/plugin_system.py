# -*- coding: utf-8 -*-
"""backend/plugin_system.py

Minimal plugin framework for LoRA Factory / Text tool.

Goals:
- Keep core modifications tiny and stable across updates.
- Allow optional plugins to add:
  (a) backend API routers
  (b) frontend UI panels/tabs (served as static ES modules)

Conventions:
- Plugins live under: <base_dir>/plugins/<plugin_name>/
- Backend entrypoint: plugins.<plugin_name>.backend_plugin
    - must expose function: get_routers() -> list[fastapi.APIRouter]
- Frontend entrypoint (static module): /plugins/<plugin_name>/frontend_plugin.js
    - is served via StaticFiles mount in lora_app.py
    - metadata is exposed through backend/routes/plugin_system_api.py

This module is content-agnostic. It only discovers and loads plugins.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from lora_config import settings


@dataclass
class FrontendPluginInfo:
    name: str
    # URL path served by StaticFiles (mounted at /plugins)
    module_url: str
    # Optional metadata from plugin.json
    title: str = ""
    description: str = ""
    # Optional hint for UI ordering
    order: int = 1000


def _plugins_root() -> Path:
    return Path(settings.base_dir) / "plugins"


def list_plugins() -> List[str]:
    root = _plugins_root()
    if not root.exists():
        return []
    out: List[str] = []
    for d in root.iterdir():
        if d.is_dir() and (d / "__init__.py").exists():
            out.append(d.name)
    return sorted(out)


def _read_plugin_manifest(plugin_dir: Path) -> Dict[str, Any]:
    manifest = plugin_dir / "plugin.json"
    if not manifest.exists():
        return {}
    try:
        import json
        return json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_frontend_plugins(mode: str = "text") -> List[FrontendPluginInfo]:
    """Return frontend plugin module URLs discoverable at runtime.

    mode is a hint for the UI. For now, plugins declare supported modes in plugin.json:
      { "frontend": { "modes": ["text"], "module": "frontend_plugin.js", ... } }
    """
    root = _plugins_root()
    infos: List[FrontendPluginInfo] = []
    for name in list_plugins():
        pdir = root / name
        manifest = _read_plugin_manifest(pdir)
        fe = (manifest.get("frontend") or {})
        modes = fe.get("modes") or ["text"]
        if mode not in modes:
            continue
        module_file = fe.get("module") or "frontend_plugin.js"
        # require that the module file exists
        if not (pdir / module_file).exists():
            continue
        title = manifest.get("title") or name
        desc = manifest.get("description") or ""
        order = int(fe.get("order") or 1000)
        infos.append(FrontendPluginInfo(
            name=name,
            module_url=f"/plugins/{name}/{module_file}",
            title=title,
            description=desc,
            order=order
        ))
    infos.sort(key=lambda x: (x.order, x.name))
    return infos


def load_backend_routers() -> List[Any]:
    """Load backend routers from plugins.

    Each plugin may expose plugins.<name>.backend_plugin.get_routers() -> list[APIRouter]
    Routers should use their own prefixes (e.g. /heretic) to avoid collisions.
    """
    routers: List[Any] = []
    for name in list_plugins():
        modname = f"plugins.{name}.backend_plugin"
        try:
            mod = importlib.import_module(modname)
        except Exception:
            # Silent fail: plugin is optional
            continue
        get_routers = getattr(mod, "get_routers", None)
        if callable(get_routers):
            try:
                rs = get_routers()
                if rs:
                    routers.extend(list(rs))
            except Exception:
                continue
    return routers
