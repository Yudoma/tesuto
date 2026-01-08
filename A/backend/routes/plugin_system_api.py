# -*- coding: utf-8 -*-
"""backend/routes/plugin_system_api.py

API endpoints for the plugin system.
This is purely discovery/metadata and does not implement any research logic.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.plugin_system import list_frontend_plugins

router = APIRouter()

@router.get("/plugins/frontend", tags=["Plugins"])
def api_list_frontend_plugins(mode: str = "text"):
    infos = list_frontend_plugins(mode=mode)
    return {
        "mode": mode,
        "plugins": [
            {
                "name": i.name,
                "module_url": i.module_url,
                "title": i.title,
                "description": i.description,
                "order": i.order,
            }
            for i in infos
        ],
    }

@router.get("/api/plugins/frontend", tags=["Plugins"])
def get_frontend_plugins_compat(mode: str = "text"):
    """Backward-compatible alias.

    Some older frontends mistakenly call /api/plugins/frontend even though the router is already mounted at /api,
    resulting in /api/api/plugins/frontend. This endpoint makes that path work too.
    """
    infos = list_frontend_plugins(mode=mode)
    return {
        "mode": mode,
        "plugins": [
            {
                "name": i.name,
                "module_url": i.module_url,
                "title": i.title,
                "description": i.description,
                "order": i.order,
            }
            for i in infos
        ],
    }
