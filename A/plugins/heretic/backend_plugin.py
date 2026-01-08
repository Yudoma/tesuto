# -*- coding: utf-8 -*-
"""plugins/heretic/backend_plugin.py

Backend entrypoint for the plugin system.

Returns BOTH routers:
- router      (prefix /heretic)
- router_api  (prefix /api/heretic)

So endpoints remain reachable across host mounting strategies.
"""

from __future__ import annotations
from typing import Any, List

def get_routers() -> List[Any]:
    try:
        from .backend.api import router, router_api
        print("[Heretic Plugin] backend router loaded OK (v5.0 full UI APIs)")
        return [router, router_api]
    except Exception as e:
        try:
            print(f"[Heretic Plugin] backend router load failed: {type(e).__name__}: {e}")
        except Exception:
            pass
        return []
