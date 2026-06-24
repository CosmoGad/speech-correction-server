"""FastAPI router for the grammar rule book. Thin HTTP layer over rules_store.

Wire it into the app with auth, e.g.:
    from rules_api import router as rules_router
    app.include_router(rules_router, dependencies=[Depends(verify_api_key)])
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

import rules_store

router = APIRouter()

# Explicit charset so non-ASCII (Cyrillic/Greek/…) decodes correctly on clients
# that fall back to latin1 when it is missing (e.g. Dart's http response.body).
_MEDIA = "application/json; charset=utf-8"


@router.get("/rules")
def list_rules(learning: str = Query(...), interface: str = Query(...)):
    """Lightweight index of available rules for a (learning, interface) pair."""
    try:
        return JSONResponse(
            content=rules_store.list_rules(learning, interface), media_type=_MEDIA
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except rules_store.RulesNotFound:
        raise HTTPException(status_code=404, detail="rules not found")


# NOTE: GET /rule and GET /resolve-rule are defined in speech_correction_server.py
# (not here) because they need the DeepSeek client + ResponseCache for lazy
# generation and error->rule resolution. See rules/DYNAMIC_RULES_SPEC.md.
