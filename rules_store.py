"""Pure data access for the grammar rule book — no web framework here, so it is
unit-testable without FastAPI. The HTTP layer lives in rules_api.py.

Reads the pre-generated static JSON in rules/rules_<learning>_<interface>.json
(see rules/README.md). On a miss the caller may fall back to on-demand
generation later; for now a miss is simply "not found".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RULES_DIR = Path(__file__).parent / "rules"

# Language codes are short ISO-style tokens (en, ru, es_MX). Validating them
# keeps user-supplied query params out of the file path (no traversal).
_CODE_RE = re.compile(r"^[a-zA-Z]{2}(_[A-Za-z]{2})?$")
_RULE_ID_RE = re.compile(r"^[a-z0-9-]+$")


class RulesNotFound(Exception):
    """No rule set for this (learning, interface) pair."""


class RuleNotFound(Exception):
    """The set exists but has no rule with this id."""


def _validate_code(code: str, name: str) -> str:
    if not isinstance(code, str) or not _CODE_RE.match(code):
        raise ValueError(f"invalid {name} code")
    return code


def _load_set(learning: str, interface: str) -> dict:
    _validate_code(learning, "learning")
    _validate_code(interface, "interface")
    path = RULES_DIR / f"rules_{learning}_{interface}.json"
    if not path.is_file():
        raise RulesNotFound(f"{learning}->{interface}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_rules(learning: str, interface: str) -> list[dict]:
    """Lightweight index: [{rule_id, title}, ...] (no heavy content)."""
    data = _load_set(learning, interface)
    return [{"rule_id": r["rule_id"], "title": r["title"]}
            for r in data.get("rules", [])]


def get_rule(learning: str, interface: str, rule_id: str) -> dict:
    """Full rule object for one rule_id."""
    if not isinstance(rule_id, str) or not _RULE_ID_RE.match(rule_id):
        raise ValueError("invalid rule_id")
    data = _load_set(learning, interface)
    for r in data.get("rules", []):
        if r.get("rule_id") == rule_id:
            return r
    raise RuleNotFound(rule_id)
