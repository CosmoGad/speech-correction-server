"""Pure data access for the grammar rule book — no web framework here, so it is
unit-testable without FastAPI. The HTTP layer lives in rules_api.py.

Reads the pre-generated static JSON in rules/rules_<learning>_<interface>.json
(see rules/README.md). On a miss the caller may fall back to on-demand
generation later; for now a miss is simply "not found".
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

RULES_DIR = Path(__file__).parent / "rules"

# Language codes are short ISO-style tokens (en, ru, es_MX). Validating them
# keeps user-supplied query params out of the file path (no traversal).
_CODE_RE = re.compile(r"^[a-zA-Z]{2}(_[A-Za-z]{2})?$")
# rule_id is only ever matched against loaded JSON (never used to build a path),
# so it may contain unicode (ö, é, IPA, Cyrillic in slugs); we only reject path
# separators / control chars and cap the length.
_RULE_ID_RE = re.compile(r"^[^/\\\x00-\x1f]{1,128}$")


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


# ── Dynamic rule book: taxonomy + prompts (see rules/DYNAMIC_RULES_SPEC.md) ──

# Learning-language code -> English display name (mirrors the client + the
# offline generator). Used only to phrase prompts clearly.
LANGUAGE_NAMES = {
    "en": "English", "ru": "Russian", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
    "pl": "Polish", "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic",
    "fa": "Persian", "ur": "Urdu", "sr": "Serbian", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "el": "Greek", "cs": "Czech",
    "ro": "Romanian", "hu": "Hungarian", "bg": "Bulgarian", "hr": "Croatian",
    "sk": "Slovak", "sl": "Slovenian", "fi": "Finnish", "lt": "Lithuanian",
    "lv": "Latvian", "et": "Estonian",
}


def load_topics(learning: str) -> list[dict]:
    """The fixed taxonomy for a learning language: [{rule_id, title}, ...].
    This is the ONLY source of rule_ids — dynamic resolution never mints new
    ids, it only selects from here, which is what prevents duplicates."""
    _validate_code(learning, "learning")
    path = RULES_DIR / f"topics_{learning}.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("topics", [])


def topic_title(learning: str, rule_id: str) -> str | None:
    for t in load_topics(learning):
        if t.get("rule_id") == rule_id:
            return t.get("title")
    return None


def error_signature(learning: str, err_type: str, original: str,
                    corrected: str) -> str:
    """Stable key for caching an error->rule_id resolution."""
    raw = json.dumps([
        learning, (err_type or "").lower(),
        (original or "").strip().lower(), (corrected or "").strip().lower(),
    ], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_rule_prompt(title: str, learning_name: str,
                      interface_name: str) -> str:
    """Mirror of the offline generator's prompt, so lazily-generated lessons
    match the pre-generated ones in shape and language split."""
    return (
        f"You are an expert {learning_name} teacher creating a short lesson "
        f"for a learner whose language is {interface_name}.\n\n"
        f"Topic (a common {learning_name} mistake): \"{title}\".\n\n"
        "Produce a compact, practical micro-lesson. Return ONLY a valid JSON "
        "object (UTF-8, no markdown) with this exact shape:\n"
        '{\n  "title": string,\n  "explanation": string,\n'
        '  "examples": [ {"wrong": string, "right": string, "note": string} ],\n'
        '  "exercises": [ {"prompt": string, "answer": string} ]\n}\n\n'
        "LANGUAGE RULES (strict):\n"
        f"- title, explanation, note and exercise prompts: in {interface_name}.\n"
        f"- wrong, right and exercise answers (the actual language samples): "
        f"in {learning_name}. Keep practice sentences in {learning_name}.\n"
        "- 2-3 examples and 2-3 exercises. Concise and beginner-friendly."
    )


def build_resolve_prompt(learning_name: str, topics: list[dict],
                         err_type: str, original: str, corrected: str,
                         explanation: str) -> str:
    catalog = "\n".join(f"- {t['rule_id']}: {t['title']}" for t in topics)
    return (
        f"You are an expert {learning_name} teacher. A learner made this "
        f"mistake:\n- wrong: \"{original}\"\n- corrected: \"{corrected}\"\n"
        f"- type: {err_type}\n- note: {explanation}\n\n"
        f"Catalog of available rule topics (rule_id: title):\n{catalog}\n\n"
        "Pick the SINGLE rule_id whose topic best explains this mistake. If "
        "none is a good fit, use null. Return ONLY a JSON object of the form "
        '{"rule_id": "<one of the rule_id values above, or null>"}.'
    )


def extract_json(raw: str) -> dict:
    """Tolerant JSON-object extraction (model usually returns clean JSON when
    response_format=json_object, but be defensive)."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e > s:
            return json.loads(raw[s:e + 1])
        raise
