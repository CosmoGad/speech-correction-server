"""Pure data access for the grammar rule book — no web framework here, so it is
unit-testable without FastAPI. The HTTP layer lives in rules_api.py.

Reads the pre-generated static JSON in rules/rules_<learning>_<interface>.json
(see rules/README.md). On a miss the caller may fall back to on-demand
generation later; for now a miss is simply "not found".
"""

from __future__ import annotations

import hashlib
import hmac
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
_QUIZ_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_CEFR_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}
_QUIZ_MIN_QUESTIONS = 3
_QUIZ_MAX_QUESTIONS = 10
_QUIZ_PROMPT_MAX_LENGTH = 500
_QUIZ_OPTION_MAX_LENGTH = 300
_QUIZ_EXPLANATION_MAX_LENGTH = 1000


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
    """Lightweight index: [{rule_id, title}, ...] (no heavy content).

    Inactive topics are filtered out — see `inactive_rule_ids`."""
    data = _load_set(learning, interface)
    hidden = inactive_rule_ids(learning)
    return [{"rule_id": r["rule_id"], "title": r["title"]}
            for r in data.get("rules", [])
            if r["rule_id"] not in hidden]


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

def _all_topics(learning: str) -> list[dict]:
    _validate_code(learning, "learning")
    path = RULES_DIR / f"topics_{learning}.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("topics", [])


def inactive_rule_ids(learning: str) -> set[str]:
    """Topics marked `"active": false` — currently the pronunciation ones.

    They are switched off rather than deleted because the pipeline cannot reach
    them: speech is transcribed to text on the device, so a mispronunciation
    either arrives as the correct word (no error at all) or as a different word
    (a vocabulary error). Nothing ever resolves to a stress or intonation rule,
    while the lessons themselves read as broken — their `wrong` and `right`
    examples are necessarily the same string, since the difference is audible
    and not spelled.

    Deleting them would throw away usable content: once real pronunciation
    analysis exists (see the TTS plan), flipping the flag brings them back."""
    return {t["rule_id"] for t in _all_topics(learning)
            if t.get("active") is False}


def load_topics(learning: str) -> list[dict]:
    """The fixed taxonomy for a learning language: [{rule_id, title}, ...].
    This is the ONLY source of rule_ids — dynamic resolution never mints new
    ids, it only selects from here, which is what prevents duplicates.

    Inactive topics are excluded, so `/resolve-rule` cannot map an error onto a
    rule the product cannot actually teach yet."""
    return [t for t in _all_topics(learning) if t.get("active") is not False]


def topic_concept_code(learning: str, rule_id: str) -> str:
    """Return an opaque, stable concept identifier for a taxonomy topic.

    The code is derived exclusively from taxonomy data.  It intentionally does
    not encode language-specific keywords or a handwritten exception list:
    adding a rule automatically creates a new exact mapping target.  The model
    sees the code next to its human title, while the server maps it back without
    another model call.
    """
    _validate_code(learning, "learning")
    if not isinstance(rule_id, str) or not _RULE_ID_RE.match(rule_id):
        raise ValueError("invalid rule_id")
    digest = hashlib.sha256(
        f"speech-correction/taxonomy-v1/{learning}/{rule_id}".encode("utf-8")
    ).hexdigest()[:20]
    return f"taxonomy.{digest}"


def topics_with_concepts(learning: str) -> list[dict]:
    """Active rule taxonomy enriched with its deterministic concept codes."""
    return [
        {**topic, "concept_code": topic_concept_code(learning, topic["rule_id"])}
        for topic in load_topics(learning)
    ]


def resolve_concept(learning: str, concept_code: str) -> str | None:
    """Map one declared concept to exactly one active rule, otherwise None."""
    if not isinstance(concept_code, str):
        return None
    matches = [
        topic["rule_id"]
        for topic in topics_with_concepts(learning)
        if hmac.compare_digest(topic["concept_code"], concept_code)
    ]
    return matches[0] if len(matches) == 1 else None


def topic_title(learning: str, rule_id: str) -> str | None:
    for t in load_topics(learning):
        if t.get("rule_id") == rule_id:
            return t.get("title")
    return None


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


def quiz_level(level: str) -> str:
    """Validate and return the exact CEFR level used for generation and cache."""
    if not isinstance(level, str) or level not in _CEFR_LEVELS:
        raise ValueError("invalid level")
    return level


def quiz_question_count(question_count: int) -> int:
    """Keep quizzes short, but let product choose a count within safe bounds."""
    if isinstance(question_count, bool) or not isinstance(question_count, int):
        raise ValueError("invalid question_count")
    if not _QUIZ_MIN_QUESTIONS <= question_count <= _QUIZ_MAX_QUESTIONS:
        raise ValueError("invalid question_count")
    return question_count


def validate_quiz_parameters(learning: str, interface: str, rule_id: str,
                             level: str, question_count: int) -> tuple[str, int]:
    """Validate path-safe quiz identifiers and return exact level and count.

    Membership in the server's configured languages is checked by the HTTP
    layer because those runtime configuration dictionaries live there.
    """
    _validate_code(learning, "learning")
    _validate_code(interface, "interface")
    if not isinstance(rule_id, str) or not _RULE_ID_RE.match(rule_id):
        raise ValueError("invalid rule_id")
    return quiz_level(level), quiz_question_count(question_count)


def _bounded_text(value, field: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid {field}")
    value = value.strip()
    if not value or len(value) > max_length:
        raise ValueError(f"invalid {field}")
    return value


def validate_rule_quiz(data: dict, *, expected_question_count: int | None = None) -> dict:
    """Strictly validate and normalize model-generated quiz JSON."""
    if not isinstance(data, dict):
        raise ValueError("quiz must be an object")
    questions = data.get("questions")
    if not isinstance(questions, list):
        raise ValueError("quiz.questions must be a list")
    expected_count = (
        quiz_question_count(expected_question_count)
        if expected_question_count is not None
        else None
    )
    if expected_count is not None and len(questions) != expected_count:
        raise ValueError(f"quiz must contain exactly {expected_count} questions")
    if expected_count is None and not _QUIZ_MIN_QUESTIONS <= len(questions) <= _QUIZ_MAX_QUESTIONS:
        raise ValueError(
            f"quiz must contain {_QUIZ_MIN_QUESTIONS} to {_QUIZ_MAX_QUESTIONS} questions")

    normalized = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for position, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError("question must be an object")
        question_id = question.get("id")
        if not isinstance(question_id, str) or not _QUIZ_ID_RE.fullmatch(question_id):
            raise ValueError("invalid question id")
        if question_id != f"q{position}":
            raise ValueError("question ids must be sequential q1 through qN")
        if question_id in seen_ids:
            raise ValueError("duplicate question id")
        seen_ids.add(question_id)

        prompt = _bounded_text(
            question.get("prompt"), "question prompt", _QUIZ_PROMPT_MAX_LENGTH)
        prompt_key = prompt.casefold()
        if prompt_key in seen_prompts:
            raise ValueError("duplicate question prompt")
        seen_prompts.add(prompt_key)

        options = question.get("options")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("question must contain exactly 4 options")
        clean_options = [
            _bounded_text(option, "option", _QUIZ_OPTION_MAX_LENGTH)
            for option in options
        ]
        if len({option.casefold() for option in clean_options}) != 4:
            raise ValueError("question options must be unique")

        correct_index = question.get("correct_index")
        if (isinstance(correct_index, bool)
                or not isinstance(correct_index, int)
                or not 0 <= correct_index <= 3):
            raise ValueError("invalid correct_index")

        explanation = _bounded_text(
            question.get("explanation"), "explanation",
            _QUIZ_EXPLANATION_MAX_LENGTH)
        normalized.append({
            "id": question_id,
            "prompt": prompt,
            "options": clean_options,
            "correct_index": correct_index,
            "explanation": explanation,
        })
    return {"questions": normalized}


def build_rule_quiz_prompt(title: str, learning_name: str,
                           interface_name: str, level: str, question_count: int,
                           rule_context: dict | None = None) -> str:
    """Build a compact structured-output prompt for a configurable quiz."""
    context = ""
    if isinstance(rule_context, dict):
        compact_context = {
            "explanation": rule_context.get("explanation", ""),
            "examples": rule_context.get("examples", []),
        }
        # Static rule files are trusted, but bounding the context prevents an
        # unexpectedly large lesson from inflating every generation request.
        serialized = json.dumps(compact_context, ensure_ascii=False)
        context = f"\nLesson source (use as ground truth):\n{serialized[:4000]}\n"

    return (
        f"Create a {level} multiple-choice practice quiz for the "
        f"{learning_name} rule \"{title}\".\n"
        f"The learner's interface language is {interface_name}.{context}\n"
        "Return ONLY one valid JSON object with exactly this shape:\n"
        '{"questions":[{"id":"q1","prompt":string,'
        '"options":[string,string,string,string],"correct_index":integer,'
        '"explanation":string}]}\n\n'
        "STRICT REQUIREMENTS:\n"
        f"- Generate exactly {question_count} distinct multiple-choice questions.\n"
        "- Every question has exactly 4 unique, non-empty options and exactly "
        "one unambiguously correct answer.\n"
        f"- Use sequential ids q1 through q{question_count} and a zero-based correct_index from 0 to 3.\n"
        f"- Write every prompt and explanation in {interface_name}.\n"
        f"- Write all answer options and language examples in {learning_name}.\n"
        "- Test application of the rule in new contexts; do not merely ask for "
        "the rule definition.\n"
        "- Keep prompt <= 500 characters, each option <= 300 characters, and "
        "each explanation <= 1000 characters.\n"
        "- Do not use markdown or add fields outside the JSON object."
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
