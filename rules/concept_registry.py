"""Data-only concept-to-rule registry for Analysis Contract V2.

The registry deliberately contains no text matching, language branching or
fallback selection.  A concept that has not explicitly been curated resolves
to ``unresolved`` instead of opening a plausible-but-wrong lesson.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


_CONCEPT_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_LANGUAGE_CODE_RE = re.compile(r"^[a-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$")


class ConceptRegistryError(ValueError):
    """Raised when a registry cannot safely be used."""


@dataclass(frozen=True)
class RuleResolution:
    """The deterministic result of resolving one V2 concept."""

    status: str
    rule_id: str | None = None

    @property
    def matched(self) -> bool:
        return self.status == "matched"


class ConceptRegistry:
    """Validated mapping from ``(learning, concept_code)`` to a rule ID."""

    def __init__(self, mappings: dict[tuple[str, str], str]) -> None:
        self._mappings = mappings

    def resolve(self, learning: str, concept_code: str | None) -> RuleResolution:
        """Return a matching rule only for an explicitly declared concept."""
        if not concept_code:
            return RuleResolution(status="unresolved")
        rule_id = self._mappings.get((learning, concept_code))
        if rule_id is None:
            return RuleResolution(status="unresolved")
        return RuleResolution(status="matched", rule_id=rule_id)


def load_concept_registry(
    registry_path: str | Path | None = None,
    taxonomy_dir: str | Path | None = None,
) -> ConceptRegistry:
    """Load and validate a registry against active taxonomy topics.

    Validation rejects duplicate concepts, malformed entries, mappings to
    unknown rules, and mappings to inactive rules.  That makes a bad content
    update fail at startup/CI rather than send users to an unrelated lesson.
    """
    rules_dir = Path(__file__).resolve().parent
    registry_file = Path(registry_path) if registry_path else rules_dir / "concept_registry.json"
    topics_dir = Path(taxonomy_dir) if taxonomy_dir else rules_dir
    document = _read_json(registry_file, "registry")

    if document.get("schema_version") != 1:
        raise ConceptRegistryError("registry.schema_version must be 1")
    entries = document.get("mappings")
    if not isinstance(entries, list):
        raise ConceptRegistryError("registry.mappings must be a list")

    active_topics_by_language: dict[str, set[str]] = {}
    mappings: dict[tuple[str, str], str] = {}
    for index, entry in enumerate(entries):
        learning, concept_code, rule_id = _validate_entry(entry, index)
        key = (learning, concept_code)
        if key in mappings:
            raise ConceptRegistryError(
                f"duplicate mapping for learning={learning!r}, concept_code={concept_code!r}"
            )
        active_topics = active_topics_by_language.setdefault(
            learning, _load_active_topics(topics_dir, learning)
        )
        if rule_id not in active_topics:
            raise ConceptRegistryError(
                f"mapping {learning}/{concept_code} references missing or inactive rule_id {rule_id!r}"
            )
        mappings[key] = rule_id
    return ConceptRegistry(mappings)


def _validate_entry(entry: Any, index: int) -> tuple[str, str, str]:
    if not isinstance(entry, dict):
        raise ConceptRegistryError(f"registry.mappings[{index}] must be an object")
    required = {"learning", "concept_code", "rule_id"}
    if set(entry) != required:
        raise ConceptRegistryError(
            f"registry.mappings[{index}] must contain exactly {sorted(required)}"
        )
    learning = entry["learning"]
    concept_code = entry["concept_code"]
    rule_id = entry["rule_id"]
    if not isinstance(learning, str) or not _LANGUAGE_CODE_RE.fullmatch(learning):
        raise ConceptRegistryError(f"registry.mappings[{index}].learning is invalid")
    if not isinstance(concept_code, str) or not _CONCEPT_CODE_RE.fullmatch(concept_code):
        raise ConceptRegistryError(f"registry.mappings[{index}].concept_code is invalid")
    if not isinstance(rule_id, str) or not rule_id.strip():
        raise ConceptRegistryError(f"registry.mappings[{index}].rule_id is invalid")
    return learning, concept_code, rule_id


def _load_active_topics(topics_dir: Path, learning: str) -> set[str]:
    document = _read_json(topics_dir / f"topics_{learning}.json", f"taxonomy for {learning}")
    topics = document.get("topics")
    if not isinstance(topics, list):
        raise ConceptRegistryError(f"taxonomy for {learning} has no topics list")
    active_ids: set[str] = set()
    for index, topic in enumerate(topics):
        if not isinstance(topic, dict) or not isinstance(topic.get("rule_id"), str):
            raise ConceptRegistryError(f"taxonomy for {learning}.topics[{index}] has no rule_id")
        if topic.get("active", True):
            active_ids.add(topic["rule_id"])
    return active_ids


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConceptRegistryError(f"{label} file is missing: {path.name}") from error
    except json.JSONDecodeError as error:
        raise ConceptRegistryError(f"{label} is not valid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ConceptRegistryError(f"{label} must be a JSON object")
    return payload
