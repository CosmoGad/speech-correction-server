"""Tests for deterministic V2 concept-to-rule registry."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from rules.concept_registry import ConceptRegistryError, load_concept_registry


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _registry_fixture(mappings: list[dict]) -> dict:
    return {"schema_version": 1, "mappings": mappings}


def _topic_fixture(*topics: dict) -> dict:
    return {"learning": "xx", "topics": list(topics)}


def _load_fixture(mappings: list[dict], *topics: dict):
    directory = TemporaryDirectory()
    root = Path(directory.name)
    _write_json(root / "concept_registry.json", _registry_fixture(mappings))
    _write_json(root / "topics_xx.json", _topic_fixture(*topics))
    registry = load_concept_registry(root / "concept_registry.json", root)
    return directory, registry


def test_real_registry_is_valid_and_does_not_guess_a_rule():
    registry = load_concept_registry()
    result = registry.resolve("ru", "verb.conjugation.present")
    assert not result.matched
    assert result.status == "unresolved"


def test_unknown_or_absent_concept_is_unresolved():
    registry = load_concept_registry()
    assert registry.resolve("ru", "verb.unknown").status == "unresolved"
    assert registry.resolve("ru", None).status == "unresolved"


def test_duplicate_learning_concept_is_rejected():
    mappings = [
        {"learning": "xx", "concept_code": "verb.present", "rule_id": "present"},
        {"learning": "xx", "concept_code": "verb.present", "rule_id": "other"},
    ]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_json(root / "concept_registry.json", _registry_fixture(mappings))
        _write_json(root / "topics_xx.json", _topic_fixture(
            {"rule_id": "present"}, {"rule_id": "other"}
        ))
        try:
            load_concept_registry(root / "concept_registry.json", root)
            assert False, "expected duplicate mapping error"
        except ConceptRegistryError as error:
            assert "duplicate mapping" in str(error)


def test_missing_rule_is_rejected():
    mappings = [{"learning": "xx", "concept_code": "verb.present", "rule_id": "missing"}]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_json(root / "concept_registry.json", _registry_fixture(mappings))
        _write_json(root / "topics_xx.json", _topic_fixture({"rule_id": "present"}))
        try:
            load_concept_registry(root / "concept_registry.json", root)
            assert False, "expected missing rule error"
        except ConceptRegistryError as error:
            assert "missing or inactive" in str(error)


def test_inactive_rule_is_rejected():
    mappings = [{"learning": "xx", "concept_code": "verb.present", "rule_id": "present"}]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_json(root / "concept_registry.json", _registry_fixture(mappings))
        _write_json(root / "topics_xx.json", _topic_fixture({"rule_id": "present", "active": False}))
        try:
            load_concept_registry(root / "concept_registry.json", root)
            assert False, "expected inactive rule error"
        except ConceptRegistryError as error:
            assert "missing or inactive" in str(error)


def test_extra_or_missing_fields_are_rejected():
    mappings = [{"learning": "xx", "concept_code": "verb.present", "rule_id": "present", "note": "no"}]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_json(root / "concept_registry.json", _registry_fixture(mappings))
        _write_json(root / "topics_xx.json", _topic_fixture({"rule_id": "present"}))
        try:
            load_concept_registry(root / "concept_registry.json", root)
            assert False, "expected schema error"
        except ConceptRegistryError as error:
            assert "exactly" in str(error)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
