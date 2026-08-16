"""Offline guardrails for text-analysis prompt and contract changes."""
import json
import re
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SCENARIOS_PATH = ROOT / "evals" / "text_analysis" / "scenarios.json"
LEARNING = {"ru", "en", "de", "uk"}
LEVELS = {"A1", "B1", "B2", "C1", "C2"}
CATEGORIES = {"grammar", "vocabulary", "spelling", "word_order", "style", "other"}
RTL = {"ar", "fa", "ur"}
INVARIANTS = {"corrected_text_changes_input", "corrected_text_preserves_input", "no_invented_errors", "no_pronunciation_claims", "uses_context_to_generate_text", "uses_context_to_preserve_meaning", "rtl_interface"}


def scenarios():
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


def test_scenarios_follow_schema():
    data = scenarios(); assert len(data) == 24
    assert len({item["id"] for item in data}) == len(data)
    for item in data:
        assert set(item) == {"id", "request", "expected", "tags"}
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", item["id"])
        request, expected = item["request"], item["expected"]
        assert request["language"] in LEARNING and request["level"] in LEVELS
        assert request["style"] in {"formal", "casual", "neutral"}
        assert request["text"].strip() or request.get("context", "").strip()
        assert expected["outcome"] in {"clean", "errors"}
        assert set(expected["allowed_error_types"]).issubset(CATEGORIES)
        assert expected["explanation_language"] == "interface"
        assert set(expected["invariants"]).issubset(INVARIANTS)
        assert "no_pronunciation_claims" in expected["invariants"]
        if expected["outcome"] == "clean":
            assert expected["min_error_count"] == 0 and not expected["allowed_error_types"]
            assert "corrected_text_preserves_input" in expected["invariants"]
        elif request["text"].strip():
            assert expected["min_error_count"] >= 1 and expected["allowed_error_types"]
        if request["interface_language"] in RTL:
            assert "rtl_interface" in expected["invariants"]


def test_coverage_matrix():
    data = scenarios()
    assert Counter(item["request"]["language"] for item in data) == {language: 6 for language in LEARNING}
    assert {item["request"]["level"] for item in data} == LEVELS
    assert RTL.issubset({item["request"]["interface_language"] for item in data})
    assert any(item["request"]["language"] == item["request"]["interface_language"] for item in data)
    assert any(item["request"]["interface_language"] == "es_MX" for item in data)
    assert any(not item["request"]["text"] for item in data)


class TextAnalysisEvalSetTests(unittest.TestCase):
    def test_scenarios_follow_schema(self):
        test_scenarios_follow_schema()

    def test_coverage_matrix(self):
        test_coverage_matrix()
