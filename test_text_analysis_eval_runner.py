"""Regression tests for recorded V2 text-analysis evaluations."""

import os
import unittest

from evals.text_analysis.run_eval import _metrics_summary, evaluate_records


def _scenario(outcome="errors"):
    return {
        "id": "example",
        "request": {"text": "I has a book."},
        "expected": {
            "outcome": outcome,
            "min_error_count": 1 if outcome == "errors" else 0,
            "allowed_error_types": ["grammar"] if outcome == "errors" else [],
        },
    }


def _record(response, language_judgement=True):
    return {"scenario_id": "example", "response": response, "language_judgement": language_judgement}


class TextAnalysisEvalRunnerTests(unittest.TestCase):
  def test_valid_record_passes_all_deterministic_gates(self):
    response = {
        "corrected_text": "I have a book.",
        "errors": [{"category": "grammar", "concept_code": "verb.conjugation.present", "confidence": 0.9, "original": "has", "corrected": "have", "explanation": "Use have with I."}],
        "summary": "One grammar correction.",
    }
    self.assertEqual(
        evaluate_records([_scenario()], [_record(response)], require_language_judgement=True), []
    )


  def test_runner_rejects_bad_anchor_and_missing_language_judgement(self):
    response = {
        "corrected_text": "I have a book.",
        "errors": [{"category": "grammar", "concept_code": "verb.conjugation.present", "confidence": 0.9, "original": "does", "corrected": "have", "explanation": "Use have with I."}],
        "summary": "One grammar correction.",
    }
    failures = evaluate_records([_scenario()], [_record(response, False)], require_language_judgement=True)
    self.assertEqual({failure.reason for failure in failures}, {
        "missing or failed explanation-language judgement",
        "error original is not an input substring",
    })


  def test_clean_scenario_requires_exact_preservation(self):
    response = {"corrected_text": "Changed", "errors": [], "summary": "No errors."}
    failures = evaluate_records([_scenario("clean")], [_record(response)])
    self.assertEqual(
        [failure.reason for failure in failures], ["clean scenario changed submitted text"]
    )

  def test_metrics_include_price_based_cost_estimate(self):
    previous_input = os.environ.get("LLM_INPUT_COST_PER_MTOK")
    previous_output = os.environ.get("LLM_OUTPUT_COST_PER_MTOK")
    try:
      os.environ["LLM_INPUT_COST_PER_MTOK"] = "2"
      os.environ["LLM_OUTPUT_COST_PER_MTOK"] = "4"
      metrics = _metrics_summary([{
          "metrics": {"latency_ms": 100, "prompt_tokens": 120, "completion_tokens": 30},
      }])
      self.assertEqual(metrics["estimated_cost_usd"], 0.00036)
    finally:
      if previous_input is None:
        os.environ.pop("LLM_INPUT_COST_PER_MTOK", None)
      else:
        os.environ["LLM_INPUT_COST_PER_MTOK"] = previous_input
      if previous_output is None:
        os.environ.pop("LLM_OUTPUT_COST_PER_MTOK", None)
      else:
        os.environ["LLM_OUTPUT_COST_PER_MTOK"] = previous_output
