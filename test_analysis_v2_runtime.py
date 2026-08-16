"""Offline contract checks for the feature-flagged V2 analysis adapter."""

import json

import rules_store
import speech_correction_server as server


def _request() -> server.CorrectionRequest:
    return server.CorrectionRequest(
        text="She go to school every day.", language="en", level="A1",
        interface_language="en", style="neutral",
    )


def test_v2_system_prompt_contains_no_learner_text_and_payload_is_separate():
    request = _request()
    prompt, payload = server.build_v2_prompt(request)
    assert request.text not in prompt
    assert json.loads(payload)["text"] == request.text


def test_v2_adapter_maps_declared_concept_without_another_model_call():
    request = _request()
    concept = next(
        topic["concept_code"] for topic in rules_store.topics_with_concepts("en")
        if topic["rule_id"] == "subject-verb-agreement")
    raw = json.dumps({
        "corrected_text": "She goes to school every day.",
        "errors": [{
            "category": "grammar", "concept_code": concept, "confidence": 0.95,
            "original": "go", "corrected": "goes",
            "explanation": "Use goes with she.",
        }],
        "summary": "One grammar correction.",
    })
    content = server.parse_v2_correction_response(raw, request)
    assert content["contract_version"] == 2
    assert content["error_analysis"][0]["rule_id"] == "subject-verb-agreement"
    assert content["pronunciation_tips"] == ""


def test_v2_adapter_never_opens_an_unresolved_or_low_confidence_rule():
    request = _request()
    raw = json.dumps({
        "corrected_text": "She goes to school every day.",
        "errors": [{
            "category": "grammar", "concept_code": "taxonomy.00000000000000000000", "confidence": 0.20,
            "original": "go", "corrected": "goes", "explanation": "Use goes with she.",
        }],
        "summary": "One grammar correction.",
    })
    assert server.parse_v2_correction_response(raw, request)["error_analysis"][0]["rule_id"] == ""


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
