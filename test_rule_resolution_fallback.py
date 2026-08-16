"""The v1 fallback for mapping a correction to its rule.

These exist because the link from an error to its lesson was dead in production:
the v2 migration removed text-based resolution while v2 itself is still off by
default, so nothing supplied a concept code and every user landed in the full
rule list instead of the rule they asked for.
"""
import asyncio
import unittest
from unittest.mock import patch

import rules_store
import speech_correction_server as srv


def _body(**kw):
    return srv.ResolveRuleRequest(learning=kw.pop("learning", "de"), **kw)


class ConceptSelectionPromptTests(unittest.TestCase):
    def test_prompt_lists_only_active_taxonomy_codes(self):
        topics = rules_store.topics_with_concepts("de")
        prompt = rules_store.build_concept_selection_prompt(
            "German", topics, "grammar", "Ich habe gegangen",
            "Ich bin gegangen", "Motion verbs take sein.")
        for topic in topics:
            self.assertIn(topic["concept_code"], prompt)
        # Inactive topics (pronunciation) must not be offered at all.
        hidden = rules_store.inactive_rule_ids("de")
        self.assertTrue(hidden, "expected some inactive topics to exist")
        for rule_id in hidden:
            self.assertNotIn(
                rules_store.topic_concept_code("de", rule_id), prompt)

    def test_prompt_allows_answering_none(self):
        topics = rules_store.topics_with_concepts("de")
        prompt = rules_store.build_concept_selection_prompt(
            "German", topics, "", "a", "b", "c")
        self.assertIn("null", prompt)


class FallbackResolutionTests(unittest.TestCase):
    def setUp(self):
        srv.response_cache.enabled = False  # exercise the model path

    def test_returns_rule_id_when_model_picks_a_valid_code(self):
        code = rules_store.topic_concept_code("de", "kasus")
        with patch.object(srv, "_deepseek_client", object()), \
             patch.object(srv, "_call_deepseek",
                          return_value='{"concept_code": "%s"}' % code):
            got = asyncio.run(srv._resolve_rule_from_text(
                "de", _body(type="grammar", original="in der Schule",
                            corrected="in die Schule", explanation="direction")))
        self.assertEqual(got, "kasus")

    def test_invented_code_resolves_to_nothing(self):
        with patch.object(srv, "_deepseek_client", object()), \
             patch.object(srv, "_call_deepseek",
                          return_value='{"concept_code": "taxonomy.deadbeef"}'):
            got = asyncio.run(srv._resolve_rule_from_text(
                "de", _body(type="grammar", original="x", corrected="y",
                            explanation="z")))
        self.assertIsNone(got, "a code outside the taxonomy must not resolve")

    def test_model_may_answer_none(self):
        with patch.object(srv, "_deepseek_client", object()), \
             patch.object(srv, "_call_deepseek",
                          return_value='{"concept_code": null}'):
            got = asyncio.run(srv._resolve_rule_from_text(
                "de", _body(type="other", original="x", corrected="y",
                            explanation="z")))
        self.assertIsNone(got)

    def test_model_failure_is_not_fatal(self):
        def boom(*a, **k):
            raise RuntimeError("upstream down")
        with patch.object(srv, "_deepseek_client", object()), \
             patch.object(srv, "_call_deepseek", side_effect=boom):
            got = asyncio.run(srv._resolve_rule_from_text(
                "de", _body(type="grammar", original="x", corrected="y",
                            explanation="z")))
        self.assertIsNone(got, "a resolver outage must not break the sheet")

    def test_empty_error_never_calls_the_model(self):
        called = []
        with patch.object(srv, "_deepseek_client", object()), \
             patch.object(srv, "_call_deepseek",
                          side_effect=lambda *a, **k: called.append(1) or "{}"):
            got = asyncio.run(srv._resolve_rule_from_text("de", _body()))
        self.assertIsNone(got)
        self.assertEqual(called, [], "no text means nothing to resolve")


if __name__ == "__main__":
    unittest.main()
