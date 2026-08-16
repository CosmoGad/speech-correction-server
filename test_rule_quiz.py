"""Offline contract tests for native rule quizzes.

Run with ``python test_rule_quiz.py``.  DeepSeek is replaced by
deterministic fakes; these tests never use the network or consume API credit.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager

from fastapi import HTTPException
from starlette.requests import Request

import rules_store
import speech_correction_server as server


def _question(number: int) -> dict:
    return {
        "id": f"q{number}",
        "prompt": f"Выберите правильный вариант {number}.",
        "options": [
            f"English option {number}A",
            f"English option {number}B",
            f"English option {number}C",
            f"English option {number}D",
        ],
        "correct_index": (number - 1) % 4,
        "explanation": f"Объяснение ответа {number}.",
    }


def _quiz(question_count: int = 5) -> dict:
    return {"questions": [_question(number) for number in range(1, question_count + 1)]}


def _request() -> Request:
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/rule-quiz",
        "headers": [],
        "client": ("198.51.100.25", 12345),
    })


class _MemoryCache:
    def __init__(self):
        self.values = {}
        self.get_keys = []
        self.put_keys = []

    def get(self, key):
        self.get_keys.append(key)
        return self.values.get(key)

    def put(self, key, value):
        self.put_keys.append(key)
        self.values[key] = value


class _Limiter:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.principals = []

    def is_allowed(self, principal):
        self.principals.append(principal)
        return self.allowed


class _UsageMeter:
    def __init__(self):
        self.cache_hits = []

    def record_cache_hit(self, **kwargs):
        self.cache_hits.append(kwargs)


@contextmanager
def _server_fakes(*, responses, allowed=True,
                  generation_allowed=True, model_delay=0):
    original_cache = server.response_cache
    original_limiter = server.rate_limiter
    original_client = server._deepseek_client
    original_call = server._call_deepseek
    original_ip_limiter = server.quiz_ip_generation_limiter
    original_global_limiter = server.quiz_global_generation_limiter
    original_locks = server._quiz_generation_locks
    original_usage_meter = server.llm_usage_meter
    cache = _MemoryCache()
    limiter = _Limiter(allowed)
    calls = []
    pending = list(responses)

    async def fake_call(client, prompt, user_text, **kwargs):
        calls.append({
            "client": client,
            "prompt": prompt,
            "user_text": user_text,
            "kwargs": kwargs,
        })
        if model_delay:
            await asyncio.sleep(model_delay)
        return pending.pop(0)

    try:
        server.response_cache = cache
        server.rate_limiter = limiter
        server._deepseek_client = object()
        server._call_deepseek = fake_call
        server.quiz_ip_generation_limiter = _Limiter(generation_allowed)
        server.quiz_global_generation_limiter = _Limiter(generation_allowed)
        server._quiz_generation_locks = {}
        server.llm_usage_meter = _UsageMeter()
        yield cache, limiter, calls
    finally:
        server.response_cache = original_cache
        server.rate_limiter = original_limiter
        server._deepseek_client = original_client
        server._call_deepseek = original_call
        server.quiz_ip_generation_limiter = original_ip_limiter
        server.quiz_global_generation_limiter = original_global_limiter
        server._quiz_generation_locks = original_locks
        server.llm_usage_meter = original_usage_meter


async def _call_async(*, learning="en", interface="ru",
                      rule_id="articles", level="B1", question_count=5):
    client = server.AuthenticatedClient(
        principal_id="uid:quiz-user",
        auth_scheme="firebase",
        app_check_status="valid",
    )
    return await server.get_rule_quiz_endpoint(
        _request(), learning, interface, rule_id, level, question_count, client)


def _call(*, learning="en", interface="ru", rule_id="articles", level="B1",
          question_count=5):
    return asyncio.run(_call_async(
        learning=learning,
        interface=interface,
        rule_id=rule_id,
        level=level,
        question_count=question_count,
    ))


def _body(response):
    return json.loads(response.body.decode("utf-8"))


def test_exact_levels_are_preserved():
    for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
        assert rules_store.quiz_level(level) == level


def test_invalid_levels_are_rejected():
    for level in ("", "a1", "B3", "native", None, 2):
        try:
            rules_store.quiz_level(level)
            assert False, f"expected ValueError for {level!r}"
        except ValueError:
            pass


def test_quiz_validator_accepts_and_normalizes_valid_shape():
    result = rules_store.validate_rule_quiz(_quiz())
    assert set(result) == {"questions"}
    assert len(result["questions"]) == 5
    assert result["questions"][0]["correct_index"] == 0


def test_quiz_validator_accepts_supported_configurable_counts():
    assert len(rules_store.validate_rule_quiz(_quiz(3))["questions"]) == 3
    assert len(rules_store.validate_rule_quiz(_quiz(7))["questions"]) == 7
    assert len(rules_store.validate_rule_quiz(
        _quiz(7), expected_question_count=7)["questions"]) == 7


def test_quiz_validator_rejects_invalid_shapes():
    mutations = [
        lambda quiz: quiz.update(questions=quiz["questions"][:2]),
        lambda quiz: quiz["questions"][0].update(options=["a", "b", "c"]),
        lambda quiz: quiz["questions"][0].update(options=["same"] * 4),
        lambda quiz: quiz["questions"][0].update(options=["a", "b", "c", ""]),
        lambda quiz: quiz["questions"][0].update(correct_index=4),
        lambda quiz: quiz["questions"][0].update(correct_index=True),
        lambda quiz: quiz["questions"][1].update(id="q1"),
        lambda quiz: quiz["questions"][1].update(id="q9"),
        lambda quiz: quiz["questions"][0].update(id="bad id"),
        lambda quiz: quiz["questions"][0].update(prompt="x" * 501),
        lambda quiz: quiz["questions"][0].update(explanation="x" * 1001),
        lambda quiz: quiz["questions"][0].update(options=["x" * 301, "b", "c", "d"]),
    ]
    for mutate in mutations:
        quiz = _quiz()
        mutate(quiz)
        try:
            rules_store.validate_rule_quiz(quiz)
            assert False, "expected invalid quiz to be rejected"
        except ValueError:
            pass


def test_quiz_prompt_enforces_language_split_and_exact_contract():
    prompt = rules_store.build_rule_quiz_prompt(
        title="Articles",
        learning_name="English",
        interface_name="Russian",
        level="B2",
        question_count=7,
        rule_context={
            "explanation": "Когда использовать артикли.",
            "examples": [{"wrong": "I bought book.", "right": "I bought a book."}],
        },
    )
    assert "exactly 7" in prompt
    assert "exactly 4" in prompt
    assert "Russian" in prompt
    assert "English" in prompt
    assert "B2" in prompt
    assert "correct_index" in prompt
    assert "I bought a book." in prompt


def test_mexican_spanish_has_human_readable_prompt_name():
    interface_name = server.catalog_display_name(server.LANGUAGE_CATALOG, "es_MX")
    assert interface_name and interface_name != "es_MX"
    prompt = rules_store.build_rule_quiz_prompt(
        title="Articles",
        learning_name="English",
        interface_name=interface_name,
        level="B2",
        question_count=5,
        rule_context=None,
    )
    assert interface_name in prompt
    assert "es_MX" not in prompt


def test_every_configured_interface_has_human_readable_prompt_name():
    assert all(
        server.catalog_display_name(server.LANGUAGE_CATALOG, code) != code
        for code in server.INTERFACE_LANGUAGES
    )


def test_mexican_spanish_interface_is_supported_by_endpoint_prompt():
    raw = json.dumps(_quiz(), ensure_ascii=False)
    with _server_fakes(responses=[raw]) as (_cache, _limiter, calls):
        response = _call(interface="es_MX")

    assert response.status_code == 200
    assert "Español Mexicano" in calls[0]["prompt"]
    assert "es_MX" not in calls[0]["prompt"]


def test_endpoint_generates_valid_quiz_and_tags_model_call():
    with _server_fakes(responses=[json.dumps(_quiz(), ensure_ascii=False)]) as (cache, limiter, calls):
        response = _call()

    data = _body(response)
    assert response.status_code == 200
    assert data["rule_id"] == "articles"
    assert data["learning"] == "en"
    assert data["interface"] == "ru"
    assert data["level_band"] == "B1"
    assert len(data["questions"]) == 5
    assert len(calls) == 1
    assert calls[0]["kwargs"] == {
        "feature": "rule_quiz",
        "max_tokens": 1800,
    }
    assert limiter.principals == ["uid:quiz-user"]
    assert cache.put_keys == ["rule-quiz::v2::en::ru::articles::B1::5"]


def test_same_level_and_question_count_hits_cache_without_second_model_call():
    raw = json.dumps(_quiz(), ensure_ascii=False)
    with _server_fakes(responses=[raw]) as (cache, _limiter, calls):
        first = _call(level="B1")
        second = _call(level="B1")

    assert _body(first) == _body(second)
    assert len(calls) == 1
    assert cache.get_keys == [
        "rule-quiz::v2::en::ru::articles::B1::5",
        "rule-quiz::v2::en::ru::articles::B1::5",
        "rule-quiz::v2::en::ru::articles::B1::5",
    ]


def test_concurrent_cache_miss_uses_single_paid_generation():
    raw = json.dumps(_quiz(), ensure_ascii=False)
    with _server_fakes(responses=[raw], model_delay=0.01) as (cache, _limiter, calls):
        async def run_pair():
            return await asyncio.gather(_call_async(), _call_async())

        first, second = asyncio.run(run_pair())

    assert _body(first) == _body(second)
    assert len(calls) == 1
    assert len(cache.put_keys) == 1
    assert server._quiz_generation_locks == {}


def test_generation_quota_blocks_cache_miss_before_model_call():
    status_code = None
    with _server_fakes(
            responses=[], generation_allowed=False) as (cache, _limiter, calls):
        try:
            _call()
            assert False, "expected HTTPException"
        except HTTPException as error:
            status_code = error.status_code

    assert status_code == 429
    assert calls == []
    assert cache.put_keys == []


def test_invalid_json_is_retried_once_then_succeeds():
    valid = json.dumps(_quiz(), ensure_ascii=False)
    with _server_fakes(responses=["not-json", valid]) as (_cache, _limiter, calls):
        response = _call(level="A2")

    assert response.status_code == 200
    assert len(calls) == 2


def test_non_text_model_response_is_retried_once():
    valid = json.dumps(_quiz(), ensure_ascii=False)
    with _server_fakes(responses=[None, valid]) as (_cache, _limiter, calls):
        response = _call(level="A1")

    assert response.status_code == 200
    assert len(calls) == 2


def test_two_invalid_model_responses_return_502_without_cache_write():
    invalid_shape = json.dumps({"questions": [_question(1)]})
    status_code = None
    with _server_fakes(responses=["not-json", invalid_shape]) as (cache, _limiter, calls):
        try:
            _call(level="C1")
            assert False, "expected HTTPException"
        except HTTPException as error:
            status_code = error.status_code

    assert status_code == 502
    assert len(calls) == 2
    assert cache.put_keys == []


def test_invalid_parameters_never_call_model():
    cases = [
        {"learning": "../en"},
        {"interface": "r/u"},
        {"interface": "zz"},
        {"rule_id": "../articles"},
        {"rule_id": "not-in-taxonomy"},
        {"level": "B3"},
    ]
    for kwargs in cases:
        with _server_fakes(responses=[]) as (cache, _limiter, calls):
            try:
                _call(**kwargs)
                assert False, f"expected HTTPException for {kwargs!r}"
            except HTTPException as error:
                assert error.status_code in {400, 404}
        assert calls == []
        assert cache.get_keys == []


def test_learning_requires_config_and_taxonomy_before_cache_or_model():
    original_configs = server.LANGUAGE_CONFIGS
    try:
        server.LANGUAGE_CONFIGS = {**original_configs, "zz": {"name": "Test"}}
        # "zz" is now configured but deliberately has no topics_zz taxonomy.
        with _server_fakes(responses=[]) as (cache, _limiter, calls):
            try:
                _call(learning="zz")
                assert False, "expected unknown learning taxonomy to be rejected"
            except HTTPException as error:
                assert error.status_code == 404
        assert cache.get_keys == []
        assert calls == []
    finally:
        server.LANGUAGE_CONFIGS = original_configs


def test_configured_learning_without_real_taxonomy_is_rejected():
    assert "es_MX" in server.LANGUAGE_CONFIGS
    with _server_fakes(responses=[]) as (cache, _limiter, calls):
        try:
            _call(learning="es_MX")
            assert False, "expected missing taxonomy to be rejected"
        except HTTPException as error:
            assert error.status_code == 404
    assert cache.get_keys == []
    assert calls == []


def test_rate_limit_matches_rule_endpoint_behavior():
    status_code = None
    with _server_fakes(responses=[], allowed=False) as (cache, limiter, calls):
        try:
            _call()
            assert False, "expected HTTPException"
        except HTTPException as error:
            status_code = error.status_code

    assert status_code == 429
    assert limiter.principals == ["uid:quiz-user"]
    assert cache.get_keys == []
    assert calls == []


def test_route_is_get_and_requires_verify_client_dependency():
    route = next(route for route in server.app.routes if route.path == "/rule-quiz")
    assert route.methods == {"GET"}
    assert any(dependency.call is server.verify_client for dependency in route.dependant.dependencies)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
