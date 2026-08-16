"""Tests for privacy-safe aggregate LLM usage telemetry."""

import os
import sqlite3
import tempfile

import speech_correction_server as server


def _rows(path: str):
    connection = sqlite3.connect(path)
    try:
        return connection.execute(
            "SELECT feature, model, model_calls, cache_hits, prompt_tokens, "
            "completion_tokens, latency_ms, estimated_cost_microdollars "
            "FROM llm_usage_daily"
        ).fetchall()
    finally:
        connection.close()


def test_usage_meter_aggregates_model_calls_and_cache_hits_without_text():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "usage.db")
        meter = server.LLMUsageMeter(
            path, input_price_per_mtok=2.0, output_price_per_mtok=4.0)
        meter.record_completion(
            feature="analysis", model="test-model", prompt_tokens=120,
            completion_tokens=30, latency_seconds=0.123)
        meter.record_cache_hit(feature="analysis", model="test-model")

        assert _rows(path) == [
            ("analysis", "test-model", 1, 1, 120, 30, 123, 360)
        ]
        with open(path, "rb") as database:
            assert b"private learner sentence" not in database.read()


def test_usage_meter_treats_missing_usage_as_zero_and_rejects_negative_prices():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "usage.db")
        meter = server.LLMUsageMeter(path)
        meter.record_completion(
            feature="analysis_stream", model="test-model", prompt_tokens=None,
            completion_tokens=-1, latency_seconds=-1)
        assert _rows(path) == [
            ("analysis_stream", "test-model", 1, 0, 0, 0, 0, 0)
        ]
        try:
            server.LLMUsageMeter(path, input_price_per_mtok=-1)
            assert False, "negative prices must be rejected"
        except ValueError:
            pass


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
