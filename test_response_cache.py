"""Regression tests for the encrypted shared response cache."""

import os
import sqlite3
import tempfile

from cryptography.fernet import Fernet

import speech_correction_server as server


def _cache(path: str, *, hmac_key: bytes = b"cache-test-hmac-key"):
    return server.ResponseCache(
        path,
        encryption_key=Fernet.generate_key(),
        hmac_key=hmac_key,
        max_entries=2,
    )


def test_cache_round_trip_is_encrypted_and_keys_are_hmacs():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "cache.db")
        cache = _cache(path)
        private_text = "I need to discuss my private medical appointment."
        key = cache.make_key(private_text, "en", "B2", "formal", "ru", "")
        value = {"original_text": private_text, "corrected_text": "Private result."}
        cache.put(key, value)

        assert cache.get(key) == value
        with open(path, "rb") as database:
            raw = database.read()
        assert private_text.encode("utf-8") not in raw
        assert key.encode("utf-8") not in raw


def test_analysis_key_includes_model_prompt_and_schema_versions():
    with tempfile.TemporaryDirectory() as directory:
        cache = _cache(os.path.join(directory, "cache.db"))
        original = (
            server.DEEPSEEK_MODEL,
            server.ANALYSIS_PROMPT_VERSION,
            server.ANALYSIS_SCHEMA_VERSION,
        )
        try:
            first = cache.make_key("same text", "en", "B2", "formal", "ru", "")
            server.ANALYSIS_PROMPT_VERSION = "v2"
            second = cache.make_key("same text", "en", "B2", "formal", "ru", "")
            assert first != second
        finally:
            (server.DEEPSEEK_MODEL,
             server.ANALYSIS_PROMPT_VERSION,
             server.ANALYSIS_SCHEMA_VERSION) = original


def test_cache_is_disabled_without_both_runtime_secrets():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "cache.db")
        cache = server.ResponseCache(path, encryption_key=Fernet.generate_key())
        assert not cache.enabled
        assert cache.make_key("text", "en", "B2", "formal", "ru", "") == ""
        cache.put("key", {"value": "private"})
        assert cache.get("key") is None
        assert not os.path.exists(path)


def test_encrypted_cache_migration_purges_plaintext_rows_and_is_bounded():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "cache.db")
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "CREATE TABLE cache (key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO cache VALUES ('legacy', 'contains private text', '2026-01-01T00:00:00')"
            )
            connection.commit()
        finally:
            connection.close()
        cache = _cache(path)
        connection = sqlite3.connect(path)
        try:
            assert connection.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0
        finally:
            connection.close()
        for index in range(3):
            cache.put(f"rule::{index}", {"result": index})
        connection = sqlite3.connect(path)
        try:
            assert connection.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 2
        finally:
            connection.close()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
