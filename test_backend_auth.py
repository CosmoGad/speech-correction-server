"""Unit tests for the Firebase/legacy authentication migration.

Run directly with `python test_backend_auth.py`. These tests do not contact
Firebase: token validation is replaced with a deterministic fake so the
authorization boundary stays testable offline.
"""

import asyncio

from fastapi import HTTPException
from starlette.requests import Request

import speech_correction_server as server


def _request(headers=None, client_host="198.51.100.1"):
    raw_headers = [
        (key.lower().encode(), value.encode())
        for key, value in (headers or {}).items()
    ]
    return Request({
        "type": "http",
        "method": "POST",
        "headers": raw_headers,
        "client": (client_host, 12345),
    })


def test_verified_firebase_token_uses_uid_as_principal():
    original_ready = server._firebase_ready
    original_verify = server.firebase_auth.verify_id_token
    try:
        server._firebase_ready = True
        server.firebase_auth.verify_id_token = lambda token, check_revoked: {"uid": "user-123"}
        client = asyncio.run(server.verify_client(
            _request({"Authorization": "Bearer signed-token"}), None))
        assert client.principal_id == "uid:user-123"
        assert client.auth_scheme == "firebase"
        assert client.app_check_status == "missing"
    finally:
        server._firebase_ready = original_ready
        server.firebase_auth.verify_id_token = original_verify


def test_invalid_firebase_token_is_rejected():
    original_ready = server._firebase_ready
    original_verify = server.firebase_auth.verify_id_token
    try:
        server._firebase_ready = True

        def reject(*_args, **_kwargs):
            raise ValueError("bad token")

        server.firebase_auth.verify_id_token = reject
        try:
            asyncio.run(server.verify_client(
                _request({"Authorization": "Bearer invalid"}), None))
            assert False, "expected 401"
        except HTTPException as error:
            assert error.status_code == 401
    finally:
        server._firebase_ready = original_ready
        server.firebase_auth.verify_id_token = original_verify


def test_legacy_key_remains_ip_scoped_during_migration():
    original_key = server._server_api_key
    original_allowed = server._allow_legacy_api_key
    try:
        server._server_api_key = "legacy-key"
        server._allow_legacy_api_key = True
        client = asyncio.run(server.verify_client(
            _request({"X-API-Key": "legacy-key"}, "203.0.113.10"), "legacy-key"))
        assert client.principal_id == "legacy:203.0.113.10"
        assert client.auth_scheme == "legacy"
    finally:
        server._server_api_key = original_key
        server._allow_legacy_api_key = original_allowed


def test_requests_without_auth_are_rejected():
    original_key = server._server_api_key
    original_allowed = server._allow_legacy_api_key
    try:
        server._server_api_key = "legacy-key"
        server._allow_legacy_api_key = False
        try:
            asyncio.run(server.verify_client(_request(), None))
            assert False, "expected 401"
        except HTTPException as error:
            assert error.status_code == 401
    finally:
        server._server_api_key = original_key
        server._allow_legacy_api_key = original_allowed


def test_invalid_app_check_token_is_observed_before_enforcement():
    original_ready = server._firebase_ready
    original_enforced = server._app_check_enforced
    original_verify = server.firebase_app_check.verify_token
    try:
        server._firebase_ready = True
        server._app_check_enforced = False

        def reject(*_args, **_kwargs):
            raise ValueError("bad App Check token")

        server.firebase_app_check.verify_token = reject
        assert server._verify_app_check(
            _request({"X-Firebase-AppCheck": "invalid"})) == "invalid"
    finally:
        server._firebase_ready = original_ready
        server._app_check_enforced = original_enforced
        server.firebase_app_check.verify_token = original_verify


def test_app_check_is_rejected_when_enforcement_is_enabled():
    original_enforced = server._app_check_enforced
    try:
        server._app_check_enforced = True
        try:
            server._verify_app_check(_request())
            assert False, "expected 401"
        except HTTPException as error:
            assert error.status_code == 401
    finally:
        server._app_check_enforced = original_enforced


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
