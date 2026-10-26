"""Security defaults of the HTTP API.

The service used to ship ``allow_origins=["*"]`` together with
``allow_credentials=True``. That pair is rejected by browsers, and the wildcard
lets any page read a locally running instance. These tests pin the replacement.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import artificial_memory.api.server as server
from artificial_memory.api.server import (
    _cors_allow_credentials,
    _cors_origins,
    cors_configuration,
    credentials_valid,
    is_public_path,
)


def test_default_cors_is_loopback_only() -> None:
    origins, credentials = cors_configuration("")

    assert "*" not in origins
    assert credentials is True
    assert all(
        "localhost" in origin or "127.0.0.1" in origin for origin in origins
    )


def test_wildcard_origin_disables_credentials() -> None:
    # A wildcard with credentials is invalid per the CORS spec; the pair must
    # never be emitted in that state.
    assert cors_configuration("*") == (["*"], False)


def test_explicit_origin_list_is_parsed_and_trimmed() -> None:
    origins, credentials = cors_configuration(
        "https://app.example.com, https://admin.example.com ,"
    )

    assert origins == ["https://app.example.com", "https://admin.example.com"]
    assert credentials is True


def test_middleware_configuration_is_coherent() -> None:
    assert not (_cors_allow_credentials and "*" in _cors_origins)
    assert _cors_origins  # never an empty allow-list


def test_public_paths_are_exempt_from_auth() -> None:
    for path in ("/health", "/auth/login", "/docs", "/redoc", "/openapi.json", "/static/app.js"):
        assert is_public_path(path), path

    for path in ("/memory", "/recall", "/context", "/ws"):
        assert not is_public_path(path), path


def test_credentials_accept_session_and_api_key() -> None:
    service = MagicMock()
    service.validate_session.side_effect = lambda token: object() if token == "session" else None
    service.validate_api_key.side_effect = lambda key: object() if key == "key" else None

    assert credentials_valid(service, "Bearer session", None)
    assert credentials_valid(service, None, "key")
    # API keys are accepted in the bearer slot too (matches /auth/me).
    assert credentials_valid(service, "Bearer key", None)


def test_credentials_reject_missing_or_invalid_values() -> None:
    service = MagicMock()
    service.validate_session.return_value = None
    service.validate_api_key.return_value = None

    assert not credentials_valid(service, None, None)
    assert not credentials_valid(service, "Bearer wrong", None)
    assert not credentials_valid(service, "Basic abc", None)
    assert not credentials_valid(service, "Bearer   ", None)
    assert not credentials_valid(service, None, "wrong")


def _fake_state(enable_auth: bool) -> SimpleNamespace:
    service = MagicMock()
    service.validate_session.side_effect = lambda token: object() if token == "good" else None
    service.validate_api_key.side_effect = lambda key: object() if key == "key" else None
    return SimpleNamespace(
        runtime=SimpleNamespace(config=SimpleNamespace(enable_auth=enable_auth)),
        auth_service=service,
    )


def test_middleware_rejects_requests_without_credentials(monkeypatch) -> None:
    monkeypatch.setattr(server, "app_state", _fake_state(enable_auth=True))
    client = TestClient(server.app)

    response = client.get("/memory", params={"topic_path": "x"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid credentials"


def test_middleware_accepts_a_valid_session(monkeypatch) -> None:
    monkeypatch.setattr(server, "app_state", _fake_state(enable_auth=True))
    client = TestClient(server.app)

    # The route itself does not exist; a 404 proves the request got past auth.
    response = client.get(
        "/definitely-not-a-route", headers={"Authorization": "Bearer good"}
    )

    assert response.status_code == 404


def test_middleware_is_pass_through_when_auth_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(server, "app_state", _fake_state(enable_auth=False))
    client = TestClient(server.app)

    # Missing query parameters -> 422 from the route, not 401 from auth.
    response = client.get("/memory")

    assert response.status_code == 422
