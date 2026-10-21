"""TernoLLMClient — provisioner errors surface as structured ProvisionerError."""

from __future__ import annotations

import urllib.error

import pytest

from terno_agent.core.exceptions import LLMError
from terno_agent.core.messages import UserMessage
from terno_agent.llm.terno_llm import ProvisionerError, TernoLLMClient

CREDITS_EXHAUSTED = {
    "status": "error",
    "error_code": "credits_exhausted",
    "message": "Insufficient credits on Sachin Sharma's account.",
    "details": {"account_name": "Sachin Sharma"},
}


@pytest.fixture
def client():
    return TernoLLMClient(api_key="k", model="m", provisioner_url="http://prov")


def _complete(client):
    return client.complete([UserMessage("hi")])


def test_non_retryable_stream_error_raises_without_fallback(client, monkeypatch):
    monkeypatch.setattr(client, "_iter_provisioner_stream", lambda _p: iter([CREDITS_EXHAUSTED]))

    def _no_fallback(_payload):
        raise AssertionError("credits_exhausted must not retry via get_response")

    monkeypatch.setattr(client, "_call_provisioner", _no_fallback)

    with pytest.raises(ProvisionerError) as info:
        _complete(client)

    exc = info.value
    assert isinstance(exc, LLMError)
    assert exc.error_code == "credits_exhausted"
    assert exc.details == {"account_name": "Sachin Sharma"}
    assert str(exc) == "Insufficient credits on Sachin Sharma's account."


def test_retryable_stream_error_falls_back_then_raises_structured(client, monkeypatch):
    error = {"status": "error", "error_code": "provisioner_error", "message": "Something went wrong."}
    monkeypatch.setattr(client, "_iter_provisioner_stream", lambda _p: iter([error]))
    calls = []

    def _fallback(payload):
        calls.append(payload["type"])
        return error

    monkeypatch.setattr(client, "_call_provisioner", _fallback)

    with pytest.raises(ProvisionerError) as info:
        _complete(client)

    assert calls == ["get_response"]
    assert info.value.error_code == "provisioner_error"
    assert str(info.value) == "Something went wrong."
    assert info.value.details == {}


def test_error_after_tokens_streamed_raises_structured(client, monkeypatch):
    chunks = [{"status": "streaming", "content": "partial"},
              {"status": "error", "error_code": "provisioner_error", "message": "Boom"}]
    monkeypatch.setattr(client, "_iter_provisioner_stream", lambda _p: iter(chunks))

    with pytest.raises(ProvisionerError) as info:
        _complete(client)

    assert info.value.error_code == "provisioner_error"


def test_unreachable_provisioner_gives_plain_message(client, monkeypatch):
    def _unreachable(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _unreachable)

    with pytest.raises(ProvisionerError) as info:
        _complete(client)

    assert info.value.error_code == "network_error"
    assert "http://prov" not in str(info.value)
    assert info.value.details == {"reason": "connection refused"}


def test_from_response_defaults_for_missing_fields():
    exc = ProvisionerError.from_response({"status": "error", "details": "not-a-dict"})
    assert exc.error_code == "unknown_error"
    assert str(exc) == "Something went wrong. Please try again."
    assert exc.details == {}
