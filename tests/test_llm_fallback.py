
from types import SimpleNamespace

import pytest

from app import llm


def test_temporary_gemini_error_uses_claude(monkeypatch):
    calls = []

    def fake_gemini(*args, **kwargs):
        raise TimeoutError("Gemini request timed out")

    def fake_claude(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"ok": true}')

    monkeypatch.setattr(llm, "_generate_with_gemini", fake_gemini)
    monkeypatch.setattr(llm, "_generate_with_claude", fake_claude)

    response = llm._generate_response(
        "system prompt",
        "user message",
        json_mode=True,
    )

    assert response.text == '{"ok": true}'
    assert len(calls) == 1
    assert calls[0]["json_mode"] is True


def test_permanent_gemini_error_does_not_use_claude(monkeypatch):
    calls = []

    def fake_gemini(*args, **kwargs):
        raise RuntimeError("API key invalid")

    def fake_claude(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(text="Claude response")

    monkeypatch.setattr(llm, "_generate_with_gemini", fake_gemini)
    monkeypatch.setattr(llm, "_generate_with_claude", fake_claude)

    with pytest.raises(RuntimeError, match="API key invalid"):
        llm._generate_response("system prompt", "user message")

    assert len(calls) == 0
