
from types import SimpleNamespace

import pytest

from app import llm


def test_temporary_gemini_error_uses_groq(monkeypatch):
    calls = []

    def fake_gemini(*args, **kwargs):
        raise TimeoutError("Gemini request timed out")

    def fake_groq(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"ok": true}')

    monkeypatch.setattr(
        llm,
        "_generate_with_gemini",
        fake_gemini,
    )

    monkeypatch.setattr(
        llm,
        "_generate_with_groq",
        fake_groq,
    )

    response = llm._generate_response(
        "system prompt",
        "user message",
        json_mode=True,
    )

    assert response.text == '{"ok": true}'
    assert len(calls) == 1
    assert calls[0]["json_mode"] is True


def test_permanent_gemini_error_does_not_use_groq(monkeypatch):
    calls = []

    def fake_gemini(*args, **kwargs):
        raise RuntimeError("API key invalid")

    def fake_groq(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(text="Groq response")

    monkeypatch.setattr(
        llm,
        "_generate_with_gemini",
        fake_gemini,
    )

    monkeypatch.setattr(
        llm,
        "_generate_with_groq",
        fake_groq,
    )

    with pytest.raises(RuntimeError, match="API key invalid"):
        llm._generate_response(
            "system prompt",
            "user message",
        )

    assert len(calls) == 0
