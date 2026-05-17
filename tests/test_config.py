"""Tests for provider auto-detection and per-role model resolution.

These use monkeypatch on env vars so they don't depend on the developer's
real environment. No LLMs are invoked — model_for() returns a model object,
which only validates the API key at invoke time."""

import pytest


def _clear_provider_env(monkeypatch):
    """Wipe every env var that could influence provider selection."""
    for var in ("LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# ─── _detect_provider ──────────────────────────────────────────────────────


def test_explicit_openai_wins_even_when_only_anthropic_key_present(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    from deep_research.config import _detect_provider
    assert _detect_provider() == "openai"


def test_explicit_anthropic_wins_even_when_only_openai_key_present(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    from deep_research.config import _detect_provider
    assert _detect_provider() == "anthropic"


def test_auto_detects_openai_when_only_openai_key_set(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    from deep_research.config import _detect_provider
    assert _detect_provider() == "openai"


def test_auto_detects_anthropic_when_only_anthropic_key_set(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    from deep_research.config import _detect_provider
    assert _detect_provider() == "anthropic"


def test_auto_prefers_openai_when_both_keys_set(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    from deep_research.config import _detect_provider
    assert _detect_provider() == "openai"


def test_raises_when_no_provider_key_set(monkeypatch):
    _clear_provider_env(monkeypatch)
    from deep_research.config import _detect_provider
    with pytest.raises(ValueError, match="No LLM provider key"):
        _detect_provider()


def test_invalid_llm_provider_value_raises(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "cohere")
    from deep_research.config import _detect_provider
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        _detect_provider()


# ─── model_for: per-role override + default selection ──────────────────────


def test_per_role_override_to_anthropic_when_both_keys_present(monkeypatch):
    """User can mix providers by overriding a single role — needs both keys."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    monkeypatch.setenv("WRITER_MODEL", "claude-opus-4-7")
    from deep_research.config import model_for
    m = model_for("writer")
    assert m.__class__.__name__ == "ChatAnthropic"


def test_per_role_override_to_openai_when_both_keys_present(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    monkeypatch.setenv("WRITER_MODEL", "gpt-4.1")
    from deep_research.config import model_for
    m = model_for("writer")
    assert m.__class__.__name__ == "ChatOpenAI"


def test_default_resolves_to_openai_when_only_openai_key(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.delenv("WRITER_MODEL", raising=False)
    from deep_research.config import model_for
    m = model_for("writer")
    assert m.__class__.__name__ == "ChatOpenAI"


def test_default_resolves_to_anthropic_when_only_anthropic_key(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("WRITER_MODEL", raising=False)
    from deep_research.config import model_for
    m = model_for("writer")
    assert m.__class__.__name__ == "ChatAnthropic"


def test_unknown_model_prefix_raises(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("WRITER_MODEL", "mistral-large")
    from deep_research.config import model_for
    with pytest.raises(ValueError, match="Unknown provider"):
        model_for("writer")
