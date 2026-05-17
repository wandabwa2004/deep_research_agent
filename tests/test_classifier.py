"""Tests for the classifier short-circuit used by --path.

If `intent` is already in state when classifier_node runs, the model must NOT
be invoked. We verify this by patching model_for to raise — any call would fail
the test loudly."""

import pytest

from deep_research.state import IntentClassification


def _exploding_model_for(*args, **kwargs):
    raise AssertionError("classifier model should not be called when intent is preset")


def test_classifier_skips_llm_when_intent_preset(monkeypatch):
    from deep_research.agents import classifier

    monkeypatch.setattr(classifier, "model_for", _exploding_model_for)

    preset = IntentClassification(
        route="deep_research",
        reasoning="forced",
        expected_output_type="report",
        research_depth="deep",
    )
    out = classifier.classifier_node({"question": "q", "intent": preset})
    # Returns empty dict — nothing to merge, intent stays as-is via state.
    assert out == {}


def test_format_progress_handles_none_update():
    """When classifier short-circuits, LangGraph streams update=None.
    The CLI's progress formatter must not crash on that."""
    from deep_research.main import _format_progress

    assert _format_progress("classifier", None) is None
    assert _format_progress("scout", None) is None
    assert _format_progress("specialist", {}) is None


def test_classifier_calls_model_when_no_intent(monkeypatch):
    """Sanity check: without a preset intent, model_for IS called."""
    from deep_research.agents import classifier
    from langchain_core.messages import AIMessage

    called = {"yes": False}

    class _StructuredOK:
        def invoke(self, _messages):
            return IntentClassification(
                route="shallow_research",
                reasoning="test",
                expected_output_type="factual_answer",
                research_depth="shallow",
            )

    class _FakeModel:
        def with_structured_output(self, _schema):
            return _StructuredOK()

        def invoke(self, _messages):
            return AIMessage(content="never reached in this path")

    def _fake_model_for(*a, **kw):
        called["yes"] = True
        return _FakeModel()

    monkeypatch.setattr(classifier, "model_for", _fake_model_for)

    out = classifier.classifier_node({"question": "q"})
    assert called["yes"] is True
    assert "intent" in out
    assert out["intent"].route == "shallow_research"
