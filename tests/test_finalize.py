"""Tests for finalize_from_partial — the early-exit report builder.

We monkeypatch `model_for` in each node module so no LLM calls happen.
"""

from langchain_core.messages import AIMessage

from deep_research import finalize as F
from deep_research.state import (
    ArchitectPlan,
    Citation,
    CitationAudit,
    IntentClassification,
    ResearchLandscape,
    ResearchTask,
    SpecialistNote,
    SynthesisBrief,
)


# ─── Fake model that satisfies both .invoke() and .with_structured_output() ──


class _FakeStructured:
    """Returns a default-constructed instance of the requested schema."""

    def __init__(self, schema):
        self.schema = schema

    def invoke(self, _messages):
        return self.schema()


class _FakeModel:
    """Stand-in for any langchain chat model."""

    def __init__(self, text: str = "fake"):
        self.text = text

    def invoke(self, _messages):
        return AIMessage(content=self.text)

    def with_structured_output(self, schema):
        return _FakeStructured(schema)


def _stub_all_models(monkeypatch, text: str = "fake"):
    """Replace `model_for` in every node module finalize touches."""
    from deep_research.agents import citation_audit, refiner, synthesis, writer

    fake = lambda *a, **kw: _FakeModel(text)
    monkeypatch.setattr(citation_audit, "model_for", fake)
    monkeypatch.setattr(refiner, "model_for", fake)
    monkeypatch.setattr(synthesis, "model_for", fake)
    monkeypatch.setattr(writer, "model_for", fake)


# ─── Branch 1: already-finished state ──────────────────────────────────────


def test_returns_existing_final_report_unchanged():
    assert F.finalize_from_partial({"final_report": "DONE"}) == "DONE"


# ─── Branch 2: draft present, only needs refining ──────────────────────────


def test_refines_draft_when_present(monkeypatch):
    _stub_all_models(monkeypatch, text="[refined] body")
    state = {"draft_report": "DRAFT BODY"}
    result = F.finalize_from_partial(state)
    assert "refined" in result


# ─── Branch 3: deep-path partial (specialist notes collected) ──────────────


def test_deep_partial_runs_synthesis_audit_writer_refiner(monkeypatch):
    _stub_all_models(monkeypatch, text="DRAFT REPORT")
    state = {
        "question": "Q",
        "plan": ArchitectPlan(
            research_brief="b",
            report_outline=["S1"],
            research_tasks=[
                ResearchTask(id="t1", specialist_role="evidence_gatherer", objective="o")
            ],
        ),
        "specialist_notes": [
            SpecialistNote(
                task_id="t1",
                specialist_role="evidence_gatherer",
                facts=["fact"],
                citations=[Citation(url="https://example.com")],
            )
        ],
    }
    result = F.finalize_from_partial(state)
    assert result
    # The pipeline ran end-to-end: synthesis, audit, writer, refiner all populated.
    assert "synthesis" in state
    assert "citation_audit" in state
    assert "draft_report" in state
    assert "final_report" in state


def test_deep_partial_bootstraps_plan_if_missing(monkeypatch):
    _stub_all_models(monkeypatch, text="REPORT")
    state = {
        "question": "Q",
        "specialist_notes": [
            SpecialistNote(
                task_id="t1",
                specialist_role="skeptic",
                facts=[],
                citations=[Citation(url="https://x")],
            )
        ],
    }
    F.finalize_from_partial(state)
    # A minimal ArchitectPlan was injected so synthesis could run.
    assert "plan" in state
    assert state["plan"].research_brief == "Q"


# ─── Branch 4: shallow-path partial ────────────────────────────────────────


def test_shallow_partial_runs_audit_then_finalize(monkeypatch):
    _stub_all_models(monkeypatch)
    state = {
        "question": "Q",
        "shallow_answer": "The answer is X. https://x.com",
        "shallow_citations": [Citation(url="https://x.com")],
    }
    result = F.finalize_from_partial(state)
    assert "The answer is X" in result
    assert "citation_audit" in state
    assert "final_report" in state


# ─── Branch 5: nothing collected ───────────────────────────────────────────


def test_no_evidence_returns_interrupted_placeholder():
    result = F.finalize_from_partial({"question": "What is X?"})
    assert "Interrupted" in result
    assert "What is X?" in result


def test_no_evidence_mentions_intent_if_present():
    intent = IntentClassification(
        route="deep_research",
        reasoning="r",
        expected_output_type="report",
        research_depth="deep",
    )
    result = F.finalize_from_partial({"question": "Q", "intent": intent})
    assert "deep_research" in result


def test_no_evidence_mentions_scout_progress_if_present():
    sf = ResearchLandscape(key_entities=["E1", "E2"])
    result = F.finalize_from_partial({"question": "Q", "scout_findings": sf})
    assert "Scout completed" in result
    assert "2 entities" in result


# ─── Idempotency / safe-replay ─────────────────────────────────────────────


def test_can_be_called_twice_safely(monkeypatch):
    _stub_all_models(monkeypatch, text="r")
    state = {"draft_report": "d"}
    first = F.finalize_from_partial(state)
    # Second call sees final_report present and short-circuits.
    second = F.finalize_from_partial(state)
    assert first == second
