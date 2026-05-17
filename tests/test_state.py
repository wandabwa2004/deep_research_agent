"""Tests for state models and the list-concatenation reducer.

These tests don't hit any LLM; they exercise the Pydantic models and reducer.
"""

from deep_research.state import (
    ArchitectPlan,
    Citation,
    GapAnalysis,
    IntentClassification,
    ResearchTask,
    SourceRecord,
    SpecialistNote,
    SynthesisBrief,
    _add_lists,
)


def test_add_lists_handles_none_on_either_side():
    assert _add_lists(None, None) == []
    assert _add_lists(None, ["a"]) == ["a"]
    assert _add_lists(["a"], None) == ["a"]
    assert _add_lists(["a"], ["b"]) == ["a", "b"]


def test_specialist_note_accumulation_via_reducer():
    """Simulates two parallel branches each returning one note."""
    branch_a = [SpecialistNote(task_id="t1", specialist_role="evidence_gatherer")]
    branch_b = [SpecialistNote(task_id="t2", specialist_role="skeptic")]
    merged = _add_lists(branch_a, branch_b)
    assert len(merged) == 2
    assert {n.task_id for n in merged} == {"t1", "t2"}


def test_intent_classification_round_trip():
    ic = IntentClassification(
        route="deep_research",
        reasoning="multi-hop comparative question",
        expected_output_type="report",
        research_depth="deep",
    )
    assert ic.route == "deep_research"
    assert ic.clarification_question is None


def test_architect_plan_defaults():
    plan = ArchitectPlan(research_brief="x")
    assert plan.report_outline == []
    assert plan.research_tasks == []
    assert plan.specialist_roles_required == []


def test_research_task_validates_role():
    # specialist_role is a Literal — pydantic should reject unknown roles
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResearchTask(id="t1", specialist_role="not_a_role", objective="x")  # type: ignore[arg-type]


def test_gap_analysis_with_follow_ups():
    from deep_research.state import FollowUpTask

    g = GapAnalysis(
        sufficient=False,
        gaps=["no 2024 data"],
        follow_up_tasks=[
            FollowUpTask(
                gap="no 2024 data",
                specialist_role="horizon_scanner",
                objective="find 2024 adoption figures",
            )
        ],
    )
    assert not g.sufficient
    assert len(g.follow_up_tasks) == 1


def test_source_record_quality_literal():
    s = SourceRecord(id="S1", url="https://x", quality="high")
    assert s.quality == "high"


def test_synthesis_brief_preserves_sources():
    b = SynthesisBrief(
        key_findings=["k1"],
        source_list=["https://a", "https://b"],
    )
    assert b.source_list == ["https://a", "https://b"]


def test_citation_minimal():
    c = Citation(url="https://x")
    assert c.title == ""
    assert c.snippet == ""
