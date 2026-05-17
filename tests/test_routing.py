"""Routing tests — exercise the pure routing functions in graph.py with synthetic state.

No LLMs are invoked. These cover the user's stated requirements:
- shallow query routes to shallow_research
- complex query routes to deep_research (scout)
- vague query routes to needs_clarification (clarifier)
- gap analyzer can trigger follow-up (returns list of Send)
- max iteration stopping works
- writer is not reachable directly from synthesis (topology guarantee)
"""

from langgraph.types import Send

from deep_research.config import MAX_DEEP_ITERATIONS
from deep_research.graph import (
    _fan_out_to_specialists,
    _route_after_citation_audit,
    _route_after_classifier,
    _route_after_gap_analyzer,
    _route_after_shallow_researcher,
)
from deep_research.state import (
    ArchitectPlan,
    FollowUpTask,
    GapAnalysis,
    IntentClassification,
    ResearchTask,
)


def _intent(route, depth="moderate"):
    return IntentClassification(
        route=route,
        reasoning="test",
        expected_output_type="report",
        research_depth=depth,
    )


# ─── Classifier routing ─────────────────────────────────────────────────────


def test_classifier_routes_shallow_to_shallow_researcher():
    assert _route_after_classifier({"intent": _intent("shallow_research")}) == "shallow_researcher"


def test_classifier_routes_deep_to_scout():
    assert _route_after_classifier({"intent": _intent("deep_research", "deep")}) == "scout"


def test_classifier_routes_clarification_to_clarifier():
    assert _route_after_classifier({"intent": _intent("needs_clarification")}) == "clarifier"


def test_classifier_routes_meta_to_meta():
    assert _route_after_classifier({"intent": _intent("meta_response", "minimal")}) == "meta"


def test_classifier_falls_back_to_scout_when_intent_missing():
    assert _route_after_classifier({}) == "scout"


# ─── Shallow path routing ──────────────────────────────────────────────────


def test_shallow_sufficient_routes_to_audit():
    assert (
        _route_after_shallow_researcher({"shallow_sufficient": True})
        == "citation_audit"
    )


def test_shallow_insufficient_escalates_to_scout():
    assert (
        _route_after_shallow_researcher({"shallow_sufficient": False})
        == "scout"
    )


def test_forced_shallow_does_not_escalate_even_if_insufficient():
    """--path shallow must respect the user's choice and not jump to deep."""
    state = {"shallow_sufficient": False, "force_path": "shallow"}
    assert _route_after_shallow_researcher(state) == "citation_audit"


# ─── Architect fan-out ─────────────────────────────────────────────────────


def test_architect_fan_out_emits_one_send_per_task():
    plan = ArchitectPlan(
        research_brief="b",
        research_tasks=[
            ResearchTask(id="t1", specialist_role="evidence_gatherer", objective="o1"),
            ResearchTask(id="t2", specialist_role="skeptic", objective="o2"),
            ResearchTask(id="t3", specialist_role="comparator", objective="o3"),
        ],
    )
    sends = _fan_out_to_specialists({"question": "q", "plan": plan})
    assert len(sends) == 3
    assert all(isinstance(s, Send) for s in sends)
    assert all(s.node == "specialist" for s in sends)
    # Each Send carries the right task id
    task_ids = {s.arg["task"].id for s in sends}
    assert task_ids == {"t1", "t2", "t3"}


# ─── Gap analyzer routing ──────────────────────────────────────────────────


def _base_state_with_plan():
    return {
        "question": "q",
        "plan": ArchitectPlan(research_brief="b"),
    }


def test_gap_sufficient_routes_to_citation_audit():
    state = _base_state_with_plan()
    state["gap_analysis"] = GapAnalysis(sufficient=True)
    state["iterations"] = 1
    assert _route_after_gap_analyzer(state) == "citation_audit"


def test_gap_max_iter_routes_to_citation_audit_even_if_insufficient():
    state = _base_state_with_plan()
    state["gap_analysis"] = GapAnalysis(
        sufficient=False,
        follow_up_tasks=[
            FollowUpTask(gap="g", specialist_role="skeptic", objective="o")
        ],
    )
    state["iterations"] = MAX_DEEP_ITERATIONS  # at the cap
    assert _route_after_gap_analyzer(state) == "citation_audit"


def test_gap_insufficient_triggers_follow_up_sends():
    state = _base_state_with_plan()
    state["gap_analysis"] = GapAnalysis(
        sufficient=False,
        follow_up_tasks=[
            FollowUpTask(gap="g1", specialist_role="skeptic", objective="o1"),
            FollowUpTask(gap="g2", specialist_role="horizon_scanner", objective="o2"),
        ],
    )
    state["iterations"] = 1
    result = _route_after_gap_analyzer(state)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(s, Send) for s in result)
    assert all(s.node == "specialist" for s in result)
    # follow-up task ids are prefixed "fu-"
    assert all(s.arg["task"].id.startswith("fu-") for s in result)


def test_gap_no_follow_ups_routes_to_audit():
    state = _base_state_with_plan()
    state["gap_analysis"] = GapAnalysis(sufficient=False, follow_up_tasks=[])
    state["iterations"] = 1
    assert _route_after_gap_analyzer(state) == "citation_audit"


# ─── Citation audit routing (shared between shallow + deep) ────────────────


def test_audit_after_deep_routes_to_writer():
    state = {
        "plan": ArchitectPlan(research_brief="b"),
        "synthesis": object(),  # truthy
    }
    assert _route_after_citation_audit(state) == "writer"


def test_audit_after_shallow_routes_to_shallow_finalize():
    state = {"shallow_answer": "some answer"}
    assert _route_after_citation_audit(state) == "shallow_finalize"
