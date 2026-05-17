"""LangGraph wiring.

Topology
--------
                                 ┌─── meta ────────────┐
                                 │                     │
                                 ├── clarifier ────────┤
START → classifier ─ROUTE ─────┤                     ├──→ END
                                 │  shallow_researcher │
                                 │    │ (sufficient)   │
                                 │    │                │
                                 │    ▼                │
                                 │  citation_audit ──→ shallow_finalize ─→ END
                                 │    │ (insufficient)
                                 │    ▼ escalate
                                 └──→ scout
                                       │
                                       ▼
                                  architect ─Send()─▶ specialist × N
                                       ▲                    │
                                       │                    ▼
                                       │              synthesis
                                       │                    │
                                       │                    ▼
                                       │              gap_analyzer
                                       │                    │
                            (Send follow-ups)        ┌──────┴──────┐
                                       └─ specialist ◄ insufficient   sufficient OR max iter
                                                                              │
                                                                              ▼
                                                                       citation_audit
                                                                              │
                                                                              ▼
                                                                          writer
                                                                              │
                                                                              ▼
                                                                          refiner ─→ END

Key safety properties enforced here:
- writer is unreachable except via citation_audit (no edge synthesis → writer)
- gap_analyzer can only loop while iterations < MAX_DEEP_ITERATIONS
- shallow path escalates to scout if shallow_researcher self-reports insufficient
"""

from __future__ import annotations

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send

from deep_research.agents import (
    architect_node,
    citation_audit_node,
    clarifier_node,
    classifier_node,
    gap_analyzer_node,
    meta_node,
    refiner_node,
    scout_node,
    shallow_finalize_node,
    shallow_researcher_node,
    specialist_node,
    synthesis_node,
    writer_node,
)
from deep_research.config import MAX_DEEP_ITERATIONS
from deep_research.state import ResearchState, ResearchTask


# ─── Routing functions ──────────────────────────────────────────────────────


def _route_after_classifier(state: ResearchState) -> str:
    intent = state.get("intent")
    if intent is None:
        return "scout"  # defensive: missing classifier output → assume deep
    if intent.route == "meta_response":
        return "meta"
    if intent.route == "needs_clarification":
        return "clarifier"
    if intent.route == "shallow_research":
        return "shallow_researcher"
    return "scout"  # deep_research


def _route_after_shallow_researcher(state: ResearchState) -> str:
    """If the shallow researcher could not answer, escalate to the deep path.

    Exception: when `--path shallow` forced this route, respect the user's
    choice and finalize regardless of the sufficiency self-judgement."""
    if state.get("force_path") == "shallow":
        return "citation_audit"
    if state.get("shallow_sufficient", True):
        return "citation_audit"
    return "scout"


def _route_after_citation_audit(state: ResearchState) -> str:
    """Citation audit is shared by both paths; route by which path we're on.

    If the shallow path put us here, finalize the shallow answer. Otherwise the
    deep path proceeds to the writer."""
    if state.get("shallow_answer") and not state.get("draft_report"):
        # We arrived here from the shallow path (deep path always has plan/synthesis
        # by this point; shallow path does not).
        if not state.get("plan"):
            return "shallow_finalize"
    return "writer"


def _fan_out_to_specialists(state: ResearchState):
    """Dispatch one Send() per research_task in the architect's plan."""
    plan = state["plan"]
    return [
        Send(
            "specialist",
            {
                "question": state["question"],
                "research_brief": plan.research_brief,
                "task": task,
            },
        )
        for task in plan.research_tasks
    ]


def _route_after_gap_analyzer(state: ResearchState):
    """Either finish (→ citation_audit → writer → refiner) or dispatch follow-ups.

    Returns a string for the "finish" path, or a list[Send] for the loop. The
    iteration cap is enforced here even if the model says sufficient=False."""
    gap = state.get("gap_analysis")
    iterations = state.get("iterations", 1)

    if gap is None or gap.sufficient or iterations >= MAX_DEEP_ITERATIONS:
        return "citation_audit"

    if not gap.follow_up_tasks:
        return "citation_audit"  # nothing actionable → finish

    plan = state["plan"]
    return [
        Send(
            "specialist",
            {
                "question": state["question"],
                "research_brief": plan.research_brief,
                "task": ResearchTask(
                    id=f"fu-{iterations}-{i}",
                    specialist_role=fu.specialist_role,
                    objective=fu.objective,
                    required_evidence=[fu.gap],
                ),
            },
        )
        for i, fu in enumerate(gap.follow_up_tasks)
    ]


# ─── Graph assembly ─────────────────────────────────────────────────────────


def build_graph():
    g = StateGraph(ResearchState)

    # Top-level routing
    g.add_node("classifier", classifier_node)
    g.add_node("meta", meta_node)
    g.add_node("clarifier", clarifier_node)

    # Shallow path
    g.add_node("shallow_researcher", shallow_researcher_node)
    g.add_node("shallow_finalize", shallow_finalize_node)

    # Deep path
    g.add_node("scout", scout_node)
    g.add_node("architect", architect_node)
    g.add_node("specialist", specialist_node)
    g.add_node("synthesis", synthesis_node)
    g.add_node("gap_analyzer", gap_analyzer_node)

    # Shared finish nodes
    g.add_node("citation_audit", citation_audit_node)
    g.add_node("writer", writer_node)
    g.add_node("refiner", refiner_node)

    # Edges
    g.add_edge(START, "classifier")
    g.add_conditional_edges(
        "classifier",
        _route_after_classifier,
        ["meta", "clarifier", "shallow_researcher", "scout"],
    )

    g.add_edge("meta", END)
    g.add_edge("clarifier", END)

    g.add_conditional_edges(
        "shallow_researcher",
        _route_after_shallow_researcher,
        ["citation_audit", "scout"],
    )

    # Deep path: scout → architect → fan-out → synthesis → gap_analyzer
    g.add_edge("scout", "architect")
    g.add_conditional_edges(
        "architect", _fan_out_to_specialists, ["specialist"]
    )
    g.add_edge("specialist", "synthesis")
    g.add_edge("synthesis", "gap_analyzer")

    # Gap analyzer either dispatches more Sends or routes to citation_audit
    g.add_conditional_edges(
        "gap_analyzer",
        _route_after_gap_analyzer,
        ["specialist", "citation_audit"],
    )

    # Citation audit fans out to either path's finalizer
    g.add_conditional_edges(
        "citation_audit",
        _route_after_citation_audit,
        ["writer", "shallow_finalize"],
    )

    # Deep path finishes
    g.add_edge("writer", "refiner")
    g.add_edge("refiner", END)
    g.add_edge("shallow_finalize", END)

    return g.compile()


# Pre-built instance for LangGraph Studio (langgraph.json points here).
graph = build_graph()
