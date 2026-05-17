"""Architect node — builds the research plan from question + scout findings.

Replaces the old `planner` which decomposed the question too early. The
architect can now see what the scout discovered before assigning specialist
roles."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.config import model_for
from deep_research.prompts import ARCHITECT_SYSTEM
from deep_research.state import ArchitectPlan, ResearchState


def architect_node(state: ResearchState) -> dict[str, Any]:
    """Produce the structured plan and bump the iteration counter."""
    model = model_for("architect").with_structured_output(ArchitectPlan)
    scout = state.get("scout_findings")
    iterations = state.get("iterations", 0)

    scout_blob = (
        scout.model_dump_json(indent=2)
        if scout
        else "(no scout findings — proceed conservatively)"
    )

    plan: ArchitectPlan = model.invoke(
        [
            SystemMessage(content=ARCHITECT_SYSTEM),
            HumanMessage(
                content=(
                    f"User question:\n{state['question']}\n\n"
                    f"Scout findings:\n{scout_blob}\n\n"
                    "Produce the research plan."
                )
            ),
        ]
    )
    return {"plan": plan, "iterations": iterations + 1}
