"""Refiner — polishes the draft without changing evidence.

The refiner's prompt is constrained: no new facts, no new or removed citations,
no changed conclusions. It exists purely for editorial cleanup.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.config import model_for
from deep_research.prompts import REFINER_SYSTEM
from deep_research.state import ResearchState


def refiner_node(state: ResearchState) -> dict[str, Any]:
    """Refine the draft report into the final report."""
    draft = state.get("draft_report")
    if not draft:
        return {
            "final_report": "(refiner skipped — no draft to refine)",
            "warnings": ["refiner: no draft_report in state"],
        }

    model = model_for("refiner", temperature=0.2)
    msg = model.invoke(
        [
            SystemMessage(content=REFINER_SYSTEM),
            HumanMessage(content=f"Refine this draft:\n\n{draft}"),
        ]
    )
    return {"final_report": msg.content}
