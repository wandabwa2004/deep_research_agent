"""Gap analyzer — decides whether to loop (with targeted follow-ups) or proceed.

Output is a GapAnalysis. The graph's routing function then either:
  - dispatches Send() to specialists with the follow_up_tasks (loop), or
  - routes to citation_audit → writer → refiner (done).

The hard iteration cap is enforced in graph.py; this node only emits a
preference.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.config import MAX_DEEP_ITERATIONS, model_for
from deep_research.prompts import GAP_ANALYZER_SYSTEM
from deep_research.state import GapAnalysis, ResearchState


def gap_analyzer_node(state: ResearchState) -> dict[str, Any]:
    plan = state.get("plan")
    synthesis = state.get("synthesis")
    iterations = state.get("iterations", 1)

    if not plan or not synthesis:
        # Defensive: with no synthesis there's nothing to analyse — accept.
        return {
            "gap_analysis": GapAnalysis(sufficient=True, max_iterations_reached=False),
            "warnings": ["gap_analyzer: missing plan or synthesis; defaulting to sufficient=True"],
        }

    model = model_for("gap_analyzer").with_structured_output(GapAnalysis)

    user = (
        f"Original question:\n{state['question']}\n\n"
        f"Research brief:\n{plan.research_brief}\n\n"
        f"Report outline:\n" + "\n".join(f"- {s}" for s in plan.report_outline) + "\n\n"
        f"Quality constraints:\n" + "\n".join(f"- {c}" for c in plan.quality_constraints) + "\n\n"
        f"Synthesis brief (JSON):\n{synthesis.model_dump_json(indent=2)}\n\n"
        f"Current iteration: {iterations} of {MAX_DEEP_ITERATIONS}\n\n"
        "Decide whether the evidence is sufficient. If not, emit ONE follow-up "
        "task per gap, choosing the best specialist role for each."
    )

    try:
        analysis: GapAnalysis = model.invoke(
            [SystemMessage(content=GAP_ANALYZER_SYSTEM), HumanMessage(content=user)]
        )
    except Exception as e:  # noqa: BLE001
        # On parse failure, accept the answer rather than looping forever.
        return {
            "gap_analysis": GapAnalysis(sufficient=True),
            "warnings": [f"gap_analyzer: parse failed ({e}); accepting current evidence"],
        }

    # Surface the iteration cap on the analysis itself for downstream visibility.
    analysis.max_iterations_reached = iterations >= MAX_DEEP_ITERATIONS
    return {"gap_analysis": analysis}
