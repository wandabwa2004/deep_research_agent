"""Synthesis node — merges specialist notes into a structured brief.

Cheaper model (gpt-4.1-mini by default) since this is high-volume compression
with no creative judgement required. Strictly preserves all source URLs.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.config import model_for
from deep_research.prompts import SYNTHESIS_SYSTEM
from deep_research.state import ResearchState, SynthesisBrief


def synthesis_node(state: ResearchState) -> dict[str, Any]:
    """Consolidate specialist notes into a structured brief.

    Always runs after every fan-out wave (initial architect plan + each
    follow-up cycle), so it sees the full accumulated note list each time."""
    notes = state.get("specialist_notes", [])
    plan = state.get("plan")
    if not notes or not plan:
        return {"warnings": ["synthesis: no notes or no plan; skipping"]}

    model = model_for("synthesis").with_structured_output(SynthesisBrief)

    notes_blob = "\n\n".join(
        f"### Task {n.task_id} ({n.specialist_role})\n"
        f"FACTS:\n" + "\n".join(f"- {f}" for f in n.facts) + "\n"
        f"INTERPRETATION: {n.interpretation}\n"
        f"UNCERTAINTY:\n" + "\n".join(f"- {u}" for u in n.uncertainty) + "\n"
        f"CITATIONS:\n" + "\n".join(f"- {c.url}" for c in n.citations)
        for n in notes
    )

    outline_blob = "\n".join(f"- {s}" for s in plan.report_outline)
    user = (
        f"Research question:\n{state['question']}\n\n"
        f"Research brief:\n{plan.research_brief}\n\n"
        f"Report outline:\n{outline_blob}\n\n"
        f"Specialist notes:\n{notes_blob}\n\n"
        "Produce the structured synthesis brief. Map evidence to outline sections."
    )

    try:
        brief: SynthesisBrief = model.invoke(
            [SystemMessage(content=SYNTHESIS_SYSTEM), HumanMessage(content=user)]
        )
        return {"synthesis": brief}
    except Exception as e:  # noqa: BLE001
        # Fall back to a minimal brief built from the notes directly so the gap
        # analyzer still has something to work with.
        sources = list(
            dict.fromkeys(c.url for n in notes for c in n.citations)
        )
        fallback = SynthesisBrief(
            key_findings=[n.interpretation for n in notes if n.interpretation],
            source_list=sources,
        )
        return {
            "synthesis": fallback,
            "warnings": [f"synthesis: structured-parse failed ({e}); used fallback brief"],
        }
