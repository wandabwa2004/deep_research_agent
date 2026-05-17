"""Writer node — composes the draft report from audited evidence.

Inputs: question, plan, synthesis brief, citation_audit (the source registry).
Output: `draft_report` markdown. The refiner picks this up next.

The writer is *explicitly* told to cite only ids present in the audited
registry — this is the safety property that keeps fabricated sources out.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.config import model_for
from deep_research.prompts import WRITER_SYSTEM
from deep_research.state import ResearchState


def writer_node(state: ResearchState) -> dict[str, Any]:
    """Produce the draft report. Must run AFTER citation_audit."""
    plan = state.get("plan")
    synthesis = state.get("synthesis")
    audit = state.get("citation_audit")

    if not (plan and synthesis and audit):
        return {
            "draft_report": "(writer skipped — missing plan, synthesis, or citation audit)",
            "warnings": ["writer: required upstream state missing"],
        }

    model = model_for("writer", temperature=0.3)

    registry_blob = "\n".join(
        f"- [{s.id}] {s.title or '(untitled)'} — {s.url} "
        f"(type={s.source_type}, quality={s.quality})"
        for s in audit.sources
    ) or "(no sources)"

    flagged_blob = (
        "\n".join(f"- {issue}" for issue in audit.flagged_issues)
        if audit.flagged_issues
        else "(none)"
    )

    user = (
        f"Research question:\n{state['question']}\n\n"
        f"Research brief:\n{plan.research_brief}\n\n"
        f"Report outline:\n" + "\n".join(f"- {s}" for s in plan.report_outline) + "\n\n"
        f"Synthesis brief (JSON):\n{synthesis.model_dump_json(indent=2)}\n\n"
        f"Audited source registry — CITE ONLY THESE IDS:\n{registry_blob}\n\n"
        f"Citation audit flagged issues (acknowledge in Limitations):\n{flagged_blob}\n\n"
        "Write the full report now."
    )

    msg = model.invoke([SystemMessage(content=WRITER_SYSTEM), HumanMessage(content=user)])
    return {"draft_report": msg.content}
