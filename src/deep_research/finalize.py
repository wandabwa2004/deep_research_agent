"""Build a best-effort final report from partial graph state.

Used by the CLI when a run is interrupted (Ctrl-C) or hits a wall-clock budget.
Walks downstream from whatever stage the partial state stopped at and runs the
remaining nodes synchronously to produce a final_report.

Branch logic
------------
final_report present     →  return as-is
draft_report present     →  refine
specialist_notes present →  (bootstrap plan if missing) synthesize → audit → write → refine
shallow_answer present   →  audit (if missing) → shallow_finalize
nothing useful           →  short placeholder explaining what was/wasn't done
"""

from __future__ import annotations

from typing import Any

from deep_research.agents.citation_audit import citation_audit_node
from deep_research.agents.refiner import refiner_node
from deep_research.agents.shallow import shallow_finalize_node
from deep_research.agents.synthesis import synthesis_node
from deep_research.agents.writer import writer_node
from deep_research.state import ArchitectPlan


def _apply(state: dict, update: dict | None) -> None:
    """Merge a node's return value into the running state.

    Finalize never fan-outs, so reducer logic isn't needed here — straight
    overwrite is correct."""
    if update:
        state.update(update)


def finalize_from_partial(state: dict[str, Any]) -> str:
    """Produce a final report from whatever evidence is available.

    Mutates `state` (adds intermediate fields). Returns the report string.
    Safe to call on completed state too — it short-circuits if final_report
    is already present.
    """
    # 1. Already finished — nothing to do.
    if state.get("final_report"):
        return state["final_report"]

    # 2. Writer already ran — just refine the draft.
    if state.get("draft_report"):
        _apply(state, refiner_node(state))
        return state.get("final_report") or state["draft_report"]

    # 3. Deep-path partial: at least one specialist completed.
    if state.get("specialist_notes"):
        if not state.get("plan"):
            # Bootstrap a minimal plan so synthesis has an outline to map to.
            state["plan"] = ArchitectPlan(
                research_brief=state.get("question", ""),
                report_outline=["Findings", "Limitations"],
            )
        if not state.get("synthesis"):
            _apply(state, synthesis_node(state))
        if not state.get("citation_audit"):
            _apply(state, citation_audit_node(state))
        _apply(state, writer_node(state))
        _apply(state, refiner_node(state))
        return (
            state.get("final_report")
            or state.get("draft_report")
            or "(finalize: writer produced no output)"
        )

    # 4. Shallow-path partial.
    if state.get("shallow_answer"):
        if not state.get("citation_audit"):
            _apply(state, citation_audit_node(state))
        _apply(state, shallow_finalize_node(state))
        return state.get("final_report") or state.get("shallow_answer", "")

    # 5. Nothing actionable was collected before exit.
    parts = ["# Interrupted — no evidence collected\n"]
    parts.append(f"**Question:** {state.get('question', '(unknown)')}")
    if state.get("intent"):
        intent = state["intent"]
        parts.append(
            f"**Classified as:** {intent.route} (depth={intent.research_depth})"
        )
    if state.get("scout_findings"):
        sf = state["scout_findings"]
        parts.append(
            f"**Scout completed:** {len(sf.findings)} findings, "
            f"{len(sf.key_entities)} entities"
        )
    if state.get("plan"):
        plan = state["plan"]
        parts.append(f"**Plan built:** {len(plan.research_tasks)} tasks queued")
    parts.append("\nNot enough evidence collected to produce a report.")
    return "\n\n".join(parts)
