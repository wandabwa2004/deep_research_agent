"""Citation manager / source audit.

Runs after synthesis is complete (deep path) OR after the shallow researcher
(shallow path). Produces a CitationAudit containing:
  - deduplicated source registry (S1, S2, ...)
  - claim → source-id mappings
  - flagged issues (weak/unsupported/mismatched citations)

The writer is instructed to cite ONLY ids from this registry.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.config import model_for
from deep_research.prompts import CITATION_AUDIT_SYSTEM
from deep_research.state import (
    Citation,
    CitationAudit,
    ResearchState,
    SourceRecord,
)


def _gather_citations(state: ResearchState) -> list[Citation]:
    """Collect every citation from either path (deep specialist notes or shallow)."""
    out: list[Citation] = []
    for note in state.get("specialist_notes", []) or []:
        out.extend(note.citations)
    out.extend(state.get("shallow_citations", []) or [])
    return out


def _dedupe_to_registry(cites: list[Citation]) -> list[SourceRecord]:
    """Group citations by URL, build a stable S1, S2, ... registry."""
    seen: dict[str, SourceRecord] = {}
    for i, c in enumerate(cites):
        if not c.url or c.url in seen:
            continue
        seen[c.url] = SourceRecord(id=f"S{len(seen) + 1}", url=c.url, title=c.title)
    return list(seen.values())


def citation_audit_node(state: ResearchState) -> dict[str, Any]:
    """Build the audited source registry that the writer will be told to obey."""
    cites = _gather_citations(state)
    if not cites:
        # Empty audit so the writer/finalizer can still proceed (e.g. meta route
        # ends up here in edge cases — we want a coherent path).
        return {
            "citation_audit": CitationAudit(flagged_issues=["no citations gathered"]),
            "warnings": ["citation_audit: no citations to audit"],
        }

    registry = _dedupe_to_registry(cites)

    # Ask the model to classify source quality and (best-effort) map claims to
    # source ids. We pre-build the registry so the model never invents source ids.
    notes_blob = "\n\n".join(
        f"### {n.task_id} ({n.specialist_role})\n"
        + "\n".join(f"- {f}" for f in n.facts)
        + (f"\nINTERPRETATION: {n.interpretation}" if n.interpretation else "")
        for n in state.get("specialist_notes", []) or []
    )
    if not notes_blob:
        notes_blob = state.get("shallow_answer", "") or "(no notes)"

    registry_blob = "\n".join(
        f"- {s.id}: {s.url}" + (f" ({s.title})" if s.title else "")
        for s in registry
    )

    model = model_for("citation_audit").with_structured_output(CitationAudit)
    user = (
        f"Research question:\n{state['question']}\n\n"
        f"Pre-built source registry (use these ids exactly, do not invent new ones):\n"
        f"{registry_blob}\n\n"
        f"Evidence / specialist notes:\n{notes_blob}\n\n"
        "Audit: classify each source, map significant claims to source ids, flag "
        "any weak/unsupported/mismatched citations. Return ONLY ids from the "
        "registry above."
    )

    try:
        audited: CitationAudit = model.invoke(
            [SystemMessage(content=CITATION_AUDIT_SYSTEM), HumanMessage(content=user)]
        )
    except Exception as e:  # noqa: BLE001
        return {
            "citation_audit": CitationAudit(
                sources=registry,
                flagged_issues=[f"audit parse failed: {e}"],
            ),
            "warnings": [f"citation_audit: parse failed ({e}); using raw registry only"],
        }

    # Defensive merge: if the model dropped any registry entries, restore them.
    audited_urls = {s.url for s in audited.sources}
    for s in registry:
        if s.url not in audited_urls:
            audited.sources.append(s)

    # And remove any source ids that aren't in the final registry from claim mappings.
    valid_ids = {s.id for s in audited.sources}
    for cm in audited.claim_source_mappings:
        cm.source_ids = [sid for sid in cm.source_ids if sid in valid_ids]

    return {"citation_audit": audited}
