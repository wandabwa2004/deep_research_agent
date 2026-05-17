"""Specialist researcher — one node, role chosen per branch.

Architecture:
1. A ReAct agent runs the search loop with a role-flavored system prompt.
2. A separate structured-output call converts the agent's final text into a
   SpecialistNote. This two-step pattern is much more reliable than asking a
   tool-calling agent to output JSON directly.
3. On parse failure we retry once with a stricter prompt; if both attempts
   fail, we emit a fallback Note carrying the raw text and append a warning.

The ReAct agent is built once per role and cached, so parallel Send() branches
reuse the same compiled subgraph.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from deep_research.config import SPECIALIST_RECURSION_LIMIT, model_for
from deep_research.prompts import (
    SPECIALIST_BASE_SYSTEM,
    SPECIALIST_OUTPUT_FORMAT,
    SPECIALIST_ROLE_TAILS,
)
from deep_research.state import Citation, SpecialistNote, SpecialistState
from deep_research.tools import make_extract_tool, make_search_tool


# Internal schema for the structured-output extractor. Mirrors SpecialistNote
# but without the bookkeeping fields the node fills in itself (task_id, role).
class _NoteSchema(BaseModel):
    facts: list[str] = Field(default_factory=list)
    interpretation: str = ""
    uncertainty: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


# Cache: role → compiled ReAct agent.
_AGENT_CACHE: dict[str, Any] = {}


def _agent_for_role(role: str):
    """Build (and cache) a ReAct agent with the role-flavored system prompt."""
    if role not in _AGENT_CACHE:
        role_tail = SPECIALIST_ROLE_TAILS.get(role, "")
        system = "\n\n".join(
            [SPECIALIST_BASE_SYSTEM, role_tail, SPECIALIST_OUTPUT_FORMAT]
        )
        _AGENT_CACHE[role] = create_react_agent(
            model=model_for("specialist"),
            tools=[make_search_tool(max_results=5), make_extract_tool()],
            prompt=system,
        )
    return _AGENT_CACHE[role]


def _format_to_note_schema(raw_text: str, max_attempts: int = 2) -> tuple[_NoteSchema | None, str | None]:
    """Use a structured-output model to parse the specialist's raw answer.

    Returns (parsed, None) on success, or (None, error_message) after retries.
    """
    formatter = model_for("specialist").with_structured_output(_NoteSchema)
    last_err: Exception | None = None

    for attempt in range(max_attempts):
        try:
            strict_hint = (
                ""
                if attempt == 0
                else "\n\nPrevious attempt failed to parse. Be strict: output a single "
                "JSON object only, no prose, no code fences."
            )
            parsed: _NoteSchema = formatter.invoke(
                [
                    SystemMessage(
                        content=(
                            "Convert the following specialist research notes into "
                            "the structured schema. Preserve URLs verbatim. Do not "
                            "invent facts or citations." + strict_hint
                        )
                    ),
                    HumanMessage(content=raw_text),
                ]
            )
            return parsed, None
        except Exception as e:  # noqa: BLE001
            last_err = e

    return None, f"specialist structured-parse failed after {max_attempts} attempts: {last_err}"


def specialist_node(state: SpecialistState) -> dict[str, Any]:
    """Run one specialist on its assigned task. Dispatched via Send()."""
    task = state["task"]
    agent = _agent_for_role(task.specialist_role)

    user = (
        f"Top-level research question (for context):\n{state['question']}\n\n"
        f"Research brief:\n{state['research_brief']}\n\n"
        f"YOUR TASK ({task.id}, role={task.specialist_role}):\n{task.objective}\n\n"
        + (
            "Required evidence:\n"
            + "\n".join(f"- {e}" for e in task.required_evidence)
            + "\n\n"
            if task.required_evidence
            else ""
        )
        + "Search, then output the final JSON object as instructed."
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content=user)]},
        config={"recursion_limit": SPECIALIST_RECURSION_LIMIT},
    )
    raw = result["messages"][-1].content

    parsed, err = _format_to_note_schema(raw)

    if parsed is not None:
        note = SpecialistNote(
            task_id=task.id,
            specialist_role=task.specialist_role,
            facts=parsed.facts,
            interpretation=parsed.interpretation,
            uncertainty=parsed.uncertainty,
            citations=parsed.citations,
        )
        return {"specialist_notes": [note]}

    # Graceful fallback — preserve the raw text so downstream agents and humans
    # can still see what the specialist found.
    fallback = SpecialistNote(
        task_id=task.id,
        specialist_role=task.specialist_role,
        facts=[],
        interpretation=raw[:4000] if isinstance(raw, str) else "",
        uncertainty=["structured parsing failed; see raw_text"],
        citations=[],
        raw_text=raw if isinstance(raw, str) else str(raw),
    )
    return {
        "specialist_notes": [fallback],
        "warnings": [f"specialist[{task.id}/{task.specialist_role}]: {err}"],
    }
