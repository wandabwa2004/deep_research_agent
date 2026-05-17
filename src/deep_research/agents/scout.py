"""Scout node — broad exploratory pass that maps the research landscape.

The scout doesn't try to answer the question. It exists to give the architect
enough situational awareness to plan smart specialist tasks (instead of guessing
sub-questions from the question alone, which the original planner did)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from deep_research.config import model_for
from deep_research.prompts import SCOUT_SYSTEM
from deep_research.state import ResearchLandscape, ResearchState
from deep_research.tools import make_search_tool

_scout_agent = None


def _get_scout_agent():
    global _scout_agent
    if _scout_agent is None:
        _scout_agent = create_react_agent(
            model=model_for("scout"),
            tools=[make_search_tool(max_results=5)],
            prompt=SCOUT_SYSTEM,
        )
    return _scout_agent


def scout_node(state: ResearchState) -> dict[str, Any]:
    """Run the scout's ReAct loop, then extract a structured landscape."""
    agent = _get_scout_agent()
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        f"Topic to scout: {state['question']}\n\n"
                        "Run 3–5 broad exploratory searches, then summarise the landscape."
                    )
                )
            ]
        }
    )
    raw = result["messages"][-1].content

    # Two-step: free-text exploration first, then a separate structured-output
    # call that converts the exploration into a ResearchLandscape. This is more
    # reliable than asking the ReAct agent to output JSON directly.
    formatter = model_for("scout").with_structured_output(ResearchLandscape)
    try:
        landscape: ResearchLandscape = formatter.invoke(
            [
                HumanMessage(
                    content=(
                        "Convert the following scout notes into the structured "
                        "ResearchLandscape schema. Preserve URLs verbatim. Do not "
                        "invent entities or controversies.\n\n"
                        f"{raw}"
                    )
                )
            ]
        )
        return {"scout_findings": landscape}
    except Exception as e:
        return {
            "scout_findings": ResearchLandscape(),
            "warnings": [f"scout: structured-parse failed ({e}); landscape is empty"],
        }
