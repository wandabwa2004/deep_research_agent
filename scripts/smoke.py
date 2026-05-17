"""End-to-end smoke test (requires API keys).

Runs the full graph against three canned questions to exercise each top-level
route. Not part of the pytest suite — invoke explicitly:

    python scripts/smoke.py
"""

from __future__ import annotations

from dotenv import load_dotenv

from deep_research.graph import build_graph

SAMPLES = [
    ("META", "What can this system do?"),
    ("SHALLOW", "What year was the LangGraph library first released?"),
    ("DEEP", "Compare LangGraph, AutoGen and CrewAI for building multi-agent research systems in 2026."),
]


def main() -> None:
    load_dotenv()
    graph = build_graph()
    for label, question in SAMPLES:
        print(f"\n{'=' * 60}\n{label}: {question}\n{'=' * 60}")
        result = graph.invoke({"question": question})
        print(result.get("final_report", "(no final report produced)"))


if __name__ == "__main__":
    main()
