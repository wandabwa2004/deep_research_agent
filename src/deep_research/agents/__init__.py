"""Agent node functions, one module per role.

Re-exported here so graph.py imports stay tidy:
    from deep_research.agents import classifier_node, scout_node, ...
"""

from deep_research.agents.architect import architect_node
from deep_research.agents.citation_audit import citation_audit_node
from deep_research.agents.classifier import classifier_node, clarifier_node, meta_node
from deep_research.agents.gap_analyzer import gap_analyzer_node
from deep_research.agents.refiner import refiner_node
from deep_research.agents.scout import scout_node
from deep_research.agents.shallow import shallow_finalize_node, shallow_researcher_node
from deep_research.agents.specialist import specialist_node
from deep_research.agents.synthesis import synthesis_node
from deep_research.agents.writer import writer_node

__all__ = [
    "architect_node",
    "citation_audit_node",
    "clarifier_node",
    "classifier_node",
    "gap_analyzer_node",
    "meta_node",
    "refiner_node",
    "scout_node",
    "shallow_finalize_node",
    "shallow_researcher_node",
    "specialist_node",
    "synthesis_node",
    "writer_node",
]
