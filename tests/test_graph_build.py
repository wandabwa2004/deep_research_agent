"""Graph compilation + topology tests.

These verify the structural guarantees we care about, without invoking any node:
- the graph compiles
- every expected node is present
- writer's only predecessor is citation_audit (so writer cannot run before audit)
- refiner's only predecessor is writer
- citation_audit is reachable from both the shallow and the deep path
"""

from deep_research.graph import build_graph


def _build():
    return build_graph()


def test_graph_compiles():
    g = _build()
    assert g is not None


def test_all_expected_nodes_present():
    g = _build()
    nodes = set(g.get_graph().nodes.keys())
    expected = {
        "classifier",
        "meta",
        "clarifier",
        "shallow_researcher",
        "shallow_finalize",
        "scout",
        "architect",
        "specialist",
        "synthesis",
        "gap_analyzer",
        "citation_audit",
        "writer",
        "refiner",
    }
    missing = expected - nodes
    assert not missing, f"missing nodes: {missing}"


def _predecessors(graph_obj, node_name: str) -> set[str]:
    """Names of nodes with an edge into `node_name`."""
    return {e.source for e in graph_obj.edges if e.target == node_name}


def test_writer_only_runs_after_citation_audit():
    """Safety property: there must be no edge synthesis → writer or
    gap_analyzer → writer. Writer's only in-edge is from citation_audit."""
    g = _build().get_graph()
    preds = _predecessors(g, "writer")
    assert preds == {"citation_audit"}, (
        f"writer should only be reachable from citation_audit, got {preds}"
    )


def test_refiner_only_runs_after_writer():
    g = _build().get_graph()
    preds = _predecessors(g, "refiner")
    assert preds == {"writer"}, f"refiner predecessors: {preds}"


def test_citation_audit_reachable_from_both_paths():
    g = _build().get_graph()
    preds = _predecessors(g, "citation_audit")
    # Reached conditionally from shallow_researcher (success) and gap_analyzer (done)
    assert "shallow_researcher" in preds
    assert "gap_analyzer" in preds


def test_specialist_reachable_from_both_architect_and_gap_analyzer():
    g = _build().get_graph()
    preds = _predecessors(g, "specialist")
    # Initial fan-out (architect) + follow-up fan-out (gap_analyzer)
    assert "architect" in preds
    assert "gap_analyzer" in preds
