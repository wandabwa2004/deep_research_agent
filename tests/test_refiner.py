"""Test that refiner reads draft_report and produces final_report.

Uses monkeypatch to replace model_for with a fake model so the test runs
without any API keys."""

from langchain_core.messages import AIMessage


class _FakeModel:
    """Minimal stand-in for a LangChain chat model — only `invoke` is used."""

    def invoke(self, messages):
        # Echo the draft back with a marker so we can assert it flowed through.
        last = messages[-1].content
        return AIMessage(content=f"[refined]\n{last}")


def test_refiner_promotes_draft_to_final(monkeypatch):
    from deep_research.agents import refiner

    monkeypatch.setattr(refiner, "model_for", lambda *a, **kw: _FakeModel())

    out = refiner.refiner_node({"draft_report": "DRAFT BODY"})
    assert "final_report" in out
    assert out["final_report"].startswith("[refined]")
    assert "DRAFT BODY" in out["final_report"]


def test_refiner_handles_missing_draft(monkeypatch):
    from deep_research.agents import refiner

    monkeypatch.setattr(refiner, "model_for", lambda *a, **kw: _FakeModel())

    out = refiner.refiner_node({})
    assert "final_report" in out
    assert "skipped" in out["final_report"].lower() or "no draft" in out["final_report"].lower()
    assert "warnings" in out
