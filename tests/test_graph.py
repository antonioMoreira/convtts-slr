import pytest

from convtts_slr.graph import Graph


class _FixedNode:
    def __init__(self, id_: str, outcome: str = "ok"):
        self.id = id_
        self.outcome = outcome

    def run(self, ctx) -> str:
        return self.outcome


def test_add_and_edge_return_the_same_graph_for_chaining():
    g = Graph()
    assert g.add(_FixedNode("a")) is g
    assert g.edge("a", "END") is g


def test_edge_rejects_an_unknown_source_or_destination_node():
    g = Graph().add(_FixedNode("a"))
    with pytest.raises(ValueError):
        g.edge("missing", "a")
    with pytest.raises(ValueError):
        g.edge("a", "missing")


def test_run_raises_when_no_edge_matches_the_outcome():
    g = Graph().add(_FixedNode("a", outcome="weird")).edge("a", "END", on="ok")
    with pytest.raises(RuntimeError, match="no edge"):
        g.run("a", ctx=None)


def test_run_raises_when_max_steps_is_exceeded():
    g = Graph().add(_FixedNode("a")).add(_FixedNode("b"))
    g.edge("a", "b").edge("b", "a")  # an infinite loop by construction
    with pytest.raises(RuntimeError, match="max_steps"):
        g.run("a", ctx=None, max_steps=3)


def test_run_invokes_on_step_for_every_transition():
    g = Graph().add(_FixedNode("a")).edge("a", "END")
    seen = []
    trace = g.run("a", ctx=None, on_step=lambda node, outcome: seen.append((node, outcome)))
    assert trace == [("a", "ok")]
    assert seen == [("a", "ok")]


def test_to_mermaid_labels_only_non_ok_outcomes():
    g = Graph().add(_FixedNode("a")).add(_FixedNode("b"))
    g.edge("a", "b", on="ok").edge("b", "END", on="custom")
    mermaid = g.to_mermaid()
    assert "a --> b" in mermaid
    assert "b -->|custom| END" in mermaid
