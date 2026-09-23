"""A deliberately small graph runtime.

Nodes are typed steps; edges are labelled with the outcome that selects them, so the
control flow (including the bounded snowball cycle) is explicit data you can print,
test and change, rather than being buried in a loop. Each transition is logged, which
together with the event store makes a crashed run resumable from its last node.

(pydantic-graph would also fit here; its API changed substantially across 1.x/2.x, so
this ~60-line runtime avoids pinning the review to one version of it.)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("convtts_slr")


class Node(Protocol):
    id: str

    def run(self, ctx: Any) -> str:
        """Do the work and return an outcome label used to pick the next edge."""


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[tuple[str, str], str] = field(default_factory=dict)  # (node, outcome) -> next node

    def add(self, node: Node) -> Graph:
        self.nodes[node.id] = node
        return self

    def edge(self, src: str, dst: str, on: str = "ok") -> Graph:
        if src not in self.nodes or (dst != "END" and dst not in self.nodes):
            raise ValueError(f"unknown node in edge {src} -> {dst}")
        self.edges[(src, on)] = dst
        return self

    def run(
        self,
        start: str,
        ctx: Any,
        max_steps: int = 100,
        stop_after: str | None = None,
        on_step: Callable[[str, str], None] | None = None,
    ) -> list[tuple[str, str]]:
        trace, current = [], start
        for _ in range(max_steps):
            if current == "END":
                return trace
            outcome = self.nodes[current].run(ctx)
            trace.append((current, outcome))
            if on_step:
                on_step(current, outcome)
            if current == stop_after:  # e.g. pause after screening to calibrate thresholds
                return trace
            nxt = self.edges.get((current, outcome))
            if nxt is None:
                raise RuntimeError(f"no edge from {current!r} on outcome {outcome!r}")
            log.info("%s --%s--> %s", current, outcome, nxt)
            current = nxt
        raise RuntimeError("max_steps exceeded: check the loop's stopping condition")

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for (src, on), dst in self.edges.items():
            label = "" if on == "ok" else f"|{on}|"
            lines.append(f"    {src} -->{label} {dst}")
        return "\n".join(lines)
