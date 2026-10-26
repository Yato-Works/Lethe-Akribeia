"""
Evidence Dependency Graph (EDG) for Lethe-Akribeia V4.

Represents dialogue turns and structured facts as nodes, with typed dependency
edges capturing temporal sequences, causal migrations, and entity coreference.
Provides bundled evidence aggregation for the Budgeted Compiler.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from artificial_memory.core.tokens import word_tokens_43


class EdgeType(StrEnum):
    TEMPORAL = "temporal"       # Sequential or calendar relation (A happened before B)
    CAUSAL = "causal"           # Direct consequence or migration (A caused/replaced B)
    ENTITY = "entity"           # Same entity / property reference across sessions
    COLLOCATION = "collocation" # Shared conversational context within same session


@dataclass(frozen=True)
class EvidenceNode:
    """A discrete unit of evidence (dialogue turn or structured memory fact)."""
    node_id: str                # Unique identifier (e.g., "S12:t4")
    session_id: str             # Parent session ID
    session_date: str           # ISO date string if available
    speaker: str                # "user" or "assistant"
    raw_text: str               # Verbatim text content
    tokens: int                 # Approximate token cost
    entities: frozenset[str]    # Extracted entity keys
    properties: frozenset[str]  # Extracted properties
    values: frozenset[str]      # Extracted values


@dataclass(frozen=True)
class DependencyEdge:
    """Directed dependency link between evidence nodes."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float               # Strength of dependency in [0.0, 1.0]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    """A tightly-coupled set of evidence nodes that must be selected together."""
    bundle_id: str
    primary_node_id: str
    member_nodes: list[EvidenceNode]
    total_tokens: int
    rationale: str              # e.g., "Temporal sequence for project migration"


class EvidenceDependencyGraph:
    """Graph structure organizing retrieved memories and their mutual dependencies."""

    def __init__(self) -> None:
        self.nodes: dict[str, EvidenceNode] = {}
        self.edges: list[DependencyEdge] = []
        self._adj: dict[str, list[DependencyEdge]] = {}
        self._rev_adj: dict[str, list[DependencyEdge]] = {}

    def add_node(self, node: EvidenceNode) -> None:
        self.nodes[node.node_id] = node
        self._adj.setdefault(node.node_id, [])
        self._rev_adj.setdefault(node.node_id, [])

    def add_edge(self, edge: DependencyEdge) -> None:
        self.edges.append(edge)
        self._adj.setdefault(edge.source_id, []).append(edge)
        self._rev_adj.setdefault(edge.target_id, []).append(edge)

    def get_dependencies(self, node_id: str) -> list[DependencyEdge]:
        """Outgoing edges: nodes that this node depends on."""
        return self._adj.get(node_id, [])

    def get_dependents(self, node_id: str) -> list[DependencyEdge]:
        """Incoming edges: nodes that depend on this node."""
        return self._rev_adj.get(node_id, [])

    def form_bundles(self, anchor_node_ids: Sequence[str], max_bundle_tokens: int = 250) -> list[EvidenceBundle]:
        """Bundle anchor nodes with their immediate high-weight dependent nodes, and bundle remaining turns by session."""
        bundles: list[EvidenceBundle] = []
        visited_nodes: set[str] = set()

        for anchor_id in anchor_node_ids:
            if anchor_id not in self.nodes or anchor_id in visited_nodes:
                continue

            anchor = self.nodes[anchor_id]
            # Window oversized anchor turns (> 180 words) to prevent knapsack starvation
            words = anchor.raw_text.split()
            if len(words) > 180:
                trunc_text = " ".join(words[:180]) + " ..."
                anchor = EvidenceNode(
                    node_id=anchor.node_id,
                    session_id=anchor.session_id,
                    session_date=anchor.session_date,
                    speaker=anchor.speaker,
                    raw_text=trunc_text,
                    tokens=word_tokens_43(trunc_text),
                    entities=anchor.entities,
                    properties=anchor.properties,
                    values=anchor.values,
                )
            members = [anchor]
            current_tokens = anchor.tokens
            visited_nodes.add(anchor_id)

            # Follow strong collocation edges (same session dialogue context)
            connected_edges = sorted(
                self.get_dependencies(anchor_id) + self.get_dependents(anchor_id),
                key=lambda e: e.weight,
                reverse=True
            )

            for edge in connected_edges:
                # Bundling is specifically for immediate adjacent conversational dialogue context (dist=1)
                if edge.edge_type != EdgeType.COLLOCATION or edge.weight < 0.95:
                    continue

                other_id = edge.target_id if edge.source_id == anchor_id else edge.source_id
                if other_id not in visited_nodes and other_id in self.nodes:
                    other_node = self.nodes[other_id]
                    if current_tokens + other_node.tokens <= max_bundle_tokens:
                        members.append(other_node)
                        current_tokens += other_node.tokens
                        visited_nodes.add(other_id)

            bundles.append(EvidenceBundle(
                bundle_id=f"B_{anchor_id}",
                primary_node_id=anchor_id,
                member_nodes=members,
                total_tokens=current_tokens,
                rationale=f"Anchored bundle for {anchor_id}",
            ))

        # COVERAGE FIX (LoCoMo full-corpus audit).  Only ANCHOR nodes received
        # bundles, so every non-anchor turn was structurally unreachable: it had
        # no bundle, therefore could never be selected, therefore could never
        # reach the reader.  Measured on the LoCoMo full corpus this was the
        # dominant failure mode for inference questions -- for 35 of 41
        # cat3 questions whose evidence turn was not retrieved, the answer turn
        # lived in a session that WAS retrieved but was never an anchor.
        # A conversation has 500-700 turns but only ~45 anchors, so the vast
        # majority of turns (including most answer turns) were invisible.
        # Give each remaining turn a single-node bundle so it can compete on
        # its own merits instead of being structurally excluded.
        for node_id, node in self.nodes.items():
            if node_id in visited_nodes:
                continue
            visited_nodes.add(node_id)
            bundles.append(EvidenceBundle(
                bundle_id=f"B_{node_id}",
                primary_node_id=node_id,
                member_nodes=[node],
                total_tokens=node.tokens,
                rationale=f"Coverage bundle for unanchored turn {node_id}",
            ))

        return bundles
