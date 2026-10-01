"""Deterministic Personalized PageRank (PPR) Associative Memory Graph (HippoRAG-style).

Implements a neurobiologically-inspired associative memory layer that replaces
heuristic multi-session widening with mathematical activation spreading over an
Entity-Turn bipartite graph.

Key Properties:
- Write LLM Calls = 0: Builds entity-turn associations purely via deterministic extraction.
- Recall LLM Calls = 0: Ranks cross-session evidence using Personalized PageRank (power iteration).
- Full Provenance: Every activation path traces back to shared entities/attributes across turns.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence


class PPREvidenceGraph:
    """Bipartite Entity-Turn Graph with Personalized PageRank for Multi-Hop Evidence Recall."""

    STOP_WORDS = frozenset({
        "the", "and", "that", "this", "with", "have", "from", "for", "was", "were",
        "are", "been", "being", "has", "had", "does", "did", "what", "when", "where",
        "which", "who", "whom", "why", "how", "all", "any", "both", "each", "few",
        "more", "most", "other", "some", "such", "than", "too", "very", "can", "will",
        "just", "should", "now", "user", "assistant", "system", "tell", "about",
        "yeah", "yes", "nope", "okay", "sure", "well", "like", "also", "into", "onto",
    })

    def __init__(self, damping: float = 0.85, max_iter: int = 30, tol: float = 1e-5) -> None:
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

        # Graph adjacency: node -> set of neighbor nodes
        self.adj: dict[str, dict[str, float]] = defaultdict(dict)
        # Mapping from turn_id -> raw content
        self.turn_content: dict[str, str] = {}
        # Mapping from turn_id -> session_id
        self.turn_session: dict[str, str] = {}
        # Set of entity nodes
        self.entity_nodes: set[str] = set()
        # Set of turn nodes
        self.turn_nodes: set[str] = set()

    def extract_entities(self, text: str) -> set[str]:
        """Extract deterministic entities and salient terms without LLM calls."""
        # 1. Capitalized proper nouns / phrases (e.g. "Samsung Galaxy", "New York", "Caroline")
        proper = set(re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", text))
        # 2. Content words of length >= 3 not in stop words
        words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower()))
        salient_words = {w for w in words if w not in self.STOP_WORDS}

        normalized = {p.lower() for p in proper if len(p) >= 3 and p.lower() not in self.STOP_WORDS}
        return normalized | salient_words

    def add_turn(self, turn_id: str, content: str, session_id: str = "") -> None:
        """Add a conversation turn node and link it to all its extracted entities."""
        self.turn_nodes.add(turn_id)
        self.turn_content[turn_id] = content
        if session_id:
            self.turn_session[turn_id] = session_id

        entities = self.extract_entities(content)
        for ent in entities:
            ent_key = f"ent:{ent}"
            self.entity_nodes.add(ent_key)
            # Undirected bipartite edge with initial uniform weight 1.0
            self.adj[turn_id][ent_key] = 1.0
            self.adj[ent_key][turn_id] = 1.0

    def build_from_records(self, records: Sequence[Any]) -> None:
        """Build the entire graph from a sequence of StructuredIR or ApexMemoryUnit objects."""
        for i, r in enumerate(records):
            raw = getattr(r, "raw_content", "") or (r.ir.raw_content if hasattr(r, "ir") else str(r))
            m_sess = re.search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]", raw)
            sid = m_sess.group(1) if m_sess else f"s{i}"
            turn_id = f"turn_{i}_{sid}"
            self.add_turn(turn_id, raw, session_id=sid)

    def personalized_pagerank(self, query: str) -> dict[str, float]:
        """Compute Personalized PageRank with personalization focused on query entities."""
        q_entities = set(self.extract_entities(query))
        from artificial_memory.recall.domain_associator import DomainAssociator
        domain_terms = DomainAssociator.expand_query(query)
        q_entities.update(domain_terms)
        matching_ent_keys = [f"ent:{e}" for e in q_entities if f"ent:{e}" in self.entity_nodes]

        # All nodes in the graph
        all_nodes = list(self.adj.keys())
        n = len(all_nodes)
        if n == 0:
            return {}

        # Construct personalization distribution (teleport set)
        p: dict[str, float] = {node: 0.0 for node in all_nodes}
        if matching_ent_keys:
            weight = 1.0 / len(matching_ent_keys)
            for k in matching_ent_keys:
                p[k] = weight
        else:
            # Fallback uniform teleport if no query entity matches
            uniform = 1.0 / n
            for node in all_nodes:
                p[node] = uniform

        # Precompute out-degrees and transition weights
        out_weights: dict[str, float] = {}
        for u in all_nodes:
            out_weights[u] = sum(self.adj[u].values())

        # Power iteration
        r = p.copy()
        for _ in range(self.max_iter):
            r_next: dict[str, float] = {node: (1.0 - self.damping) * p[node] for node in all_nodes}

            for u, neighbors in self.adj.items():
                deg = out_weights[u]
                if deg <= 0:
                    continue
                push = self.damping * r[u] / deg
                for v in neighbors:
                    r_next[v] += push

            # Check convergence L1 norm
            diff = sum(abs(r_next[node] - r[node]) for node in all_nodes)
            r = r_next
            if diff < self.tol:
                break

        return r

    def rank_turns(self, query: str, top_k: int = 10) -> list[tuple[str, float, str]]:
        """Rank turn nodes by stationary probability.

        Returns list of tuples: (turn_id, score, raw_content).
        """
        ranks = self.personalized_pagerank(query)
        # Filter to turn nodes only
        turn_ranks = [(t, ranks.get(t, 0.0), self.turn_content.get(t, "")) for t in self.turn_nodes]
        turn_ranks.sort(key=lambda x: x[1], reverse=True)
        return turn_ranks[:top_k]

    def explain_connection(self, turn_id: str, query: str) -> list[str]:
        """Explain the associative entities connecting the turn to query terms."""
        q_entities = {f"ent:{e}" for e in self.extract_entities(query)}
        turn_neighbors = set(self.adj.get(turn_id, {}).keys())
        shared = q_entities & turn_neighbors
        return [s.replace("ent:", "") for s in shared]
