"""
Graph Builder: Builds EvidenceDependencyGraph from candidate turns and StructuredIR.

Analyzes cross-turn entity recurrence, temporal progressions, and semantic
connections to establish typed dependency edges.
"""

from __future__ import annotations

import datetime
import re
from typing import Sequence

from artificial_memory.compiler.graph.dependency_graph import (
    DependencyEdge,
    EdgeType,
    EvidenceDependencyGraph,
    EvidenceNode,
)
from artificial_memory.core.ir.structured import IRRelation, IRStatus, StructuredIR


def parse_iso_date(date_str: str) -> datetime.date | None:
    if not date_str:
        return None
    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", date_str)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


class GraphBuilder:
    """Constructs EvidenceDependencyGraph from dialogue turns and IR records."""

    _STOP_WORDS = frozenset({
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "can", "could", "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each",
        "few", "for", "from", "further", "had", "hadn", "has", "hasn", "have", "haven", "having", "he",
        "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
        "isn", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not",
        "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
        "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
        "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
        "to", "too", "under", "until", "up", "very", "was", "wasn", "we", "were", "weren", "what", "when",
        "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours",
        "yourself", "yourselves", "great", "good", "well", "really", "think", "know", "like", "also",
        "want", "need", "help", "sure", "make", "always", "something", "everything", "nothing", "awesome",
        "going", "look", "time", "first", "last", "back", "come", "take", "give", "tell", "even", "much"
    })

    def build(self, turns: Sequence[tuple[str, str, str, str]], records: Sequence[StructuredIR]) -> EvidenceDependencyGraph:
        """
        Build EDG from turns and corresponding records.
        turns: list of (session_id, session_date, speaker, content)
        """
        graph = EvidenceDependencyGraph()

        # Map session/turn to records via the deterministic "[sid ...]" header
        # prefix (no substring matching: avoids O(N*M) cost and FP hits from
        # short generic turns like "Received." matching unrelated records).
        turn_to_records: dict[str, list[StructuredIR]] = {}
        header_sid_re = re.compile(r"^\[([^\s\]]+)")
        for r in records:
            m = header_sid_re.match(r.raw_content or "")
            if not m:
                continue
            rsid = m.group(1)
            for idx, (sid, s_date, spk, content) in enumerate(turns):
                if sid != rsid:
                    continue
                # raw_content is f"[{sid} ...] Speaker: {content}"; match the
                # exact turn by suffix instead of substring containment.
                if (r.raw_content or "").endswith(content[-200:] if len(content) > 200 else content):
                    turn_to_records.setdefault(f"{sid}:t{idx}", []).append(r)
                    break

        # 1. Create Nodes
        created_nodes: list[EvidenceNode] = []
        for idx, (sid, s_date, speaker, content) in enumerate(turns):
            node_id = f"{sid}:t{idx}"
            approx_tokens = max(10, len(content.split()) * 4 // 3)

            recs = turn_to_records.get(node_id, [])
            entities = {r.entity.lower() for r in recs if r.entity}
            properties = {r.property.lower() for r in recs if r.property}
            values = {r.value.lower() for r in recs if r.value}

            # Extract capitalized entities and specific keywords
            cap_entities = set(re.findall(r"\b[A-Z][a-z0-9_-]+\b", content))
            words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", content.lower())) - self._STOP_WORDS
            entities.update(cap_entities)
            entities.update([w for w in words if len(w) > 4])

            node = EvidenceNode(
                node_id=node_id,
                session_id=sid,
                session_date=s_date,
                speaker=speaker.lower(),
                raw_text=content,
                tokens=approx_tokens,
                entities=frozenset(entities),
                properties=frozenset(properties),
                values=frozenset(values),
            )
            graph.add_node(node)
            created_nodes.append(node)

        # 2. Establish Edges
        num_nodes = len(created_nodes)
        for i in range(num_nodes):
            n1 = created_nodes[i]
            d1 = parse_iso_date(n1.session_date)

            for j in range(i + 1, num_nodes):
                n2 = created_nodes[j]
                d2 = parse_iso_date(n2.session_date)

                # A. Same Session Collocation (distance-weighted dialogue locality)
                if n1.session_id == n2.session_id:
                    dist = abs(i - j)
                    if dist == 1:
                        w = 0.99
                    elif dist == 2:
                        w = 0.92
                    elif dist == 3:
                        w = 0.82
                    elif dist == 4:
                        w = 0.72
                    else:
                        w = 0.20
                    graph.add_edge(DependencyEdge(
                        source_id=n1.node_id,
                        target_id=n2.node_id,
                        edge_type=EdgeType.COLLOCATION,
                        weight=w,
                    ))
                    continue

                # B. Shared Entity / Topic across sessions (requires genuine entity overlap)
                shared_entities = n1.entities & n2.entities
                if len(shared_entities) >= 2:
                    weight = min(1.0, 0.5 + len(shared_entities) * 0.15)
                    
                    # Temporal sequence if dates differ
                    if d1 and d2 and d1 != d2:
                        src, tgt = (n1.node_id, n2.node_id) if d1 < d2 else (n2.node_id, n1.node_id)
                        graph.add_edge(DependencyEdge(
                            source_id=src,
                            target_id=tgt,
                            edge_type=EdgeType.TEMPORAL,
                            weight=weight,
                            metadata={"earlier": min(d1, d2).isoformat(), "later": max(d1, d2).isoformat()},
                        ))
                    else:
                        graph.add_edge(DependencyEdge(
                            source_id=n1.node_id,
                            target_id=n2.node_id,
                            edge_type=EdgeType.ENTITY,
                            weight=weight,
                            metadata={"shared": ",".join(list(shared_entities)[:3])},
                        ))

        return graph
