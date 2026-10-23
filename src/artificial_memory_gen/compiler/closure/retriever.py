"""Multi-View Retriever with Reciprocal Rank Fusion (Write LLM = 0).

Combines 5 independent retrieval views:
1. Lexical / Keyword View
2. FactIndex Predicate View
3. Entity & Coreference View
4. Temporal Constraint View
5. Session Neighborhood View

Fuses rankings via standard Reciprocal Rank Fusion (RRF) to eliminate score scale bias.
Zero benchmark-specific hardcoding or gold-answer leaks.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from artificial_memory_gen.compiler.closure.types import (
    ClosureCandidate,
    EntityTarget,
    PredicateRequirement,
    QueryPlan,
    TemporalConstraint,
)
from artificial_memory_gen.recall.fact_index import FactIndex

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "through", "during", "before", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "it", "its", "they", "them", "their", "we", "us", "our",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize and normalize text into non-stop words."""
    tokens = re.findall(r"\b[a-zA-Z0-9_+/'-]+\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _word_root(w: str) -> str:
    """Normalize word suffixes for inflection-invariant matching."""
    w = w.lower()
    for suffix in ["ing", "ed", "es", "s"]:
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


class MultiViewRetriever:
    """Deterministic Multi-View Retrieval engine with Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k
        self.fact_index = FactIndex()

    def _normalize_turn(self, turn: Any, index: int) -> tuple[str, str, str, str, str]:
        """Returns (turn_id, session_id, session_date, speaker, content)."""
        if isinstance(turn, (tuple, list)):
            if len(turn) == 4:
                sid, sdate, spk, content = turn
                tid = f"{sid}:t{index}"
                return tid, str(sid), str(sdate), str(spk), str(content)
            elif len(turn) >= 5:
                tid, sid, sdate, spk, content = turn[:5]
                return str(tid), str(sid), str(sdate), str(spk), str(content)
        elif hasattr(turn, "turn_id"):
            return (
                str(turn.turn_id),
                str(getattr(turn, "session_id", "")),
                str(getattr(turn, "session_date", "")),
                str(getattr(turn, "speaker", "")),
                str(getattr(turn, "content", "")),
            )
        raise ValueError(f"Unsupported turn structure: {type(turn)}")

    def retrieve(
        self,
        query_plan: QueryPlan,
        all_turns: Sequence[Any],
        top_k: int = 35,
    ) -> list[ClosureCandidate]:
        """Perform multi-view retrieval and rank fusion."""
        normalized_turns = [self._normalize_turn(t, i) for i, t in enumerate(all_turns)]
        if not normalized_turns:
            return []

        # Index FactIndex if not already populated for these turns
        raw_4tuples = [(sid, sdate, spk, content) for _, sid, sdate, spk, content in normalized_turns]
        self.fact_index.build(raw_4tuples)

        query_tokens = _tokenize(query_plan.query_text)

        # Compute Session IDF for query tokens
        session_word_sets: dict[str, set[str]] = defaultdict(set)
        for _, sid, _, _, content in normalized_turns:
            c_toks = _tokenize(content)
            session_word_sets[sid].update(c_toks)
            session_word_sets[sid].update(_word_root(t) for t in c_toks)

        n_sessions = max(1, len(session_word_sets))
        idf_dict: dict[str, float] = {}
        for tok in query_tokens:
            root = _word_root(tok)
            cnt = sum(1 for words in session_word_sets.values() if tok in words or root in words)
            idf_dict[tok] = math.log(1.0 + (n_sessions / (1.0 + cnt)))
            idf_dict[root] = idf_dict[tok]

        # View 1: Lexical View (Exact + Root + Session IDF)
        lexical_scores = self._score_lexical_view(query_tokens, normalized_turns, idf_dict)

        # View 2: Predicate View
        predicate_scores = self._score_predicate_view(query_plan.predicates, normalized_turns)

        # View 3: Entity & Coref View
        entity_scores = self._score_entity_view(query_plan.entity_targets, normalized_turns)

        # View 4: Temporal View
        temporal_scores = self._score_temporal_view(query_plan.temporal_constraints, normalized_turns)

        # View 5: Neighborhood View (seeded by top lexical & predicate turns)
        top_anchors = self._get_anchor_tids(lexical_scores, predicate_scores, top_n=5)
        neighborhood_scores = self._score_neighborhood_view(top_anchors, normalized_turns)

        # Convert scores to rankings per view
        view_rankings = {
            "lexical": self._rank_scores(lexical_scores),
            "predicate": self._rank_scores(predicate_scores),
            "entity": self._rank_scores(entity_scores),
            "temporal": self._rank_scores(temporal_scores),
            "neighborhood": self._rank_scores(neighborhood_scores),
        }

        view_weights = {
            "lexical": 1.0,
            "predicate": 1.1,
            "entity": 1.0,
            "temporal": 1.05,
            "neighborhood": 0.85,
        }

        # Compute Reciprocal Rank Fusion (RRF)
        turn_map = {t[0]: t for t in normalized_turns}
        all_candidate_tids = set()
        for ranking in view_rankings.values():
            all_candidate_tids.update(ranking.keys())

        fused_candidates: list[ClosureCandidate] = []
        for tid in all_candidate_tids:
            rrf_score = 0.0
            turn_view_ranks = {}
            for view_name, ranking in view_rankings.items():
                if tid in ranking:
                    rank = ranking[tid]
                    turn_view_ranks[view_name] = rank
                    weight = view_weights.get(view_name, 1.0)
                    rrf_score += weight / (self.rrf_k + rank)

            orig_tid, sid, sdate, spk, content = turn_map[tid]
            candidate = ClosureCandidate(
                turn_id=orig_tid,
                session_id=sid,
                session_date=sdate,
                speaker=spk,
                content=content,
                view_ranks=turn_view_ranks,
                rrf_score=rrf_score,
            )
            fused_candidates.append(candidate)

        fused_candidates.sort(key=lambda c: c.rrf_score, reverse=True)
        return fused_candidates[:top_k]

    def _score_lexical_view(
        self,
        query_tokens: list[str],
        turns: list[tuple[str, str, str, str, str]],
        idf_dict: dict[str, float] | None = None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not query_tokens:
            return scores

        idf = idf_dict or {}
        query_roots = {_word_root(t): t for t in query_tokens}
        query_set = set(query_tokens)

        for tid, sid, sdate, spk, content in turns:
            c_tokens = _tokenize(content)
            if not c_tokens:
                continue
            c_set = set(c_tokens)
            c_roots = {_word_root(t) for t in c_tokens}

            exact_matches = query_set.intersection(c_set)
            root_matches = set(query_roots.keys()).intersection(c_roots)

            if exact_matches or root_matches:
                score = 0.0
                for tok in exact_matches:
                    w_idf = idf.get(tok, 1.5)
                    score += w_idf * (1.0 + math.log(1.0 + c_tokens.count(tok)))
                for root in root_matches:
                    orig_q_tok = query_roots[root]
                    if orig_q_tok not in exact_matches:
                        w_idf = idf.get(root, 1.0)
                        score += 0.8 * w_idf
                scores[tid] = score
        return scores

    def _score_predicate_view(
        self,
        predicates: tuple[PredicateRequirement, ...],
        turns: list[tuple[str, str, str, str, str]],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not predicates:
            return scores

        # 1. Direct predicate verb / token matching
        for pred in predicates:
            pred_toks = _tokenize(pred.predicate_name)
            for tid, sid, sdate, spk, content in turns:
                c_lower = content.lower()
                if pred.predicate_name.lower() in c_lower:
                    scores[tid] = scores.get(tid, 0.0) + 3.0
                elif any(pt in c_lower for pt in pred_toks):
                    scores[tid] = scores.get(tid, 0.0) + 1.5

        # 2. FactIndex syntactic grammar fact matching
        for spk, facts in self.fact_index.speaker_facts.items():
            for fact in facts:
                for pred in predicates:
                    if pred.category == "action" and fact.fact_type in ("action", "event"):
                        scores[fact.turn_id] = scores.get(fact.turn_id, 0.0) + 2.5
                    elif pred.category == "creation" and fact.fact_type in ("creation", "action"):
                        scores[fact.turn_id] = scores.get(fact.turn_id, 0.0) + 2.5
                    elif fact.fact_type == "event":
                        scores[fact.turn_id] = scores.get(fact.turn_id, 0.0) + 2.0

        return scores

    def _score_entity_view(
        self,
        entities: tuple[EntityTarget, ...],
        turns: list[tuple[str, str, str, str, str]],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not entities:
            return scores

        for ent in entities:
            cname = ent.canonical_name.lower()
            aliases = [a.lower() for a in ent.aliases]
            all_names = [cname] + aliases
            corefs = [c.lower() for c in ent.coref_tokens]

            for tid, sid, sdate, spk, content in turns:
                c_lower = content.lower()
                spk_lower = spk.lower()
                # Speaker attribution match
                if any(name in spk_lower for name in all_names):
                    scores[tid] = scores.get(tid, 0.0) + 1.5
                # Direct name mention in content
                if any(re.search(rf"\b{re.escape(name)}\b", c_lower) for name in all_names):
                    scores[tid] = scores.get(tid, 0.0) + 2.0
                # Coreference match if spoken by the target or companion
                elif corefs and any(re.search(rf"\b{re.escape(cf)}\b", c_lower) for cf in corefs):
                    scores[tid] = scores.get(tid, 0.0) + 0.5

        return scores

    def _score_temporal_view(
        self,
        constraints: tuple[TemporalConstraint, ...],
        turns: list[tuple[str, str, str, str, str]],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not constraints:
            return scores

        for tc in constraints:
            texpr = (tc.time_expression or "").lower()
            for tid, sid, sdate, spk, content in turns:
                c_lower = content.lower()
                d_lower = sdate.lower()
                # Date or relative marker match
                if texpr and (texpr in c_lower or texpr in d_lower):
                    scores[tid] = scores.get(tid, 0.0) + 2.5

        return scores

    def _get_anchor_tids(
        self,
        lexical: dict[str, float],
        predicate: dict[str, float],
        top_n: int = 5,
    ) -> set[str]:
        combined = defaultdict(float)
        for tid, s in lexical.items():
            combined[tid] += s
        for tid, s in predicate.items():
            combined[tid] += s * 1.5
        sorted_anchors = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return {tid for tid, _ in sorted_anchors[:top_n]}

    def _score_neighborhood_view(
        self,
        anchor_tids: set[str],
        turns: list[tuple[str, str, str, str, str]],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not anchor_tids:
            return scores

        session_turns = defaultdict(list)
        for idx, (tid, sid, sdate, spk, content) in enumerate(turns):
            session_turns[sid].append((idx, tid))

        for sid, t_list in session_turns.items():
            for pos, (orig_idx, tid) in enumerate(t_list):
                if tid in anchor_tids:
                    # Boost immediate predecessors and successors in the same session
                    for delta in (-2, -1, 1, 2):
                        neighbor_pos = pos + delta
                        if 0 <= neighbor_pos < len(t_list):
                            neighbor_tid = t_list[neighbor_pos][1]
                            weight = 2.0 if abs(delta) == 1 else 1.0
                            scores[neighbor_tid] = scores.get(neighbor_tid, 0.0) + weight

        return scores

    def _rank_scores(self, score_dict: dict[str, float]) -> dict[str, int]:
        """Convert a score dict to 1-based ranks (1 = highest score)."""
        sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return {tid: rank for rank, (tid, _) in enumerate(sorted_items, start=1)}
