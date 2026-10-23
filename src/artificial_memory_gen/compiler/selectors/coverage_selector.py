"""
Coverage-Aware Selector (Phase 3 of the V4 directive) — EXPERIMENTAL, NOT ADOPTED.

MEASURED NEGATIVE RESULT (Dev-250, budget 1300, no LLM):
    selector        multi recall_selected   temporal recall_selected
    budgeted (BCC)          81.0%                     77.3%
    this selector           37.4%                     31.9%

Why it failed: the selection budget is already saturated (mean 1296/1300 tokens,
~8 turns) by the baseline compiler, so granting a large per-new-session bonus
(λ) does not add turns — it *displaces* the deeper, higher-relevance selection
that the baseline gets right, dropping gold sessions that ranked below the
spread. At fixed B there is no leftover budget for a non-displacing coverage
pass either. Improving multi/temporal recall therefore needs re-allocation
(e.g. Phase 4 dynamic budget), not an unconstrained coverage term.

Kept as the reproducible record of experiment Phase-3-A. It is NOT wired into
the default path; ``--selector coverage`` opts in.

Improves on the plain relevance-greedy budgeted compiler by maximizing

    F(S) = Σ_{t∈S} rel(t, q)  +  λ · |sessions(S)|  +  μ · |entities_covered(S)|

under a token budget B with a greedy knapsack rule

    argmax_t  [F(S ∪ {t}) − F(S)] / tokens(t) ** density_exponent  .

* ``rel`` reuses the UniversalEvidenceScorer vocabulary/semantic expansions and
  per-node IDF weighting.
* ``λ`` and ``μ`` scale with a continuous ``coverage_signal`` (aggregation /
  temporal demand). It is ~0 for single-session facts, so those queries fall
  back to pure relevance greedy and never spread across irrelevant sessions.
* Selection is strictly **turn-level** (never session bundles), which avoids the
  "2774-token session blob" failure mode. Each selected turn is later annotated
  with structured ``[session_id on date]`` metadata by the compiler formatter.

The formulation is monotone submodular, so the greedy knapsack solution carries
the standard (1 − 1/e) approximation guarantee.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence

from artificial_memory.core.tokens import word_tokens_floored
from artificial_memory_gen.compiler.budgeted_compiler import CompiledContext
from artificial_memory_gen.compiler.graph.dependency_graph import EvidenceNode
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer

_STOP = frozenset(UniversalEvidenceScorer.STOP_WORDS)
_EXP = UniversalEvidenceScorer.SEMANTIC_EXPANSIONS
_WORD_RE = re.compile(r"\b[a-z0-9_-]+\b")


def _terms(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _root(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _expand(words: set[str]) -> set[str]:
    out = set(words) | {_root(w) for w in words}
    for w in list(words):
        out.update(_EXP.get(w, ()))
        out.update(_EXP.get(_root(w), ()))
    return out


def _approx_tokens(text: str) -> int:
    return word_tokens_floored(text)  # canonical (core.tokens)


class CoverageAwareSelector:
    """Turn-level greedy selector maximizing relevance + session/entity coverage."""

    def __init__(
        self,
        session_weight: float = 40.0,
        entity_weight: float = 6.0,
        density_exponent: float = 0.15,
        min_relevance: float = 0.0,
    ) -> None:
        self.session_weight = session_weight
        self.entity_weight = entity_weight
        self.density_exponent = density_exponent
        self.min_relevance = min_relevance

    # -- relevance -----------------------------------------------------------

    def _query_terms(self, question: str) -> set[str]:
        raw = _terms(question) - _STOP
        if not raw:
            raw = _terms(question)
        return _expand(raw)

    def _idf(self, nodes: Sequence[EvidenceNode]) -> dict[str, float]:
        df: dict[str, int] = {}
        for n in nodes:
            for w in _expand(_terms(n.raw_text)):
                df[w] = df.get(w, 0) + 1
        n_docs = max(1, len(nodes))
        return {
            w: max(0.2, math.log(1.0 + (n_docs - c + 0.5) / (c + 0.5)))
            for w, c in df.items()
        }

    def relevance(self, question: str, nodes: Sequence[EvidenceNode]) -> dict[str, float]:
        q_terms = self._query_terms(question)
        idfs = self._idf(nodes)
        rel: dict[str, float] = {}
        for n in nodes:
            overlap = q_terms & _expand(_terms(n.raw_text))
            rel[n.node_id] = sum(idfs.get(w, 1.0) for w in overlap) + 0.15 * len(overlap)
        return rel

    # -- selection -----------------------------------------------------------

    def select(
        self,
        question: str,
        nodes: Sequence[EvidenceNode],
        token_budget: int,
        cert_tokens: int = 0,
        coverage_signal: float = 0.0,
    ) -> list[EvidenceNode]:
        """Select turns greedily; returns nodes sorted chronologically."""
        if not nodes:
            return []

        rel = self.relevance(question, nodes)
        effective_budget = max(120, token_budget - cert_tokens)

        lam = self.session_weight * 3.0 * coverage_signal
        mu = self.entity_weight * coverage_signal

        selected: list[EvidenceNode] = []
        covered_sessions: set[str] = set()
        covered_entities: set[str] = set()
        tokens = 0
        pool = list(nodes)

        while True:
            best: EvidenceNode | None = None
            best_density = -1.0
            best_gain = 0.0

            for n in pool:
                if tokens + n.tokens > effective_budget:
                    continue

                # Gate: a turn must be relevant, or continue an already-selected
                # session. New irrelevant sessions can never enter the context.
                if rel[n.node_id] <= self.min_relevance and n.session_id not in covered_sessions:
                    continue

                new_session = 0.0 if n.session_id in covered_sessions else 1.0
                new_entities = len(set(n.entities) - covered_entities)
                gain = rel[n.node_id] + lam * new_session + mu * new_entities
                if gain <= 0.0:
                    continue

                density = gain / (max(80, n.tokens) ** self.density_exponent)
                if density > best_density:
                    best_density = density
                    best = n
                    best_gain = gain

            if best is None or best_gain <= 0.0:
                break

            selected.append(best)
            tokens += best.tokens
            covered_sessions.add(best.session_id)
            covered_entities |= set(best.entities)
            pool = [n for n in pool if n.node_id != best.node_id]

        selected.sort(key=lambda n: (n.session_date or "", n.session_id, n.node_id))
        return selected


def format_selected(nodes: Sequence[EvidenceNode], certificate_block: str = "", cert_tokens: int = 0) -> tuple[str, int]:
    """Render selected turns with structured ``[session_id on date]`` metadata."""
    body_lines = []
    for n in nodes:
        date_str = f" on {n.session_date}" if n.session_date else ""
        body_lines.append(f"[{n.session_id}{date_str}] {n.speaker.capitalize()}: {n.raw_text}")

    parts: list[str] = []
    if certificate_block:
        parts.append(certificate_block)
    if body_lines:
        parts.append("[VERIFIED CONVERSATION EVIDENCE]\n" + "\n\n".join(body_lines))
    else:
        parts.append("[No specific conversation evidence was selected under budget.]")

    body_tokens = sum(n.tokens for n in nodes)
    return "\n\n".join(parts), cert_tokens + body_tokens


def coverage_compile(
    question: str,
    nodes: Sequence[EvidenceNode],
    token_budget: int,
    certificate_block: str = "",
    coverage_signal: float = 0.0,
    selector: CoverageAwareSelector | None = None,
) -> CompiledContext:
    """Convenience wrapper producing a ``CompiledContext`` compatible with the BCC."""
    selector = selector or CoverageAwareSelector()
    cert_tokens = _approx_tokens(certificate_block) if certificate_block else 0
    selected = selector.select(
        question=question,
        nodes=nodes,
        token_budget=token_budget,
        cert_tokens=cert_tokens,
        coverage_signal=coverage_signal,
    )
    context_text, total_tokens = format_selected(selected, certificate_block, cert_tokens)
    return CompiledContext(
        context_text=context_text,
        total_tokens=total_tokens,
        selected_bundle_ids=[],
        selected_node_ids=[n.node_id for n in selected],
        utility_score=0.0,
        budget_limit=token_budget,
        provenance_trail=[f"{n.node_id} ({n.tokens} tok)" for n in selected],
    )
