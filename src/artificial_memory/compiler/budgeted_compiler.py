"""
Budgeted Cognitive Compiler (BCC) for Lethe-Akribeia V4.

Maximizes evidence coverage and coherence under a strict token budget B using
submodular knapsack optimization over Evidence Bundles.
Produces deterministic, auditable context bundles with mathematical certificates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.compiler.certificates.coprocessor_engine import CoProcessorBundle
from artificial_memory.compiler.graph.dependency_graph import EvidenceBundle, EvidenceNode
from artificial_memory.compiler.speaker_normalizer import SpeakerAttributionNormalizer
from artificial_memory.compiler.sufficiency_judge import EvidenceSufficiencyJudge, SufficiencyAssessment
from artificial_memory.context.temporal_normalizer import TemporalNormalizer


@dataclass(frozen=True)
class CompiledContext:
    """The final compiled evidence context prepared for Reader ingestion."""
    context_text: str
    total_tokens: int
    selected_bundle_ids: list[str]
    selected_node_ids: list[str]
    utility_score: float
    budget_limit: int
    provenance_trail: list[str]
    sufficiency_assessment: SufficiencyAssessment | None = None


class BudgetedCognitiveCompiler:
    """Submodular Knapsack Evidence Compiler under strict token budgets."""

    def __init__(self) -> None:
        self.speaker_normalizer = SpeakerAttributionNormalizer()
        self.temporal_normalizer = TemporalNormalizer()
        self.judge = EvidenceSufficiencyJudge()

    @property
    def _STOP_WORDS(self) -> frozenset[str]:
        from artificial_memory.recall.evidence_scorer import UniversalEvidenceScorer
        return frozenset(UniversalEvidenceScorer.STOP_WORDS | {
            "can", "could", "would", "should", "some", "any", "suggest", "suggestion", "suggestions",
            "recommend", "recommendation", "recommendations", "advice", "tips", "ideas", "thoughts",
            "opinion", "what", "where", "when", "which", "who", "whom", "how", "tell", "help", "give",
            "think", "wondering", "please", "look", "looking", "interested",
            "back", "previous", "conversation", "conversations", "remind", "remember", "recall", "talked", "discussed", "mention", "mentioned"
        })

    def compile(
        self,
        question: str,
        bundles: Sequence[EvidenceBundle],
        token_budget: int,
        co_processor_bundle: CoProcessorBundle | None = None,
        anchor_scores: dict[str, float] | None = None,
        is_multi_session: bool = False,
        anchor_prefill: bool = True,
        adaptive_escalation: bool = False,
    ) -> CompiledContext:
        """Select optimal subset of bundles under token_budget and format context."""
        import math
        # Expand question words with roots and synonyms
        from artificial_memory.recall.evidence_scorer import UniversalEvidenceScorer
        expansions = UniversalEvidenceScorer.SEMANTIC_EXPANSIONS
        stop_words = self._STOP_WORDS
        raw_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", question.lower())) - stop_words
        if not raw_words:
            raw_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", question.lower()))
        q_words = set(raw_words)
        for w in raw_words:
            for suffix in ["ing", "ed", "es", "s"]:
                if len(w) > len(suffix) + 2 and w.endswith(suffix):
                    q_words.add(w[:-len(suffix)])
            if w in expansions:
                q_words.update(expansions[w])

        # Reserve tokens for certificates
        cert_tokens = 0
        cert_block = ""
        if co_processor_bundle and co_processor_bundle.assembled_certificate_block:
            cert_block = co_processor_bundle.assembled_certificate_block
            cert_tokens = len(cert_block.split()) * 4 // 3

        effective_budget = max(150, token_budget - cert_tokens)

        # 1. Greedy Density Selection (Submodular Knapsack)
        selected_bundles: list[EvidenceBundle] = []
        selected_sids: set[str] = set()
        covered_words: set[str] = set()
        current_tokens = 0

        # Pool of valid bundles within budget
        valid_candidates = [b for b in bundles if b.total_tokens <= effective_budget]

        # H1 anchor-priority pre-fill (opt-in): admit the bundles of the
        # TOP-ranked anchors BEFORE density fill.  Without this, the
        # submodular gain starves the answer turn: once an earlier,
        # topically-redundant turn covered the query words, the answer
        # bundle scores ~zero new words, loses the density race, and the
        # budget is already spent (trace: rank-1 answer bundle never fit).
        # Fit-or-skip: an oversized anchor bundle does not block the rest.
        if anchor_prefill and anchor_scores:

            def _max_anc(b: EvidenceBundle) -> float:
                return max((anchor_scores.get(n.node_id, 0.0)
                            for n in b.member_nodes), default=0.0)

            # Admit as many top evidence anchors as fit within 85% of budget
            prefill_budget = int(effective_budget * 0.85)
            prefill = sorted((b for b in valid_candidates if _max_anc(b) > 0.0),
                             key=_max_anc, reverse=True)
            prefill_ids = set()
            session_counts: dict[str, int] = {}
            for b in prefill:
                if current_tokens + b.total_tokens > prefill_budget:
                    continue
                selected_bundles.append(b)
                current_tokens += b.total_tokens
                prefill_ids.add(b.bundle_id)
                for n in b.member_nodes:
                    covered_words.update(re.findall(r"\b[a-zA-Z0-9_-]+\b", n.raw_text.lower()))
                    selected_sids.add(n.session_id)
                    session_counts[n.session_id] = session_counts.get(n.session_id, 0) + 1
            if prefill_ids:
                valid_candidates = [b for b in valid_candidates
                                    if b.bundle_id not in prefill_ids]

        while valid_candidates:
            best_bundle = None
            best_density = -1.0
            best_gain = 0.0

            for b in valid_candidates:
                if current_tokens + b.total_tokens > effective_budget:
                    continue

                # Bundle word and session set
                b_words = set()
                b_sids = set()
                for n in b.member_nodes:
                    b_words.update(re.findall(r"\b[a-zA-Z0-9_-]+\b", n.raw_text.lower()))
                    b_sids.add(n.session_id)

                overlap = b_words & q_words
                in_sel_sid = bool(b_sids & selected_sids)

                # Anchor score injection
                anc_score = 0.0
                if anchor_scores:
                    anc_score = max([anchor_scores.get(n.node_id, 0.0) for n in b.member_nodes] + [0.0])

                # Gate: bundle must either have anchor score, cover query terms, or continue an already-selected evidence session
                if not overlap and not in_sel_sid and anc_score <= 0.0:
                    continue
                new_words = overlap - covered_words
                # If it contributes NO new query words and has no anchor score and is not continuing an existing session, skip it
                if not in_sel_sid and not new_words and anc_score <= 0.0:
                    continue

                # Adaptive marginal utility:
                # Diminish score if session is already represented and contributes no new query terms,
                # while preserving high-relevance complementary turns and rewarding unseen candidate sessions.
                if is_multi_session:
                    if in_sel_sid:
                        k = max((session_counts.get(sid, 0) for sid in b_sids), default=1)
                        if not new_words and anc_score < 10.0:
                            anc_term = (anc_score * 2.5) / (1.0 + 0.8 * k)
                        else:
                            anc_term = anc_score * 2.5
                        gain = anc_term + len(new_words) * 8.0
                    else:
                        # Unseen candidate session: full anchor score + exploration bonus
                        gain = anc_score * 2.5 + len(new_words) * 8.0 + (6.0 if anc_score > 0 else 2.0)
                else:
                    gain = anc_score * 2.5 + len(new_words) * 8.0 + (6.0 if in_sel_sid else 0.0)

                novel_entities = min(3.0, len(set().union(*[n.entities for n in b.member_nodes])) * 0.1) if b.member_nodes else 0.0
                gain += novel_entities

                # Soft token scaling: prevent tiny empty fragments from beating substantive evidence turns
                density = gain / (max(80, b.total_tokens) ** 0.15)
                if density > best_density:
                    best_density = density
                    best_bundle = b
                    best_gain = gain

            if not best_bundle or best_gain <= 0.0:
                break

            selected_bundles.append(best_bundle)
            current_tokens += best_bundle.total_tokens
            
            for n in best_bundle.member_nodes:
                covered_words.update(re.findall(r"\b[a-zA-Z0-9_-]+\b", n.raw_text.lower()))
                selected_sids.add(n.session_id)
                session_counts[n.session_id] = session_counts.get(n.session_id, 0) + 1

            valid_candidates = [b for b in valid_candidates if b.bundle_id != best_bundle.bundle_id]

        # 2. Extract and Deduplicate Member Nodes
        seen_nodes: set[str] = set()
        ordered_nodes: list[EvidenceNode] = []

        for b in selected_bundles:
            for n in b.member_nodes:
                if n.node_id not in seen_nodes:
                    seen_nodes.add(n.node_id)
                    ordered_nodes.append(n)

        # Sort chronologically by session date, session id, and turn index
        ordered_nodes.sort(key=lambda n: (n.session_date or "", n.session_id, n.node_id))

        # 3. Format Context
        body_lines: list[str] = []
        provenance: list[str] = []

        for n in ordered_nodes:
            date_str = f" on {n.session_date}" if n.session_date else ""
            spk_str = n.speaker.capitalize()
            norm_text = self.speaker_normalizer.normalize_turn(n.raw_text, speaker=n.speaker)
            if n.session_date:
                norm_text = self.temporal_normalizer.normalize(norm_text, n.session_date)
            line = f"[{n.session_id}{date_str}] {spk_str}: {norm_text}"
            body_lines.append(line)
            provenance.append(f"{n.node_id} ({n.tokens} tok)")

        spk_cert = self.speaker_normalizer.generate_attribution_certificate(
            question, [(n.speaker, n.raw_text) for n in ordered_nodes]
        )
        if spk_cert:
            cert_block = f"{cert_block}\n\n{spk_cert}".strip() if cert_block else spk_cert

        context_parts = []
        if cert_block:
            context_parts.append(cert_block)

        if body_lines:
            context_parts.append(f"[VERIFIED CONVERSATION EVIDENCE]\n" + "\n\n".join(body_lines))
        else:
            context_parts.append("[No specific conversation evidence was selected under budget.]")

        full_context = "\n\n".join(context_parts)
        final_tokens = cert_tokens + current_tokens

        compiled_ctx = CompiledContext(
            context_text=full_context,
            total_tokens=final_tokens,
            selected_bundle_ids=[b.bundle_id for b in selected_bundles],
            selected_node_ids=list(seen_nodes),
            utility_score=len(covered_words & q_words),
            budget_limit=token_budget,
            provenance_trail=provenance,
            sufficiency_assessment=None,
        )

        top1_anc = max(anchor_scores.values()) if anchor_scores else 0.0
        assessment = self.judge.evaluate(
            query=question,
            plan=None,
            compiled_context=compiled_ctx,
            top1_anchor_score=top1_anc,
            selected_session_count=len(selected_sids),
            is_multi_session=is_multi_session,
            candidate_count=len(bundles),
        )

        if adaptive_escalation and not assessment.is_sufficient and not anchor_prefill:
            escalated_budget = min(2200, int(token_budget * assessment.budget_escalation_factor))
            escalated_prefill = "anchor_prefill" in assessment.recommended_escalations or not anchor_prefill
            escalated_ctx = self.compile(
                question=question,
                bundles=bundles,
                token_budget=escalated_budget,
                co_processor_bundle=co_processor_bundle,
                anchor_scores=anchor_scores,
                is_multi_session=is_multi_session,
                anchor_prefill=escalated_prefill,
                adaptive_escalation=False,
            )
            escalated_trail = list(escalated_ctx.provenance_trail) + [
                f"[ADAPTIVE ESCALATION: {','.join(assessment.distress_reasons)} factor={assessment.budget_escalation_factor}]"
            ]
            return CompiledContext(
                context_text=escalated_ctx.context_text,
                total_tokens=escalated_ctx.total_tokens,
                selected_bundle_ids=escalated_ctx.selected_bundle_ids,
                selected_node_ids=escalated_ctx.selected_node_ids,
                utility_score=escalated_ctx.utility_score,
                budget_limit=escalated_budget,
                provenance_trail=escalated_trail,
                sufficiency_assessment=assessment,
            )

        return CompiledContext(
            context_text=full_context,
            total_tokens=final_tokens,
            selected_bundle_ids=[b.bundle_id for b in selected_bundles],
            selected_node_ids=list(seen_nodes),
            utility_score=len(covered_words & q_words),
            budget_limit=token_budget,
            provenance_trail=provenance,
            sufficiency_assessment=assessment,
        )
