#!/usr/bin/env python3
"""
Phase 0 Runner: Label-Free Baseline Evaluation for Lethe-Akribeia V4.

CRITICAL: This script runs V4 WITHOUT using gold question_type labels.
All routing decisions use LabelFreeTypeClassifierV2 predictions only.

Compliance checks (enforced by tests/unit/test_no_label_leak.py):
- NO gold question_type in inference pipeline
- NO gold answer/answer_session_ids in prompt construction
- Holdout split is never loaded
- LLM outputs are cached (question id + prompt hash) for reproducibility

Design principles:
- Gold labels enter ONLY the grading stage (`OfficialLongMemEvalGrader`) and the
  offline recall measurement (`answer_session_ids`), never the inference path.
- Errors are never swallowed into a 0.0 / empty answer: each failure is recorded
  with `status="error"` + traceback and surfaced as an explicit count.
- Evidence recall uses structural session-id set operations over the compiled
  node metadata, not substring matching on the rendered context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory.compiler.graph.dependency_graph import EvidenceBundle
from artificial_memory.compiler.graph.graph_builder import GraphBuilder
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.compiler.router.label_free_classifier import LabelFreeTypeClassifierV2
from artificial_memory.compiler.selectors import CoverageAwareSelector, coverage_compile
from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler
from artificial_memory.recall.state_timeline import StateTimelineEngine
from artificial_memory.research.benchmarks.llm import OllamaAnswerer


# ---------- Label-Free V4 Pipeline ----------

class LabelFreeV4Pipeline:
    """
    V4 pipeline with label-free routing.

    The inference path (``run`` + its helpers) accepts ONLY ``question``,
    ``question_date`` and ``haystack_*``.  Gold ``question_type`` / ``answer`` /
    ``answer_session_ids`` are never parameters and never referenced here; the
    regression test scans these methods to guarantee it stays that way.
    """

    PROMPT_TEMPLATES = {
        "fact": (
            "[VERIFIED CONVERSATION EVIDENCE]\n{context}\n\n"
            "Question: {question}\nAnswer:"
        ),
        "temporal": (
            "{certificates}\n\n"
            "[VERIFIED CONVERSATION EVIDENCE]\n{context}\n\n"
            "Question: {question}\n\n"
            "Based on the verified temporal calculation above, what is the final answer? "
            "State the exact number, date, or duration.\nAnswer:"
        ),
        "aggregation": (
            "{certificates}\n\n"
            "[VERIFIED CONVERSATION EVIDENCE]\n{context}\n\n"
            "Question: {question}\n\n"
            "Based on the verified aggregation above, what is the final total?\nAnswer:"
        ),
        "knowledge_update": (
            "[INSTRUCTION: KNOWLEDGE UPDATE & STATE EVOLUTION]\n"
            "The user's state changes over time. Answer with the CURRENT or LATEST state.\n\n"
            "{certificates}\n\n"
            "[VERIFIED CONVERSATION EVIDENCE]\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer directly and concisely:\nAnswer:"
        ),
        "preference": (
            "[INSTRUCTION: PREFERENCE & RECOMMENDATION]\n"
            "Tailor your answer to the user's stated preferences, past equipment, "
            "or background. Do NOT say 'I don't know'. Give concrete, specific suggestions.\n\n"
            "[USER PROFILE & PREFERENCES]\n{profile_block}\n\n"
            "[VERIFIED CONVERSATION EVIDENCE]\n{context}\n\n"
            "Question: {question}\n\nAnswer:"
        ),
        "default": (
            "[VERIFIED CONVERSATION EVIDENCE]\n{context}\n\n"
            "Question: {question}\nAnswer:"
        ),
    }

    def __init__(
        self,
        token_budget: int = 1300,
        reader_url: str = "http://localhost:11434",
        reader_model: str = "qwen2.5:7b-instruct",
        cache_dir: str | Path = "cache/phase0",
        use_llm_cache: bool = True,
        selector: str = "budgeted",
        soft_routing: bool = False,
    ):
        self.token_budget = token_budget
        self.selector = selector
        self.soft_routing = soft_routing
        self.coverage_selector = CoverageAwareSelector()
        self.reader_model = reader_model
        self.use_llm_cache = use_llm_cache
        self.cache_dir = Path(cache_dir)
        if self.use_llm_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.classifier = LabelFreeTypeClassifierV2(use_embeddings=True)
        self.ir_extractor = UniversalIRExtractor()
        self.graph_builder = GraphBuilder()
        self.budgeted_compiler = BudgetedCognitiveCompiler()
        self.coprocessor = CoProcessorEngine()
        self.planner = CognitiveExecutionPlanner()
        self.msc_compiler = MinimumSufficientContextCompiler()
        self.timeline_engine = StateTimelineEngine()
        self.reader = OllamaAnswerer(
            base_url=reader_url,
            model=reader_model,
            num_ctx=8192,
        )
        self.reader.max_tokens = 256
        self.reader.stop = ["\n\n[INSTRUCTION", "\n\nUser:"]

    def _classify(self, question: str) -> str:
        """Classify question type using label-free classifier only."""
        return self.classifier.classify(question).predicted_type

    def _turns_to_records_and_ir(
        self, sessions: list[list[dict]], session_dates: list[str], session_ids: list[str]
    ) -> tuple[list, list]:
        """Convert haystack sessions into StructuredIR records and turn tuples."""
        all_records = []
        turns = []

        for s_idx, session in enumerate(sessions):
            s_date = session_dates[s_idx] if s_idx < len(session_dates) else ""
            sid = session_ids[s_idx] if s_idx < len(session_ids) else f"session_{s_idx}"

            for turn_idx, turn in enumerate(session):
                speaker = turn.get("role", "user")
                content = turn.get("content", "")
                if not content.strip():
                    continue

                turns.append((sid, s_date, speaker, content))
                recs = self.ir_extractor.extract(content, default_source=speaker)
                for r in recs:
                    r.raw_content = f"[{sid} on {s_date}] {speaker}: {content}" if s_date else f"[{sid}] {speaker}: {content}"
                    r.time_scope = s_date
                    all_records.append(r)

        return all_records, turns

    def _build_bundles(self, edg) -> list:
        """Group graph nodes into bundles for compiler.

        Creates one bundle per turn (individual evidence units), so the compiler
        can select individual turns regardless of session size.
        """
        bundles = []
        for node_id, node in edg.nodes.items():
            bundle = EvidenceBundle(
                bundle_id=f"bundle_{node_id}",
                primary_node_id=node_id,
                member_nodes=[node],
                total_tokens=node.tokens,
                rationale=f"Evidence turn from session {node.session_id}",
            )
            bundles.append(bundle)
        return bundles

    def _get_anchor_scores(self, edg, question: str) -> dict:
        """Compute relevance scores for evidence nodes using keyword overlap."""
        import re as _re
        from artificial_memory.recall.evidence_scorer import UniversalEvidenceScorer
        scorer = UniversalEvidenceScorer()
        q_tokens = _re.findall(r'[a-z0-9_]+', question.lower())
        q_words = set(w for w in q_tokens if w not in scorer.STOP_WORDS)
        scores = {}
        for node_id, node in edg.nodes.items():
            text_tokens = _re.findall(r'[a-z0-9_]+', node.raw_text.lower())
            text_words = set(text_tokens)
            overlap = q_words & text_words
            scores[node_id] = float(len(overlap))
        return scores

    # Phase 2 soft-routing instruction blocks. Multiple blocks may co-activate
    # (e.g. temporal + aggregation); routing is a continuous prior, not a
    # mutually exclusive branch on the classifier label.
    _SOFT_INSTRUCTIONS = {
        "temporal": "[INSTRUCTION: TEMPORAL] Compute the exact date, duration, or ordering from the dated evidence. State the exact number/date.",
        "aggregation": "[INSTRUCTION: AGGREGATION] Check EVERY session in the evidence, then give the exact count or total.",
        "state_update": "[INSTRUCTION: STATE UPDATE] The user's state changes over time. Answer with the CURRENT / LATEST value.",
        "preference": "[INSTRUCTION: PREFERENCE] Tailor the answer to the user's stated preferences and past experience. Be concrete; do not refuse.",
        "assistant_recall": "[INSTRUCTION: ASSISTANT RECALL] Answer with what you (the assistant) previously recommended or said.",
        "fact": "[INSTRUCTION: EVIDENCE-FIRST] Answer directly from the evidence below. Never reply that you don't know.",
    }

    def _build_soft_prompt(self, question, compiled_context, coproc_bundle, exec_plan, profile_block):
        """Compose a prompt from continuously-activated demand signals."""
        blocks: list[str] = []
        if exec_plan.temporal_demand >= 0.50:
            blocks.append(self._SOFT_INSTRUCTIONS["temporal"])
        if exec_plan.aggregation_demand >= 0.50 or exec_plan.requires_event_aggregation:
            blocks.append(self._SOFT_INSTRUCTIONS["aggregation"])
        if exec_plan.state_evolution_demand >= 0.50:
            blocks.append(self._SOFT_INSTRUCTIONS["state_update"])
        if exec_plan.preference_demand >= 0.50:
            blocks.append(self._SOFT_INSTRUCTIONS["preference"])
        if exec_plan.assistant_recall_demand >= 0.50:
            blocks.append(self._SOFT_INSTRUCTIONS["assistant_recall"])
        if not blocks:
            blocks.append(self._SOFT_INSTRUCTIONS["fact"])

        parts = ["\n".join(blocks)]
        if exec_plan.preference_demand >= 0.50 and profile_block:
            parts.append(f"[USER PROFILE & PREFERENCES]\n{profile_block}")
        cert = coproc_bundle.assembled_certificate_block
        if cert:
            parts.append(cert)
        parts.append(f"[VERIFIED CONVERSATION EVIDENCE]\n{compiled_context}")
        parts.append(f"Question: {question}\nAnswer:")
        return "\n\n".join(parts)

    def _compile(self, question, edg, coproc_bundle, exec_plan, predicted_type):
        """Compile the retrieved evidence graph under the configured selector."""
        if self.selector == "coverage":
            coverage_signal = max(
                exec_plan.aggregation_demand,
                0.7 * exec_plan.temporal_demand,
                0.5 * exec_plan.assistant_recall_demand,
            )
            return coverage_compile(
                question=question,
                nodes=list(edg.nodes.values()),
                token_budget=self.token_budget,
                certificate_block=coproc_bundle.assembled_certificate_block,
                coverage_signal=coverage_signal,
                selector=self.coverage_selector,
            )
        # Phase 2: under soft routing, aggregation demand (not the hard label)
        # decides multi-session compilation behaviour.
        if self.soft_routing:
            is_multi = exec_plan.aggregation_demand >= 0.50 or exec_plan.requires_event_aggregation
        else:
            is_multi = predicted_type == "multi-session"
        bundles = self._build_bundles(edg)
        anchor_scores = self._get_anchor_scores(edg, question)
        return self.budgeted_compiler.compile(
            question=question,
            bundles=bundles,
            token_budget=self.token_budget,
            co_processor_bundle=coproc_bundle,
            anchor_scores=anchor_scores,
            is_multi_session=is_multi,
        )

    def _answer_cached(self, question: str, ctx: str) -> tuple[str, int, int, float, bool]:
        """Call the frozen reader, memoized on (model, question, prompt) hash.

        Returns (text, prompt_tokens, completion_tokens, latency_ms, cache_hit).
        The prompt hash makes re-runs and re-scoring bit-for-bit reproducible.
        """
        key = hashlib.sha256(
            (self.reader_model + "\x00" + question + "\x00" + ctx).encode("utf-8")
        ).hexdigest()[:24]
        cache_file = self.cache_dir / f"{key}.json"
        if self.use_llm_cache and cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return (
                data["text"],
                int(data["prompt_tokens"]),
                int(data["completion_tokens"]),
                float(data["latency_ms"]),
                True,
            )

        res = self.reader.answer(question, ctx)
        if self.use_llm_cache:
            payload = {
                "model": self.reader_model,
                "question": question,
                "prompt_sha256": hashlib.sha256(ctx.encode("utf-8")).hexdigest(),
                "text": res.text,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "latency_ms": res.latency_ms,
            }
            cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return res.text, res.prompt_tokens, res.completion_tokens, res.latency_ms, False

    def run(
        self,
        question: str,
        sessions: list[list[dict]],
        session_dates: list[str],
        session_ids: list[str],
        reference_date: str | None = None,
    ) -> dict:
        """
        Run the full V4 pipeline with label-free routing.

        NEVER uses: question_type, answer, answer_session_ids.
        """
        t0 = time.perf_counter()

        # 1. Convert to IR records and turns
        records, turns = self._turns_to_records_and_ir(sessions, session_dates, session_ids)

        # 2. LABEL-FREE type classification
        predicted_type = self._classify(question)

        # 3. Memory-Conditioned Execution Plan (query text + memory topology only)
        exec_plan = self.planner.plan(question, records, reference_date=reference_date)

        # 4. Build Evidence Dependency Graph
        edg = self.graph_builder.build(turns, records)

        # 5. Run Co-processors (Certificates from RETRIEVED evidence only)
        coproc_bundle = self.coprocessor.process(
            question=question,
            records=records,
            plan=exec_plan,
            reference_date=reference_date,
        )

        # Check for deterministic override (e.g., pure temporal math)
        if coproc_bundle.deterministic_answer_override:
            return {
                "status": "ok",
                "predicted_answer": coproc_bundle.deterministic_answer_override,
                "predicted_type": predicted_type,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "deterministic_override": True,
                "cache_hit": False,
                "context_tokens": 0,
                "certificates": coproc_bundle.assembled_certificate_block,
                # Selected sessions are derived purely from the retrieved
                # records' own session ids — never from gold labels — so this
                # stays invariant under the label-invariance test.
                "selected_session_ids": sorted({
                    getattr(r, "metadata", {}).get("session_id")
                    for r in records
                    if getattr(r, "metadata", {}).get("session_id")
                }),
                "provenance": [],
            }

        # 6. Compile context (bundled BCC or coverage-aware selector)
        compiled = self._compile(question, edg, coproc_bundle, exec_plan, predicted_type)

        # 7. Build profile block for preference queries
        profile_block = ""
        if predicted_type == "single-session-preference":
            try:
                profile_block = self.msc_compiler.extract_profile_grounding(records)
            except Exception:
                profile_block = ""

        # 8. Build prompt based on PREDICTED type (not gold).
        # Phase 2 (--soft-routing) overrides this with demand-composed blocks.
        if predicted_type == "temporal-reasoning":
            ctx = self.PROMPT_TEMPLATES["temporal"].format(
                certificates=coproc_bundle.assembled_certificate_block,
                context=compiled.context_text,
                question=question,
            )
        elif predicted_type == "multi-session":
            ctx = self.PROMPT_TEMPLATES["aggregation"].format(
                certificates=coproc_bundle.assembled_certificate_block,
                context=compiled.context_text,
                question=question,
            )
        elif predicted_type == "knowledge-update":
            ctx = self.PROMPT_TEMPLATES["knowledge_update"].format(
                certificates=coproc_bundle.assembled_certificate_block,
                context=compiled.context_text,
                question=question,
            )
        elif predicted_type == "single-session-preference":
            ctx = self.PROMPT_TEMPLATES["preference"].format(
                profile_block=profile_block,
                context=compiled.context_text,
                question=question,
            )
        elif predicted_type in ("single-session-user", "single-session-assistant"):
            ctx = self.PROMPT_TEMPLATES["fact"].format(
                context=compiled.context_text,
                question=question,
            )
        else:
            ctx = self.PROMPT_TEMPLATES["default"].format(
                context=compiled.context_text,
                question=question,
            )

        if self.soft_routing:
            ctx = self._build_soft_prompt(
                question, compiled.context_text, coproc_bundle, exec_plan, profile_block,
            )

        prompt_tokens_estimate = len(ctx.split()) * 4 // 3

        # 9. Call Reader (cached)
        text, prompt_tokens, completion_tokens, latency_ms, cache_hit = self._answer_cached(question, ctx)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "status": "ok",
            "predicted_answer": text,
            "predicted_type": predicted_type,
            "prompt_tokens": prompt_tokens or prompt_tokens_estimate,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms or elapsed_ms,
            "deterministic_override": False,
            "cache_hit": cache_hit,
            "context_tokens": compiled.total_tokens,
            "certificates": coproc_bundle.assembled_certificate_block,
            "provenance": compiled.provenance_trail,
            "budget_limit": self.token_budget,
            # Structural provenance for strict recall measurement.
            "selected_session_ids": sorted({n.session_id for n in edg.nodes.values() if n.node_id in compiled.selected_node_ids}),
            "retrieved_session_ids": sorted({n.session_id for n in edg.nodes.values()}),
        }


# ---------- Evidence Recall Evaluation (No LLM) ----------

def _gold_turn_ids(item: dict) -> set[str] | None:
    """Return turn-level gold ids when the dataset provides them, else None."""
    for key in ("answer_turn_ids", "evidence_turn_ids", "has_answer"):
        val = item.get(key)
        if val:
            return set(map(str, val)) if not isinstance(val, str) else {val}
    return None


def evaluate_evidence_recall(
    pipeline: LabelFreeV4Pipeline,
    dev_items: list[dict],
    budget: int,
) -> dict:
    """
    Evaluate Evidence Recall @ Budget WITHOUT calling LLM.

    Metrics (strict, structural session-id set operations):
      * recall_retrieved: gold sessions present in the candidate evidence graph
        (the current pipeline indexes the full haystack, so this is 1.0 by
        construction; kept explicit so a future retrieval stage is measurable).
      * recall_selected:  gold sessions with >=1 turn selected by the budgeted
        compiler (the L2 bottleneck).
      * gold_turn_recall: turn-level recall when gold turn ids exist, else None.

    For each question:
    1. Predict type (label-free)
    2. Retrieve + compile context
    3. Compare selected session-id set against gold ``answer_session_ids``
    """
    results = []
    error_results = []

    for i, item in enumerate(dev_items):
        qid = item["question_id"]
        question = item["question"]
        gold_sessions = set(map(str, item.get("answer_session_ids", []) or []))
        gold_turns = _gold_turn_ids(item)

        try:
            records, turns = pipeline._turns_to_records_and_ir(
                item["haystack_sessions"],
                item["haystack_dates"],
                item["haystack_session_ids"],
            )

            predicted_type = pipeline._classify(question)
            exec_plan = pipeline.planner.plan(
                question, records, reference_date=item.get("question_date")
            )
            edg = pipeline.graph_builder.build(turns, records)

            coproc_bundle = pipeline.coprocessor.process(
                question=question,
                records=records,
                plan=exec_plan,
                reference_date=item.get("question_date"),
            )

            compiled = pipeline._compile(question, edg, coproc_bundle, exec_plan, predicted_type)

            # Structural session-id sets (no string containment)
            retrieved_sids = {n.session_id for n in edg.nodes.values()}
            selected_sids = {
                edg.nodes[nid].session_id
                for nid in compiled.selected_node_ids
                if nid in edg.nodes
            }

            n_gold = len(gold_sessions)
            recall_retrieved = len(retrieved_sids & gold_sessions) / n_gold if n_gold else 1.0
            recall_selected = len(selected_sids & gold_sessions) / n_gold if n_gold else 1.0

            gold_turn_recall = None
            if gold_turns:
                selected_node_ids = set(compiled.selected_node_ids)
                gold_turn_recall = len(selected_node_ids & gold_turns) / len(gold_turns)

            results.append({
                "question_id": qid,
                "question_type": item.get("question_type", "unknown"),
                "predicted_type": predicted_type,
                "type_mismatch": item.get("question_type", "") != predicted_type,
                "n_gold_sessions": n_gold,
                "n_gold_sessions_retrieved": len(retrieved_sids & gold_sessions),
                "n_gold_sessions_selected": len(selected_sids & gold_sessions),
                "recall_retrieved": recall_retrieved,
                "recall_selected": recall_selected,
                "gold_turn_recall": gold_turn_recall,
                # Legacy combined field = post-budget session recall.
                "recall": recall_selected,
                "budget_used": compiled.total_tokens,
                "budget_limit": budget,
                "n_bundles": len(compiled.selected_bundle_ids),
                "n_nodes": len(compiled.selected_node_ids),
                "deterministic_override": bool(coproc_bundle.deterministic_answer_override),
            })

        except Exception as e:
            # Never swallow into a silent 0.0: record the traceback, count it.
            tb = traceback.format_exc()
            print(f"  [ERROR] qid={qid}: {e}")
            print(tb, file=sys.stderr)
            error_results.append({
                "question_id": qid,
                "question_type": item.get("question_type", "unknown"),
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb,
            })

        if (i + 1) % 50 == 0:
            print(f"  Recall eval: {i+1}/{len(dev_items)}")

    # Aggregate
    recalls = [r["recall_selected"] for r in results]
    recall_mean = float(np.mean(recalls)) if recalls else 0
    recall_median = float(np.median(recalls)) if recalls else 0
    retrieved_mean = float(np.mean([r["recall_retrieved"] for r in results])) if results else 0
    selected_mean = float(np.mean([r["recall_selected"] for r in results])) if results else 0
    turn_vals = [r["gold_turn_recall"] for r in results if r["gold_turn_recall"] is not None]
    turn_mean = float(np.mean(turn_vals)) if turn_vals else None

    # Per-type recall
    type_recall: dict[str, list[dict]] = {}
    for r in results:
        type_recall.setdefault(r.get("question_type", "unknown"), []).append(r)

    type_recall_summary = {
        t: {
            "mean": float(np.mean([x["recall_selected"] for x in v])) if v else 0,
            "median": float(np.median([x["recall_selected"] for x in v])) if v else 0,
            "mean_recall_retrieved": float(np.mean([x["recall_retrieved"] for x in v])) if v else 0,
            "mean_recall_selected": float(np.mean([x["recall_selected"] for x in v])) if v else 0,
            "mean_gold_turn_recall": (
                float(np.mean([x["gold_turn_recall"] for x in v if x["gold_turn_recall"] is not None]))
                if any(x["gold_turn_recall"] is not None for x in v) else None
            ),
            "n": len(v),
        }
        for t, v in type_recall.items()
    }

    return {
        "budget": budget,
        "mean_recall": recall_mean,
        "median_recall": recall_median,
        "mean_recall_retrieved": retrieved_mean,
        "mean_recall_selected": selected_mean,
        "mean_gold_turn_recall": turn_mean,
        "error_count": len(error_results),
        "errors": error_results,
        "per_type_recall": type_recall_summary,
        "individual": results,
    }


# ---------- Helpers ----------

def _bootstrap_ci(values: list[float], n_bootstrap: int = 5000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% CI (scipy-free, deterministic)."""
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def mcnemar_p(prev: list[bool], curr: list[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (stdlib only)."""
    from math import comb
    if len(prev) != len(curr):
        raise ValueError("paired runs must have the same length")
    b = sum(1 for p, c in zip(prev, curr) if p and not c)  # prev correct, curr wrong
    c = sum(1 for p, c in zip(prev, curr) if not p and c)  # prev wrong, curr correct
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "significant": False}
    # Two-sided exact binomial p-value under p=0.5
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    p = min(1.0, p)
    return {"b": b, "c": c, "n_discordant": n, "p_value": p, "significant": p < 0.05}


def _load_prev_results(path: str | None) -> dict[str, bool] | None:
    if not path or not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("results", data) if isinstance(data, dict) else data
    if isinstance(rows, dict):
        rows = rows.get("predictions", [])
    out = {}
    for r in rows:
        if "question_id" in r and "is_correct" in r:
            out[r["question_id"]] = bool(r["is_correct"])
    return out or None


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Phase 0: Label-Free V4 Baseline")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--budget", type=int, default=1300)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N dev questions")
    parser.add_argument("--cache-dir", type=str, default="cache/phase0")
    parser.add_argument("--no-cache", action="store_true", help="Disable the LLM response cache")
    parser.add_argument("--compare-to", type=str, default=None,
                        help="Previous results JSON; reports exact McNemar p vs this run")
    parser.add_argument("--selector", type=str, default="budgeted", choices=["budgeted", "coverage"],
                        help="Evidence selection strategy (coverage is experimental and regressed recall; see coverage_selector.py)")
    parser.add_argument("--soft-routing", action="store_true",
                        help="Phase 2: compose prompts from continuous demand signals instead of hard type branches")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 0: LABEL-FREE V4 BASELINE")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Budget: {args.budget}")
    print(f"Output: {out_dir}")
    print()

    # Load split
    with open(args.split, "r") as f:
        dev_qids = json.load(f)
    if isinstance(dev_qids, dict) and "qids" in dev_qids:
        dev_qids = dev_qids["qids"]

    # Load dataset
    print("Loading full dataset...")
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    ds_by_id = {item["question_id"]: item for item in dataset}

    dev_items = [ds_by_id[qid] for qid in dev_qids if qid in ds_by_id]
    if args.limit > 0:
        dev_items = dev_items[:args.limit]
    print(f"Loaded {len(dev_items)} Dev questions")

    # Initialize pipeline
    pipeline = LabelFreeV4Pipeline(
        token_budget=args.budget,
        cache_dir=args.cache_dir,
        use_llm_cache=not args.no_cache,
        selector=args.selector,
        soft_routing=args.soft_routing,
    )
    print(f"Selector: {args.selector}")
    print(f"Routing: {'soft (Phase 2)' if args.soft_routing else 'hard type branch (Phase 1)'}")

    # Step 1: Classifier evaluation
    print("\n--- Step 1: Label-Free Type Classifier Evaluation ---")
    cls_results = pipeline.classifier.evaluate_on_dev(dev_items)
    print(f"Type classification accuracy: {cls_results['accuracy']:.1%}")
    for qtype, m in sorted(cls_results['per_type'].items()):
        if m['support'] > 0:
            print(f"  {qtype:30s} F1: {m['f1']:.3f}  (n={m['support']})")

    with open(out_dir / "classifier_confusion_matrix.json", "w") as f:
        json.dump({
            "accuracy": cls_results["accuracy"],
            "per_type": cls_results["per_type"],
            "confusion_matrix": cls_results["confusion_matrix"],
            "method": cls_results.get("method", "ensemble_v3"),
        }, f, indent=2, default=str)

    # Step 2: Evidence Recall @ Budget
    print("\n--- Step 2: Evidence Recall @ Budget ---")
    recall_results = evaluate_evidence_recall(pipeline, dev_items, args.budget)
    print(f"  Mean Recall@Retrieved: {recall_results['mean_recall_retrieved']:.1%}")
    print(f"  Mean Recall@Selected:  {recall_results['mean_recall_selected']:.1%}")
    print(f"  Mean Recall@1300 (legacy alias): {recall_results['mean_recall']:.1%}")
    if recall_results["mean_gold_turn_recall"] is not None:
        print(f"  Mean Gold-Turn Recall: {recall_results['mean_gold_turn_recall']:.1%}")
    else:
        print("  Mean Gold-Turn Recall: N/A (dataset has no turn-level gold)")
    print(f"  Recall errors: {recall_results['error_count']}")
    for qtype, m in sorted(recall_results["per_type_recall"].items()):
        print(f"    {qtype:30s} selected: {m['mean_recall_selected']:.1%}  retrieved: {m['mean_recall_retrieved']:.1%}  (n={m['n']})")

    with open(out_dir / "evidence_recall_at_budget.json", "w") as f:
        json.dump(recall_results, f, indent=2, default=str)

    # Step 3: Full pipeline evaluation (if LLM available)
    predictions = []
    accuracy = None
    mean_tokens = None
    acc_ci = tok_ci = None
    type_stats = {}
    type_correct = 0
    error_count = 0
    cache_hits = 0
    overrides = 0
    mcnemar = None

    if not args.skip_llm:
        print("\n--- Step 3: V4 Full Pipeline Evaluation ---")

        # Check LLM availability
        reader_available = False
        try:
            import httpx
            resp = httpx.get(f"{pipeline.reader.base_url}/api/tags", timeout=5)
            reader_available = resp.status_code == 200
        except Exception as e:
            print(f"  (Ollama probe failed: {type(e).__name__}: {e})")

        if not reader_available:
            print("  Ollama reader not available. Skipping LLM evaluation.")
        else:
            print(f"  LLM available ({pipeline.reader.model}), running {len(dev_items)} questions...")

            for i, item in enumerate(dev_items):
                try:
                    result = pipeline.run(
                        question=item["question"],
                        sessions=item["haystack_sessions"],
                        session_dates=item["haystack_dates"],
                        session_ids=item["haystack_session_ids"],
                        reference_date=item.get("question_date"),
                    )

                    # Grade with the official grader. Gold question_type is used
                    # for SCORING ONLY, never for routing.
                    from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader
                    grader = OfficialLongMemEvalGrader()
                    grade_result = grader.grade(
                        question_id=item["question_id"],
                        question_type=item["question_type"],  # Gold, grading only
                        predicted=result["predicted_answer"],
                        ground_truth=item.get("answer", ""),
                    )

                    cache_hits += int(result.get("cache_hit", False))
                    overrides += int(result.get("deterministic_override", False))
                    predictions.append({
                        "question_id": item["question_id"],
                        "question_type": item["question_type"],  # Gold, for grading/scoring
                        "predicted_type": result["predicted_type"],
                        "type_correct": item["question_type"] == result["predicted_type"],
                        "question": item["question"],
                        "predicted_answer": result["predicted_answer"],
                        "ground_truth": item.get("answer", ""),
                        "is_correct": grade_result.is_correct,
                        "status": "ok",
                        "prompt_tokens": result["prompt_tokens"],
                        "context_tokens": result["context_tokens"],
                        "latency_ms": result["latency_ms"],
                        "deterministic_override": result["deterministic_override"],
                        "cache_hit": result.get("cache_hit", False),
                    })

                except Exception as e:
                    # Fail loudly: keep the row (counted as incorrect) AND record
                    # status + traceback so the error count is explicit.
                    tb = traceback.format_exc()
                    print(f"  [ERROR] {item['question_id']}: {type(e).__name__}: {e}")
                    print(tb, file=sys.stderr)
                    error_count += 1
                    predictions.append({
                        "question_id": item["question_id"],
                        "question_type": item["question_type"],
                        "predicted_type": "error",
                        "type_correct": False,
                        "question": item["question"],
                        "predicted_answer": "",
                        "ground_truth": item.get("answer", ""),
                        "is_correct": False,
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": tb,
                    })

                if (i + 1) % 25 == 0:
                    correct_so_far = sum(1 for p in predictions if p.get("is_correct"))
                    print(f"  {i+1}/{len(dev_items)} ({correct_so_far}/{i+1} correct, {error_count} errors)")

            # Save predictions
            with open(out_dir / "lme_dev250_v4_label_free.json", "w") as f:
                json.dump(predictions, f, indent=2, default=str)

            # Compute metrics
            total = len(predictions)
            correct = sum(1 for p in predictions if p["is_correct"])
            accuracy = correct / total if total else 0.0
            prompt_tokens = [p["prompt_tokens"] for p in predictions if p.get("prompt_tokens")]
            mean_tokens = float(np.mean(prompt_tokens)) if prompt_tokens else 0.0

            print(f"\n  Accuracy: {accuracy:.1%} ({correct}/{total})")
            print(f"  Mean tokens: {mean_tokens:.0f}")
            print(f"  Errors: {error_count}")

            # Per-type
            print("\n  Per-Type:")
            for p in predictions:
                qtype = p["question_type"]
                s = type_stats.setdefault(qtype, {"correct": 0, "total": 0, "tokens": [], "errors": 0})
                s["total"] += 1
                s["tokens"].append(p.get("prompt_tokens", 0))
                if p["is_correct"]:
                    s["correct"] += 1
                if p.get("status") == "error":
                    s["errors"] += 1

            for qtype, s in sorted(type_stats.items()):
                acc = s["correct"] / s["total"] if s["total"] else 0.0
                tok = float(np.mean(s["tokens"])) if s["tokens"] else 0.0
                print(f"    {qtype:30s} {acc:.1%} ({s['correct']}/{s['total']})  {tok:.0f} tok")

            type_correct = sum(1 for p in predictions if p.get("type_correct"))
            print(f"\n  Type prediction accuracy: {type_correct}/{total} ({type_correct/total:.1%})")

            # Bootstrap CI (stdlib implementation; matches StatisticalTester)
            acc_vals = [float(p["is_correct"]) for p in predictions]
            acc_ci = _bootstrap_ci(acc_vals, args.n_bootstrap)
            tok_ci = _bootstrap_ci([float(p.get("prompt_tokens", 0)) for p in predictions], args.n_bootstrap) if prompt_tokens else None
            print("\n  95% CI:")
            print(f"    Accuracy: [{acc_ci[0]:.1%}, {acc_ci[1]:.1%}]")
            if tok_ci:
                print(f"    Tokens: [{tok_ci[0]:.0f}, {tok_ci[1]:.0f}]")

            # McNemar vs previous run (1 change = 1 experiment)
            prev = _load_prev_results(args.compare_to)
            if prev:
                paired_prev, paired_curr, used = [], [], 0
                for p in predictions:
                    if p["question_id"] in prev:
                        paired_prev.append(prev[p["question_id"]])
                        paired_curr.append(bool(p["is_correct"]))
                        used += 1
                if used:
                    mcnemar = mcnemar_p(paired_prev, paired_curr)
                    mcnemar["n_paired"] = used
                    mcnemar["compare_to"] = args.compare_to
                    print(f"\n  McNemar vs {args.compare_to} (n={used}): "
                          f"b={mcnemar['b']} c={mcnemar['c']} p={mcnemar['p_value']:.4f} "
                          f"({'significant' if mcnemar['significant'] else 'n.s.'})")

    # Step 4: Generate report
    print("\n--- Step 4: Generating phase0_report.md ---")

    acc_ci_txt = f"[{acc_ci[0]:.1%}, {acc_ci[1]:.1%}]" if acc_ci else "N/A"
    tok_ci_txt = f"[{tok_ci[0]:.0f}, {tok_ci[1]:.0f}]" if tok_ci else "N/A"
    err_note = f"{error_count}" if not args.skip_llm else "N/A (skipped)"
    acc_txt = f"{accuracy:.1%}" if accuracy is not None else "N/A"
    tok_txt = f"{mean_tokens:.0f}" if mean_tokens is not None else "N/A"
    acc_per_1k = f"{accuracy * 1000 / mean_tokens:.2f}" if (accuracy is not None and mean_tokens) else "N/A"

    report = f"""# Phase 0 Baseline Report: Label-Free V4 (leak-free)

## Configuration
- Dataset: `{args.dataset}`
- Split: Dev ({len(dev_items)} questions)
- Token Budget: {args.budget} tokens
- Reader Model: {pipeline.reader.model}
- Type Classifier: {cls_results.get("method", "ensemble_v3")}
- LLM cache: `{args.cache_dir}` ({cache_hits} cache hits)

## Leakage Boundary (regression-tested)
- NO gold `question_type` used in routing or inference
- NO gold `answer` or `answer_session_ids` used in prompt construction
- Label-free classifier uses question text only
- Gold labels enter ONLY the grader (scoring) and offline recall measurement
- Holdout split (`lme_holdout_qids.json`) never accessed
- Guarded by `tests/unit/test_no_label_leak.py`

## 1. Label-Free Type Classification
- **Accuracy: {cls_results['accuracy']:.1%}** ({cls_results['correct']}/{cls_results['total']})

### Per-Type Classification Metrics
| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|----|---------|
"""
    for qtype, m in sorted(cls_results['per_type'].items()):
        if m['support'] > 0:
            report += f"| {qtype} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.3f} | {m['support']} |\n"

    report += f"""
## 2. Evidence Recall @ Budget (structural session-id sets)
- **Mean Recall@Retrieved (candidate pool):** {recall_results['mean_recall_retrieved']:.1%}
- **Mean Recall@Selected (post-budget):** {recall_results['mean_recall_selected']:.1%}
- Mean Gold-Turn Recall: {f"{recall_results['mean_gold_turn_recall']:.1%}" if recall_results['mean_gold_turn_recall'] is not None else "N/A (no turn-level gold)"}
- Recall evaluation errors: {recall_results['error_count']}

> Note: the current pipeline indexes the full haystack into the evidence graph,
> so `recall_retrieved` is 1.0 by construction and is reported for completeness.
> `recall_selected` (the L2 bottleneck) is the meaningful signal.

### Per-Type Recall
| Type | Recall@Retrieved | Recall@Selected | N |
|------|------------------|-----------------|----|
"""
    for qtype, m in sorted(recall_results["per_type_recall"].items()):
        report += f"| {qtype} | {m['mean_recall_retrieved']:.1%} | {m['mean_recall_selected']:.1%} | {m['n']} |\n"

    if predictions:
        report += f"""
## 3. V4 Pipeline Results (Label-Free, leak-free)

### Overall Metrics
- **Accuracy: {acc_txt}**
- **Mean Prompt Tokens: {tok_txt}**
- **Acc/1k tokens: {acc_per_1k}**  (Acc / (tokens/1000))
- **Errors: {err_note}** (recorded with `status="error"` + traceback, never silently dropped)

### 95% Bootstrap Confidence Intervals
- Accuracy: {acc_ci_txt}
- Tokens: {tok_ci_txt}

### Per-Type Pipeline Results
| Type | Accuracy | Correct | Total | Mean Tokens | Errors |
|------|----------|---------|-------|-------------|--------|
"""
        for qtype, s in sorted(type_stats.items()):
            acc = s["correct"] / s["total"] if s["total"] else 0.0
            tok = float(np.mean(s["tokens"])) if s["tokens"] else 0.0
            report += f"| {qtype} | {acc:.1%} | {s['correct']} | {s['total']} | {tok:.0f} | {s['errors']} |\n"

        report += f"""
### Type Prediction Accuracy
- **{type_correct}/{len(predictions)} ({type_correct/len(predictions):.1%})** classified correctly

### Deterministic 0-token Overrides
- {overrides}/{len(predictions)} questions answered by certificates without a reader call
"""
        if mcnemar:
            report += f"""
## 4. McNemar Test vs `{mcnemar.get('compare_to')}`
- Paired n = {mcnemar.get('n_paired')}, b = {mcnemar['b']}, c = {mcnemar['c']}
- Exact two-sided p = {mcnemar['p_value']:.4f} ({'significant' if mcnemar['significant'] else 'not significant'})
"""

    with open(out_dir / "phase0_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*70}")
    print("PHASE 0 COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {out_dir}")
    print("  - classifier_confusion_matrix.json")
    print("  - evidence_recall_at_budget.json")
    if predictions:
        print("  - lme_dev250_v4_label_free.json")
    print("  - phase0_report.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
