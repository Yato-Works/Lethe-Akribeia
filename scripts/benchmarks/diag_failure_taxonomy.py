"""Classify every failure into evidence / reader / scoring, per question.

Eli's correction: "83.8% evidence coverage, so the rest is the reader" is a
logical jump.  16.2% of 1,986 is ~322 questions with no explanation at all, and
they could be any of:
  NO_EVIDENCE     the gold evidence is not in the compiled context
  HAS_EVIDENCE    it is there and the reader still failed
  SCORING_SUSPECT prediction and gold agree semantically but the scorer said no

This walks the real pipeline per question, rebuilds the context, and files every
failure into exactly one bucket.  No LLM calls.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from eval_v4_pipeline_dev import V4CandidateRetriever  # noqa: E402
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner  # noqa: E402
from artificial_memory.compiler.graph.graph_builder import GraphBuilder  # noqa: E402
from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"
RUN = ROOT / "benchmark_results" / "locomo_full_20261012_211601.json"

STOP = frozenset({
    "i", "im", "ive", "id", "ill", "my", "me", "mine", "we", "our", "us",
    "you", "your", "the", "a", "an", "and", "or", "but", "to", "of", "in",
    "on", "at", "for", "with", "was", "were", "is", "are", "be", "been",
    "it", "its", "this", "that", "these", "those", "so", "very", "just",
    "went", "go", "going", "had", "has", "have", "did", "do", "does",
    "when", "what", "where", "which", "who", "how", "why", "would", "could",
    "should", "likely", "still", "want", "wants", "she", "her", "he", "his",
    "they", "them", "their", "melanie", "caroline", "jon", "gina",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "yesterday",
    "today", "tomorrow", "week", "weeks", "weekend", "month", "months",
    "year", "years", "ago", "before", "after", "last", "next",
})


def content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower())
            if w not in STOP and len(w) > 2 and not w.isdigit()}


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    recorded = {r["question_id"]: r
                for r in json.loads(RUN.read_text(encoding="utf-8"))["results"]}
    retriever = V4CandidateRetriever()
    planner = CognitiveExecutionPlanner()
    gb = GraphBuilder()
    comp = BudgetedCognitiveCompiler()

    buckets = Counter()
    by_cat = defaultdict(Counter)
    examples = {}

    for conv in data:
        turns, _ = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        for q in E.load_questions(conv):
            r = recorded.get(q["question_id"])
            if r is None or r["is_correct"]:
                continue                      # only failures
            cat = q["category"]
            pred = str(r["predicted_answer"])
            gold = str(q["answer"])
            gold_words = content(gold)

            top, _ = retriever.rank_turns(q["question"], turns, top_k=45)
            plan = planner.plan(q["question"], [])
            t_map = {(t[0], t[3]): f"{t[0]}:t{k}" for k, t in enumerate(top)}
            anchors = [t_map[(t[0], t[3])] for t in top[:12]
                       if (t[0], t[3]) in t_map]
            edg = gb.build(list(top), [])
            bundles = edg.form_bundles(anchors, max_bundle_tokens=150)
            budget = E._scaled_budget(
                plan.recommended_budget,
                sum(len(t[3].split()) * 4 // 3 for t in top))
            try:
                out = comp.compile(
                    question=q["question"], bundles=bundles, token_budget=budget,
                    anchor_scores={a: 5.0 for a in anchors},
                    is_multi_session=False, adaptive_escalation=True)
                ctx = out.context_text
            except Exception:
                ctx = ""

            ctx_words = set(re.findall(r"[a-z0-9]+", ctx.lower()))
            cov = (len(gold_words & ctx_words) / len(gold_words)) if gold_words else 1.0

            # SCORING_SUSPECT: high token overlap either way, scorer said no
            pg = set(norm(pred).split()) & set(norm(gold).split())
            suspect = (norm(pred) and norm(gold) and
                       len(pg) / max(1, min(len(norm(pred).split()),
                                            len(norm(gold).split()))) >= 0.6)

            if suspect:
                key = "SCORING_SUSPECT"
            elif cov >= 0.5:
                key = "HAS_EVIDENCE_reader_failed"
            else:
                key = "NO_EVIDENCE"
            buckets[key] += 1
            by_cat[cat][key] += 1
            if key not in examples:
                examples[key] = (q["question"], gold, pred, round(cov, 2))

    total = sum(buckets.values())
    print("=" * 78)
    print("EVERY FAILURE CLASSIFIED -- evidence / reader / scoring")
    print("=" * 78)
    print(f"total failures: {total}   (of 1,986 questions)")
    print()
    labels = {
        "NO_EVIDENCE": "gold evidence NOT in context   -> retrieval",
        "HAS_EVIDENCE_reader_failed": "evidence present, reader failed -> reader",
        "SCORING_SUSPECT": "semantically agrees, scorer said no -> scoring",
    }
    for k in ("HAS_EVIDENCE_reader_failed", "NO_EVIDENCE", "SCORING_SUSPECT"):
        c = buckets[k]
        print(f"  {labels[k]:48s} {c:4d}  ({c/total*100:5.1f}%)")
        if k in examples:
            qq, gt, pred, cov = examples[k]
            print(f"      e.g. {qq[:58]}")
            print(f"           GT {gt[:40]}  | PRED {pred[:28]}  | cov {cov}")
    print()
    print("per category:")
    print(f"  {'cat':5s} {'n':>5s} {'reader':>8s} {'retrieval':>11s} {'scoring':>9s}")
    print("  " + "-" * 42)
    for cat in sorted(by_cat):
        d = by_cat[cat]
        n = sum(d.values())
        print(f"  cat{cat:<2d} {n:5d} {d['HAS_EVIDENCE_reader_failed']:8d} "
              f"{d['NO_EVIDENCE']:11d} {d['SCORING_SUSPECT']:9d}")
    print()
    retr = buckets["NO_EVIDENCE"]
    print(f"THE 322-QUESTION QUESTION, ANSWERED: {total} failures split as above.")
    print(f"Only {retr} of them ({retr/total*100:.1f}%) are evidence-absence, so")
    print("retrieval work has that ceiling -- and 'the reader is to blame' covers")
    print(f"only {buckets['HAS_EVIDENCE_reader_failed']} of them "
          f"({buckets['HAS_EVIDENCE_reader_failed']/total*100:.1f}%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
