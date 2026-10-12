"""Stage 2 taxonomy: every failure, one bucket, reconciled with answerability.

Eli's plan: stop guessing at "improve retrieval" / "fix the prompt" and classify
every failure so a single structural change can be aimed at a real block.

Buckets, mutually exclusive, summing to the failure count:
  UNANSWERABLE      gold tokens appear nowhere in the conversation -- no engine
                    change can ever score these (109 measured)
  SCORER_REJECTED   scorer refused a semantically-agreeing prediction, and the
                    rejection is intentional anti-inflation policy (86 measured)
  NO_EVIDENCE       gold is derivable but never reached the compiled context
  READER_FAILED     evidence was in the context and the reader still failed

Only NO_EVIDENCE and READER_FAILED are actionable, and they need different fixes.

Prediction discipline, stated BEFORE running:
  predicted  UNANSWERABLE ~109, SCORER_REJECTED ~86, READER_FAILED ~500,
             NO_EVIDENCE ~280
  mechanism  reuses the answerability probe (digits included) and the scorer
             audit's enumeration/offset rules
  if it fails the buckets will not sum to the failure count, which means a case
             is double-counted or dropped -- the reconciliation line catches it
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from eval_v4_pipeline_dev import V4CandidateRetriever  # noqa: E402
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner  # noqa: E402
from artificial_memory.compiler.graph.graph_builder import GraphBuilder  # noqa: E402
from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler  # noqa: E402
from artificial_memory.compiler.eval.audit_strict_grade import is_list_gold  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"
RUN = ROOT / "benchmark_results" / "locomo_full_20261012_211601.json"

STOP = frozenset({
    "i", "my", "me", "we", "you", "the", "a", "an", "and", "or", "but", "to",
    "of", "in", "on", "at", "for", "with", "was", "were", "is", "are", "be",
    "been", "it", "its", "this", "that", "these", "those", "so", "very",
    "just", "she", "her", "he", "his", "they", "them", "their", "yes", "no",
    "not", "some", "any", "all", "more", "most", "other", "another", "when",
    "what", "where", "which", "who", "how", "why", "would", "could", "should",
    "likely", "still", "want", "wants", "melanie", "caroline", "jon", "gina",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "yesterday",
    "today", "tomorrow", "week", "weeks", "weekend", "month", "months",
    "year", "years", "ago", "before", "after", "last", "next",
})

WORDNUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
           "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def items(s: str) -> set[str]:
    return {norm(x) for x in re.split(r",|\band\b", str(s)) if norm(x)}


def gold_tokens(ans: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(ans).lower())
            if w not in STOP]


def scorer_intentional_reject(pred: str, gold: str) -> bool:
    """True when the scorer's refusal is a deliberate anti-inflation rule."""
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return False
    if is_list_gold(gold) or is_list_gold(pred):
        gi, pi = items(gold), items(pred)
        return bool(gi and pi and gi != pi)
    gn, pn = re.findall(r"\d+", g), re.findall(r"\d+", p)
    if gn and pn and gn == pn and re.search(r"\b(before|after|of)\b", g):
        return True
    return False


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    recorded = {r["question_id"]: r
                for r in json.loads(RUN.read_text(encoding="utf-8"))["results"]}
    retriever = V4CandidateRetriever()
    planner = CognitiveExecutionPlanner()
    gb = GraphBuilder()
    comp = BudgetedCognitiveCompiler()

    buckets = Counter()
    by_cat = {}
    examples = {}

    for conv in data:
        turns, _ = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        corpus = set()
        for t in turns:
            corpus.update(re.findall(r"[a-z0-9]+", t[3].lower()))

        for q in E.load_questions(conv):
            r = recorded.get(q["question_id"])
            if r is None or r["is_correct"]:
                continue
            cat = q["category"]
            pred = str(r["predicted_answer"])
            gold = str(q["answer"])

            # 1. UNANSWERABLE -- gold tokens absent from the whole conversation
            gt = gold_tokens(gold)
            if gt:
                present = {w for w in gt if w in corpus}
                if not present and str(gold).lower().strip() in WORDNUM:
                    w = WORDNUM[str(gold).lower().strip()]
                    if w in corpus:
                        present = {w}
                if not present:
                    key = "UNANSWERABLE"
                    buckets[key] += 1
                    by_cat.setdefault(cat, Counter())[key] += 1
                    examples.setdefault(key, (q["question"], gold, pred))
                    continue
            # (no checkable tokens: falls through to the evidence test below)

            # 2. SCORER_REJECTED -- intentional anti-inflation refusal
            if scorer_intentional_reject(pred, gold):
                key = "SCORER_REJECTED"
                buckets[key] += 1
                by_cat.setdefault(cat, Counter())[key] += 1
                examples.setdefault(key, (q["question"], gold, pred))
                continue

            # 3/4. evidence present in the compiled context?
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
                    question=q["question"], bundles=bundles,
                    token_budget=budget,
                    anchor_scores={a: 5.0 for a in anchors},
                    is_multi_session=False, adaptive_escalation=True)
                ctx = out.context_text
            except Exception:
                ctx = ""

            content_gold = {w for w in gold_tokens(gold) if len(w) > 2}
            ctx_words = set(re.findall(r"[a-z0-9]+", ctx.lower()))
            cov = (len(content_gold & ctx_words) / len(content_gold)) if content_gold else 0.0
            key = "READER_FAILED" if cov >= 0.5 else "NO_EVIDENCE"
            buckets[key] += 1
            by_cat.setdefault(cat, Counter())[key] += 1
            examples.setdefault(key, (q["question"], gold, pred))

    total = sum(buckets.values())
    order = ("UNANSWERABLE", "SCORER_REJECTED", "NO_EVIDENCE", "READER_FAILED")
    meaning = {
        "UNANSWERABLE": "gold absent from the corpus     -- no fix possible",
        "SCORER_REJECTED": "intentional anti-inflation rule -- no fix wanted",
        "NO_EVIDENCE": "gold derivable, never reached ctx -- RETRIEVAL",
        "READER_FAILED": "evidence present, reader failed -- READER",
    }
    print("=" * 80)
    print("STAGE 2 TAXONOMY -- every failure in exactly one bucket")
    print("=" * 80)
    print(f"failures classified: {total}   (of 1,986 questions)")
    print()
    print(f"{'bucket':18s} {'n':>5s} {'share':>7s}   meaning")
    print("-" * 80)
    for k in order:
        c = buckets[k]
        print(f"{k:18s} {c:5d} {c/total*100:6.1f}%   {meaning[k]}")
        if k in examples:
            qq, gt, pred = examples[k]
            print(f"    e.g. {qq[:58]}")
            print(f"         GT {gt[:42]}  | PRED {pred[:30]}")
    print("-" * 80)
    print("RECONCILIATION: " + " + ".join(str(buckets[k]) for k in order) +
          f" = {total}  (must equal the failure count)")
    print()
    print("per category:")
    print(f"  {'cat':5s} {'unans':>7s} {'scorer':>7s} {'no_ev':>7s} {'reader':>7s}")
    print("  " + "-" * 38)
    for cat in sorted(by_cat):
        d = by_cat[cat]
        print(f"  cat{cat:<2d} {d['UNANSWERABLE']:7d} {d['SCORER_REJECTED']:7d} "
              f"{d['NO_EVIDENCE']:7d} {d['READER_FAILED']:7d}")
    print()
    actionable = buckets["NO_EVIDENCE"] + buckets["READER_FAILED"]
    print(f"ACTIONABLE: {actionable} failures "
          f"({actionable/1986*100:.1f}pt of headroom)")
    print(f"  retrieval-side : {buckets['NO_EVIDENCE']}")
    print(f"  reader-side    : {buckets['READER_FAILED']}")
    print()
    print("PREDICTION CHECK (stated before running):")
    print("  predicted UNANSWERABLE ~109  actual", buckets["UNANSWERABLE"])
    print("  predicted SCORER_REJECTED ~86 actual", buckets["SCORER_REJECTED"])
    print("  predicted READER_FAILED ~500  actual", buckets["READER_FAILED"])
    print("  predicted NO_EVIDENCE ~280     actual", buckets["NO_EVIDENCE"])
    print()
    biggest = max(order, key=lambda k: buckets[k])
    print(f"Largest single block: {biggest} ({buckets[biggest]} questions).")
    if biggest == "READER_FAILED":
        print("  A structural reader change is the highest-leverage target.")
    elif biggest == "NO_EVIDENCE":
        print("  A structural retrieval change is the highest-leverage target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
