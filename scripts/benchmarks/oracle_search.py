"""ORACLE SEARCH -- upper bound on what better retrieval could buy.

Purpose is NOT to raise the score.  It is to answer: if retrieval were perfect,
how many of these questions would the CURRENT reader still get wrong?  That
splits the failure population into
  A  fixed by evidence alone        -> retrieval is worth investing in
  B  still wrong with gold evidence -> reader / reasoning / scoring
and either answer sets the next research direction.

METHOD, and the constraints that make the comparison valid:
  * the gold ANSWER is never shown to the model -- only the evidence turn
  * same model, same prompt builder, same scorer as the normal run
  * the normal retrieval result is KEPT; the gold evidence is APPENDED
  * per-question token cost of the appended evidence is recorded

Population: the 1,455 questions whose annotation genuinely supports the answer
(diag_reconcile_audits.py).  The 213 unrelated-annotation questions are excluded
because forcing their evidence in tests tolerance of irrelevant text, not
retrieval.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from eval_v4_pipeline_dev import V4CandidateRetriever  # noqa: E402
from artificial_memory.research.benchmarks.llm import OllamaAnswerer  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"

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


def is_temporal_gold(gold: str) -> bool:
    g = str(gold).lower().strip()
    if re.search(r"\b(19|20)\d{2}\b", g):
        return True
    if re.search(r"\b(january|february|march|april|may|june|july|august|"
                 r"september|october|november|december)\b", g):
        return True
    return bool(re.search(r"\b(monday|tuesday|wednesday|thursday|friday|"
                          r"saturday|sunday)\b", g))


def build_jobs(data, limit: int):
    """Questions whose annotation supports the answer AND whose evidence is
    already in the retrieved pool -- i.e. the oracle changes nothing for them.

    Wait: that would make the experiment vacuous.  Include ALL annotation-valid
    questions, and record separately whether the evidence was already present,
    so the two sub-populations can be compared.
    """
    jobs = []
    for conv in data:
        turns, dia_map = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        for q in E.load_questions(conv):
            if is_temporal_gold(q["answer"]):
                continue
            gold_words = content(q["answer"])
            ev = [e for e in (q.get("evidence") or []) if dia_map.get(e)]
            if not ev or not gold_words:
                continue
            ev_words = set()
            for e in ev:
                ev_words |= content(dia_map[e])
            if not (ev_words & gold_words):
                continue          # annotation unrelated -> not a retrieval test
            jobs.append((conv, q, turns, dia_map, ev))
            if limit and len(jobs) >= limit:
                return jobs
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/external/locomo10.json")
    ap.add_argument("--limit", type=int, default=120,
                    help="questions to run; 0 = all 1455")
    ap.add_argument("--num-ctx", type=int, default=8192)
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    jobs = build_jobs(data, args.limit)
    print(f"Oracle Search: {len(jobs)} annotation-valid questions")
    print("Normal retrieval vs normal-retrieval+gold-evidence, same reader.\n")

    retriever = V4CandidateRetriever()
    answerer = OllamaAnswerer(num_ctx=args.num_ctx)

    rows = []
    t0 = time.perf_counter()
    for i, (conv, q, turns, dia_map, ev) in enumerate(jobs):
        top, _ = retriever.rank_turns(q["question"], turns, top_k=45)
        cand = list(top)

        # ---- normal arm ------------------------------------------------
        try:
            ctx_norm = E._compile_context(q, cand) if hasattr(E, "_compile_context") else None
        except Exception:
            ctx_norm = None
        if ctx_norm is None:
            # replicate the harness compile path
            from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner
            from artificial_memory.compiler.graph.graph_builder import GraphBuilder
            from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler
            plan = CognitiveExecutionPlanner().plan(q["question"], [])
            t_map = {(t[0], t[3]): f"{t[0]}:t{k}" for k, t in enumerate(cand)}
            anchors = [t_map[(t[0], t[3])] for t in cand[:12]
                       if (t[0], t[3]) in t_map]
            edg = GraphBuilder().build(cand, [])
            bundles = edg.form_bundles(anchors, max_bundle_tokens=150)
            budget = E._scaled_budget(
                plan.recommended_budget,
                sum(len(t[3].split()) * 4 // 3 for t in cand))
            out = BudgetedCognitiveCompiler().compile(
                question=q["question"], bundles=bundles, token_budget=budget,
                anchor_scores={a: 5.0 for a in anchors},
                is_multi_session=len({t[0] for t in cand}) > 1,
                adaptive_escalation=True)
            ctx_norm = out.context_text

        # ---- oracle arm: append the gold EVIDENCE (never the answer) ----
        gold_lines = []
        for e in ev:
            txt = dia_map[e]
            date = next((t[1] for t in turns if t[3] == txt), "")
            spk = next((t[2] for t in turns if t[3] == txt), "")
            gold_lines.append(f"[ORACLE EVIDENCE on {date}] {spk}: {txt}")
        ctx_oracle = ctx_norm + "\n" + "\n".join(gold_lines)
        extra_tokens = sum(len(l.split()) * 4 // 3 for l in gold_lines)

        def ask(context: str) -> tuple[str, bool]:
            try:
                pred = answerer.answer(
                    q["question"],
                    E.build_reader_prompt(q["question"], context)).text.strip()
            except Exception as exc:
                pred = f"<ERROR: {type(exc).__name__}>"
            if pred.endswith(".") and len(pred.split()) <= 2:
                pred = pred[:-1].strip()
            return pred, E.score(pred, q["answer"])

        p_norm, ok_norm = ask(ctx_norm)
        p_orc, ok_orc = ask(ctx_oracle)
        was_present = any(
            w in ctx_norm.lower() for e in ev for w in list(content(dia_map[e]))[:3])

        rows.append({
            "question_id": q["question_id"], "category": q["category"],
            "question": q["question"], "ground_truth": q["answer"],
            "pred_normal": p_norm, "ok_normal": ok_norm,
            "pred_oracle": p_orc, "ok_oracle": ok_orc,
            "evidence_already_present": was_present,
            "extra_tokens": extra_tokens,
        })

        if (i + 1) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"[{i+1}/{len(jobs)}] {el:.0f}s  "
                  f"ETA {el/(i+1)*(len(jobs)-i-1)/60:.1f}min")

    # ---- report --------------------------------------------------------
    fixed = [r for r in rows if r["ok_oracle"] and not r["ok_normal"]]
    broke = [r for r in rows if r["ok_normal"] and not r["ok_oracle"]]
    n = len(rows)
    acc_n = sum(1 for r in rows if r["ok_normal"]) / n * 100
    acc_o = sum(1 for r in rows if r["ok_oracle"]) / n * 100

    print()
    print("=" * 74)
    print(f"ORACLE SEARCH RESULT  (n={n})")
    print("=" * 74)
    print(f"  normal retrieval        : {acc_n:5.1f}%")
    print(f"  retrieval + gold evidence: {acc_o:5.1f}%   ({acc_o-acc_n:+.1f}pt)")
    print(f"  fixed by evidence       : {len(fixed)}")
    print(f"  broken by evidence      : {len(broke)}")
    print(f"  net                     : {len(fixed)-len(broke):+d}")
    print()
    by_cat = {}
    for c in sorted({r["category"] for r in rows}):
        sel = [r for r in rows if r["category"] == c]
        if not sel:
            continue
        a = sum(1 for r in sel if r["ok_normal"]) / len(sel) * 100
        b = sum(1 for r in sel if r["ok_oracle"]) / len(sel) * 100
        by_cat[str(c)] = {"normal": round(a, 1), "oracle": round(b, 1)}
        print(f"  cat{c}  n={len(sel):4d}  {a:5.1f}% -> {b:5.1f}%  ({b-a:+.1f}pt)")
    print()
    still_wrong = [r for r in rows if not r["ok_oracle"]]
    print(f"STILL WRONG WITH GOLD EVIDENCE: {len(still_wrong)} "
          f"({len(still_wrong)/n*100:.1f}%)")
    print("  -> these are reader / reasoning / scoring failures, not retrieval.")
    print()
    if len(still_wrong) > len(fixed):
        print("VERDICT: even with perfect evidence most failures persist.")
        print("  Reader-side work is justified; retrieval is not the lever.")
    else:
        print("VERDICT: evidence alone fixes more than it leaves broken.")
        print("  Retrieval-side work is justified.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / "benchmark_results" / f"oracle_search_{stamp}.json"
    out.write_text(json.dumps({
        "run_stamp": stamp, "n": n,
        "acc_normal": acc_n, "acc_oracle": acc_o,
        "fixed": len(fixed), "broke": len(broke),
        "still_wrong_with_gold": len(still_wrong),
        "per_category": by_cat, "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
