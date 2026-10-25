"""Probe: does a HYBRID context (R7 compile + lexical top-k) reach more gold?

0 LLM calls.  The two channels are measured as complementary (AM_current
probe: R7 reach 68.9% vs lexical 75%, but opposite category profiles --
V4_PROGRAM.md 4.13 / am_current probe).  This probe prices the union at the
context level, inside the same 2,000-token arena contract:

    ctx = R7 compiled context (budget A) + lexical top-K turns dedup.

Variants measured (all 0 LLM):
    r7_only_A     : R7 compiled alone at budget A
    r7_only_2000  : R7 compiled alone at the full budget (reference)
    lex_only      : the 20 top IDF-weighted turns alone (reference)
    hybrid_A      : R7@A first, lexical turns fill the remainder

Run:
    python scripts/benchmarks/probe_hybrid_ctx.py [--budget-a 1000] [--top-k 20]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from artificial_memory.research.benchmarks.arena import ARENA_VERSION, build_dataset  # noqa: E402
from am_current_player import (  # noqa: E402
    build_pipeline,
    compile_r7_context,
    fill_context_from_lines,
    hybrid_context,
)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "e"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]{3,}", text.lower())}


def lexical_top_k(query: str, turns: list, top_k: int) -> list:
    """The runtime's IDF-weighted overlap, applied to raw pipeline turns."""
    docs = [_terms(f"{t[2]}: {t[3]}") for t in turns]
    df: Counter = Counter()
    for doc in docs:
        df.update(doc)
    n = max(1, len(docs))
    query_terms = _terms(query)
    max_idf = sum(math.log((n + 1) / (df[t] + 1)) for t in query_terms) or 1.0
    scored = []
    for turn, doc in zip(turns, docs):
        raw = sum(math.log((n + 1) / (df[t] + 1)) for t in query_terms & doc)
        scored.append((raw, turn))
    scored.sort(key=lambda p: (-p[0], p[1][0], p[1][3]))
    return [t for raw, t in scored[:top_k] if raw > 0]


def render_line(turn) -> str:
    return f"[{turn[0]}] {turn[2]}: {turn[3]}"


def est_tokens(text: str) -> int:
    return len(text) // 4


def main() -> int:
    parser = argparse.ArgumentParser(description="hybrid context reachability probe")
    parser.add_argument("--budget-a", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default="benchmark_results/am_current/probe_hybrid.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    dataset = build_dataset(version=ARENA_VERSION, created_at=datetime.now().isoformat())
    all_turns = []
    for scenario in dataset.scenarios:
        for turn in scenario.turns:
            all_turns.append((scenario.scenario_id, str(scenario.day),
                              turn.speaker, turn.content))
    pipeline = build_pipeline()
    fact_index = pipeline[5]
    fact_index.build(all_turns)

    # The lexical runtime stack, exactly as the AM v0.2.0 player drives it
    # (user turns only, facade.remember synchronously).
    import asyncio
    from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig
    lex_runtime = ArtificialMemoryRuntime(RuntimeConfig(
        database_path=":memory:", memory_files_path=None, vector_index_path=None,
        retrieval_strategy="classic", lexical_recall=True))
    for scenario in dataset.scenarios:
        for turn in scenario.turns:
            if turn.speaker == "user":
                asyncio.run(lex_runtime.remember(turn.content, topic="Arena"))

    per_cat: dict = {}
    rows = []
    t0 = time.perf_counter()
    lex_cache = {id(q) if False else q.question_id: lexical_top_k(q.question, all_turns, args.top_k)
                 for q in dataset.questions}
    elapsed_lex = time.perf_counter() - t0
    print(f"lexical ranking: {elapsed_lex:.1f}s total")

    def reach(ctx: str, question) -> bool:
        low = ctx.lower()
        groups = [any(term.lower() in low for term in group)
                  for group in question.ground_truth]
        return bool(groups) and all(groups)

    def forb_hit(ctx: str, question) -> int:
        """Negative-evidence dimension: forbidden (decoy) terms in context.

        On abstention questions this is precisely the planted temptation
        that flips a correct refusal into a fluent wrong answer; on
        distractor questions it marks a decoy line riding along with the
        gold.  Uses ONLY the official scorer's own term vocabulary.
        """
        low = ctx.lower()
        return sum(1 for term in question.forbidden if term.lower() in low)

    t1 = time.perf_counter()
    for question in dataset.questions:
        cat = question.category.value
        ctx_full, _ = compile_r7_context(
            pipeline, all_turns, question.question, arena_budget=2000,
            fact_index=fact_index)
        ctx_a, _ = compile_r7_context(
            pipeline, all_turns, question.question, arena_budget=args.budget_a,
            fact_index=fact_index)

        # runtime-fill: the lexical runtime's OWN recall stack fills the rest
        remaining = 2000 - est_tokens(ctx_a)
        rt_result = asyncio.run(lex_runtime.recall(
            question.question, topic="Arena", level=2, max_tokens=remaining))
        fill_lines = [m.semantic_content.content for m in rt_result.memories]
        ctx_rt, rt_meta = fill_context_from_lines(ctx_a, fill_lines, arena_budget=2000)

        # hybrid: R7@A first, lexical turns appended until the arena budget
        used = est_tokens(ctx_a)
        hybrid_lines = [ctx_a]
        seen_texts = set()
        top = lex_cache[question.question_id]
        for turn in top:
            line = render_line(turn)
            key = turn[3].strip().lower()
            if key in seen_texts or key in ctx_a.lower():
                continue
            seen_texts.add(key)
            cost = est_tokens(line) + 1
            if used + cost > 2000:
                break
            hybrid_lines.append(line)
            used += cost
        ctx_hybrid = "\n".join(hybrid_lines)

        # lexical-only reference: the same rendered lines under the full budget
        used_l = 0
        lex_lines = []
        seen_l = set()
        for turn in top:
            line = render_line(turn)
            key = turn[3].strip().lower()
            if key in seen_l:
                continue
            seen_l.add(key)
            cost = est_tokens(line) + 1
            if used_l + cost > 2000:
                break
            lex_lines.append(line)
            used_l += cost
        ctx_lex = "\n".join(lex_lines)

        recs = {
            "r7_only": reach(ctx_full, question),
            "r7_a": reach(ctx_a, question),
            "hybrid": reach(ctx_hybrid, question),
            "hybrid_rt": reach(ctx_rt, question),
            "lex_only": reach(ctx_lex, question),
        }
        forb = {
            "r7_only": forb_hit(ctx_full, question),
            "r7_a": forb_hit(ctx_a, question),
            "hybrid": forb_hit(ctx_hybrid, question),
            "hybrid_rt": forb_hit(ctx_rt, question),
            "lex_only": forb_hit(ctx_lex, question),
        }
        rows.append({
            "question_id": question.question_id,
            "category": cat,
            "expected_abstention": question.expected_abstention,
            "ctx_tokens": {"r7_a": est_tokens(ctx_a), "hybrid": est_tokens(ctx_hybrid),
                           "lex_only": est_tokens(ctx_lex), "hybrid_rt": est_tokens(ctx_rt)},
            **recs,
            **{f"forb_{k}": v for k, v in forb.items()},
            "forbidden_terms": list(question.forbidden),
        })
        if not question.expected_abstention:
            for name, v in recs.items():
                d = per_cat.setdefault(cat, {name: 0 for name in recs})
                d[name] = d.get(name, 0) + (1 if v else 0)
                d.setdefault("n", 0)
            per_cat[cat]["n"] += 1
    elapsed = time.perf_counter() - t1
    print(f"compiled {len(rows)} contexts in {elapsed:.1f}s")

    variants = ["r7_only", "lex_only", "r7_a", "hybrid", "hybrid_rt"]
    header = f"{'category':<28}" + "".join(f"{v:>12}" for v in variants)
    print("\n" + header)
    print("-" * len(header))
    cats = sorted(per_cat)
    for cat in cats:
        n = per_cat[cat]["n"]
        line = f"{cat:<28}"
        for v in variants:
            hit = per_cat[cat].get(v, 0)
            line += f"{hit}/{n}".rjust(8) + f"({100 * hit / max(1, n):>4.0f}%)"[:12].rjust(12)
        print(line)
    n_all = sum(per_cat[c]["n"] for c in cats)
    line = f"{'ALL':<28}"
    for v in variants:
        hit = sum(per_cat[c].get(v, 0) for c in cats)
        line += f"{hit}/{n_all}".rjust(8) + f"({100 * hit / n_all:>4.0f}%)"[:12].rjust(12)
    print(line)
    print(f"mean hybrid ctx tokens: "
          f"{sum(r['ctx_tokens']['hybrid'] for r in rows) / len(rows):.0f}")

    # Negative-evidence dimension: forbidden-hit rate per variant per category.
    print("\n  forbidden (decoy) term presence per context")
    header2 = f"{'category':<28}" + "".join(f"{v:>15}" for v in variants)
    print(header2)
    print("-" * len(header2))
    for cat in cats:
        rows_cat = [r for r in rows if r["category"] == cat]
        line = f"{cat:<28}"
        for v in variants:
            hit = sum(1 for r in rows_cat if r[f"forb_{v}"] > 0)
            line += f"{hit}/{len(rows_cat)}".rjust(15)
        print(line)
    n_abs = sum(1 for r in rows if r["expected_abstention"])
    line = f"{'abstention (temptation)':<28}"
    for v in variants:
        hit = sum(1 for r in rows if r["expected_abstention"] and r[f"forb_{v}"] > 0)
        line += f"{hit}/{n_abs}".rjust(15)
    print(line)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "budget_a": args.budget_a,
                               "top_k": args.top_k}, indent=2), encoding="utf-8")
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
