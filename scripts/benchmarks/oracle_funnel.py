#!/usr/bin/env python3
"""Funnel diagnostic for LoCoMo oracle-recall misses (failure-driven step 4).

Reproduces the compile path stage by stage and reports, per failing question,
the first stage at which the ground-truth evidence disappears:

    S0 corpus        evidence turn present in the IR records handed to compile
    S1 pool          present in ``_working_records`` (bounded candidate pool)
    S2 rank          rank in ``StateReconstructor.reconstruct_world`` output
    S3 widening      rank after ``_apply_evidence_widening`` merge
    S4 window        rank < max_units (the selection window the loop inspects)
    S5 budget        would survive the token-budget / session-diversity gates
    S6 context       present in the compiled context  (= oracle hit)

Nothing here calls an LLM, so it is a Layer-1 only instrument.

Usage:
  uv run python scripts/benchmarks/oracle_funnel.py --limit 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler  # noqa: E402
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter  # noqa: E402

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


class Funnel:
    """Records the intermediate candidate lists produced by one compile()."""

    def __init__(self, compiler: MinimumSufficientContextCompiler) -> None:
        self.compiler = compiler
        self.stage2: list = []
        self.stage3: list = []
        self.max_units = 0
        self.rescue_quota = 0
        self._patch()

    def _patch(self) -> None:
        c = self.compiler
        orig_recon = c.reconstructor.reconstruct_world
        orig_widen = c._apply_evidence_widening

        def recon(*a, **kw):
            intent, units = orig_recon(*a, **kw)
            self.stage2 = list(units)
            return intent, units

        def widen(*a, **kw):
            merged = orig_widen(*a, **kw)
            self.stage3 = list(merged)
            return merged

        c.reconstructor.reconstruct_world = recon  # type: ignore[method-assign]
        c._apply_evidence_widening = widen  # type: ignore[method-assign]


def rank_of(units: list, label: str) -> int:
    for i, u in enumerate(units):
        if label in u.ir.raw_content:
            return i
    return -1


def simulate_selection(compiler, query, units, max_units, ev_label):
    """Replay the Phase-1 selection loop and report why the evidence dropped.

    Mirrors the loop in ``MinimumSufficientContextCompiler.compile`` closely
    enough for diagnosis; a mismatch is reported as ``not_selected_unknown``.
    """
    selected = []
    curr = 0
    reason = "not_reached"
    for idx, u in enumerate(units[:max_units]):
        raw_words = len(u.ir.raw_content.split())
        is_ast = "assistant:" in u.ir.raw_content.lower()
        u_tok = min(raw_words, 35) if is_ast else min(raw_words, 120)
        if ev_label in u.ir.raw_content:
            if selected and curr + u_tok > 500 + compiler.rescue_token_bonus_per_unit * min(
                compiler._last_rescue_count, compiler.selection_window_cap
            ):
                return "budget", idx, len(selected)
            is_multi = any(w in query.lower() for w in
                           ["before", "after", "while", "during", "between", "both", "all",
                            "most", "least", "which", "compare", "difference"])
            if is_multi:
                m = re.search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]", u.ir.raw_content)
                if m:
                    sid = m.group(1)
                    cnt = sum(1 for su in selected if re.search(
                        rf"\[{re.escape(sid)}(?:\s+on\s+[^\]]+)?\]", su.ir.raw_content))
                    cap = 4
                    if any(w in query.lower() for w in ["most", "least", "which", "compare", "difference"]):
                        cap = 2
                    if cnt >= cap:
                        return "session_diversity_cap", idx, len(selected)
            return "would_select", idx, len(selected)
        if selected and curr + u_tok > 500 + compiler.rescue_token_bonus_per_unit * min(
            compiler._last_rescue_count, compiler.selection_window_cap
        ):
            continue
        selected.append(u)
        curr += u_tok
        if len(units) <= idx + 1:
            reason = "window_exhausted"
    return reason, -1, len(selected)


def main() -> int:
    ap = argparse.ArgumentParser(description="Oracle-miss funnel diagnostic")
    ap.add_argument("--results", default="benchmark_results/locomo1540/locomo_final.json")
    ap.add_argument("--limit", type=int, default=40, help="number of misses to trace")
    ap.add_argument("--out", default="benchmark_results/retrieval/oracle_funnel.json")
    args = ap.parse_args()

    adapter = LoCoMoAdapter()
    payload = json.loads((REPO / args.results).read_text(encoding="utf-8"))
    rows = payload["results"] if isinstance(payload, dict) else payload
    misses = [r for r in rows if not r.get("oracle_recall")]
    if args.limit:
        misses = misses[: args.limit]

    traced: list[dict] = []
    sample_ids = _sample_ids(adapter)
    compiler = MinimumSufficientContextCompiler()
    funnel = Funnel(compiler)
    for conv_idx in range(10):
        # conversation sample_id is the qid prefix before "-qa-"
        want = [m for m in misses if m["question_id"].split("-qa-")[0] == sample_ids[conv_idx]]
        if not want:
            continue
        _, questions, ir = adapter.load_conversation(conv_idx)
        qmap = {q.question_id: q for q in questions}
        for m in want:
            q = qmap.get(m["question_id"])
            if q is None or not q.evidence_ids:
                continue
            label = q.evidence_ids[0].strip()
            pcc = compiler.compile(q.question, ir)
            entry = {"qid": q.question_id, "category": q.category, "question": q.question,
                     "evidence": list(q.evidence_ids), "label": label}
            # S0
            entry["s0_corpus"] = any(label in r.raw_content for r in ir)
            # S1
            pool = compiler._working_records(q.question, ir)
            entry["s1_pool"] = any(label in r.raw_content for r in pool)
            # S2/S3
            r2 = rank_of(funnel.stage2, label)
            merged = funnel.stage3 or funnel.stage2
            r3 = rank_of(merged, label)
            entry["s2_rank"] = r2
            entry["s3_rank"] = r3
            entry["n_stage2"] = len(funnel.stage2)
            entry["n_stage3"] = len(funnel.stage3)
            rescue = compiler._last_rescue_count
            max_units = min(compiler.selection_window_cap, 6 + rescue) if compiler.evidence_widening else 12
            entry["rescue_quota"] = rescue
            entry["max_units"] = max_units
            entry["s4_window"] = 0 <= r3 < max_units
            reason, pos, nsel = simulate_selection(compiler, q.question, merged, max_units, label)
            entry["s5_selection"] = reason
            entry["s5_position"] = pos
            entry["s5_selected_before"] = nsel
            entry["s6_context"] = label in pcc.context_text
            entry["token_cost"] = pcc.token_cost
            traced.append(entry)
        print(f"conv {conv_idx}: traced {len(traced)}", flush=True)

    # ---- aggregate: first failing stage -------------------------------------
    def first_fail(e: dict) -> str:
        if not e["s0_corpus"]:
            return "S0 corpus (evidence turn absent from IR)"
        if not e["s1_pool"]:
            return "S1 pool (dropped by bounded candidate pool)"
        if e["s2_rank"] < 0:
            return "S2 rank (never surfaced by reconstruct_world)"
        if e["s3_rank"] < 0:
            return "S3 widening (absent from merged candidate list)"
        if not e["s4_window"]:
            return "S4 window (rank >= max_units selection window)"
        if e["s5_selection"] == "budget":
            return "S5 budget (token budget skipped it)"
        if e["s5_selection"] == "session_diversity_cap":
            return "S5 session diversity cap"
        if e["s5_selection"] == "would_select":
            return "S5+ selected but not in output (later stage)"
        return f"S5 {e['s5_selection']} (window exhausted before reaching it)"

    counts = Counter(first_fail(e) for e in traced)
    print("\n=== FIRST FAILING STAGE (misses traced: %d) ===" % len(traced))
    for k, v in counts.most_common():
        print(f"  {k:58s} {v:4d}  ({100 * v / max(1, len(traced)):5.1f}%)")

    print("\n=== rank distribution of evidence inside merged candidates ===")
    ranks = [e["s3_rank"] for e in traced if e["s3_rank"] >= 0]
    print(f"  present: {len(ranks)}/{len(traced)}   "
          f"min={min(ranks) if ranks else '-'} max={max(ranks) if ranks else '-'}")
    buckets = Counter()
    for r in ranks:
        buckets["0-5" if r < 6 else "6-11" if r < 12 else "12-23" if r < 24 else "24-59" if r < 60 else "60+"] += 1
    print("  " + ", ".join(f"{k}:{v}" for k, v in sorted(buckets.items())))

    print("\n=== rescue quota / window on misses ===")
    print("  rescue_quota:", dict(Counter(e["rescue_quota"] for e in traced).most_common(8)))
    print("  max_units   :", dict(Counter(e["max_units"] for e in traced).most_common(8)))

    by_cat: dict[str, Counter] = defaultdict(Counter)
    for e in traced:
        by_cat[CATEGORY_NAMES.get(e["category"], str(e["category"]))][first_fail(e)] += 1
    print("\n=== category x first failing stage ===")
    for cat in ("multi-hop", "temporal", "open-domain", "single-hop"):
        if cat in by_cat:
            print(f"  {cat:12s} " + ", ".join(f"{k.split(' ')[0]}={v}" for k, v in by_cat[cat].most_common()))

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"traced": traced,
                               "first_fail": dict(counts)}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten -> {out.relative_to(REPO)}")
    return 0


def _sample_ids(adapter: LoCoMoAdapter) -> list[str]:
    raw = json.loads(adapter.dataset_path.read_text(encoding="utf-8"))
    return [c.get("sample_id", "") for c in raw]


if __name__ == "__main__":
    raise SystemExit(main())
