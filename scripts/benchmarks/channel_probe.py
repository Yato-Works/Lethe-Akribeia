#!/usr/bin/env python3
"""Where does the ground-truth evidence live inside the WideSlicer channels?

Layer-1 only (no reader, no LLM).  For every question this reports, per
annotated evidence id:

  * whether it is in the corpus at all
  * which retrieval channel(s) surface it, and its rank inside that channel
  * its position under three candidate orderings of the selection window:
        union   -- current flatten order (channel concatenation)
        pool    -- rescue block sorted by union position
        roundrobin -- channel-interleaved (channel diversity inside window N)

The output answers one question: does an ordering change move ground truth
inside the selection window, or is the evidence simply never retrieved?

Usage:
  uv run python scripts/benchmarks/channel_probe.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter  # noqa: E402
from artificial_memory.steroid.wide_slicer import WideSlicer  # noqa: E402

CHANNELS = ["lexical", "entity", "temporal", "relation", "session"]
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def roundrobin(channels: dict[str, list]) -> list:
    """Interleave channels so every channel contributes to the head of the list."""
    out: list = []
    idx = {k: 0 for k in CHANNELS}
    remaining = sum(len(channels.get(k, [])) for k in CHANNELS)
    while remaining > 0:
        for k in CHANNELS:
            i = idx[k]
            ch = channels.get(k) or []
            if i < len(ch):
                out.append(ch[i])
                idx[k] = i + 1
                remaining -= 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="WideSlicer channel probe")
    ap.add_argument("--results", default="benchmark_results/retrieval/oracle_recall_baseline.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--window", type=int, default=24, help="selection window to evaluate against")
    ap.add_argument("--out", default="benchmark_results/retrieval/channel_probe.json")
    args = ap.parse_args()

    adapter = LoCoMoAdapter()
    slicer = WideSlicer(per_channel_budget=40)
    base = json.loads((REPO / args.results).read_text(encoding="utf-8"))
    rows = base["rows"]
    if args.limit:
        rows = rows[: args.limit]

    recs_by_conv: dict[str, list] = {}
    q_by_conv: dict[str, dict] = {}
    raw_convs = json.loads(adapter.dataset_path.read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in raw_convs]

    probe: list[dict] = []
    for i, row in enumerate(rows, 1):
        sid = row["question_id"].split("-qa-")[0]
        conv_idx = sample_ids.index(sid)
        if sid not in recs_by_conv:
            _, questions, ir = adapter.load_conversation(conv_idx)
            recs_by_conv[sid] = list(ir)
            q_by_conv[sid] = {q.question_id: q.question for q in questions}
        records = recs_by_conv[sid]
        qtext = q_by_conv[sid].get(row["question_id"], row.get("question", ""))
        res = slicer.slice(qtext, records)
        union_pos = {id(r): p for p, r in enumerate(res.candidate_records)}
        rr = roundrobin(res.channels)
        rr_pos = {id(r): p for p, r in enumerate(rr)}

        ev = [e.strip() for e in row["evidence_ids"]]
        per_ev = []
        for e in ev:
            in_corpus = any(e in r.raw_content for r in records)
            hit_ch = {}
            for k in CHANNELS:
                ch = res.channels.get(k) or []
                pos = next((p for p, r in enumerate(ch) if e in r.raw_content), -1)
                if pos >= 0:
                    hit_ch[k] = pos
            u_pos = next((p for p, r in enumerate(res.candidate_records) if e in r.raw_content), -1)
            r_pos = next((p for p, r in enumerate(rr) if e in r.raw_content), -1)
            per_ev.append({"id": e, "in_corpus": in_corpus, "channels": hit_ch,
                           "union_pos": u_pos, "rr_pos": r_pos})
        probe.append({
            "qid": row["question_id"],
            "category": row["category"],
            "oracle": row["oracle_recall"],
            "n_ev": len(ev),
            "in_pool": any(p["union_pos"] >= 0 for p in per_ev),
            "ev": per_ev,
            "union_size": len(res.candidate_records),
            "channel_counts": dict(res.channel_counts),
        })
        if i % 200 == 0:
            print(f"  probed {i}/{len(rows)}", flush=True)

    # ---- aggregate -----------------------------------------------------------
    def pos_under(e: dict, mode: str) -> int:
        if mode == "union":
            return e["union_pos"]
        if mode == "rr":
            return e["rr_pos"]
        if mode == "best_channel":
            return (min(e["channels"].values()) if e["channels"] else -1)
        return -1

    def min_pos(p: dict, mode: str) -> int:
        """Best (smallest) evidence position; -1 when no evidence id is placed."""
        vals = [v for v in (pos_under(e, mode) for e in p["ev"]) if v >= 0]
        return min(vals) if vals else -1

    W = args.window
    print("\n=== evidence presence ===")
    miss = [p for p in probe if not p["oracle"]]
    hit = [p for p in probe if p["oracle"]]
    print(f"  questions {len(probe)}   hits {len(hit)}   misses {len(miss)}")
    print(f"  misses with evidence in corpus : {sum(1 for p in miss if any(e['in_corpus'] for e in p['ev']))}")
    print(f"  misses with evidence in pool   : {sum(1 for p in miss if p['in_pool'])}  "
          f"({100 * sum(1 for p in miss if p['in_pool']) / max(1, len(miss)):.1f}%)")

    print("\n=== channel coverage of evidence (per evidence id, misses only) ===")
    ch_hit = Counter()
    for p in miss:
        for e in p["ev"]:
            for k in e["channels"]:
                ch_hit[k] += 1
            if not e["channels"]:
                ch_hit["(none)"] += 1
    n_ev_miss = sum(len(p["ev"]) for p in miss)
    for k in CHANNELS + ["(none)"]:
        print(f"    {k:10s} {ch_hit[k]:5d} / {n_ev_miss}  ({100 * ch_hit[k] / max(1, n_ev_miss):5.1f}%)")

    print(f"\n=== window eligibility at W={W} (upper bound, NOT a measured oracle) ===")
    print("    a question is 'eligible' if its BEST evidence position falls inside the")
    print("    window under that ordering; the real compiler also applies token budget,")
    print("    session caps and early stop, so measured recall will be <= these numbers.")
    for mode in ("union", "rr", "best_channel"):
        m = sum(1 for p in miss if 0 <= min_pos(p, mode) < W)
        h = sum(1 for p in hit if 0 <= min_pos(p, mode) < W)
        print(f"    {mode:14s} misses eligible {m:4d}/{len(miss)}   "
              f"hits eligible {h:4d}/{len(hit)}   "
              f"-> eligible questions {100 * (m + h) / len(probe):.2f}%")

    print("\n=== per category: misses eligible under round-robin (W=%d) ===" % W)
    grid: dict[str, Counter] = defaultdict(Counter)
    for p in miss:
        cat = CATEGORY_NAMES.get(p["category"], str(p["category"]))
        grid[cat]["n"] += 1
        if 0 <= min_pos(p, "rr") < W:
            grid[cat]["rescued_rr"] += 1
        if 0 <= min_pos(p, "union") < W:
            grid[cat]["rescued_union"] += 1
    for cat in ("multi-hop", "temporal", "open-domain", "single-hop"):
        if cat in grid:
            v = grid[cat]
            print(f"    {cat:12s} misses={v['n']:4d}  rr={v['rescued_rr']:4d}  union={v['rescued_union']:4d}")

    print("\n=== round-robin position of misses that are eligible (best case) ===")
    positions = sorted(min_pos(p, "rr") for p in miss if 0 <= min_pos(p, "rr") < W)
    print(f"    n={len(positions)}  min={positions[0] if positions else '-'}  "
          f"max={positions[-1] if positions else '-'}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
