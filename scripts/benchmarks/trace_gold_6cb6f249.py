"""Precise numeric reachability trace for 6cb6f249 (GT=17 days).

Enumerates every turn mentioning a *duration-shaped* break
(e.g. '10-day break', 'week-long break', 'took a week off')
and traces it through the V4 aggregation stages.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory_gen.compiler.samplers.per_session_sampler import PerSessionSampler

DUR_RE = re.compile(
    r"(\d+\s*(?:-\s*)?day|week[\s-]*long|took\s+a\s+week\s+off|"
    r"\d+\s*weeks?\s+off|\d+\s*days?\s+(?:break|off))",
    re.IGNORECASE,
)


def load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_v4_pipeline_dev",
        str(REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json",
              encoding="utf-8") as f:
        items = json.load(f)
    item = next(x for x in items if x["question_id"] == "6cb6f249")
    q = item["question"]
    q_date = item.get("question_date", "")

    all_turns = []
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])
    for s_idx, session in enumerate(item.get("haystack_sessions", [])):
        s_date = dates[s_idx] if s_idx < len(dates) else ""
        sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
        for turn in session:
            all_turns.append((sid, s_date, turn.get("role", "user"),
                              turn.get("content", "")))

    gold = [(i, t) for i, t in enumerate(all_turns) if DUR_RE.search(t[3])]
    print(f"Duration-shaped break mentions: {len(gold)}")
    for i, (sid, date, speaker, content) in gold:
        m = DUR_RE.search(content)
        print(f"  - idx={i} [{sid}] {speaker}: ...{m.group(0)!r} :: "
              f"{content[:160]}")

    mod = load_eval_module()
    retriever = mod.V4CandidateRetriever()
    top_turns, _ = retriever.rank_turns(q, all_turns, top_k=60,
                                        reference_date=q_date)
    top_sids = {t[0] for t in top_turns[:30]}
    print(f"\n[STAGE 1] gold turns surviving top-30 prefilter: "
          f"{sum(1 for _, t in gold if t[0] in top_sids)}/{len(gold)}")
    for i, (sid, date, speaker, content) in gold:
        mark = "KEEP" if sid in top_sids else "LOST"
        print(f"  [{mark}] idx={i} [{sid}] {speaker}: {content[:130]}")

    candidate_turns = [t for t in all_turns if t[0] in top_sids]
    for tps in (2, 3):
        sampler = PerSessionSampler(turns_per_session=tps)
        sampled = sampler.sample(q, candidate_turns, max_total_tokens=2200)
        sampled_texts = {e.content for e in sampled}
        n = sum(1 for _, t in gold if t[3] in sampled_texts)
        print(f"\n[STAGE 2 turns={tps}] gold duration turns in context: "
              f"{n}/{len(gold)} ({len(sampled)} turns total)")
        for e in sampled:
            flag = "*" if DUR_RE.search(e.content) else " "
            print(f"  {flag} [{e.session_id}] {e.speaker} "
                  f"(score={e.score:.1f}): {e.content[:130]}")


if __name__ == "__main__":
    main()
