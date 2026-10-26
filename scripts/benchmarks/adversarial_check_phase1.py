"""Adversarial check on the Phase 1 conclusion: is B/C really zero?

trace_no_carrying_turn.py reported A=37, E=54, B=0, C=0 across 91 evidence-absent
cases.  'The retriever and compiler never lose the evidence' is the single most
consequential claim in the trace, and a zero is exactly the number that should be
distrusted -- it could mean the check itself is broken rather than the pipeline clean.

Three ways the zero could be an artifact, each tested:

  1  DEAD JOIN.  If the sample_id join failed, raw_hits would be 0 for every case and
     every case would be mislabelled A.  Checked by counting how many cases resolved
     a conversation AND how many raw turns that conversation actually has.
  2  TOO-STRICT MATCH.  The strict day+month pattern may simply never match the
     corpus's phrasing, again forcing everything into A.  Checked by relaxing to
     month-only and seeing how many cases gain a raw hit.
  3  WRONG TARGET SET.  If the evidence-absent set were mis-derived, the whole trace
     would be scoped wrong.  Checked by re-deriving it from the frozen cache directly.

Reported as counts plus, for any case where relaxing the match changes the verdict,
the concrete raw line -- so a genuine B/C cannot hide behind the strict pattern.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "datasets/external/locomo10.json"
TRACE = ROOT / "benchmark_results/when_date_map/raw_trace_no_carrying_turn.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"

MONTHS = ("january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december")


def main() -> int:
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    raw = {c["sample_id"]: c for c in data if c.get("sample_id")}
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    cases = trace["cases"]

    print("=" * 74)
    print("CHECK 1 -- did the sample_id join actually resolve?")
    print("=" * 74)
    resolved = [c for c in cases if c.get("raw_turn_count")]
    print(f"  cases with a resolved conversation : {len(resolved)}/{len(cases)}")
    print(f"  cases with ZERO raw turns          : "
          f"{len(cases) - len(resolved)}  <- these would be forced into 'A'")
    zero_turn = [c["question_id"] for c in cases if not c.get("raw_turn_count")]
    if zero_turn:
        print(f"    ids: {zero_turn[:8]}")
    turn_counts = Counter(c.get("raw_turn_count") for c in resolved)
    print(f"  distinct raw-turn counts           : "
          f"{sorted(k for k in turn_counts if k)[:8]} ...")
    print()
    print("  A healthy join shows hundreds of raw turns per conversation.  Zeros here")
    print("  would mean the join failed and every case was mislabelled 'A'.")
    print()

    print("=" * 74)
    print("CHECK 2 -- does relaxing to month-only change any verdict?")
    print("=" * 74)
    promoted = []
    for c in cases:
        gd = c.get("gold_date_searched")
        if not gd:
            continue
        conv = raw.get(c.get("conv_id"))
        if not conv:
            continue
        body = conv.get("conversation") or {}
        month = gd.split()[1]
        month_re = re.compile(rf"\b{month}\b", re.I)
        for key in body:
            if re.fullmatch(r"session_\d+", key):
                for turn in body[key] or []:
                    if isinstance(turn, dict) and month_re.search(
                            turn.get("text") or ""):
                        promoted.append((c["question_id"], gd, month,
                                         (turn.get("text") or "")[:100]))
                        break
                else:
                    continue
                break
    print(f"  cases that gain a raw hit under month-only matching : {len(promoted)}")
    print()
    print("  These are the cases the strict day+month pattern may have wrongly")
    print("  forced into 'A_no_evidence'.  Each is printed so it can be judged:")
    for qid, gd, month, text in promoted[:12]:
        print(f"    {qid:<17} gold-day-month={gd!r}  month {month!r} found in raw:")
        print(f"        {text}")
    if len(promoted) > 12:
        print(f"    ... {len(promoted) - 12} more")
    print()

    print("=" * 74)
    print("CHECK 3 -- is the evidence-absent target set derived correctly?")
    print("=" * 74)
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row
    map_data = json.loads(
        (ROOT / "benchmark_results/when_date_map/when_date_failure_map.json")
        .read_text(encoding="utf-8"))
    from collections import Counter as C2
    cls = C2(r["engine_class"] for r in map_data["records"])
    print(f"  map engine classes : {dict(cls)}")
    absent = [r for r in map_data["records"]
              if r["engine_class"] in ("ENGINE_WRONG_EVIDENCE_ABSENT",)
              or (r["engine_class"] == "ENGINE_ABSTAIN"
                  and r["gold_in_context"] is False)]
    print(f"  re-derived evidence-absent set : {len(absent)}")
    print(f"  traced in phase 1              : {len(cases)}")
    print(f"  match                          : {len(absent) == len(cases)}")
    print()

    # The decisive one: is the GOLD itself wrong?
    print("=" * 74)
    print("CHECK 4 -- is the gold date itself present anywhere in the raw corpus?")
    print("=" * 74)
    blob_by_conv = {k: json.dumps(v.get("conversation") or {}, ensure_ascii=False).lower()
                    for k, v in raw.items()}
    checked = 0
    absent_everywhere = 0
    examples = []
    for c in cases:
        gd = c.get("gold_date_searched")
        if not gd:
            continue
        checked += 1
        blob = blob_by_conv.get(c.get("conv_id"), "")
        d, mon = gd.split()
        if not (re.search(rf"\b{d}\s+{mon}\b", blob)
                or re.search(rf"\b{mon}\s+{d}(?:st|nd|rd|th)?\b", blob)):
            absent_everywhere += 1
            if len(examples) < 6:
                examples.append((c["question_id"], c["gold"]))
    print(f"  cases with a searchable day-month : {checked}")
    print(f"  of those, gold date absent from the ENTIRE raw conversation : "
          f"{absent_everywhere}")
    for qid, gold in examples:
        print(f"    {qid:<17} gold={gold!r}")
    print()
    print("READ THIS AS")
    print("  If absent_everywhere ~= checked, the dataset's gold names a date the")
    print("  conversation never states.  No retriever, compiler, or engine change can")
    print("  recover that -- the answer is not in the data.  Those cases are a DATA")
    print("  defect, not an engine defect, and the honest classification is 'A'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
