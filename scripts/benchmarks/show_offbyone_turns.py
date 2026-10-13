"""Show the raw evidence turn for the +1-day parsing errors.

cause_of_offbyone.py found 22 of 26 cases where BOTH gold and gold+1 are written
in the context -- so the engine selects the right turn and then resolves the date
one day late.  This prints the verbatim turn text so the date FORMAT that is being
misread is visible, which is what a parsing fix has to target.

Printed for the first few cases only; this is inspection, not a measurement.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"

TARGETS = ["lc_conv-26_0", "lc_conv-26_16", "lc_conv-26_58", "lc_conv-30_33"]


def main() -> int:
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row
    atlas = {}
    for line in ATLAS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            atlas[row["question_id"]] = row

    for qid in TARGETS:
        row = atlas.get(qid)
        if not row:
            continue
        ctx = (cache.get(row.get("cache_qid")) or {}).get("context") or ""
        print("=" * 74)
        print(qid)
        print(f"  Q    : {row['question']}")
        print(f"  gold : {row['gold']!r}")
        print(f"  pred : {row.get('predicted_answer')!r}")
        print(f"  evidence_turns : {row.get('evidence_turns')}")
        lines = ctx.splitlines()
        for tid in (row.get("evidence_turns") or [])[:5]:
            for line in lines:
                if line.strip().startswith(f"[{tid} "):
                    print(f"  [{tid}] {line.strip()[:200]}")
                    break
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
