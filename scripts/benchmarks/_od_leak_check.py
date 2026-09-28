"""Count boolean-answer leaks on non-boolean Open-Domain questions."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
BOOL = re.compile(r"^(likely\s+)?(yes|no)\b", re.I)
GTBOOL = re.compile(r"^(likely|yes|no)\b", re.I)


def main() -> int:
    for tag in sys.argv[1:] or ["cap24_coder7b", "locomo_fixes_applied", "od_probe"]:
        path = REPO / "benchmark_results" / "locomo1540" / f"{tag}.json"
        rows = [x for x in json.load(open(path, encoding="utf-8"))["results"]
                if x["category"] == 3]
        leaks = [x for x in rows
                 if BOOL.match(str(x["prediction"]).strip())
                 and not GTBOOL.match(str(x["ground_truth"]).strip())]
        hit = sum(1 for x in rows if x["is_correct"])
        q50 = next((x for x in rows if x["question_id"] == "conv-26-qa-050"), None)
        pred = repr(q50["prediction"]) if q50 else "n/a"
        ok = q50["is_correct"] if q50 else "-"
        print(f"{tag:<24} OD {hit}/{len(rows)} | bool-leaks {len(leaks)}"
              f" | qa-050 -> {pred} (correct={ok})")
        for x in leaks[:5]:
            print(f"      Q={x['question'][:64]}")
            print(f"      GT={str(x['ground_truth'])[:44]} | PRED={x['prediction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
