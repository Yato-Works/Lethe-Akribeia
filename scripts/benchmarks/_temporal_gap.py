"""Expansion rate of relative temporal expressions, per pattern.

Tells apart two different bugs:
  * a pattern with a rule in temporal_normalizer but a low expansion rate
    => the turn's ``time_scope`` was empty, so normalize() bailed out early;
  * a pattern with no rule at all => the rule itself is missing.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]

PATTERNS = [
    ("last <weekday>", r"\blast\s+(?:mon|monday|tue|tues|tuesday|wed|weds|wednesday"
                       r"|thu|thur|thurs|thursday|fri|friday|sat|saturday|sun|sunday)\b"),
    ("last year", r"\blast year\b"),
    ("this month", r"\bthis month\b"),
    ("next month", r"\bnext month\b"),
    ("last week", r"\blast week\b"),
    ("last weekend", r"\blast weekend\b"),
    ("yesterday", r"\byesterday\b"),
    ("tomorrow", r"\btomorrow\b"),
    ("N days ago", r"\b\d+\s+days?\s+ago\b"),
    ("N weeks ago", r"\b\d+\s+weeks?\s+ago\b"),
    ("two days ago", r"\btwo days ago\b"),
    ("this week", r"\bthis week\b"),
    ("next week", r"\bnext week\b"),
    ("last <season>", r"\blast\s+(?:summer|winter|spring|fall)\b"),
]

HAS_RULE = {
    "last <weekday>": True, "last year": True, "this month": True,
    "next month": True, "last week": True, "last weekend": True,
    "yesterday": True, "tomorrow": True, "N days ago": True,
    "N weeks ago": True, "two days ago": True,
    "this week": True, "next week": True, "last <season>": True,
}

EXPANDED = re.compile(
    r"\(\s*the\s+[^)]*before\b|\(\d{4}\)|\(\s*\d{1,2}\s+[A-Za-z]+\s+\d{4}\)"
    r"|\(\s*[A-Za-z]+\s+\d{4}\s*\)|\(\s*the\s+week\s+(?:of|after)\b"
    r"|\(\s*the\s+(?:summer|winter|spring|fall)\s+of\b",
    re.IGNORECASE,
)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else \
        str(REPO / "benchmark_results" / "locomo_context_cache.jsonl")
    stat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if row.get("category") == 5:
            continue
        for text in row["context"].split("\n"):
            for name, pat in PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    stat[name][0] += 1
                    if EXPANDED.search(text):
                        stat[name][1] += 1

    print(f"{'pattern':<20}{'rule?':>7}{'lines':>8}{'expanded':>10}{'rate':>9}")
    for name, _ in PATTERNS:
        total, done = stat[name]
        if not total:
            continue
        rule = "yes" if HAS_RULE[name] else "NO"
        print(f"{name:<20}{rule:>7}{total:>8}{done:>10}{100 * done / total:>8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
