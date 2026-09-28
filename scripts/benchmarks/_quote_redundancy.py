"""Measure how much of a compiled LoCoMo context is removable without losing facts.

Reads benchmark_results/locomo_context_cache.jsonl (cap24, frozen) and reports:
  * word split: header / (In reply to ...) quote / speaker / body
  * what fraction of quotes are redundant (their quoted text already appears in
    another selected turn's body) vs load-bearing
  * how many words that leaves on the table
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "benchmark_results" / "locomo_context_cache.jsonl"

QUOTE_RE = re.compile(r"\(In reply to ([^:]+): \"([^\"]{10,})")
HEADER_RE = re.compile(r"^\[D\d+:\d+[^\]]*\]\s*")
SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z .'-]{0,30}):\s*")


def split_line(line: str) -> tuple[int, int, int, str]:
    header = quote = speaker = 0
    rest = line
    m = HEADER_RE.match(rest)
    if m:
        header = len(m.group(0).split())
        rest = rest[m.end():]
    q = quote_span(rest)
    if q:
        quote = len(rest[:q[1]].split())
        rest = rest[q[1]:].lstrip()
    m = SPEAKER_RE.match(rest)
    if m:
        speaker = len(m.group(0).split())
        rest = rest[m.end():]
    return header, quote, speaker, len(rest.split())


def quote_span(text: str) -> tuple[int, int] | None:
    """Locate an '(In reply to ...)' block.  Quoted text may itself contain ')',
    so the block terminates at the closing quote+paren sequence: '")'."""
    start = text.find("(In reply to ")
    if start != 0:
        return None
    end = text.find('")', start)
    if end == -1:
        return None
    return start, end + 2


def main() -> int:
    tot = {"header": 0, "quote": 0, "speaker": 0, "body": 0}
    red = nonred = 0
    red_words = nonred_words = 0
    per_ctx: list[int] = []
    n = 0

    for line in CACHE.open(encoding="utf-8"):
        d = json.loads(line)
        if d.get("category") == 5:
            continue
        n += 1
        lines = d["context"].split("\n")
        per_ctx.append(len(d["context"].split()))

        bodies = []
        for ln in lines:
            h, q, s, b = split_line(ln)
            tot["header"] += h
            tot["quote"] += q
            tot["speaker"] += s
            tot["body"] += b
            rest = HEADER_RE.sub("", ln)
            q = quote_span(rest)
            if q:
                rest = rest[q[1]:].lstrip()
            bodies.append(rest)

        for i, ln in enumerate(lines):
            mq = QUOTE_RE.search(ln)
            if not mq:
                continue
            needle = mq.group(2)[:30].strip()
            found = any(needle in bodies[j] for j in range(len(bodies)) if j != i)
            q = quote_span(HEADER_RE.sub("", ln))
            qlen = len(ln[q[0]:q[1]].split()) if q else 0
            if found:
                red += 1
                red_words += qlen
            else:
                nonred += 1
                nonred_words += qlen

    mean_ctx = statistics.mean(per_ctx)
    total_words = sum(tot.values())
    print(f"n={n}  mean context={mean_ctx:.0f} words")
    print("\ncomponent share of context:")
    for k, v in tot.items():
        print(f"  {k:8s} {v / n:7.1f}/q  ({100 * v / total_words:5.1f}%)")

    print(f"\nquotes: {red} redundant / {nonred} load-bearing "
          f"({100 * red / (red + nonred):.1f}% redundant)")
    print(f"  dropping redundant only : save {red_words / n:.0f} words/q "
          f"-> {mean_ctx - red_words / n:.0f} words total")
    print(f"  dropping load-bearing too: save {nonred_words / n:.0f} more words/q "
          f"-> {mean_ctx - (red_words + nonred_words) / n:.0f} words total")
    print(f"  headers -> compact form  : save up to {tot['header'] / n:.0f} words/q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
