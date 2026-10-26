"""What is the deterministic-extractor ceiling on the real union target?

The identity module reached 1 of 242 gold items while 142 were present verbatim,
so extraction is the constraint.  Before writing another extractor, this measures
the CEILING of the plausible rule patterns, because a hand-written extractor that
can never pass a low ceiling is not worth building.

For each gold item present verbatim in its context, it tests a battery of
introduction patterns and reports how many items each one reaches.  The union of
the battery is the ceiling -- the most any combination of these rules could
recover.  If the ceiling is high, a combined extractor is worth writing.  If it is
low, the items are introduced in ways rules cannot reliably capture, and that is
the model dependency the Cognee digest predicted.

Every pattern is reported separately so a pattern that reaches nothing is visible
rather than hidden inside a total.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"

#: Candidate introduction patterns.  Each returns the matched span(s).
PATTERNS: dict[str, re.Pattern] = {
    "named_X": re.compile(r"\b(?:named|called)\s+([A-Z][a-z]{2,})\b"),
    "quoted": re.compile(r"[\"'\u201c\u2018]([^\"'\u201d\u2019]{2,40})[\"'\u201d\u2019]"),
    "my_favorite": re.compile(
        r"\bmy favou?rite (?:is|was|ones? are|song is|band is)\s+"
        r"(?:the\s+)?([A-Za-z][\w' -]{2,30})", re.I),
    "i_love_i_like": re.compile(
        r"\bI (?:love|like|enjoy|adore)\s+(?:the\s+)?([A-Za-z][\w' -]{2,30})", re.I),
    "doing_activity": re.compile(
        r"\bI(?:'m| am) (?:doing|into|taking up)\s+([a-z][\w' -]{2,30})", re.I),
    "the_X_is": re.compile(r"\b[Tt]he ([a-z][\w' -]{2,30}) (?:is|was|matters?)\b"),
    "these_X": re.compile(r"\b[Tt]hese ([a-z][\w' -]{2,30})", re.I),
}


def gold_items(gold: str) -> list[str]:
    return [x.strip().strip('"').strip("'").strip() for x in str(gold).split(",")
            if x.strip()]


def main() -> int:
    atlas = [json.loads(l) for l in ATLAS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    cohort = [
        r for r in atlas
        if r.get("shape") in ("what_thing", "what_person")
        and r.get("gold_in_context") == "full"
        and r.get("rescored_correct") is False
        and len(gold_items(r["gold"])) >= 2
    ]

    present_total = 0
    per_pattern = Counter()
    covered: set[int] = set()          # indices of present items any pattern reaches
    item_index = 0
    present_indices: list[tuple] = []

    for row in cohort:
        ctx = cache.get(row.get("cache_qid"))
        if not ctx:
            continue
        context = ctx.get("context") or ""
        lowered = context.lower()
        for item in gold_items(row["gold"]):
            key = item.lower()
            if key not in lowered:
                continue
            present_total += 1
            present_indices.append((item_index, item, context))
            for name, pat in PATTERNS.items():
                for m in pat.finditer(context):
                    if key in m.group(0).lower() or m.group(1).strip().lower() == key:
                        per_pattern[name] += 1
                        covered.add(item_index)
                        break
            item_index += 1

    print(f"gold items PRESENT VERBATIM in context : {present_total}")
    print()
    print("reach per introduction pattern (items that pattern can capture)")
    print(f"  {'pattern':<18} {'items':>6} {'% of present':>13}")
    for name in PATTERNS:
        n = per_pattern[name]
        print(f"  {name:<18} {n:>6} {n / max(1, present_total) * 100:>12.1f}%")
    print()
    reachable = len(covered)
    print(f"UNION of all patterns (the CEILING)     : {reachable} "
          f"({reachable / max(1, present_total) * 100:.1f}% of present)")
    print(f"  unreachable by any of these rules     : "
          f"{present_total - reachable} "
          f"({(present_total - reachable) / max(1, present_total) * 100:.1f}%)")
    print()
    print("HOW TO READ THIS")
    print("  A high union means the items ARE introduced in regular ways and a")
    print("  combined deterministic extractor is worth writing -- the identity")
    print("  module is already proven safe at zero wrong merges, so it can be")
    print("  wired in front of it unchanged.")
    print("  A low union means the items are introduced in free-form ways that")
    print("  rules cannot capture.  That is the model dependency, and writing")
    print("  more regexes would be motion rather than progress.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
