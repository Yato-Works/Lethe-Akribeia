"""Next target selection: classify the committer's abstentions and wrong answers.

Eli's rule: pick the next thing to fix from MEASURED failures, with a reproducible
question and a concrete goal -- not a theory.

Scope: the same 321 category-2 questions, frozen cache.  For every question the
committer did NOT answer correctly, record why, so the biggest fixable block is
chosen on evidence.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from artificial_memory.skills.answer_committer import commit_answer  # noqa: E402

CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
DATA = ROOT / "datasets/external/locomo10.json"

_SHAPES = (
    ("when_date", r"\bwhen\b.*\b(?:did|does|is|was|has|have|go|going|get|got)\b"),
    ("when_bare", r"^when\b"),
    ("what_count", r"\bhow (?:many|much)\b"),
    ("what_person", r"\bwho\b"),
    ("what_thing", r"\bwhat\b"),
    ("where_place", r"\bwhere\b"),
    ("why_reason", r"\bwhy\b"),
    ("yesno", r"^(?:did|does|is|was|were|has|have|can|could|would|will)\b"),
)


def shape(q: str) -> str:
    ql = q.lower().strip()
    for name, pat in _SHAPES:
        if re.search(pat, ql):
            return name
    return "other"


def gold_present(gold: str, context: str) -> bool:
    g = [w for w in re.sub(r"[^a-z0-9 ]", " ", gold.lower()).split() if len(w) > 2]
    if not g:
        return False
    c = set(re.sub(r"[^a-z0-9 ]", " ", context.lower()).split())
    return sum(1 for w in g if w in c) >= max(1, len(g) // 2)


def main() -> None:
    golds = {}
    for conv in json.load(open(DATA, encoding="utf-8")):
        for qa in conv.get("qa", []):
            if "question" in qa and "answer" in qa:
                golds[str(qa["question"]).lower().strip()] = str(qa["answer"])

    rows = [json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if str(r.get("category")) == "2"]

    abstain_shape, abstain_evidence = Counter(), Counter()
    wrong_shape = Counter()
    abstain_examples, wrong_examples = [], []

    for r in rows:
        gold = golds.get(r["question"].lower().strip())
        if gold is None:
            continue
        sh = shape(r["question"])
        ans = commit_answer(r["question"], r["context"], 1)

        if ans.used and E.score(str(ans.answer), gold):
            continue

        if not ans.used:
            abstain_shape[sh] += 1
            has = gold_present(gold, r["context"])
            abstain_evidence["gold_in_context" if has else "gold_absent"] += 1
            if len(abstain_examples) < 200:
                abstain_examples.append((r["question"], gold, has, ans.detail))
        else:
            wrong_shape[sh] += 1
            if len(wrong_examples) < 200:
                wrong_examples.append((r["question"], gold, str(ans.answer),
                                       ans.detail))

    print(f"=== category-2, 321 questions, current committer ===")
    print(f"abstained: {sum(abstain_shape.values())}   wrong: {sum(wrong_shape.values())}")
    print()
    print("=== abstentions by question shape ===")
    for k, v in abstain_shape.most_common():
        print(f"  {k:12s}: {v}")
    print()
    print("=== abstentions by evidence availability ===")
    for k, v in abstain_evidence.most_common():
        print(f"  {k:14s}: {v}")
    print()
    print("=== wrong answers by question shape ===")
    for k, v in wrong_shape.most_common():
        print(f"  {k:12s}: {v}")
    print()
    print("=== abstain examples where the gold IS in the context (fixable) ===")
    shown = 0
    for q, g, has, det in abstain_examples:
        if not has or shown >= 6:
            continue
        shown += 1
        print(f"  Q : {q[:58]}")
        print(f"  GT: {g[:44]!r}")
        print(f"  why: {str(det)[:56]!r}")
    print()
    print("=== wrong-answer examples ===")
    for q, g, a, det in wrong_examples[:6]:
        print(f"  Q : {q[:58]}")
        print(f"  GT: {g[:32]!r}   got {a[:32]!r}")
        print(f"  why: {str(det)[:56]!r}")


if __name__ == "__main__":
    main()
