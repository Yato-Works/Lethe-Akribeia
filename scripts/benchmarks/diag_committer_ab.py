"""Before/after committer comparison on ONE frozen context cache.

Eli's rule: same questions, same conditions, gold from the dataset, nothing
estimated.

Baseline already on disk (locomo_cat2_temporal_claims_current.json), measured on
locomo_context_cache_rules.jsonl:
  n=321, claimed=180, commit_correct=124

This re-measures the SAME 321 questions against the CURRENT committer revision
(the speaker-attribution fixes in 9e0abe9) and reports:
  claimed / correct / wrong / abstained, split out as Eli asked
  per-question deltas: FIXED (was wrong, now correct) and
                       BROKE (was correct, now wrong)
  speaker attribution correctness on the claims
  wall time and LLM calls (should be 0 -- the committer is deterministic)
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory.skills.answer_committer import (  # noqa: E402
    commit_answer,
    parse_turns,
)

CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
BASELINE = ROOT / "benchmark_results/committer_metrics/locomo_cat2_temporal_claims_current.json"
DATA = ROOT / "datasets/external/locomo10.json"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def main() -> None:
    # --- gold answers, keyed by normalised question -------------------------
    golds: dict[str, str] = {}
    convs = json.load(open(DATA, encoding="utf-8"))
    for conv in convs:
        for qa in conv.get("qa", []):
            if "question" not in qa or "answer" not in qa:
                continue
            golds[norm(qa["question"])] = str(qa["answer"])

    # --- the baseline's own per-question record ----------------------------
    # Keyed on question_id, NOT question text -- the baseline stores only the id,
    # and matching on normalised question text silently matched nothing (which
    # reported 0 fixed / 0 broke against a real 124 -> 159 jump).
    base_recs = {}
    b = json.load(open(BASELINE, encoding="utf-8"))
    for r in b.get("records", []):
        base_recs[str(r.get("question_id", ""))] = r

    # question_id -> normalised question, so a cache row can find its baseline
    qid_by_norm: dict[str, str] = {}
    for conv in convs:
        for i, qa in enumerate(conv.get("qa", [])):
            if "question" not in qa:
                continue
            qid_by_norm[norm(qa["question"])] = f"{conv.get('sample_id','')}-qa-{i:03d}"

    # --- re-measure on the frozen cache ------------------------------------
    rows = [json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()]
    # the baseline covered 321 category-2 questions; keep the same scope
    rows = [r for r in rows if str(r.get("category")) == "2"]

    t0 = time.time()
    counts = Counter()
    fixed, broke = [], []
    speaker_checked = speaker_ok = 0

    for r in rows:
        q = r["question"]
        gold = golds.get(norm(q))
        if gold is None:
            counts["no_gold"] += 1
            continue
        counts["n"] += 1

        ans = commit_answer(q, r["context"], 1)
        turns = parse_turns(r["context"])

        if not ans.used:
            counts["abstained"] += 1
            continue

        counts["claimed"] += 1
        now_ok = bool(_strict_correct(str(ans.answer), gold))
        counts["correct" if now_ok else "wrong"] += 1

        # speaker attribution: does the evidence turn belong to the subject?
        subj = _subject(q)
        if subj and getattr(ans, "evidence_turn", ""):
            ev = next((t for t in turns if t.dia_id == ans.evidence_turn), None)
            if ev is not None:
                speaker_checked += 1
                if ev.speaker.lower() == subj.lower():
                    speaker_ok += 1

        # delta vs baseline
        qid = qid_by_norm.get(norm(q))
        br = base_recs.get(qid) if qid else None
        if br is not None:
            was_ok = bool(br.get("commit_correct"))
            if now_ok and not was_ok:
                fixed.append((q, str(ans.answer), gold))
            elif was_ok and not now_ok:
                broke.append((q, str(ans.answer), gold))

    dt = time.time() - t0

    print("=== same 321 category-2 questions, frozen cache, current committer ===")
    print(f"  evaluated      : {counts['n']}")
    print(f"  claimed        : {counts['claimed']}")
    print(f"    correct      : {counts['correct']}")
    print(f"    wrong        : {counts['wrong']}")
    print(f"  abstained      : {counts['abstained']}")
    if counts["no_gold"]:
        print(f"  no gold found  : {counts['no_gold']}")
    print()
    print(f"  speaker attribution correct: {speaker_ok}/{speaker_checked}"
          f" ({speaker_ok/speaker_checked*100 if speaker_checked else 0:.1f}%)")
    print()
    print("=== delta vs the on-disk baseline (claimed 180 / correct 124) ===")
    print(f"  FIXED (was wrong, now correct) : {len(fixed)}")
    print(f"  BROKE (was correct, now wrong) : {len(broke)}")
    print(f"  net                             : {len(fixed) - len(broke):+d}")
    print()
    print(f"  wall time: {dt:.1f}s   LLM calls: 0 (committer is deterministic)")
    print()
    if fixed:
        print("=== FIXED examples ===")
        for q, a, g in fixed[:6]:
            print(f"  Q : {q[:56]}")
            print(f"  -> {a[:32]!r}   GT: {g[:26]!r}")
    if broke:
        print("=== BROKE examples ===")
        for q, a, g in broke[:6]:
            print(f"  Q : {q[:56]}")
            print(f"  -> {a[:32]!r}   GT: {g[:26]!r}")


def _subject(question: str) -> str | None:
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", question):
        w = m.group(1)
        if w.lower() in {"what", "when", "where", "who", "which", "why", "how",
                         "did", "does", "is", "was", "has", "have", "the"}:
            continue
        return w
    return None


def _strict_correct(pred: str, gold: str) -> bool:
    """Strict: no partial credit.  Imported lazily to keep this script standalone."""
    sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
    import eval_locomo_full as E
    return E.score(pred, gold)


if __name__ == "__main__":
    main()
