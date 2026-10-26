#!/usr/bin/env python3
"""M30-42 Phase 4b: Baseline vs ablation on the unseen generic set.

Deterministic retrieval ranking only (no LLM), so the comparison is exact.
Measures whether benchmark-specific mechanisms help on data they were NOT
tuned for (they should not, since the names never appear).
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory_gen.recall.state_reconstructor import StateReconstructor
from artificial_memory.recall import evidence_scorer as es

MONTHS = {m.lower(): i for i, m in enumerate([
    "january","february","march","april","may","june","july","august",
    "september","october","november","december"], start=1)}


def flatten(conv):
    turns = []
    for k in sorted(conv.keys()):
        if k.endswith("_date_time") or not isinstance(conv[k], list):
            continue
        for t in conv[k]:
            turns.append(t)
    return turns


def make_records(turns):
    return [StructuredIR(
        entity=t.get("speaker",""), property="statement", value=t.get("text",""),
        source=t.get("speaker",""),
        raw_content=f"[{t.get('dia_id','')}] {t.get('speaker','')}: {t.get('text','')}",
    ) for t in turns]


def main():
    data = json.loads((PROJECT_ROOT / "results/m30_42/frozen_generic.json").read_text(encoding="utf-8"))
    convs, qs = data["conversations"], data["questions"]

    recon = StateReconstructor()
    ORIG_ACTOR = StateReconstructor._extract_target_actor
    ORIG_KNOWN = list(es.UniversalEvidenceScorer.KNOWN_ACTORS)
    ORIG_EXP = es.UniversalEvidenceScorer.SEMANTIC_EXPANSIONS

    print("=" * 78)
    print("M30-42 Phase 4b: Unseen generic set — Baseline vs ablation")
    print("=" * 78)
    print(f"Questions: {len(qs)}")
    print()

    def run(label, disable):
        StateReconstructor._extract_target_actor = ORIG_ACTOR
        es.UniversalEvidenceScorer.KNOWN_ACTORS = ORIG_KNOWN
        es.UniversalEvidenceScorer.SEMANTIC_EXPANSIONS = ORIG_EXP
        if "A" in disable:
            es.UniversalEvidenceScorer.KNOWN_ACTORS = []
        if "B" in disable:
            StateReconstructor._extract_target_actor = lambda self, q: None
        if "E" in disable:
            es.UniversalEvidenceScorer.SEMANTIC_EXPANSIONS = {}

        rows = []
        for q in qs:
            turns = flatten(convs[q["conversation_id"]])
            try:
                intent, units = recon.reconstruct_world(q["question"], make_records(turns))
            except Exception:
                rows.append(None); continue
            rank = None
            for i, u in enumerate(units, 1):
                m = re.search(r"\[(D\d+:\d+)\]", u.ir.raw_content)
                if m and m.group(1) == q["answer_turn"]:
                    rank = i; break
            rows.append(rank)
        return rows

    def summ(rows, label):
        r1 = sum(1 for r in rows if r == 1)
        r5 = sum(1 for r in rows if r and r <= 5)
        r10 = sum(1 for r in rows if r and r <= 10)
        found = sum(1 for r in rows if r is not None)
        print(f"{label:28s} top1={r1:2d}  top5={r5:2d}  top10={r10:2d}  found={found:2d}  (n={len(rows)})")
        return {"top1": r1, "top5": r5, "top10": r10, "found": found}

    res = {}
    for label, dis in [("Baseline (all ON)",""), ("A: KNOWN_ACTORS off","A"),
                       ("B: name-list off","B"), ("E: SEMANTIC_EXPANSIONS off","E"),
                       ("F: A+B+E off","ABE")]:
        res[label] = summ(run(label, dis), label)
    print()

    StateReconstructor._extract_target_actor = ORIG_ACTOR
    es.UniversalEvidenceScorer.KNOWN_ACTORS = ORIG_KNOWN
    es.UniversalEvidenceScorer.SEMANTIC_EXPANSIONS = ORIG_EXP

    b = res["Baseline (all ON)"]
    print("DELTA vs Baseline (unseen data):")
    for label in list(res)[1:]:
        s = res[label]
        print(f"  {label:28s} top1={s['top1']-b['top1']:+d}  top5={s['top5']-b['top5']:+d}  top10={s['top10']-b['top10']:+d}")
    print()

    out = PROJECT_ROOT / "results/m30_42"
    (out / "unseen_ablation.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'unseen_ablation.json'}")


if __name__ == "__main__":
    main()
