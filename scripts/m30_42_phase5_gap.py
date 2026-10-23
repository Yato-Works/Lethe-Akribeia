#!/usr/bin/env python3
"""M30-42 Phase 5: Generalization gap analysis.

Bucket A top-1 = 16/28 (57%) but unseen generic top-1 = 4/120 (3.3%).
Is this a real capability gap, or an artifact of the synthetic data
(different lexical overlap, question phrasing)?

Diagnostic: compare the lexical overlap between question and answer-turn
in each set, which is the primary retrieval signal.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STOP = {"the","and","for","are","was","were","did","does","has","have","had",
        "will","would","could","should","can","may","might","must","this","that",
        "when","where","what","which","who","why","how","not","all","any","both",
        "with","from","into","onto","upon","out","up","far","near","about","after",
        "before","during","between","through","over","under","again","then","once",
        "january","february","march","april","may","june","july","august",
        "september","october","november","december"}


def kw(t):
    return set(w for w in re.findall(r"\b[a-z]{3,}\b", t.lower()) if w not in STOP)


def analyze(questions, convs, label):
    overlaps = []
    for q in questions:
        turns = []
        conv = convs[q["conversation_id"]]
        for k in sorted(conv.keys()):
            if k.endswith("_date_time") or not isinstance(conv[k], list):
                continue
            turns.extend(conv[k])
        ans = next((t for t in turns if t["dia_id"] == q["answer_turn"]), None)
        if not ans:
            continue
        qk = kw(q["question"])
        ak = kw(ans["text"])
        # exclude month names from the overlap count (they leak the answer)
        ov = (qk & ak)
        overlaps.append(len(ov))

    avg = sum(overlaps)/len(overlaps) if overlaps else 0
    zero = sum(1 for o in overlaps if o == 0)
    print(f"{label}: n={len(overlaps)}  avg_q_a_overlap={avg:.2f}  zero_overlap={zero} ({100*zero/len(overlaps):.0f}%)")
    return {"n": len(overlaps), "avg_overlap": avg, "zero_overlap": zero}


def main():
    # Bucket A
    b = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    b_cases = {c["question_id"]: c for c in b["cases"]}
    b_convs = b["conversations"]
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    b_targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")
    # Build bucket-A question records with a synthetic "answer_turn" = gold evidence
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    gold_by_q = {}
    for conv in locomo:
        sid = conv.get("sample_id","")
        for qa in conv.get("qa", []):
            gold_by_q[(sid, qa.get("question","").strip().lower())] = qa.get("evidence", []) or []

    b_questions = []
    for qid in b_targets:
        c = b_cases[qid]
        gold = gold_by_q.get((c["source_conversation"], c["question"].strip().lower()), [])
        if not gold:
            continue
        b_questions.append({
            "question": c["question"],
            "conversation_id": c["source_conversation"],
            "answer_turn": gold[0],
        })
    # Bucket A conversations are flat lists; wrap for the analyzer
    b_convs_wrapped = {}
    for k, v in b_convs.items():
        if isinstance(v, list):
            b_convs_wrapped[k] = {"session_1": v}

    # Unseen generic
    g = json.loads((PROJECT_ROOT / "results/m30_42/frozen_generic.json").read_text(encoding="utf-8"))

    print("=" * 78)
    print("M30-42 Phase 5: Generalization gap analysis")
    print("=" * 78)
    print()
    a = analyze(b_questions, b_convs_wrapped, "Bucket A 28")
    print()
    c = analyze(g["questions"], g["conversations"], "Unseen generic 120")
    print()

    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print()
    print(f"Bucket A:     avg question<->answer keyword overlap = {a['avg_overlap']:.2f}")
    print(f"Unseen gen:   avg question<->answer keyword overlap = {c['avg_overlap']:.2f}")
    print()
    if c["avg_overlap"] < a["avg_overlap"]:
        print("The unseen set has LOWER lexical overlap between question and answer.")
        print("This is a DATA-DESIGN difference, not (only) a capability gap:")
        print("the synthetic questions use generic words ('attend', 'event') while")
        print("the answer turns use specific words ('kiln firing', 'Verdant Labs').")
        print()
        print("=> The 3.3% vs 57% gap conflates:")
        print("   (a) genuine generalization difficulty, and")
        print("   (b) a lexical-overlap difference in the test design.")
    print()

    out = PROJECT_ROOT / "results/m30_42"
    (out / "generalization_gap.json").write_text(json.dumps(
        {"bucket_a": a, "unseen_generic": c}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'generalization_gap.json'}")


if __name__ == "__main__":
    main()
