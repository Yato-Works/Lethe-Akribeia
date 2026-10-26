#!/usr/bin/env python3
"""M30-39 Phase 3: Verify format-mismatch claims with the actual scorer.

For each format-mismatch case, test whether a format-normalized version of
the Reader's (already correct) answer would pass the existing scorer.

This does NOT change the scorer; it only measures how much of the gap is
answer-format vs genuine reasoning.
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
paired = json.loads((PROJECT_ROOT / "results/m30_38/paired_results.json").read_text(encoding="utf-8"))

fails = [p for p in paired if p["treatment_answer_in_input"] and not p["treatment_correct"]]


def score_answer(predicted, expected):
    """Same scorer logic used in M30-37/38."""
    MONTHS = {m.lower(): i for i, m in enumerate([
        "january","february","march","april","may","june","july","august",
        "september","october","november","december"], start=1)}
    if expected == "I don't know.":
        return "don't know" in predicted.lower()
    if predicted.lower().strip().rstrip(".") == expected.lower().strip().rstrip("."):
        return True
    def ed(t):
        m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})", t, re.I)
        if m: return (MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3)))
        m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", t)
        if m: return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", t, re.I)
        if m: return (MONTHS[m.group(1).lower()], None, int(m.group(2)))
        m = re.search(r"\b(19|20)\d{2}\b", t)
        if m: return (None, None, int(m.group(0)))
        return None
    pd, gd = ed(predicted), ed(expected)
    if pd and gd: return pd == gd
    if re.match(r"^\d{4}$", expected.strip()): return expected.strip() in predicted
    m = r"(\d{1,2})/(\d{4})"
    pm, em = re.search(m, predicted), re.search(m, expected)
    if pm and em: return pm.groups() == em.groups()
    dp = re.search(r"(\d+)\s+days?", predicted, re.I)
    de = re.search(r"(\d+)\s+days?", expected, re.I)
    if dp and de: return dp.group(1) == de.group(1)
    sp = re.search(r"session\s+(\d+)", predicted, re.I)
    se = re.search(r"session\s+(\d+)", expected, re.I)
    if sp and se: return sp.group(1) == se.group(1)
    return False


def normalize(predicted, expected, category):
    """Deterministic answer-format normalization (NOT a scorer change)."""
    out = predicted

    # answerability: map semantic abstention to the literal marker
    if expected == "I don't know.":
        low = out.lower()
        if any(s in low for s in ["does not provide", "no information", "not specified",
                                  "cannot determine", "unable to", "doesn't provide"]):
            return "I don't know."

    # multi_session: extract "Session N" from a descriptive sentence
    if category == "multi_session":
        m = re.search(r"session[s]?\s+(\d+)", out, re.I)
        if m:
            return f"Session {m.group(1)}"
        # Reader may say "the sessions from <date>" - look for ordinal hints
        if re.search(r"\b(1st|first)\b", out, re.I):
            return "Session 1"
        if re.search(r"\b(2nd|second)\b", out, re.I):
            return "Session 2"
        if re.search(r"\b(3rd|third)\b", out, re.I):
            return "Session 3"

    # granularity: reduce a full date to M/YYYY when Gold asks for month+year
    if category == "granularity" and re.match(r"^\d{1,2}/\d{4}$", expected.strip()):
        m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(\d{4})", out, re.I)
        if m:
            MONTHS = {mm.lower(): i for i, mm in enumerate([
                "january","february","march","april","may","june","july","august",
                "september","october","november","december"], start=1)}
            return f"{MONTHS[m.group(1).lower()]}/{m.group(2)}"

    return out


print("=" * 78)
print("M30-39 Phase 3: Format-Mismatch Verification")
print("=" * 78)
print()

recovered_by_format = 0
still_wrong = 0
rows = []

for p in fails:
    cat = p["category"]
    exp = p["expected"]
    reader = p["treatment_reader"]

    before = score_answer(p["treatment_final"], exp)
    normalized = normalize(reader, exp, cat)
    after = score_answer(normalized, exp)

    if not before and after:
        recovered_by_format += 1
        tag = "FORMAT_FIX"
    elif before:
        tag = "already correct (unexpected)"
    else:
        still_wrong += 1
        tag = "still wrong"

    rows.append({
        "qid": p["question_id"], "category": cat, "expected": exp,
        "reader": reader[:70], "normalized": normalized[:40], "tag": tag,
    })

for r in rows:
    if r["tag"] == "FORMAT_FIX":
        print(f"  [FORMAT_FIX] {r['qid']} [{r['category']}] expected={r['expected']}")
        print(f"    reader:     {r['reader']}")
        print(f"    normalized: {r['normalized']}")
print()

print(f"Recovered by deterministic format normalization: {recovered_by_format}/{len(fails)}")
print(f"Still wrong after normalization: {still_wrong}/{len(fails)}")
print()

# Net effect on Frozen 120
base_correct = 28
treat_correct = 32
projected = treat_correct + recovered_by_format
print(f"Frozen projection:")
print(f"  Baseline:                 {base_correct}/120")
print(f"  Treatment (M30-38):       {treat_correct}/120")
print(f"  Treatment + format norm:  {projected}/120 (projected)")
print()

out = PROJECT_ROOT / "results/m30_39"
out.mkdir(parents=True, exist_ok=True)
(out / "format_verification.json").write_text(json.dumps({
    "recovered_by_format": recovered_by_format,
    "still_wrong": still_wrong,
    "projected_frozen": projected,
    "rows": rows,
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved: {out / 'format_verification.json'}")
