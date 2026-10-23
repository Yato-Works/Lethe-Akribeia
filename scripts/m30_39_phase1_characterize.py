#!/usr/bin/env python3
"""M30-39 Phase 1: Characterize the Reader wrong-turn-selection failure.

For the 27 cases where evidence was in Reader input but Reader failed:
- How many distinct dated turns were in the input?
- Was the correct turn uniquely identifiable (unique speaker/date)?
- Did Reader pick wrong speaker, or right speaker wrong date?
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

paired = json.loads((PROJECT_ROOT / "results/m30_38/paired_results.json").read_text(encoding="utf-8"))
frozen = json.loads((PROJECT_ROOT / "results/m30_37/frozen_data.json").read_text(encoding="utf-8"))
conversations = frozen["conversations"]
q_by_id = {q["question_id"]: q for q in frozen["questions"]}

DATE_RE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})", re.I)
MONTHS = {m.lower(): i for i, m in enumerate([
    "january","february","march","april","may","june","july","august",
    "september","october","november","december"], start=1)}


def flatten(conv):
    turns = []
    for key in sorted(conv.keys()):
        if key.endswith("_date_time") or not isinstance(conv[key], list):
            continue
        for turn in conv[key]:
            turns.append({"dia_id": turn["dia_id"], "speaker": turn["speaker"], "text": turn["text"]})
    return turns


def parse_expected(exp):
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", exp)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def main():
    print("=" * 78)
    print("M30-39 Phase 1: Reader Wrong-Turn-Selection Characterization")
    print("=" * 78)
    print()

    fails = [p for p in paired if p["treatment_answer_in_input"] and not p["treatment_correct"]]
    print(f"Cases (evidence in input, Reader wrong): {len(fails)}")
    print()

    categories = {
        "UNIQUE_CORRECT_TURN": 0,      # only one dated turn -> Reader should get it
        "AMBIGUOUS_SAME_SPEAKER": 0,   # multiple dated turns by the target speaker
        "AMBIGUOUS_MULTI_SPEAKER": 0,  # multiple dated turns, different speakers
        "NO_DATED_TURN": 0,            # no dated turn in input
    }

    details = []
    for p in fails:
        q = q_by_id[p["question_id"]]
        turns = flatten(conversations[q["conversation_id"]])
        input_ids = set(p.get("evidence_ids_in_input", []))
        input_turns = [t for t in turns if t["dia_id"] in input_ids]

        dated = []
        for t in input_turns:
            for m in DATE_RE.finditer(t["text"]):
                dated.append((t["speaker"], (MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3))), t["dia_id"]))

        speakers = set(d[0] for d in dated)
        target_speaker = q.get("question", "")
        # speaker the question asks about (first capitalized name in question)
        names = re.findall(r"\b([A-Z][a-z]{2,})\b", q["question"])
        target = names[0] if names else None

        if not dated:
            cat = "NO_DATED_TURN"
        elif len(dated) == 1:
            cat = "UNIQUE_CORRECT_TURN"
        elif len(speakers) == 1:
            cat = "AMBIGUOUS_SAME_SPEAKER"
        else:
            cat = "AMBIGUOUS_MULTI_SPEAKER"
        categories[cat] += 1

        # Did Reader pick a turn by the target speaker?
        reader_date = DATE_RE.search(p["treatment_reader"])
        picked_speaker = None
        if reader_date:
            rd = (MONTHS[reader_date.group(1).lower()], int(reader_date.group(2)), int(reader_date.group(3)))
            for sp, dt, did in dated:
                if dt == rd:
                    picked_speaker = sp
                    break

        details.append({
            "qid": p["question_id"],
            "category": p["category"],
            "target_speaker": target,
            "n_dated_turns": len(dated),
            "n_speakers": len(speakers),
            "classification": cat,
            "expected": p["expected"],
            "reader_final": p["treatment_final"][:40],
            "picked_speaker": picked_speaker,
        })

    print("Classification:")
    for k, v in categories.items():
        print(f"  {k}: {v}")
    print()

    print("Details:")
    for d in details[:15]:
        print(f"  {d['qid']} [{d['category']}] target={d['target_speaker']}")
        print(f"    dated_turns={d['n_dated_turns']}, speakers={d['n_speakers']}, {d['classification']}")
        print(f"    expected={d['expected']}, reader={d['reader_final']}, picked_speaker={d['picked_speaker']}")
    print()

    out = PROJECT_ROOT / "results/m30_39"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reader_failure_characterization.json").write_text(
        json.dumps({"categories": categories, "details": details}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"Saved: {out / 'reader_failure_characterization.json'}")


if __name__ == "__main__":
    main()
