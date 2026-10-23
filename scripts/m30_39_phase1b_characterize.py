#!/usr/bin/env python3
"""M30-39 Phase 1b: Re-derive evidence IDs from frozen data + M30-38 retrieval.

The paired_results.json lacks evidence_ids_in_input. Recompute using the
same retrieval functions from M30-37/38.
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STOPWORDS = {
    "the","and","for","are","was","were","did","does","has","have","had","will",
    "would","could","should","can","may","might","must","this","that","these",
    "those","when","where","what","which","who","whom","whose","why","how","not",
    "all","any","both","each","few","more","most","other","some","such","only",
    "own","same","than","too","very","just","also","now","then","here","there",
    "once","again","further","else","ever","never","always","often","sometimes",
    "usually","already","yet","still","even","back","away","down","off","over",
    "under","above","below","between","through","during","before","after","about",
    "into","onto","upon","out","up","far","near","inside","outside","behind",
    "beyond","across","around","along","among","against","within","without",
    "toward","towards","per","via","like","unlike","despite","except","regarding",
    "concerning","according","due","owing","thanks","regardless","irrespective",
    "notwithstanding","whether","either","neither","both","whatever","whenever",
    "wherever","however","whichever","whoever","whomever","whosever","whatsoever",
    "wheresoever","whensoever","howsoever","whosoever","whomsoever","whosesoever",
    "january","february","march","april","may","june","july","august",
    "september","october","november","december",
}
DATE_RE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})", re.I)
MONTHS = {m.lower(): i for i, m in enumerate([
    "january","february","march","april","may","june","july","august",
    "september","october","november","december"], start=1)}


def extract_keywords(text):
    return set(w for w in re.findall(r"\b[a-z]{3,}\b", text.lower()) if w not in STOPWORDS)


def flatten(conv):
    turns = []
    for key in sorted(conv.keys()):
        if key.endswith("_date_time") or not isinstance(conv[key], list):
            continue
        for turn in conv[key]:
            turns.append({"dia_id": turn["dia_id"], "speaker": turn["speaker"], "text": turn["text"]})
    return turns


def retrieval_treatment(question, turns):
    q_kw = extract_keywords(question)
    q_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", question))
    scored = []
    for i, t in enumerate(turns):
        score = len(q_kw & extract_keywords(t["text"]))
        entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", t["text"]))
        score += len(entities & q_entities) * 5
        if t["speaker"] in q_entities:
            score += 8
        if score > 0:
            scored.append((score, t["dia_id"], t["text"], i))
    scored.sort(key=lambda x: -x[0])
    return scored


def main():
    paired = json.loads((PROJECT_ROOT / "results/m30_38/paired_results.json").read_text(encoding="utf-8"))
    frozen = json.loads((PROJECT_ROOT / "results/m30_37/frozen_data.json").read_text(encoding="utf-8"))
    conversations = frozen["conversations"]
    q_by_id = {q["question_id"]: q for q in frozen["questions"]}

    fails = [p for p in paired if p["treatment_answer_in_input"] and not p["treatment_correct"]]
    print(f"Cases: {len(fails)}")
    print()

    cats = {"UNIQUE_DATED_TURN": 0, "MULTI_DATED_TURNS": 0, "NO_DATED_TURN_IN_INPUT": 0,
            "MULTI_DATED_DIFF_SPEAKER": 0}
    details = []

    for p in fails:
        q = q_by_id[p["question_id"]]
        turns = flatten(conversations[q["conversation_id"]])
        results = retrieval_treatment(q["question"], turns)
        input_ids = [x[1] for x in results[:8]]
        input_turns = [t for t in turns if t["dia_id"] in input_ids]

        dated = []
        for t in input_turns:
            for m in DATE_RE.finditer(t["text"]):
                dated.append((t["speaker"], (MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3))), t["dia_id"]))

        speakers = set(d[0] for d in dated)
        names = re.findall(r"\b([A-Z][a-z]{2,})\b", q["question"])
        target = names[0] if names else None

        if not dated:
            cat = "NO_DATED_TURN_IN_INPUT"
        elif len(dated) == 1:
            cat = "UNIQUE_DATED_TURN"
        elif len(speakers) == 1:
            cat = "MULTI_DATED_TURNS"
        else:
            cat = "MULTI_DATED_DIFF_SPEAKER"
        cats[cat] += 1

        # Did the correct date appear among input dated turns?
        exp = q["expected_answer"]
        m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", exp)
        correct_in_input = False
        if m:
            ed = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            correct_in_input = any(d[1] == ed for d in dated)

        details.append({
            "qid": p["question_id"], "category": p["category"],
            "target": target, "n_dated": len(dated), "n_speakers": len(speakers),
            "classification": cat, "correct_date_in_input": correct_in_input,
            "expected": exp, "reader_final": p["treatment_final"][:40],
        })

    print("Classification:")
    for k, v in cats.items():
        print(f"  {k}: {v}")
    print()

    correct_in_input = sum(1 for d in details if d["correct_date_in_input"])
    print(f"Correct date present in Reader input: {correct_in_input}/{len(fails)}")
    print()

    print("Details:")
    for d in details:
        print(f"  {d['qid']} [{d['category']}] {d['classification']}")
        print(f"    dated={d['n_dated']} speakers={d['n_speakers']} correct_date_in_input={d['correct_date_in_input']}")
        print(f"    expected={d['expected']} reader={d['reader_final']}")
    print()

    out = PROJECT_ROOT / "results/m30_39"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reader_failure_characterization.json").write_text(
        json.dumps({"categories": cats, "correct_date_in_input": correct_in_input, "details": details},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'reader_failure_characterization.json'}")


if __name__ == "__main__":
    main()
