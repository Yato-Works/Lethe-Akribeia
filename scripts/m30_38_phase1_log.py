#!/usr/bin/env python3
"""M30-38 Phase 1: Re-run M30-37 frozen evaluation with per-question logging.

Reuses the exact same retrieval functions and scoring from M30-37.
Saves per-question results for Baseline and Treatment.
"""
import json, re, sys, urllib.request
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}

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


def extract_keywords(text):
    return set(w for w in re.findall(r"\b[a-z]{3,}\b", text.lower()) if w not in STOPWORDS)


def flatten_conversation(conv):
    turns = []
    for key in sorted(conv.keys()):
        if key.endswith("_date_time") or not isinstance(conv[key], list):
            continue
        session_dt = conv.get(f"{key}_date_time", "")
        for turn in conv[key]:
            turns.append({"dia_id": turn["dia_id"], "speaker": turn["speaker"],
                          "text": turn["text"], "session_datetime": session_dt})
    return turns


def retrieval_baseline(question, turns):
    q_kw = extract_keywords(question)
    q_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", question))
    scored = []
    for i, t in enumerate(turns):
        score = len(q_kw & extract_keywords(t["text"]))
        entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", t["text"]))
        score += len(entities & q_entities) * 5
        if score > 0:
            scored.append((score, t["dia_id"], t["text"], i))
    scored.sort(key=lambda x: -x[0])
    return scored


def retrieval_treatment(question, turns):
    q_kw = extract_keywords(question)
    q_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", question))
    target_speakers = q_entities
    scored = []
    for i, t in enumerate(turns):
        score = len(q_kw & extract_keywords(t["text"]))
        entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", t["text"]))
        score += len(entities & q_entities) * 5
        if t["speaker"] in target_speakers:
            score += 8
        if score > 0:
            scored.append((score, t["dia_id"], t["text"], i))
    scored.sort(key=lambda x: -x[0])
    return scored


def call_ollama(question, evidence_text, model="qwen2.5:7b-instruct"):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer the question based on the conversation context. Be concise."},
            {"role": "user", "content": f"Context:\n{evidence_text}\n\nQuestion: {question}\n\nAnswer:"},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]["content"].strip()


def extract_date(text):
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})", text, re.I)
    if m: return (MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m: return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", text, re.I)
    if m: return (MONTHS[m.group(1).lower()], None, int(m.group(2)))
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m: return (None, None, int(m.group(0)))
    return None


def score_answer(predicted, expected):
    if expected == "I don't know.":
        return "don't know" in predicted.lower()
    if predicted.lower().strip().rstrip(".") == expected.lower().strip().rstrip("."):
        return True
    pd, ed = extract_date(predicted), extract_date(expected)
    if pd and ed:
        return pd == ed
    if re.match(r"^\d{4}$", expected.strip()):
        return expected.strip() in predicted
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


def extract_answer(text, category):
    if category in ["direct_fact", "paraphrase", "relative_date", "granularity"]:
        for pat in [
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
            r"\d{1,2}/\d{1,2}/\d{4}",
            r"\b(19|20)\d{2}\b",
        ]:
            m = re.search(pat, text, re.I)
            if m: return m.group(0)
    return text


def evaluate_with_log(questions, conversations, retrieval_fn, label):
    flat = {cid: flatten_conversation(c) for cid, c in conversations.items()}
    rows = []
    
    for q in questions:
        turns = flat[q["conversation_id"]]
        results = retrieval_fn(q["question"], turns)
        ids = [x[1] for x in results]
        answer_dia = q["answer_turn"]
        
        rank = None
        for i, rid in enumerate(ids, 1):
            if rid == answer_dia:
                rank = i
                break
        
        evidence_turns = results[:8]
        evidence_ids = [x[1] for x in evidence_turns]
        
        evidence = "\n".join(f"- {d} | {turns[i].get('session_datetime','')} | {txt}" for _, d, txt, i in evidence_turns)
        
        reader_out = ""
        final_out = ""
        correct = False
        try:
            reader_out = call_ollama(q["question"], evidence)
            final_out = extract_answer(reader_out, q["category"])
            correct = score_answer(final_out, q["expected_answer"])
        except Exception as e:
            reader_out = f"ERROR: {e}"
        
        rows.append({
            "question_id": q["question_id"],
            "category": q["category"],
            "question": q["question"],
            "expected": q["expected_answer"],
            "answer_turn": answer_dia,
            "retrieval_rank": rank,
            "retrieval_count": len(results),
            "evidence_ids_in_input": evidence_ids,
            "answer_turn_in_input": answer_dia in evidence_ids,
            "reader_output": reader_out,
            "final_output": final_out,
            "correct": correct,
        })
    
    n_correct = sum(1 for r in rows if r["correct"])
    print(f"  {label}: {n_correct}/{len(rows)}")
    return rows


def main():
    print("=" * 78)
    print("M30-38 Phase 1: Per-question logging (Frozen 120)")
    print("=" * 78)
    print()
    
    frozen = json.loads((PROJECT_ROOT / "results/m30_37/frozen_data.json").read_text(encoding="utf-8"))
    
    print("Baseline:")
    base_rows = evaluate_with_log(frozen["questions"], frozen["conversations"], retrieval_baseline, "Baseline")
    print("Treatment:")
    treat_rows = evaluate_with_log(frozen["questions"], frozen["conversations"], retrieval_treatment, "Treatment")
    print()
    
    # Pair and classify
    base_by_id = {r["question_id"]: r for r in base_rows}
    paired = []
    for tr in treat_rows:
        br = base_by_id[tr["question_id"]]
        if br["correct"] and not tr["correct"]:
            cls = "REGRESSED"
        elif not br["correct"] and tr["correct"]:
            cls = "RECOVERED"
        elif br["correct"] and tr["correct"]:
            cls = "UNCHANGED_CORRECT"
        else:
            cls = "UNCHANGED_WRONG"
        
        paired.append({
            "question_id": tr["question_id"],
            "category": tr["category"],
            "expected": tr["expected"],
            "answer_turn": tr["answer_turn"],
            "baseline_correct": br["correct"],
            "baseline_rank": br["retrieval_rank"],
            "baseline_answer_in_input": br["answer_turn_in_input"],
            "baseline_reader": br["reader_output"],
            "baseline_final": br["final_output"],
            "treatment_correct": tr["correct"],
            "treatment_rank": tr["retrieval_rank"],
            "treatment_answer_in_input": tr["answer_turn_in_input"],
            "treatment_reader": tr["reader_output"],
            "treatment_final": tr["final_output"],
            "classification": cls,
        })
    
    from collections import Counter
    counts = Counter(p["classification"] for p in paired)
    
    print("=" * 78)
    print("CLASSIFICATION")
    print("=" * 78)
    for k in ["RECOVERED", "REGRESSED", "UNCHANGED_CORRECT", "UNCHANGED_WRONG"]:
        print(f"  {k}: {counts.get(k, 0)}")
    print(f"  Total: {sum(counts.values())}")
    print()
    
    base_total = sum(1 for p in paired if p["baseline_correct"])
    treat_total = sum(1 for p in paired if p["treatment_correct"])
    print(f"Baseline correct: {base_total}/120")
    print(f"Treatment correct: {treat_total}/120")
    print(f"Net: {treat_total - base_total}")
    print()
    
    out_dir = PROJECT_ROOT / "results/m30_38"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_results.json").write_text(json.dumps(paired, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_dir / 'paired_results.json'}")


if __name__ == "__main__":
    main()
