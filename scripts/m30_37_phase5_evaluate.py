#!/usr/bin/env python3
"""M30-37 Phase 5: Development selection + Frozen evaluation.

ONE change: speaker-aware retrieval.
When the question mentions a person's name, boost turns spoken BY that person.

Development (40): compare Baseline vs Treatment, choose.
Frozen (120): evaluate once, no tuning.
"""
import json, re, sys, urllib.request
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}

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
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return set(w for w in words if w not in STOPWORDS)


def flatten_conversation(conv):
    turns = []
    for key in sorted(conv.keys()):
        if key.endswith("_date_time") or not isinstance(conv[key], list):
            continue
        session_dt = conv.get(f"{key}_date_time", "")
        for turn in conv[key]:
            turns.append({
                "dia_id": turn["dia_id"], "speaker": turn["speaker"],
                "text": turn["text"], "session_datetime": session_dt,
            })
    return turns


def retrieval_baseline(question, turns):
    """Original: keyword overlap + entity match."""
    q_kw = extract_keywords(question)
    q_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", question))
    
    scored = []
    for i, t in enumerate(turns):
        text = t["text"]
        score = len(q_kw & extract_keywords(text))
        entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
        score += len(entities & q_entities) * 5
        if score > 0:
            scored.append((score, t["dia_id"], text, i))
    scored.sort(key=lambda x: -x[0])
    return scored


def retrieval_treatment(question, turns):
    """Treatment: speaker-aware retrieval.
    
    If the question mentions a person's name, boost turns spoken BY that person.
    This addresses the failure where the question asks about "Olivia" but the
    answer turn is by the other speaker and never contains "Olivia".
    """
    q_kw = extract_keywords(question)
    q_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", question))
    
    # Identify the person the question is about
    target_speakers = q_entities  # e.g., {"Olivia"}
    
    scored = []
    for i, t in enumerate(turns):
        text = t["text"]
        score = len(q_kw & extract_keywords(text))
        entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
        score += len(entities & q_entities) * 5
        
        # NEW: boost turns spoken BY the person the question asks about
        if t["speaker"] in target_speakers:
            score += 8
        
        if score > 0:
            scored.append((score, t["dia_id"], text, i))
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
    if m:
        return (MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", text, re.I)
    if m:
        return (MONTHS[m.group(1).lower()], None, int(m.group(2)))
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        return (None, None, int(m.group(0)))
    return None


def score_answer(predicted, expected):
    if expected == "I don't know.":
        return "don't know" in predicted.lower()
    pred_norm = predicted.lower().strip().rstrip(".")
    exp_norm = expected.lower().strip().rstrip(".")
    if pred_norm == exp_norm:
        return True
    pd, ed = extract_date(predicted), extract_date(expected)
    if pd and ed:
        return pd == ed
    if re.match(r"^\d{4}$", exp_norm):
        return exp_norm in pred_norm
    m = r"(\d{1,2})/(\d{4})"
    pm, em = re.search(m, predicted), re.search(m, expected)
    if pm and em:
        return pm.groups() == em.groups()
    dp = re.search(r"(\d+)\s+days?", predicted, re.I)
    de = re.search(r"(\d+)\s+days?", expected, re.I)
    if dp and de:
        return dp.group(1) == de.group(1)
    sp = re.search(r"session\s+(\d+)", predicted, re.I)
    se = re.search(r"session\s+(\d+)", expected, re.I)
    if sp and se:
        return sp.group(1) == se.group(1)
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
            if m:
                return m.group(0)
    return text


def evaluate(questions, conversations, retrieval_fn, label):
    flat = {cid: flatten_conversation(c) for cid, c in conversations.items()}
    correct = 0
    total = 0
    cat_correct, cat_total = {}, {}
    recall1 = recall5 = recall10 = 0
    
    for q in questions:
        turns = flat[q["conversation_id"]]
        cat = q["category"]
        cat_total[cat] = cat_total.get(cat, 0) + 1
        
        results = retrieval_fn(q["question"], turns)
        ids = [x[1] for x in results]
        if q["answer_turn"] in ids[:1]: recall1 += 1
        if q["answer_turn"] in ids[:5]: recall5 += 1
        if q["answer_turn"] in ids[:10]: recall10 += 1
        
        evidence = "\n".join(f"- {d} | {turns[i].get('session_datetime','')} | {txt}" for _, d, txt, i in results[:8])
        
        try:
            out = call_ollama(q["question"], evidence)
            final = extract_answer(out, cat)
            if score_answer(final, q["expected_answer"]):
                correct += 1
                cat_correct[cat] = cat_correct.get(cat, 0) + 1
        except Exception as e:
            print(f"    ERROR {q['question_id']}: {e}")
        total += 1
    
    print(f"  {label}: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"    Recall@1={recall1}/{len(questions)}, @5={recall5}, @10={recall10}")
    return {
        "correct": correct, "total": total,
        "recall_at_1": recall1, "recall_at_5": recall5, "recall_at_10": recall10,
        "category_correct": cat_correct, "category_total": cat_total,
    }


def main():
    print("=" * 78)
    print("M30-37 Phase 5: Development + Frozen Evaluation")
    print("=" * 78)
    print()
    
    dev = json.loads((PROJECT_ROOT / "results/m30_37/dev_data.json").read_text(encoding="utf-8"))
    frozen = json.loads((PROJECT_ROOT / "results/m30_37/frozen_data.json").read_text(encoding="utf-8"))
    
    print("--- Development (40) ---")
    dev_base = evaluate(dev["questions"], dev["conversations"], retrieval_baseline, "Baseline")
    dev_treat = evaluate(dev["questions"], dev["conversations"], retrieval_treatment, "Treatment (speaker-aware)")
    print()
    
    print("--- Frozen Evaluation (120) ---")
    frozen_base = evaluate(frozen["questions"], frozen["conversations"], retrieval_baseline, "Baseline")
    frozen_treat = evaluate(frozen["questions"], frozen["conversations"], retrieval_treatment, "Treatment (speaker-aware)")
    print()
    
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Development:  Baseline {dev_base['correct']}/40 -> Treatment {dev_treat['correct']}/40")
    print(f"Frozen:       Baseline {frozen_base['correct']}/120 -> Treatment {frozen_treat['correct']}/120")
    print(f"Frozen Recall@1: {frozen_base['recall_at_1']} -> {frozen_treat['recall_at_1']}")
    print(f"Frozen Recall@5: {frozen_base['recall_at_5']} -> {frozen_treat['recall_at_5']}")
    print()
    
    out = {
        "development": {"baseline": dev_base, "treatment": dev_treat},
        "frozen": {"baseline": frozen_base, "treatment": frozen_treat},
    }
    path = PROJECT_ROOT / "results/m30_37/evaluation_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
