#!/usr/bin/env python3
"""M30-36 Phase 1: Paired Track A/C failure attribution.

Re-runs Track A and Track C with per-question output saved.
Then pairs results to classify each question.
"""
import json, re, sys, urllib.request
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}

STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "did", "does", "has", "have",
    "had", "will", "would", "could", "should", "can", "may", "might", "must",
    "this", "that", "these", "those", "when", "where", "what", "which", "who",
    "whom", "whose", "why", "how", "not", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "than",
    "too", "very", "just", "also", "now", "then", "here", "there", "once",
    "again", "further", "else", "ever", "never", "always", "often", "sometimes",
    "usually", "already", "yet", "still", "even", "back", "away", "down", "off",
    "over", "under", "above", "below", "between", "through", "during", "before",
    "after", "about", "into", "onto", "upon", "out", "up", "far", "near",
    "inside", "outside", "behind", "beyond", "across", "around", "along",
    "among", "against", "within", "without", "toward", "towards", "per", "via",
    "like", "unlike", "despite", "except", "regarding", "concerning",
    "according", "due", "owing", "thanks", "regardless", "irrespective",
    "notwithstanding", "whether", "either", "neither", "both", "whatever",
    "whenever", "wherever", "however", "whichever", "whoever", "whomever",
    "whosever", "whatsoever", "wheresoever", "whensoever", "howsoever",
    "whosoever", "whomsoever", "whosesoever",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


def extract_keywords(text):
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return set(w for w in words if w not in STOPWORDS)


def parse_session_datetime(s):
    if not s:
        return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", s)
    if m and m.group(2).lower() in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None


def flatten_conversation(conv):
    turns = []
    for key in sorted(conv.keys()):
        if key.endswith("_date_time") or not isinstance(conv[key], list):
            continue
        session_dt = conv.get(f"{key}_date_time", "")
        for turn in conv[key]:
            turns.append({
                "dia_id": turn["dia_id"],
                "speaker": turn["speaker"],
                "text": turn["text"],
                "session_datetime": session_dt,
            })
    return turns


def expanded_retrieval(question, turns):
    question_kw = extract_keywords(question)
    q_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", question))
    
    scored = []
    for i, t in enumerate(turns):
        text = t["text"]
        turn_kw = extract_keywords(text)
        overlap = question_kw & turn_kw
        score = len(overlap)
        
        entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
        entity_overlap = entities & q_entities
        score += len(entity_overlap) * 5
        
        rel_patterns = [
            r"\bnext month\b", r"\blast month\b", r"\bthis month\b",
            r"\bnext year\b", r"\blast year\b", r"\bone year ago\b",
            r"\b\d+\s+years?\s+ago\b", r"\byesterday\b", r"\btomorrow\b",
        ]
        for pat in rel_patterns:
            if re.search(pat, text, re.I):
                score += 3
                break
        
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
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["message"]["content"].strip()


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
    
    pred_date = extract_date(predicted)
    exp_date = extract_date(expected)
    
    if pred_date and exp_date:
        return pred_date == exp_date
    
    if re.match(r"^\d{4}$", exp_norm):
        return exp_norm in pred_norm
    
    my_pattern = r"(\d{1,2})/(\d{4})"
    pred_my = re.search(my_pattern, predicted)
    exp_my = re.search(my_pattern, expected)
    if pred_my and exp_my:
        return pred_my.groups() == exp_my.groups()
    
    days_pattern = r"(\d+)\s+days?"
    pred_days = re.search(days_pattern, predicted, re.I)
    exp_days = re.search(days_pattern, expected, re.I)
    if pred_days and exp_days:
        return pred_days.group(1) == exp_days.group(1)
    
    sess_pattern = r"session\s+(\d+)"
    pred_sess = re.search(sess_pattern, predicted, re.I)
    exp_sess = re.search(sess_pattern, expected, re.I)
    if pred_sess and exp_sess:
        return pred_sess.group(1) == exp_sess.group(1)
    
    return False


def resolve_temporal_expressions(text, turns):
    rel_patterns = [
        (r"\byesterday\b", "yesterday"),
        (r"\btomorrow\b", "tomorrow"),
        (r"\blast month\b", "last_month"),
        (r"\bnext month\b", "next_month"),
        (r"\bone year ago\b", "one_year_ago"),
    ]
    
    new_text = text
    for pattern, expr_type in rel_patterns:
        for m in re.finditer(pattern, text, re.I):
            expr_text = m.group(0)
            for t in turns:
                if expr_text.lower() in t["text"].lower():
                    ts = parse_session_datetime(t.get("session_datetime", ""))
                    if ts:
                        dt = datetime(*ts)
                        if expr_type == "yesterday":
                            dt -= timedelta(days=1)
                        elif expr_type == "tomorrow":
                            dt += timedelta(days=1)
                        elif expr_type == "last_month":
                            dt = datetime(dt.year, dt.month, 1) - timedelta(days=1)
                            dt = datetime(dt.year, dt.month, 1)
                        elif expr_type == "next_month":
                            dt = datetime(dt.year, dt.month, 1) + timedelta(days=32)
                            dt = datetime(dt.year, dt.month, 1)
                        elif expr_type == "one_year_ago":
                            dt = datetime(dt.year - 1, dt.month, dt.day)
                        month_name = MONTH_NAMES.get(dt.month, "")
                        date_str = f"{month_name} {dt.day}, {dt.year}"
                        new_text = new_text.replace(expr_text, date_str)
                    break
    return new_text


def extract_answer(text, category):
    if category in ["direct_fact", "paraphrase", "relative_date", "granularity"]:
        m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", text, re.I)
        if m:
            return m.group(0)
        m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}", text, re.I)
        if m:
            return m.group(0)
        m = re.search(r"\d{1,2}/\d{1,2}/\d{4}", text)
        if m:
            return m.group(0)
        m = re.search(r"\b(19|20)\d{2}\b", text)
        if m:
            return m.group(0)
    return text


def main():
    print("=" * 78)
    print("M30-36 Phase 1: Paired Track A/C Failure Attribution")
    print("=" * 78)
    print()
    
    test_data = json.loads((PROJECT_ROOT / "results/m30_35/test_data.json").read_text(encoding="utf-8"))
    questions = test_data["questions"]
    conversations = test_data["conversations"]
    
    flat_conversations = {}
    for conv_id, conv in conversations.items():
        flat_conversations[conv_id] = flatten_conversation(conv)
    
    results = []
    
    for q in questions:
        conv_id = q["conversation_id"]
        turns = flat_conversations[conv_id]
        cat = q["category"]
        expected = q["expected_answer"]
        
        # Track A: Reader-only with correct evidence
        answer_turn = None
        for t in turns:
            if t["dia_id"] == q["answer_turn"]:
                answer_turn = t
                break
        
        track_a_output = ""
        track_a_correct = None
        if answer_turn:
            evidence = f"- {answer_turn['dia_id']} | {answer_turn['speaker']} | {answer_turn['text']}"
            try:
                track_a_output = call_ollama(q["question"], evidence)
                track_a_correct = score_answer(track_a_output, expected)
            except Exception as e:
                track_a_output = f"ERROR: {e}"
                track_a_correct = None
        
        # Track C: End-to-End
        results_retrieval = expanded_retrieval(q["question"], turns)
        evidence_lines = []
        for score, dia_id, text, i in results_retrieval[:8]:
            ts = turns[i].get("session_datetime", "N/A")
            evidence_lines.append(f"- {dia_id} | {ts} | {text}")
        evidence_text = "\n".join(evidence_lines)
        
        track_c_output = ""
        track_c_correct = None
        try:
            reader_output = call_ollama(q["question"], evidence_text)
            resolved_output = resolve_temporal_expressions(reader_output, turns)
            track_c_output = extract_answer(resolved_output, cat)
            track_c_correct = score_answer(track_c_output, expected)
        except Exception as e:
            track_c_output = f"ERROR: {e}"
            track_c_correct = None
        
        # Classification
        if track_a_correct is None or track_c_correct is None:
            classification = "INVALID_OR_UNVERIFIED"
        elif track_a_correct and not track_c_correct:
            classification = "A_CORRECT_C_WRONG"
        elif not track_a_correct and track_c_correct:
            classification = "A_WRONG_C_CORRECT"
        elif track_a_correct and track_c_correct:
            classification = "BOTH_CORRECT"
        else:
            classification = "BOTH_WRONG"
        
        results.append({
            "question_id": q["question_id"],
            "category": cat,
            "expected": expected,
            "answer_turn": q["answer_turn"],
            "track_a_output": track_a_output,
            "track_a_correct": track_a_correct,
            "track_c_output": track_c_output,
            "track_c_correct": track_c_correct,
            "classification": classification,
        })
    
    # Summary
    from collections import Counter
    class_counts = Counter(r["classification"] for r in results)
    
    print("Classification summary:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    print(f"  Total: {len(results)}")
    print()
    
    # Show A_CORRECT_C_WRONG cases
    a_correct_c_wrong = [r for r in results if r["classification"] == "A_CORRECT_C_WRONG"]
    print(f"A_CORRECT_C_WRONG: {len(a_correct_c_wrong)} cases")
    for r in a_correct_c_wrong[:5]:
        print(f"  {r['question_id']} [{r['category']}]: A={r['track_a_output'][:40]} C={r['track_c_output'][:40]}")
    print()
    
    # Save
    output_path = PROJECT_ROOT / "results/m30_36/paired_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
