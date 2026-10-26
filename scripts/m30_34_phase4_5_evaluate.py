#!/usr/bin/env python3
"""M30-34 Phase 4-5: Track A/B/C evaluation on frozen test set.

Track A: Reader-only (correct evidence, measure Reader ability)
Track B: Retrieval-only (recall@k)
Track C: End-to-End (full pipeline)

Uses frozen test data from results/m30_34/test_data.json.
No Bucket A or holdout data is used.
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


def score_answer(predicted, expected, category):
    if expected == "I don't know.":
        return "don't know" in predicted.lower()
    
    pred_norm = predicted.lower().strip().rstrip(".")
    exp_norm = expected.lower().strip().rstrip(".")
    
    if pred_norm == exp_norm:
        return True
    
    # Date matching
    date_pattern = r"(\d{1,2})/(\d{1,2})/(\d{4})"
    pred_date = re.search(date_pattern, predicted)
    exp_date = re.search(date_pattern, expected)
    
    if pred_date and exp_date:
        pm, pd, py = int(pred_date.group(1)), int(pred_date.group(2)), int(pred_date.group(3))
        em, ed, ey = int(exp_date.group(1)), int(exp_date.group(2)), int(exp_date.group(3))
        return (pm, pd, py) == (em, ed, ey)
    
    # Month name matching
    month_pattern = r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s*(\d{4})"
    pred_month = re.search(month_pattern, predicted, re.I)
    exp_month = re.search(month_pattern, expected, re.I)
    
    if pred_month and exp_month:
        pm = MONTHS.get(pred_month.group(1).lower(), 0)
        pd = int(pred_month.group(2))
        py = int(pred_month.group(3))
        em = MONTHS.get(exp_month.group(1).lower(), 0)
        ed = int(exp_month.group(2))
        ey = int(exp_month.group(3))
        return (pm, pd, py) == (em, ed, ey)
    
    # Year only
    if re.match(r"^\d{4}$", exp_norm):
        return exp_norm in pred_norm
    
    # Month/Year
    my_pattern = r"(\d{1,2})/(\d{4})"
    pred_my = re.search(my_pattern, predicted)
    exp_my = re.search(my_pattern, expected)
    if pred_my and exp_my:
        return pred_my.groups() == exp_my.groups()
    
    # Days
    days_pattern = r"(\d+)\s+days?"
    pred_days = re.search(days_pattern, predicted, re.I)
    exp_days = re.search(days_pattern, expected, re.I)
    if pred_days and exp_days:
        return pred_days.group(1) == exp_days.group(1)
    
    # Session
    sess_pattern = r"session\s+(\d+)"
    pred_sess = re.search(sess_pattern, predicted, re.I)
    exp_sess = re.search(sess_pattern, expected, re.I)
    if pred_sess and exp_sess:
        return pred_sess.group(1) == exp_sess.group(1)
    
    return False


def resolve_temporal_expressions(text, turns):
    """Resolve relative date expressions using turn timestamps."""
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
            # Find the turn with this expression
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
    """Extract answer from Reader output."""
    if category in ["direct_fact", "paraphrase", "relative_date", "granularity"]:
        # Try "Month Day, Year" first
        m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", text, re.I)
        if m:
            return m.group(0)
        # Try "Month Year"
        m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}", text, re.I)
        if m:
            return m.group(0)
        # Try "MM/DD/YYYY"
        m = re.search(r"\d{1,2}/\d{1,2}/\d{4}", text)
        if m:
            return m.group(0)
        # Year only
        m = re.search(r"\b(19|20)\d{2}\b", text)
        if m:
            return m.group(0)
    return text


def main():
    print("=" * 78)
    print("M30-34 Phase 4-5: Track A/B/C Evaluation")
    print("=" * 78)
    print()
    
    # Load frozen test data
    test_data = json.loads((PROJECT_ROOT / "results/m30_34/test_data.json").read_text(encoding="utf-8"))
    questions = test_data["questions"]
    conversations = test_data["conversations"]
    
    print(f"Questions: {len(questions)}")
    print(f"Conversations: {len(conversations)}")
    print()
    
    # Flatten all conversations
    flat_conversations = {}
    for conv_id, conv in conversations.items():
        flat_conversations[conv_id] = flatten_conversation(conv)
    
    # =========================================================================
    # Track B: Retrieval-only
    # =========================================================================
    print("--- Track B: Retrieval-only ---")
    
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    total_answerable = 0
    
    for q in questions:
        conv_id = q["conversation_id"]
        turns = flat_conversations[conv_id]
        answer_dia_id = q["answer_turn"]
        
        results = expanded_retrieval(q["question"], turns)
        retrieved_ids = [r[1] for r in results]
        
        if answer_dia_id in retrieved_ids[:1]:
            recall_at_1 += 1
        if answer_dia_id in retrieved_ids[:5]:
            recall_at_5 += 1
        if answer_dia_id in retrieved_ids[:10]:
            recall_at_10 += 1
        
        if q["expected_answer"] != "I don't know.":
            total_answerable += 1
    
    print(f"Recall@1: {recall_at_1}/{len(questions)} ({100*recall_at_1/len(questions):.1f}%)")
    print(f"Recall@5: {recall_at_5}/{len(questions)} ({100*recall_at_5/len(questions):.1f}%)")
    print(f"Recall@10: {recall_at_10}/{len(questions)} ({100*recall_at_10/len(questions):.1f}%)")
    print(f"Answerable: {total_answerable}")
    print()
    
    # =========================================================================
    # Track A: Reader-only (correct evidence)
    # =========================================================================
    print("--- Track A: Reader-only ---")
    
    track_a_correct = 0
    track_a_total = 0
    track_a_abstain_correct = 0
    track_a_abstain_total = 0
    
    for q in questions:
        conv_id = q["conversation_id"]
        turns = flat_conversations[conv_id]
        
        # Find the answer turn
        answer_turn = None
        for t in turns:
            if t["dia_id"] == q["answer_turn"]:
                answer_turn = t
                break
        
        if not answer_turn:
            continue
        
        evidence = f"- {answer_turn['dia_id']} | {answer_turn['speaker']} | {answer_turn['text']}"
        
        try:
            reader_output = call_ollama(q["question"], evidence)
            correct = score_answer(reader_output, q["expected_answer"], q["category"])
            
            if q["expected_answer"] == "I don't know.":
                track_a_abstain_total += 1
                if correct:
                    track_a_abstain_correct += 1
            else:
                track_a_total += 1
                if correct:
                    track_a_correct += 1
                    
        except Exception as e:
            print(f"  ERROR {q['question_id']}: {e}")
    
    print(f"Track A (answerable): {track_a_correct}/{track_a_total} ({100*track_a_correct/track_a_total:.1f}%)")
    print(f"Track A (abstention): {track_a_abstain_correct}/{track_a_abstain_total}")
    print()
    
    # =========================================================================
    # Track C: End-to-End
    # =========================================================================
    print("--- Track C: End-to-End ---")
    
    track_c_correct = 0
    track_c_total = 0
    category_correct = {}
    category_total = {}
    
    for q in questions:
        conv_id = q["conversation_id"]
        turns = flat_conversations[conv_id]
        cat = q["category"]
        category_total[cat] = category_total.get(cat, 0) + 1
        
        # Run retrieval
        results = expanded_retrieval(q["question"], turns)
        
        # Build evidence text (top 8)
        evidence_lines = []
        for score, dia_id, text, i in results[:8]:
            ts = turns[i].get("session_datetime", "N/A")
            evidence_lines.append(f"- {dia_id} | {ts} | {text}")
        evidence_text = "\n".join(evidence_lines)
        
        try:
            reader_output = call_ollama(q["question"], evidence_text)
            
            # Temporal resolution
            resolved_output = resolve_temporal_expressions(reader_output, turns)
            
            # Extract answer
            final_output = extract_answer(resolved_output, cat)
            
            correct = score_answer(final_output, q["expected_answer"], cat)
            if correct:
                track_c_correct += 1
                category_correct[cat] = category_correct.get(cat, 0) + 1
            track_c_total += 1
            
        except Exception as e:
            print(f"  ERROR {q['question_id']}: {e}")
    
    print(f"Track C: {track_c_correct}/{track_c_total} ({100*track_c_correct/track_c_total:.1f}%)")
    print()
    
    # Category breakdown
    print("Category breakdown (Track C):")
    for cat in sorted(category_total.keys()):
        c = category_correct.get(cat, 0)
        t = category_total[cat]
        print(f"  {cat}: {c}/{t} ({100*c/t:.1f}%)")
    
    print()
    
    # Interpretation
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print()
    
    track_a_rate = track_a_correct / track_a_total if track_a_total > 0 else 0
    track_c_rate = track_c_correct / track_c_total if track_c_total > 0 else 0
    
    print(f"Track A (Reader-only): {track_a_rate:.1%}")
    print(f"Track C (End-to-End): {track_c_rate:.1%}")
    print(f"Track B (Recall@5): {recall_at_5/len(questions):.1%}")
    print()
    
    if track_a_rate > 0.5 and track_c_rate < 0.3:
        print("DIAGNOSIS: Retrieval/evidence delivery is the primary bottleneck.")
    elif track_a_rate < 0.3 and track_c_rate < 0.3:
        print("DIAGNOSIS: Both Reader and Retrieval have significant issues.")
    elif track_a_rate > 0.5 and track_c_rate > 0.5:
        print("DIAGNOSIS: System performs well on synthetic data.")
    else:
        print("DIAGNOSIS: Mixed results - further investigation needed.")
    
    # Save
    output = {
        "track_a": {
            "correct": track_a_correct,
            "total": track_a_total,
            "abstain_correct": track_a_abstain_correct,
            "abstain_total": track_a_abstain_total,
        },
        "track_b": {
            "recall_at_1": recall_at_1,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "total": len(questions),
        },
        "track_c": {
            "correct": track_c_correct,
            "total": track_c_total,
            "category_correct": category_correct,
            "category_total": category_total,
        },
    }
    
    output_path = PROJECT_ROOT / "results/m30_34/track_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
