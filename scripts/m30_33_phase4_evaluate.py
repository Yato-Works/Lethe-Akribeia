#!/usr/bin/env python3
"""M30-33 Phase 4: Track A/B/C evaluation on independent test set.

Track A: Reader-only (give correct evidence, measure Reader ability)
Track B: Retrieval-only (measure recall@k)
Track C: End-to-End (full pipeline)

Uses synthetic data from results/m30_33/test_data.json.
No Bucket A or holdout data is used.
"""
import json, re, sys
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request

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
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def flatten_conversation(conv):
    """Convert synthetic conversation format to flat turn list."""
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
    """Expanded retrieval on synthetic conversation."""
    question_kw = extract_keywords(question)
    q_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", question))
    
    scored = []
    for i, t in enumerate(turns):
        text = t["text"]
        text_lower = text.lower()
        
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
    """Call Ollama API directly."""
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
    """Custom scorer for synthetic data."""
    if expected == "I don't know.":
        return "don't know" in predicted.lower()
    
    pred_norm = predicted.lower().strip().rstrip(".")
    exp_norm = expected.lower().strip().rstrip(".")
    
    if pred_norm == exp_norm:
        return True
    
    # Date matching: "6/27/2023" vs "June 27, 2023"
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


def main():
    print("=" * 78)
    print("M30-33 Phase 4: Track A/B/C Evaluation")
    print("=" * 78)
    print()
    
    # Load test data
    test_data = json.loads((PROJECT_ROOT / "results/m30_33/test_data.json").read_text(encoding="utf-8"))
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
        
        # Get the answer turn
        answer_dia_id = q["answer_turn"]
        
        # Run retrieval
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
    print()
    
    # =========================================================================
    # Track A: Reader-only (give correct evidence)
    # =========================================================================
    print("--- Track A: Reader-only (correct evidence) ---")
    
    track_a_correct = 0
    track_a_total = 0
    
    for q in questions[:30]:  # Subset for speed
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
            if correct:
                track_a_correct += 1
            track_a_total += 1
        except Exception as e:
            print(f"  ERROR {q['question_id']}: {e}")
    
    print(f"Track A: {track_a_correct}/{track_a_total} correct ({100*track_a_correct/track_a_total:.1f}%)")
    print()
    
    # =========================================================================
    # Track C: End-to-End
    # =========================================================================
    print("--- Track C: End-to-End ---")
    
    track_c_correct = 0
    track_c_total = 0
    category_correct = {}
    category_total = {}
    
    for q in questions[:30]:  # Subset for speed
        conv_id = q["conversation_id"]
        turns = flat_conversations[conv_id]
        
        cat = q["category"]
        category_total[cat] = category_total.get(cat, 0) + 1
        
        # Run retrieval
        results = expanded_retrieval(q["question"], turns)
        
        # Build evidence text (top 8)
        evidence_lines = []
        for score, dia_id, text, i in results[:8]:
            # Get timestamp
            ts = turns[i].get("session_datetime", "N/A")
            evidence_lines.append(f"- {dia_id} | {ts} | {text}")
        evidence_text = "\n".join(evidence_lines)
        
        try:
            reader_output = call_ollama(q["question"], evidence_text)
            
            # Simple temporal resolution
            final_output = reader_output
            expressions = re.findall(r"\b(yesterday|tomorrow|today|next month|last month|this month|next year|last year|one year ago)\b", reader_output, re.I)
            if expressions:
                for expr in expressions:
                    # Find the turn with this expression
                    for score, dia_id, text, i in results[:8]:
                        if expr.lower() in text.lower():
                            ts = parse_session_datetime(turns[i].get("session_datetime", ""))
                            if ts:
                                dt = datetime(*ts)
                                if expr.lower() == "yesterday":
                                    dt -= timedelta(days=1)
                                elif expr.lower() == "tomorrow":
                                    dt += timedelta(days=1)
                                month_name = MONTH_NAMES.get(dt.month, "")
                                date_str = f"{month_name} {dt.day}, {dt.year}"
                                final_output = final_output.replace(expr, date_str)
                            break
            
            # Extract answer (for temporal questions)
            if cat in ["direct_fact", "paraphrase", "relative_date", "granularity"]:
                date_match = re.search(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", final_output, re.I)
                if date_match:
                    final_output = date_match.group(0)
                else:
                    date_match = re.search(r"\d{1,2}/\d{1,2}/\d{4}", final_output)
                    if date_match:
                        final_output = date_match.group(0)
            
            correct = score_answer(final_output, q["expected_answer"], cat)
            if correct:
                track_c_correct += 1
                category_correct[cat] = category_correct.get(cat, 0) + 1
            track_c_total += 1
            
        except Exception as e:
            print(f"  ERROR {q['question_id']}: {e}")
    
    print(f"Track C: {track_c_correct}/{track_c_total} correct ({100*track_c_correct/track_c_total:.1f}%)")
    print()
    
    # Category breakdown
    print("Category breakdown (Track C):")
    for cat in sorted(category_total.keys()):
        c = category_correct.get(cat, 0)
        t = category_total[cat]
        print(f"  {cat}: {c}/{t} ({100*c/t:.1f}%)")
    
    print()
    
    # Save results
    output = {
        "track_b": {
            "recall_at_1": recall_at_1,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "total": len(questions),
        },
        "track_a": {
            "correct": track_a_correct,
            "total": track_a_total,
        },
        "track_c": {
            "correct": track_c_correct,
            "total": track_c_total,
            "category_correct": category_correct,
            "category_total": category_total,
        },
    }
    
    output_path = PROJECT_ROOT / "results/m30_33/phase4_results.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
