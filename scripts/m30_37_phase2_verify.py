#!/usr/bin/env python3
"""M30-37 Phase 2: Verify whether Track C's retrieval actually missed evidence.

For the 22 A_CORRECT_C_WRONG cases:
1. Was the answer_turn retrieved by Track C's retrieval?
2. If yes, at what rank?
3. Did it reach the Reader input (top 8)?
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}

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


def main():
    print("=" * 78)
    print("M30-37 Phase 2: Verify Retrieval Miss for A_CORRECT_C_WRONG")
    print("=" * 78)
    print()
    
    paired = json.loads((PROJECT_ROOT / "results/m30_36/paired_results.json").read_text(encoding="utf-8"))
    test_data = json.loads((PROJECT_ROOT / "results/m30_35/test_data.json").read_text(encoding="utf-8"))
    conversations = test_data["conversations"]
    
    flat_conversations = {cid: flatten_conversation(c) for cid, c in conversations.items()}
    q_by_id = {q["question_id"]: q for q in test_data["questions"]}
    
    a_correct_c_wrong = [r for r in paired if r["classification"] == "A_CORRECT_C_WRONG"]
    print(f"A_CORRECT_C_WRONG cases: {len(a_correct_c_wrong)}")
    print()
    
    stats = {
        "answer_turn_in_top1": 0,
        "answer_turn_in_top5": 0,
        "answer_turn_in_top8": 0,
        "answer_turn_in_retrieval": 0,
        "answer_turn_not_retrieved": 0,
        "answer_turn_no_date": 0,
    }
    
    for r in a_correct_c_wrong:
        qid = r["question_id"]
        q = q_by_id[qid]
        conv_id = q["conversation_id"]
        turns = flat_conversations[conv_id]
        answer_dia_id = q["answer_turn"]
        
        # Check if answer turn has a date
        answer_turn = next((t for t in turns if t["dia_id"] == answer_dia_id), None)
        if answer_turn:
            has_date = any(m in answer_turn["text"].lower() for m in MONTHS.keys())
            if not has_date:
                stats["answer_turn_no_date"] += 1
        
        # Run retrieval
        results = expanded_retrieval(q["question"], turns)
        retrieved_ids = [x[1] for x in results]
        
        if answer_dia_id in retrieved_ids[:1]:
            stats["answer_turn_in_top1"] += 1
        if answer_dia_id in retrieved_ids[:5]:
            stats["answer_turn_in_top5"] += 1
        if answer_dia_id in retrieved_ids[:8]:
            stats["answer_turn_in_top8"] += 1
        if answer_dia_id in retrieved_ids:
            stats["answer_turn_in_retrieval"] += 1
        else:
            stats["answer_turn_not_retrieved"] += 1
    
    print("Retrieval results for A_CORRECT_C_WRONG cases:")
    for k, v in stats.items():
        print(f"  {k}: {v}/{len(a_correct_c_wrong)}")
    print()
    
    # Show detail for a few
    print("Detail (first 5):")
    for r in a_correct_c_wrong[:5]:
        qid = r["question_id"]
        q = q_by_id[qid]
        conv_id = q["conversation_id"]
        turns = flat_conversations[conv_id]
        answer_dia_id = q["answer_turn"]
        
        results = expanded_retrieval(q["question"], turns)
        retrieved_ids = [x[1] for x in results]
        
        rank = None
        for i, rid in enumerate(retrieved_ids, 1):
            if rid == answer_dia_id:
                rank = i
                break
        
        print(f"  {qid} [{q['category']}]:")
        print(f"    Expected: {r['expected']}")
        print(f"    Answer turn: {answer_dia_id}, rank: {rank}")
        print(f"    A output: {r['track_a_output'][:50]}")
        print(f"    C output: {r['track_c_output'][:50]}")
        print()
    
    # Save
    output_path = PROJECT_ROOT / "results/m30_37/phase2_retrieval_verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "total_a_correct_c_wrong": len(a_correct_c_wrong),
        "stats": stats,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
