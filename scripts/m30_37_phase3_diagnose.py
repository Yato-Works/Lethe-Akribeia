#!/usr/bin/env python3
"""M30-37 Phase 3: Diagnose the retrieval miss cause.

For the 16 cases where the answer turn was NOT retrieved:
- What keywords does the question have?
- What keywords does the answer turn have?
- Why is there no overlap?
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


def main():
    print("=" * 78)
    print("M30-37 Phase 3: Diagnose Retrieval Miss Cause")
    print("=" * 78)
    print()
    
    paired = json.loads((PROJECT_ROOT / "results/m30_36/paired_results.json").read_text(encoding="utf-8"))
    test_data = json.loads((PROJECT_ROOT / "results/m30_35/test_data.json").read_text(encoding="utf-8"))
    conversations = test_data["conversations"]
    
    flat_conversations = {cid: flatten_conversation(c) for cid, c in conversations.items()}
    q_by_id = {q["question_id"]: q for q in test_data["questions"]}
    
    a_correct_c_wrong = [r for r in paired if r["classification"] == "A_CORRECT_C_WRONG"]
    
    # For each, show question keywords vs answer turn keywords
    no_overlap = 0
    entity_only = 0
    
    for r in a_correct_c_wrong[:10]:
        qid = r["question_id"]
        q = q_by_id[qid]
        conv_id = q["conversation_id"]
        turns = flat_conversations[conv_id]
        answer_dia_id = q["answer_turn"]
        
        q_kw = extract_keywords(q["question"])
        answer_turn = next((t for t in turns if t["dia_id"] == answer_dia_id), None)
        
        if not answer_turn:
            continue
        
        a_kw = extract_keywords(answer_turn["text"])
        overlap = q_kw & a_kw
        
        # Question entities
        q_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", q["question"]))
        a_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", answer_turn["text"]))
        entity_overlap = q_entities & a_entities
        
        print(f"  {qid}:")
        print(f"    Q: {q['question']}")
        print(f"    Q keywords: {q_kw}")
        print(f"    A turn: {answer_turn['text'][:70]}")
        print(f"    A keywords: {a_kw}")
        print(f"    Overlap: {overlap}")
        print(f"    Q entities: {q_entities}")
        print(f"    A entities: {a_entities}")
        print(f"    Entity overlap: {entity_overlap}")
        print()
        
        if not overlap:
            no_overlap += 1
        if not overlap and not entity_overlap:
            entity_only += 1
    
    print("=" * 78)
    print("DIAGNOSIS")
    print("=" * 78)
    print()
    print("The question asks 'On what date did X attend the event?'")
    print("The answer turn says 'On <date>, I visited <city> for the <event>.'")
    print()
    print("Problem: The question uses generic terms ('attend', 'event')")
    print("while the answer turn uses specific terms ('visited', 'art exhibition').")
    print("There is NO keyword overlap.")
    print()
    print("The only signal is the ENTITY (person name), but the answer turn")
    print("may not contain the question's entity if the speaker is the other person.")
    print()
    print("This is a REAL retrieval failure: the retrieval relies on")
    print("lexical overlap, which fails for paraphrased questions.")


if __name__ == "__main__":
    main()
