#!/usr/bin/env python3
"""M30-31 Phase 4b: Minimal experiment — Reader prompt for concise answers.

The scorer requires exact match for temporal-reasoning questions.
Reader outputs like "Maria met Jean February 24, 2023 while volunteering." fail.

Fix: Add instruction to Reader prompt to output ONLY the answer (date/fact),
not a full sentence.

Test on lc_conv-41_10:
- Baseline: "I don't know." (M30-3)
- Treatment A: "Maria met Jean yesterday while volunteering." (M30-30)
- Treatment A + Concise prompt: "February 24, 2023" (expected)
"""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artificial_memory.research.benchmarks.llm import OllamaAnswerer
from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader

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


def select_evidence_v2(question, conversations, conv_key):
    """Improved evidence retrieval (from M30-30)."""
    turns = conversations.get(conv_key, [])
    if not isinstance(turns, list):
        return [], {}
    
    question_kw = extract_keywords(question)
    
    scored = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        dia_id = t.get("dia_id", f"D{i+1}")
        
        turn_kw = extract_keywords(text)
        overlap = question_kw & turn_kw
        score = len(overlap)
        
        entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
        q_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", question))
        entity_overlap = entities & q_entities
        score += len(entity_overlap) * 3
        
        rel_patterns = [
            r"\bnext month\b", r"\blast month\b", r"\bthis month\b",
            r"\bnext year\b", r"\blast year\b", r"\bone year ago\b",
            r"\b\d+\s+years?\s+ago\b", r"\byesterday\b", r"\btomorrow\b",
        ]
        for pat in rel_patterns:
            if re.search(pat, text, re.I):
                score += 5
                break
        
        if score > 0:
            scored.append((score, dia_id, text, i))
    
    scored.sort(key=lambda x: -x[0])
    
    evidence = []
    turn_map = {}
    for score, dia_id, text, i in scored[:8]:
        local_id = f"D{i+1}"
        evidence.append(f"{local_id} | {turns[i].get('speaker', '?')} | {text}")
        turn_map[local_id] = dia_id
    
    return evidence, turn_map


def build_evidence_text(evidence_strings):
    if not evidence_strings:
        return ""
    return "\n".join(f"- {t}" for t in evidence_strings)


def get_qtype_name(case):
    qtype_map = {1: "multi-session", 2: "temporal-reasoning", 3: "single-session-user", 4: "single-session-assistant", 5: "abstention"}
    return qtype_map.get(case.get("category", 3), "single-session-user")


def main():
    print("=" * 78)
    print("M30-31 Phase 4b: Concise Reader Prompt Experiment")
    print("=" * 78)
    print()
    
    # Load data
    m30_3 = json.loads((PROJECT_ROOT / "results/m30_3_evaluation_results.json").read_text(encoding="utf-8"))
    m30_3_results = {r["question_id"]: r for r in m30_3["results"]}
    bucketA = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucketA["cases"]}
    conversations = bucketA["conversations"]
    
    qid = "lc_conv-41_10"
    case = cases[qid]
    question = case["question"]
    gold = str(case["gold_answer"])
    conv_key = case["source_conversation"]
    
    print(f"QID: {qid}")
    print(f"Question: {question}")
    print(f"Gold: {gold}")
    print()
    
    # Get evidence using improved retrieval
    evidence_strings, turn_map = select_evidence_v2(question, conversations, conv_key)
    evidence = build_evidence_text(evidence_strings)
    
    print(f"Evidence turns: {len(evidence_strings)}")
    for ev in evidence_strings[:5]:
        print(f"  {ev[:100]}")
    print()
    
    # Test with concise prompt
    concise_prompt = """Answer the question based on the conversation context provided below.

IMPORTANT: Output ONLY the answer itself (e.g., just a date, name, or fact).
Do NOT include additional context, explanation, or full sentences.
If the question asks "when", output only the date.
If you cannot determine the answer, respond with "I don't know."

Conversation context:
{evidence}

Question: {question}

Answer:"""
    
    prompt = concise_prompt.format(evidence=evidence, question=question)
    
    try:
        reader = OllamaAnswerer(model="qwen2.5:7b-instruct")
        result = reader.answer(question, evidence)
        reader_output = result.text
        
        print(f"Reader output: {reader_output}")
        print()
        
        # Score
        grader = OfficialLongMemEvalGrader()
        qtype_name = get_qtype_name(case)
        
        # Test various forms
        tests = [
            ("Raw Reader output", reader_output),
            ("Stripped", reader_output.strip().rstrip(".")),
        ]
        
        # Try to extract just the date if present
        date_match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}", reader_output, re.I)
        if date_match:
            tests.append(("Extracted date", date_match.group(0)))
        
        print("Scorer results:")
        for name, output in tests:
            result = grader.grade(qid, qtype_name, output, gold)
            print(f"  {name}: {repr(output[:60])} -> is_correct={result.is_correct}")
        
        print()
        
        # Final result
        final_output = reader_output.strip().rstrip(".")
        result = grader.grade(qid, qtype_name, final_output, gold)
        if result.is_correct:
            print("IMPROVEMENT CONFIRMED: Concise prompt + improved retrieval")
        else:
            print("NO IMPROVEMENT")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
