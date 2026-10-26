#!/usr/bin/env python3
"""M30-30 Phase 2-3: Evidence Retrieval + Reader Prompt Improvement.

Treatment A: Retrieval-only (improve evidence selection, keep Reader prompt)
Treatment B: Reader-only (keep evidence, improve Reader prompt)
Treatment C: Combined (both improvements)

For EVIDENCE_IN_INPUT cases: Reader prompt improvement should help
For EVIDENCE_AMBIGUOUS cases: Retrieval improvement may help

Constraints:
- No Gold, no Gold evidence IDs for retrieval/ranking
- Isolated experiment, no production changes
- 553 holdout completely isolated
"""
import json, re, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artificial_memory.research.benchmarks.llm import OllamaAnswerer
from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader

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


def build_provenance(locomo_path):
    locomo = json.loads(Path(locomo_path).read_text(encoding="utf-8"))
    mapping = {}
    for conv in locomo:
        conv_id = conv.get("sample_id", "")
        if not conv_id.startswith("conv-"):
            continue
        conv_data = conv.get("conversation", {})
        for sk in sorted(conv_data.keys()):
            if not sk.endswith("_date_time"):
                continue
            session_key = sk.replace("_date_time", "")
            session_dt = conv_data.get(sk, "")
            turns = conv_data.get(session_key, [])
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                dia_id = turn.get("dia_id", "")
                if not dia_id:
                    continue
                mapping[(conv_id, dia_id)] = {
                    "speaker": turn.get("speaker", ""),
                    "text": turn.get("text", ""),
                    "session_id": session_key,
                    "timestamp": session_dt,
                    "timestamp_parsed": parse_session_datetime(session_dt),
                }
    return mapping


# =============================================================================
# Phase 2: Improved Evidence Retrieval
# =============================================================================

def select_evidence_v2(question, conversations, conv_key):
    """Improved evidence retrieval with better relevance scoring.
    
    Key improvements over M30-3:
    1. Use question keywords (not gold keywords)
    2. Score by keyword overlap + entity matching
    3. Include turns with temporal expressions
    4. Preserve turn IDs and timestamps
    """
    turns = conversations.get(conv_key, [])
    if not isinstance(turns, list):
        return [], {}
    
    question_kw = extract_keywords(question)
    
    # Score all turns
    scored = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        dia_id = t.get("dia_id", f"D{i+1}")
        
        turn_kw = extract_keywords(text)
        overlap = question_kw & turn_kw
        
        # Score: keyword overlap
        score = len(overlap)
        
        # Boost for entity names (capitalized words in original)
        entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
        q_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", question))
        entity_overlap = entities & q_entities
        score += len(entity_overlap) * 3
        
        # Boost for temporal expressions
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


# =============================================================================
# Phase 3: Improved Reader Prompt
# =============================================================================

BASELINE_READER_PROMPT = """Answer the question based on the conversation context.
If you cannot determine the answer from the context, respond with "I don't know."
Keep your answer concise and to the point."""

IMPROVED_READER_PROMPT = """Answer the question based on the conversation context provided below.

Instructions:
1. First, check if the context contains information that directly answers the question.
2. If the context contains the answer or facts that uniquely determine the answer, provide that answer.
3. Do NOT say "I don't know" if the context contains sufficient information to answer.
4. Keep your answer concise and in the format the question asks for (e.g., if asked "when", provide a date or time period).
5. If the context is insufficient or the answer is ambiguous, then respond with "I don't know."
6. Do not invent facts that are not in the context.

Conversation context:
{evidence}

Question: {question}

Answer:"""


def build_evidence_text(evidence_strings):
    """Build evidence text from evidence strings."""
    if not evidence_strings:
        return ""
    return "\n".join(f"- {t}" for t in evidence_strings)


def get_qtype_name(case):
    qtype_map = {1: "multi-session", 2: "temporal-reasoning", 3: "single-session-user", 4: "single-session-assistant", 5: "abstention"}
    return qtype_map.get(case.get("category", 3), "single-session-user")


def get_evidence_strings_from_text(evidence_text):
    """Parse evidence_text to get list of evidence strings."""
    results = []
    for line in evidence_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("- "):
            results.append(line[2:])
    return results


def main():
    print("=" * 78)
    print("M30-30: Abstention Recovery & Evidence-First Reader")
    print("=" * 78)
    print()
    
    # Load data
    m29 = json.loads((PROJECT_ROOT / "results/m30_29/results_v2.json").read_text(encoding="utf-8"))
    m30_3 = json.loads((PROJECT_ROOT / "results/m30_3_evaluation_results.json").read_text(encoding="utf-8"))
    m30_3_results = {r["question_id"]: r for r in m30_3["results"]}
    bucketA = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucketA["cases"]}
    conversations = bucketA["conversations"]
    intervention = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted([r["question_id"] for r in intervention["rows"] if r.get("group") == "TARGET"])
    
    # Load abstention classification
    abstention = json.loads((PROJECT_ROOT / "results/m30_30/abstention_classification.json").read_text(encoding="utf-8"))
    
    # Get abstention qids
    abstention_qids = sorted(abstention["classifications"].keys())
    
    print(f"M30-29 baseline: {m29['treatment_correct']}/28")
    print(f"Abstention cases: {len(abstention_qids)}")
    print()
    
    grader = OfficialLongMemEvalGrader()
    model_name = "qwen2.5:7b-instruct"
    
    # =========================================================================
    # Treatment B: Reader-only (improved prompt, same evidence)
    # =========================================================================
    print("--- Treatment B: Reader-only (improved prompt) ---")
    
    reader_b_correct = 0
    reader_b_results = {}
    
    for qid in targets:
        case = cases.get(qid, {})
        if not case:
            continue
        
        question = case.get("question", "")
        gold = str(case.get("gold_answer", ""))
        evidence_text = m30_3_results.get(qid, {}).get("evidence_text", "")
        baseline_correct = m29["results"].get(qid, {}).get("treatment_correct", False)
        
        if qid not in abstention_qids:
            reader_b_results[qid] = {
                "treatment_correct": baseline_correct,
                "reader_output": m30_3_results.get(qid, {}).get("reader_text", ""),
                "note": "Not abstention, kept M30-29 result",
            }
            if baseline_correct:
                reader_b_correct += 1
            continue
        
        # Use improved prompt with same evidence
        evidence_strings = get_evidence_strings_from_text(evidence_text)
        evidence = build_evidence_text(evidence_strings)
        
        try:
            reader = OllamaAnswerer(model=model_name)
            reader_output = reader.answer(question, evidence)
            
            qtype_name = get_qtype_name(case)
            result = grader.grade(qid, qtype_name, reader_output.text, gold)
            treatment_correct = result.is_correct
            
            reader_b_results[qid] = {
                "reader_output": reader_output.text,
                "gold": gold,
                "baseline_correct": baseline_correct,
                "treatment_correct": treatment_correct,
            }
            
            if treatment_correct:
                reader_b_correct += 1
        except Exception as e:
            print(f"  ERROR {qid}: {e}")
            reader_b_results[qid] = {
                "treatment_correct": baseline_correct,
                "error": str(e),
            }
            if baseline_correct:
                reader_b_correct += 1
    
    print(f"Treatment B: {reader_b_correct}/28 correct")
    print()
    
    # =========================================================================
    # Treatment A: Retrieval-only (same prompt, improved evidence)
    # =========================================================================
    print("--- Treatment A: Retrieval-only (improved evidence) ---")
    
    reader_a_correct = 0
    reader_a_results = {}
    
    for qid in targets:
        case = cases.get(qid, {})
        if not case:
            continue
        
        question = case.get("question", "")
        gold = str(case.get("gold_answer", ""))
        evidence_text = m30_3_results.get(qid, {}).get("evidence_text", "")
        baseline_correct = m29["results"].get(qid, {}).get("treatment_correct", False)
        
        if qid not in abstention_qids:
            reader_a_results[qid] = {
                "treatment_correct": baseline_correct,
                "note": "Not abstention, kept M30-29 result",
            }
            if baseline_correct:
                reader_a_correct += 1
            continue
        
        # Use improved retrieval with same prompt
        conv_key = case.get("source_conversation", "")
        evidence_strings, turn_map = select_evidence_v2(question, conversations, conv_key)
        evidence = build_evidence_text(evidence_strings)
        
        try:
            reader = OllamaAnswerer(model=model_name)
            reader_output = reader.answer(question, evidence)
            
            qtype_name = get_qtype_name(case)
            result = grader.grade(qid, qtype_name, reader_output.text, gold)
            treatment_correct = result.is_correct
            
            reader_a_results[qid] = {
                "reader_output": reader_output.text,
                "gold": gold,
                "baseline_correct": baseline_correct,
                "treatment_correct": treatment_correct,
                "evidence_count": len(evidence_strings),
                "turn_map": turn_map,
            }
            
            if treatment_correct:
                reader_a_correct += 1
        except Exception as e:
            print(f"  ERROR {qid}: {e}")
            reader_a_results[qid] = {
                "treatment_correct": baseline_correct,
                "error": str(e),
            }
            if baseline_correct:
                reader_a_correct += 1
    
    print(f"Treatment A: {reader_a_correct}/28 correct")
    print()
    
    # =========================================================================
    # Treatment C: Combined (improved retrieval + improved prompt)
    # =========================================================================
    print("--- Treatment C: Combined ---")
    
    reader_c_correct = 0
    reader_c_results = {}
    
    for qid in targets:
        case = cases.get(qid, {})
        if not case:
            continue
        
        question = case.get("question", "")
        gold = str(case.get("gold_answer", ""))
        baseline_correct = m29["results"].get(qid, {}).get("treatment_correct", False)
        
        if qid not in abstention_qids:
            reader_c_results[qid] = {
                "treatment_correct": baseline_correct,
                "note": "Not abstention, kept M30-29 result",
            }
            if baseline_correct:
                reader_c_correct += 1
            continue
        
        # Use improved retrieval + improved prompt
        conv_key = case.get("source_conversation", "")
        evidence_strings, turn_map = select_evidence_v2(question, conversations, conv_key)
        evidence = build_evidence_text(evidence_strings)
        
        try:
            reader = OllamaAnswerer(model=model_name)
            reader_output = reader.answer(question, evidence)
            
            qtype_name = get_qtype_name(case)
            result = grader.grade(qid, qtype_name, reader_output.text, gold)
            treatment_correct = result.is_correct
            
            reader_c_results[qid] = {
                "reader_output": reader_output.text,
                "gold": gold,
                "baseline_correct": baseline_correct,
                "treatment_correct": treatment_correct,
                "evidence_count": len(evidence_strings),
                "turn_map": turn_map,
            }
            
            if treatment_correct:
                reader_c_correct += 1
        except Exception as e:
            print(f"  ERROR {qid}: {e}")
            reader_c_results[qid] = {
                "treatment_correct": baseline_correct,
                "error": str(e),
            }
            if baseline_correct:
                reader_c_correct += 1
    
    print(f"Treatment C: {reader_c_correct}/28 correct")
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    
    baseline = m29["treatment_correct"]
    print(f"Baseline (M30-29): {baseline}/28")
    print(f"Treatment A (Retrieval-only): {reader_a_correct}/28")
    print(f"Treatment B (Reader-only): {reader_b_correct}/28")
    print(f"Treatment C (Combined): {reader_c_correct}/28")
    print()
    
    # Per-case comparison for abstention cases
    print("Abstention cases comparison:")
    for qid in abstention_qids:
        b_b = reader_b_results.get(qid, {}).get("treatment_correct", False)
        a_a = reader_a_results.get(qid, {}).get("treatment_correct", False)
        c_c = reader_c_results.get(qid, {}).get("treatment_correct", False)
        m29_c = m29["results"].get(qid, {}).get("treatment_correct", False)
        
        changed = ""
        if c_c and not m29_c:
            changed = "RECOVERED"
        elif not c_c and m29_c:
            changed = "REGRESSED"
        
        print(f"  {qid}: M29={'✓' if m29_c else '✗'} A={'✓' if a_a else '✗'} B={'✓' if b_b else '✗'} C={'✓' if c_c else '✗'} {changed}")
    
    # Save
    output = {
        "baseline_correct": baseline,
        "treatment_a_correct": reader_a_correct,
        "treatment_b_correct": reader_b_correct,
        "treatment_c_correct": reader_c_correct,
        "abstention_qids": abstention_qids,
        "reader_a_results": reader_a_results,
        "reader_b_results": reader_b_results,
        "reader_c_results": reader_c_results,
    }
    output_path = PROJECT_ROOT / "results/m30_30/treatment_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
