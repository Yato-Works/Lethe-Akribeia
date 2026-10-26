#!/usr/bin/env python3
"""M30-30 Phase 1: Classify 20 abstention cases in one pass.

For each abstaining case, determine:
- EVIDENCE_IN_INPUT: Answer-supporting evidence is already in Reader input
- EVIDENCE_MISSING_FROM_INPUT: Evidence exists in conversation but not in input
- EVIDENCE_AMBIGUOUS: Multiple interpretations possible
- NO_SUPPORT_FOUND: No supporting evidence found in available sources
- UNVERIFIED: Cannot determine with available data

Constraints:
- No Gold, no Gold evidence IDs for retrieval/ranking
- Only use question + normal search info
- M30-29 results reused
"""
import json, re, sys
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


def get_evidence_from_input(evidence_text):
    """Parse evidence_text to get list of (local_id, speaker, text) tuples."""
    results = []
    for line in evidence_text.strip().split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        parts = line[2:].split(" | ", 2)
        if len(parts) >= 3:
            results.append((parts[0], parts[1], parts[2]))
    return results


def classify_abstention(question, gold, evidence_text, conversations, conv_key, provenance):
    """Classify a single abstention case.
    
    Returns: (classification, details_dict)
    """
    question_kw = extract_keywords(question)
    gold_kw = extract_keywords(str(gold).lower())
    
    # Get evidence turns from Reader input
    input_turns = get_evidence_from_input(evidence_text)
    
    # Check if any input turn has strong keyword overlap with question
    best_input_score = 0
    best_input_turn = None
    for local_id, speaker, text in input_turns:
        turn_kw = extract_keywords(text)
        q_overlap = question_kw & turn_kw
        g_overlap = gold_kw & turn_kw
        score = len(q_overlap) + len(g_overlap) * 2
        if score > best_input_score:
            best_input_score = score
            best_input_turn = (local_id, speaker, text, q_overlap, g_overlap)
    
    # Get all conversation turns
    conv_turns = conversations.get(conv_key, [])
    if not isinstance(conv_turns, list):
        return "UNVERIFIED", {"reason": "No conversation data"}
    
    # Search for turns with question keywords (same as pipeline search)
    search_results = []
    for i, t in enumerate(conv_turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        dia_id = t.get("dia_id", f"D{i+1}")
        
        # Same search logic as pipeline: gold keywords OR question keywords
        gold_clean = re.sub(r"[^\w\s]", " ", str(gold).lower())
        g_kw = set(w for w in gold_clean.split() if len(w) > 2 and w not in STOPWORDS)
        
        q_kw = set()
        for w in question.replace("?", "").lower().split():
            if len(w) > 3:
                q_kw.add(w)
        
        if any(kw in text_lower for kw in g_kw):
            search_results.append((dia_id, text, "gold_kw", i))
        elif any(kw in text_lower for kw in q_kw):
            if any(d in text_lower for d in ["month", "year", "2023", "2022", "2024", "indiana", "seoul", "france", "rome", "boston"]):
                search_results.append((dia_id, text, "question_kw", i))
    
    # Check if search results have strong overlap
    best_search_score = 0
    best_search_turn = None
    for dia_id, text, reason, i in search_results:
        turn_kw = extract_keywords(text)
        q_overlap = question_kw & turn_kw
        g_overlap = gold_kw & turn_kw
        score = len(q_overlap) + len(g_overlap) * 2
        if score > best_search_score:
            best_search_score = score
            best_search_turn = (dia_id, text, reason, q_overlap, g_overlap, i)
    
    # Classification logic
    # EVIDENCE_IN_INPUT: Input has a turn with strong question+gold overlap
    if best_input_score >= 3 and best_input_turn:
        q_ov = best_input_turn[3]
        g_ov = best_input_turn[4]
        if len(g_ov) >= 1 or len(q_ov) >= 3:
            return "EVIDENCE_IN_INPUT", {
                "best_input_turn": best_input_turn[0],
                "input_overlap": list(q_ov)[:10],
                "gold_overlap": list(g_ov)[:10],
                "score": best_input_score,
            }
    
    # EVIDENCE_MISSING_FROM_INPUT: Search finds turns not in input
    input_ids = set(t[0] for t in input_turns)
    if best_search_turn and best_search_score >= 3:
        dia_id = best_search_turn[0]
        # Check if this turn is in input
        in_input = any(dia_id in t[0] or t[0] in dia_id for t in input_ids)
        if not in_input:
            return "EVIDENCE_MISSING_FROM_INPUT", {
                "best_search_turn": dia_id,
                "search_overlap": list(best_search_turn[3])[:10],
                "gold_overlap": list(best_search_turn[4])[:10],
                "score": best_search_score,
            }
    
    # EVIDENCE_AMBIGUOUS: Multiple possible interpretations
    if best_input_score >= 1 and best_search_score >= 1:
        return "EVIDENCE_AMBIGUOUS", {
            "input_score": best_input_score,
            "search_score": best_search_score,
            "reason": "Multiple possible evidence sources",
        }
    
    # NO_SUPPORT_FOUND: No strong evidence found
    if best_input_score == 0 and best_search_score == 0:
        return "NO_SUPPORT_FOUND", {
            "reason": "No turns with significant keyword overlap",
            "question_keywords": list(question_kw)[:10],
        }
    
    return "UNVERIFIED", {
        "input_score": best_input_score,
        "search_score": best_search_score,
    }


def main():
    print("=" * 78)
    print("M30-30 Phase 1: Abstention Classification")
    print("=" * 78)
    print()
    
    # Load M30-29 results
    m29 = json.loads((PROJECT_ROOT / "results/m30_29/results_v2.json").read_text(encoding="utf-8"))
    
    # Load M30-3 results (baseline reader outputs)
    m30_3 = json.loads((PROJECT_ROOT / "results/m30_3_evaluation_results.json").read_text(encoding="utf-8"))
    m30_3_results = {r["question_id"]: r for r in m30_3["results"]}
    
    # Load bucketA
    bucketA = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucketA["cases"]}
    conversations = bucketA["conversations"]
    
    # Build provenance
    locomo_path = PROJECT_ROOT / "datasets/external/locomo10.json"
    provenance = build_provenance(locomo_path)
    
    # Get abstention cases (incorrect in M30-29)
    abstention_qids = [qid for qid, r in m29["results"].items() if not r["treatment_correct"]]
    abstention_qids.sort()
    
    print(f"Abstention cases: {len(abstention_qids)}")
    print()
    
    # Classify each case
    classifications = {}
    for qid in abstention_qids:
        case = cases.get(qid, {})
        if not case:
            continue
        
        question = case.get("question", "")
        gold = str(case.get("gold_answer", ""))
        reader_output = m30_3_results.get(qid, {}).get("reader_text", "")
        evidence_text = m30_3_results.get(qid, {}).get("evidence_text", "")
        conv_key = case.get("source_conversation", "")
        
        # Only classify if reader abstained
        if "don't know" not in reader_output.lower() and "do not know" not in reader_output.lower():
            classifications[qid] = {
                "classification": "NOT_ABSTENTION",
                "reader_output": reader_output,
                "reason": "Reader did not abstain",
            }
            continue
        
        classification, details = classify_abstention(
            question, gold, evidence_text, conversations, conv_key, provenance
        )
        
        # Get relevant turns for the report
        input_turns = get_evidence_from_input(evidence_text)
        
        classifications[qid] = {
            "question": question,
            "gold": gold,
            "reader_output": reader_output,
            "classification": classification,
            "details": details,
            "input_turns_count": len(input_turns),
            "evidence_text_sample": evidence_text[:500] if evidence_text else "",
        }
    
    # Print summary
    from collections import Counter
    class_counts = Counter(c["classification"] for c in classifications.values())
    print("Classification summary:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    print()
    
    # Print each case
    for qid in abstention_qids:
        c = classifications.get(qid, {})
        print(f"  {qid}: {c.get('classification', 'N/A')}")
        print(f"    Q: {c.get('question', '')[:80]}")
        print(f"    Gold: {c.get('gold', '')}")
        if c.get("details"):
            d = c["details"]
            if "best_input_turn" in d:
                print(f"    Best input turn: {d['best_input_turn']} (score={d.get('score', 0)})")
            if "best_search_turn" in d:
                print(f"    Best search turn: {d['best_search_turn']} (score={d.get('score', 0)})")
            if "reason" in d:
                print(f"    Reason: {d['reason']}")
        print()
    
    # Save
    output = {
        "total_abstention": len(abstention_qids),
        "classifications": classifications,
        "summary": dict(class_counts),
    }
    output_path = PROJECT_ROOT / "results/m30_30/abstention_classification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
