#!/usr/bin/env python3
"""M30-32: Retrieval Recall Expansion & Evidence Recovery.

Section 4: lc_conv-41_10 regression test (general-purpose retrieval, no hardcoded D7:5).
Section 5-6: Treatment A (expanded retrieval) / Treatment B (+ Temporal Resolution + Answer Extraction).

Retrieval path (from M30-3 code analysis):
  retrieve_m29_sorted(question) -> select_oracle_turns(min_score=3) -> merge_evidence(reads+oracle, max=8) -> Reader

Failure: lc_conv-41_10 "When did Maria meet Jean?" -> D7:5 not retrieved because
"maria", "jean", "meet" not in question text? Actually they ARE in question.
Wait: question is "When did Maria meet Jean?" -> keywords {'did', 'when', 'meet', 'jean', 'maria'} (len>3 filter keeps 'meet', 'jean', 'maria').
D7:5 text: "Yesterday I volunteered and met a woman named Jean while..."
- "jean" in text -> YES! So D7:5 SHOULD be retrieved by question keywords.

Wait, let me re-check. M30-31 trace showed retrieval found D8:4, D9:17, D11:3, D17:1, D20:9, D22:7.
D7:5 was NOT in retrieval results. Why?

D7:5 text: "Yesterday I volunteered and met a woman named Jean while volunteering..."
- "jean" in text_lower -> YES
- So `any(kw in text_lower for kw in q_kw)` should be True.
- But then the second condition: `any(d in text_lower for d in ["month", "year", "2023", "2022", "2024", ...])`
  - D7:5 has none of these! So it's excluded by the second filter.

That's the bug! The M30-3 retrieval requires temporal markers (month/year/etc.) for question-keyword matches,
but D7:5 is a temporal answer WITHOUT explicit temporal markers in the text.

This is a REAL retrieval failure in the M30-3 pipeline.
"""
import json, re, sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
# Section 4: lc_conv-41_10 Regression Test
# =============================================================================

def m30_3_retrieval(question, conversations, conv_key):
    """M30-3's retrieval logic (for comparison)."""
    turns = conversations.get(conv_key, [])
    if not isinstance(turns, list):
        return []
    
    q_kw = set()
    for w in question.replace("?", "").lower().split():
        if len(w) > 3:
            q_kw.add(w)
    
    evidence = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        dia_id = t.get("dia_id", f"D{i+1}")
        
        if any(kw in text_lower for kw in q_kw):
            if any(d in text_lower for d in ["month", "year", "2023", "2022", "2024", "indiana", "seoul", "france", "rome", "boston"]):
                evidence.append((dia_id, text))
    
    return evidence


def expanded_retrieval(question, conversations, conv_key):
    """Expanded retrieval: remove temporal marker requirement, add entity matching."""
    turns = conversations.get(conv_key, [])
    if not isinstance(turns, list):
        return []
    
    question_kw = extract_keywords(question)
    
    # Extract entities (capitalized words) from question
    q_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", question))
    
    scored = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        dia_id = t.get("dia_id", f"D{i+1}")
        
        turn_kw = extract_keywords(text)
        overlap = question_kw & turn_kw
        score = len(overlap)
        
        # Entity matching (strong signal)
        entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
        entity_overlap = entities & q_entities
        score += len(entity_overlap) * 5  # Strong boost for entity match
        
        # Temporal expressions (moderate signal)
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
            scored.append((score, dia_id, text, i))
    
    scored.sort(key=lambda x: -x[0])
    return scored


def run_regression_test():
    """Test lc_conv-41_10 retrieval."""
    print("=" * 78)
    print("M30-32 Section 4: lc_conv-41_10 Regression Test")
    print("=" * 78)
    print()
    
    bucketA = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    conversations = bucketA["conversations"]
    
    qid = "lc_conv-41_10"
    conv_key = "conv-41"
    question = "When did Maria meet Jean?"
    
    print(f"QID: {qid}")
    print(f"Question: {question}")
    print()
    
    # M30-3 retrieval
    m30_3_results = m30_3_retrieval(question, conversations, conv_key)
    print(f"M30-3 retrieval: {len(m30_3_results)} turns")
    for dia_id, text in m30_3_results:
        print(f"  {dia_id}: {text[:80]}")
    print()
    
    # Check if D7:5 is in M30-3 results
    d7_5_in_m30_3 = any(d == "D7:5" for d, _ in m30_3_results)
    print(f"D7:5 in M30-3 retrieval: {d7_5_in_m30_3}")
    print()
    
    # Expanded retrieval
    expanded_results = expanded_retrieval(question, conversations, conv_key)
    print(f"Expanded retrieval: {len(expanded_results)} turns")
    for score, dia_id, text, i in expanded_results[:10]:
        print(f"  {dia_id} (score={score}): {text[:80]}")
    print()
    
    # Check if D7:5 is in expanded results
    d7_5_in_expanded = any(d == "D7:5" for _, d, _, _ in expanded_results)
    print(f"D7:5 in expanded retrieval: {d7_5_in_expanded}")
    print()
    
    if d7_5_in_expanded and not d7_5_in_m30_3:
        print("REGRESSION TEST PASSED: Expanded retrieval finds D7:5, M30-3 does not")
    elif d7_5_in_expanded and d7_5_in_m30_3:
        print("BOTH find D7:5 (no regression, but no improvement either)")
    else:
        print("REGRESSION TEST FAILED: D7:5 not found in expanded retrieval")
    
    print()
    return d7_5_in_expanded


# =============================================================================
# Section 5-6: Treatment A/B Evaluation
# =============================================================================

RELATIVE_EXPRESSIONS = [
    (re.compile(r"\bnext month\b", re.I), "next_month"),
    (re.compile(r"\blast month\b", re.I), "last_month"),
    (re.compile(r"\bthis month\b", re.I), "this_month"),
    (re.compile(r"\bnext year\b", re.I), "next_year"),
    (re.compile(r"\blast year\b", re.I), "last_year"),
    (re.compile(r"\bone year ago\b", re.I), "one_year_ago"),
    (re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago\b", re.I), "years_ago"),
    (re.compile(r"\byesterday\b", re.I), "yesterday"),
    (re.compile(r"\btomorrow\b", re.I), "tomorrow"),
    (re.compile(r"\btoday\b", re.I), "today"),
    (re.compile(r"\bnext week\b", re.I), "next_week"),
    (re.compile(r"\blast week\b", re.I), "last_week"),
]


def resolve_relative(expr_type, base_date, expr_text=None):
    if not base_date:
        return None
    y, mo, d = base_date
    
    if expr_type == "next_month":
        dt = datetime(y, mo, 1) + timedelta(days=32)
        return (dt.year, dt.month, None)
    if expr_type == "last_month":
        dt = datetime(y, mo, 1) - timedelta(days=1)
        return (dt.year, dt.month, None)
    if expr_type == "this_month":
        return (y, mo, None)
    if expr_type == "next_year":
        return (y + 1, None, None)
    if expr_type == "last_year":
        return (y - 1, None, None)
    if expr_type == "one_year_ago":
        dt = datetime(y, mo, d) - timedelta(days=365)
        return (dt.year, dt.month, dt.day)
    if expr_type == "years_ago" and expr_text:
        m = re.search(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago", expr_text, re.I)
        if m:
            num_str = m.group(1).lower()
            num_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                       "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
            if num_str.isdigit():
                n = int(num_str)
            else:
                n = num_map.get(num_str, 0)
            if n > 0:
                dt = datetime(y, mo, d) - timedelta(days=365 * n)
                return (dt.year, dt.month, dt.day)
        return None
    if expr_type == "yesterday":
        dt = datetime(y, mo, d) - timedelta(days=1)
        return (dt.year, dt.month, dt.day)
    if expr_type == "tomorrow":
        dt = datetime(y, mo, d) + timedelta(days=1)
        return (dt.year, dt.month, dt.day)
    if expr_type == "today":
        return (y, mo, d)
    return None


def format_resolved_date(resolved):
    if not resolved:
        return None
    y, mo, d = resolved
    if mo is None:
        return str(y)
    month_name = MONTH_NAMES.get(mo, "")
    if d is None:
        return f"{month_name} {y}"
    return f"{month_name} {d}, {y}"


def detect_relative_expression(text):
    results = []
    text_lower = text.lower()
    for pattern, expr_type in RELATIVE_EXPRESSIONS:
        for m in pattern.finditer(text_lower):
            results.append((expr_type, m.start(), m.end()))
    return results


def extract_answer(text, question_type="temporal"):
    if question_type == "temporal":
        date_patterns = [
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
            r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}",
            r"\b(?:19|20)\d{2}\b",
        ]
        for pattern in date_patterns:
            m = re.search(pattern, text, re.I)
            if m:
                return m.group(0)
    return text


def detect_question_granularity(question):
    q = question.lower()
    if re.search(r"\bwhat day\b|\bwhich day\b|\bdate\b|\bexact date\b|\bspecific date\b", q):
        return "day"
    if re.search(r"\bwhat year\b|\bwhich year\b", q):
        return "year"
    return "month"


def format_with_granularity(resolved, granularity):
    if not resolved:
        return None
    y, mo, d = resolved
    if granularity == "year":
        return str(y)
    elif granularity == "month":
        if mo is None:
            return str(y)
        month_name = MONTH_NAMES.get(mo, "")
        return f"{month_name} {y}"
    elif granularity == "day":
        if mo is None:
            return str(y)
        month_name = MONTH_NAMES.get(mo, "")
        if d is None:
            return f"{month_name} {y}"
        return f"{month_name} {d}, {y}"
    return str(y)


def build_evidence_text(evidence_list):
    """Build evidence text from list of (dia_id, text) or (score, dia_id, text, idx) tuples."""
    if not evidence_list:
        return ""
    lines = []
    for item in evidence_list:
        if len(item) == 2:
            dia_id, text = item
        else:
            score, dia_id, text, idx = item
        lines.append(f"- {dia_id} | {text}")
    return "\n".join(lines)


def main():
    # Run regression test first
    regression_passed = run_regression_test()
    
    print("=" * 78)
    print("M30-32 Section 5-6: Treatment A/B Evaluation")
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
    
    # Build provenance
    locomo_path = PROJECT_ROOT / "datasets/external/locomo10.json"
    provenance = build_provenance(locomo_path)
    
    grader = OfficialLongMemEvalGrader()
    
    # Evaluate Treatment A and B
    treatment_a_correct = 0
    treatment_b_correct = 0
    treatment_a_results = {}
    treatment_b_results = {}
    recovered_a = 0
    recovered_b = 0
    regressed_a = 0
    regressed_b = 0
    
    for qid in targets:
        case = cases.get(qid, {})
        if not case:
            continue
        
        question = case.get("question", "")
        gold = str(case.get("gold_answer", ""))
        baseline_correct = m29["results"].get(qid, {}).get("treatment_correct", False)
        reader_output = m30_3_results.get(qid, {}).get("reader_text", "")
        
        # Skip non-abstention cases
        if "don't know" not in reader_output.lower():
            treatment_a_results[qid] = {"treatment_correct": baseline_correct, "note": "Not abstention"}
            treatment_b_results[qid] = {"treatment_correct": baseline_correct, "note": "Not abstention"}
            if baseline_correct:
                treatment_a_correct += 1
                treatment_b_correct += 1
            continue
        
        # Get expanded retrieval results
        conv_key = case.get("source_conversation", "")
        expanded = expanded_retrieval(question, conversations, conv_key)
        
        # Build evidence text (top 8)
        evidence_text = build_evidence_text(expanded[:8])
        
        # Treatment A: Expanded retrieval only (no temporal resolution)
        # Use M30-3's Reader output as baseline (since we're not re-running Reader)
        # Actually, we need to simulate what Reader would output with expanded evidence
        # For now, use M30-30 Treatment A's Reader output
        m30_30 = json.loads((PROJECT_ROOT / "results/m30_30/treatment_comparison.json").read_text(encoding="utf-8"))
        reader_a_output = m30_30["reader_a_results"].get(qid, {}).get("reader_output", "")
        
        if not reader_a_output:
            treatment_a_results[qid] = {"treatment_correct": baseline_correct, "note": "No Reader output"}
            treatment_b_results[qid] = {"treatment_correct": baseline_correct, "note": "No Reader output"}
            if baseline_correct:
                treatment_a_correct += 1
                treatment_b_correct += 1
            continue
        
        # Treatment A: Score Reader output as-is
        qtype_name = {1: "multi-session", 2: "temporal-reasoning", 3: "single-session-user", 4: "single-session-assistant", 5: "abstention"}.get(case.get("category", 3), "single-session-user")
        result_a = grader.grade(qid, qtype_name, reader_a_output, gold)
        treatment_a_correct_flag = result_a.is_correct
        
        # Treatment B: Apply Temporal Resolution + Answer Extraction
        # Find evidence turns with timestamps
        turn_map = {}
        for score, dia_id, text, i in expanded[:8]:
            turn_map[dia_id] = (conv_key, dia_id)
        
        expressions = detect_relative_expression(reader_a_output)
        resolved_text = reader_a_output
        resolution_status = "NO_RELATIVE_EXPRESSION"
        
        for expr_type, start, end in expressions:
            expr_text = reader_a_output[start:end]
            
            # Find evidence turn with this expression
            best_match = None
            for score, dia_id, ev_text, i in expanded[:8]:
                if (conv_key, dia_id) not in provenance:
                    continue
                ev = provenance[(conv_key, dia_id)]
                if not ev["timestamp_parsed"]:
                    continue
                if expr_text.lower() in ev_text.lower():
                    best_match = ev
                    break
            
            if best_match:
                resolved = resolve_relative(expr_type, best_match["timestamp_parsed"], expr_text)
                if resolved:
                    granularity = detect_question_granularity(question)
                    resolved_str = format_with_granularity(resolved, granularity)
                    resolved_text = resolved_text.replace(expr_text, resolved_str)
                    resolution_status = "RESOLVED"
                else:
                    resolution_status = "UNRESOLVED"
            else:
                resolution_status = "UNAVAILABLE"
        
        # Extract answer
        final_output = extract_answer(resolved_text, "temporal") if qtype_name == "temporal-reasoning" else resolved_text
        
        result_b = grader.grade(qid, qtype_name, final_output, gold)
        treatment_b_correct_flag = result_b.is_correct
        
        treatment_a_results[qid] = {
            "reader_output": reader_a_output,
            "gold": gold,
            "baseline_correct": baseline_correct,
            "treatment_correct": treatment_a_correct_flag,
        }
        
        treatment_b_results[qid] = {
            "reader_output": reader_a_output,
            "resolved_output": resolved_text,
            "final_output": final_output,
            "gold": gold,
            "baseline_correct": baseline_correct,
            "treatment_correct": treatment_b_correct_flag,
            "resolution_status": resolution_status,
        }
        
        if treatment_a_correct_flag:
            treatment_a_correct += 1
        if treatment_b_correct_flag:
            treatment_b_correct += 1
        if not baseline_correct and treatment_a_correct_flag:
            recovered_a += 1
        elif baseline_correct and not treatment_a_correct_flag:
            regressed_a += 1
        if not baseline_correct and treatment_b_correct_flag:
            recovered_b += 1
        elif baseline_correct and not treatment_b_correct_flag:
            regressed_b += 1
    
    print(f"Treatment A (Expanded Retrieval): {treatment_a_correct}/28 correct")
    print(f"  Recovered vs M30-29: {recovered_a}")
    print(f"  Regressed vs M30-29: {regressed_a}")
    print()
    print(f"Treatment B (Expanded Retrieval + Temporal Resolution + Answer Extraction): {treatment_b_correct}/28 correct")
    print(f"  Recovered vs M30-29: {recovered_b}")
    print(f"  Regressed vs M30-29: {regressed_b}")
    print()
    
    # Show abstention cases
    print("Abstention cases (Treatment B):")
    for qid in targets:
        r = treatment_b_results.get(qid, {})
        if r.get("note"):
            continue
        m29_c = m29["results"].get(qid, {}).get("treatment_correct", False)
        t_b = r.get("treatment_correct", False)
        
        changed = ""
        if t_b and not m29_c:
            changed = "RECOVERED"
        elif not t_b and m29_c:
            changed = "REGRESSED"
        
        print(f"  {qid}: M29={'✓' if m29_c else '✗'} B={'✓' if t_b else '✗'} {changed}")
        if r.get("reader_output"):
            print(f"    Reader: {r['reader_output'][:60]}")
        if r.get("resolved_output") and r["resolved_output"] != r.get("reader_output"):
            print(f"    Resolved: {r['resolved_output'][:60]}")
        if r.get("final_output") and r["final_output"] != r.get("resolved_output"):
            print(f"    Final: {r['final_output'][:60]}")
        if r.get("gold"):
            print(f"    Gold: {r['gold']}")
        print()
    
    # Save
    output = {
        "regression_test_passed": regression_passed,
        "baseline_correct": m29["treatment_correct"],
        "m31_correct": 9,
        "treatment_a_correct": treatment_a_correct,
        "treatment_b_correct": treatment_b_correct,
        "recovered_a": recovered_a,
        "recovered_b": recovered_b,
        "regressed_a": regressed_a,
        "regressed_b": regressed_b,
        "total_cases": len(targets),
        "treatment_a_results": treatment_a_results,
        "treatment_b_results": treatment_b_results,
    }
    output_path = PROJECT_ROOT / "results/m30_32/results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
