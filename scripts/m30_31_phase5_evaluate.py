#!/usr/bin/env python3
"""M30-31 Phase 5: 28-case evaluation with Temporal Resolution + Answer Extraction.

For each abstention case where Reader provides a relative date:
1. Apply M30-29 Temporal Resolution Engine
2. Extract just the answer (date/fact)
3. Score with existing scorer

Also test on cases where Reader already outputs dates (may regress).

Baseline: M30-29's 8/28
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


def select_evidence_v2(question, conversations, conv_key):
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


def resolve_temporal_expressions(text, base_date):
    new_text = text
    expressions = detect_relative_expression(text)
    
    for expr_type, start, end in expressions:
        expr_text = text[start:end]
        resolved = resolve_relative(expr_type, base_date, expr_text)
        if resolved:
            resolved_str = format_resolved_date(resolved)
            new_text = new_text.replace(expr_text, resolved_str)
    
    return new_text


def extract_answer(text, question_type="temporal"):
    if question_type == "temporal":
        # Most specific first: "Month Day, Year" or "Month Year"
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


def main():
    print("=" * 78)
    print("M30-31 Phase 5: 28-case Evaluation")
    print("=" * 78)
    print()
    
    # Load data
    m29 = json.loads((PROJECT_ROOT / "results/m30_29/results_v2.json").read_text(encoding="utf-8"))
    m30_3 = json.loads((PROJECT_ROOT / "results/m30_3_evaluation_results.json").read_text(encoding="utf-8"))
    m30_3_results = {r["question_id"]: r for r in m30_3["results"]}
    m30_30 = json.loads((PROJECT_ROOT / "results/m30_30/treatment_comparison.json").read_text(encoding="utf-8"))
    bucketA = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucketA["cases"]}
    conversations = bucketA["conversations"]
    intervention = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted([r["question_id"] for r in intervention["rows"] if r.get("group") == "TARGET"])
    
    # Build provenance
    locomo_path = PROJECT_ROOT / "datasets/external/locomo10.json"
    provenance = build_provenance(locomo_path)
    
    grader = OfficialLongMemEvalGrader()
    
    # Evaluate all 28 cases
    treatment_correct = 0
    treatment_results = {}
    recovered = 0
    regressed = 0
    
    for qid in targets:
        case = cases.get(qid, {})
        if not case:
            continue
        
        question = case.get("question", "")
        gold = str(case.get("gold_answer", ""))
        baseline_correct = m29["results"].get(qid, {}).get("treatment_correct", False)
        reader_output = m30_3_results.get(qid, {}).get("reader_text", "")
        
        # Skip non-abstention cases (keep M30-29 results)
        if "don't know" not in reader_output.lower():
            treatment_results[qid] = {
                "treatment_correct": baseline_correct,
                "note": "Not abstention, kept M30-29 result",
            }
            if baseline_correct:
                treatment_correct += 1
            continue
        
        # For abstention cases, use Treatment A's Reader output (improved retrieval)
        m30_30_output = m30_30["reader_a_results"].get(qid, {}).get("reader_output", "")
        
        if not m30_30_output or m30_30_output == reader_output:
            # No change from Treatment A
            treatment_results[qid] = {
                "treatment_correct": baseline_correct,
                "note": "No change from Treatment A",
            }
            if baseline_correct:
                treatment_correct += 1
            continue
        
        # Apply Temporal Resolution
        conv_key = case.get("source_conversation", "")
        
        # Get evidence turns with timestamps
        evidence_strings, turn_map = select_evidence_v2(question, conversations, conv_key)
        
        # Find the turn that contains the relative expression
        expressions = detect_relative_expression(m30_30_output)
        resolved_text = m30_30_output
        resolution_status = "NO_RELATIVE_EXPRESSION"
        
        for expr_type, start, end in expressions:
            expr_text = m30_30_output[start:end]
            
            # Find the evidence turn with this expression
            best_match = None
            for ev_str in evidence_strings:
                parts = ev_str.split(" | ", 2)
                if len(parts) >= 3:
                    local_id = parts[0]
                    ev_text = parts[2]
                    dia_id = turn_map.get(local_id, "")
                    if not dia_id or (conv_key, dia_id) not in provenance:
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
        qtype_name = {1: "multi-session", 2: "temporal-reasoning", 3: "single-session-user", 4: "single-session-assistant", 5: "abstention"}.get(case.get("category", 3), "single-session-user")
        
        final_output = extract_answer(resolved_text, "temporal") if qtype_name == "temporal-reasoning" else resolved_text
        
        result = grader.grade(qid, qtype_name, final_output, gold)
        treatment_correct_flag = result.is_correct
        
        treatment_results[qid] = {
            "reader_output": m30_30_output,
            "resolved_output": resolved_text,
            "final_output": final_output,
            "gold": gold,
            "baseline_correct": baseline_correct,
            "treatment_correct": treatment_correct_flag,
            "resolution_status": resolution_status,
        }
        
        if treatment_correct_flag:
            treatment_correct += 1
        if not baseline_correct and treatment_correct_flag:
            recovered += 1
        elif baseline_correct and not treatment_correct_flag:
            regressed += 1
    
    print(f"Treatment (Temporal Resolution + Answer Extraction): {treatment_correct}/28 correct")
    print(f"Recovered vs M30-29: {recovered}")
    print(f"Regressed vs M30-29: {regressed}")
    print()
    
    # Show abstention cases
    print("Abstention cases:")
    for qid in targets:
        r = treatment_results.get(qid, {})
        if r.get("note"):
            continue
        m29_c = m29["results"].get(qid, {}).get("treatment_correct", False)
        t_c = r.get("treatment_correct", False)
        
        changed = ""
        if t_c and not m29_c:
            changed = "RECOVERED"
        elif not t_c and m29_c:
            changed = "REGRESSED"
        
        print(f"  {qid}: M29={'✓' if m29_c else '✗'} T={'✓' if t_c else '✗'} {changed}")
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
        "baseline_correct": m29["treatment_correct"],
        "treatment_correct": treatment_correct,
        "recovered": recovered,
        "regressed": regressed,
        "total_cases": len(targets),
        "results": treatment_results,
    }
    output_path = PROJECT_ROOT / "results/m30_31/results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
