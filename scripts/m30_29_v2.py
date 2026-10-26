#!/usr/bin/env python3
"""M30-29 v2: Fix granularity + improve evidence selection.

Fix 1: Precision-preserving normalization
  - Detect question's required granularity (year/month/day)
  - Round resolved date to that precision
  - No Gold, no QID-specific rules

Fix 2: Better evidence-to-question alignment
  - Prefer turns with same entity/topic as question
  - Penalize turns with relative expressions but unrelated topic
"""
import json, re, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

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


@dataclass(frozen=True)
class TemporalEvidence:
    turn_id: str
    speaker: str
    text: str
    session_id: Optional[str]
    timestamp: Optional[str]
    timestamp_parsed: Optional[tuple]


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


def build_temporal_provenance(locomo_path):
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
            parsed_dt = parse_session_datetime(session_dt)
            turns = conv_data.get(session_key, [])
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                dia_id = turn.get("dia_id", "")
                if not dia_id:
                    continue
                mapping[(conv_id, dia_id)] = TemporalEvidence(
                    turn_id=dia_id,
                    speaker=turn.get("speaker", ""),
                    text=turn.get("text", ""),
                    session_id=session_key,
                    timestamp=session_dt,
                    timestamp_parsed=parsed_dt,
                )
    return mapping


# =============================================================================
# Phase 2.1: Improved Evidence-to-Question Alignment
# =============================================================================

def score_turn_relevance(question, turn_text):
    """Score turn relevance to question.
    
    Key improvement: prefer turns that share topic/entity keywords
    AND have relative date expressions.
    """
    q_kw = extract_keywords(question)
    t_kw = extract_keywords(turn_text)
    
    # Basic keyword overlap
    overlap = q_kw & t_kw
    score = len(overlap)
    
    # Strong boost for topic overlap (entity names, specific nouns)
    topic_overlap = q_kw & t_kw
    score += len(topic_overlap) * 2
    
    # Boost for relative date expressions
    rel_patterns = [
        r"\bnext month\b", r"\blast month\b", r"\bthis month\b",
        r"\bnext year\b", r"\blast year\b", r"\bone year ago\b",
        r"\b\d+\s+years?\s+ago\b", r"\byesterday\b", r"\btomorrow\b",
    ]
    has_rel = False
    for pat in rel_patterns:
        if re.search(pat, turn_text, re.I):
            score += 10  # Strong boost
            has_rel = True
            break
    
    # Penalize turns with relative dates but no topic overlap
    if has_rel and len(topic_overlap) == 0:
        score -= 5
    
    return score


def select_evidence_turns_v2(question, gold, conversations, conv_key):
    """Improved evidence selection with better relevance scoring."""
    turns = conversations.get(conv_key, [])
    if not isinstance(turns, list):
        return [], {}
    
    scored = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        dia_id = t.get("dia_id", f"D{i+1}")
        
        score = score_turn_relevance(question, text)
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
# Phase 2.2: Temporal Resolution Engine
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
]


def resolve_relative(expr_type, base_date, expr_text=None):
    if not base_date:
        return None
    y, mo, d = base_date
    
    if expr_type == "next_month":
        return (y + (1 if mo == 12 else 0), 1 if mo == 12 else mo + 1, None)
    if expr_type == "last_month":
        return (y - (1 if mo == 1 else 0), 12 if mo == 1 else mo - 1, None)
    if expr_type == "this_month":
        return (y, mo, None)
    if expr_type == "next_year":
        return (y + 1, None, None)
    if expr_type == "last_year":
        return (y - 1, None, None)
    if expr_type == "one_year_ago":
        return (y - 1, mo, d)  # Preserve month/day precision
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
                return (y - n, mo, d)  # Preserve month/day precision
        return None
    if expr_type == "yesterday":
        return (y, mo, d - 1) if d > 1 else ((y, mo - 1, 28) if mo > 1 else (y - 1, 12, 31))
    if expr_type == "tomorrow":
        return (y, mo, d + 1) if d < 31 else ((y, mo + 1, 1) if mo < 12 else (y + 1, 1, 1))
    if expr_type == "today":
        return (y, mo, d)
    return None


def detect_relative_expression(text):
    results = []
    text_lower = text.lower()
    for pattern, expr_type in RELATIVE_EXPRESSIONS:
        for m in pattern.finditer(text_lower):
            results.append((expr_type, m.start(), m.end()))
    return results


# =============================================================================
# Phase 2.3: Precision-Preserving Normalization
# =============================================================================

def detect_question_granularity(question):
    """Detect the granularity required by the question.
    
    Returns: "year", "month", or "day"
    """
    q = question.lower()
    
    # Check for day-level indicators
    day_patterns = [
        r"\bwhat day\b", r"\bwhich day\b", r"\bdate\b",
        r"\bexact date\b", r"\bspecific date\b",
    ]
    for pat in day_patterns:
        if re.search(pat, q):
            return "day"
    
    # Check for month-level indicators
    month_patterns = [
        r"\bwhat month\b", r"\bwhich month\b", r"\bwhen\b",
    ]
    for pat in month_patterns:
        if re.search(pat, q):
            return "month"
    
    # Check for year-level indicators
    year_patterns = [
        r"\bwhat year\b", r"\bwhich year\b",
    ]
    for pat in year_patterns:
        if re.search(pat, q):
            return "year"
    
    # Default to month for temporal questions
    if re.search(r"\bwhen\b", q):
        return "month"
    
    return "month"


def format_with_granularity(resolved, granularity):
    """Format resolved date with appropriate granularity.
    
    (2022, 10, 8) with "month" -> "October 2022"
    (2022, 10, 8) with "year" -> "2022"
    (2022, 10, 8) with "day" -> "8 October 2022"
    """
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
        return f"{d} {month_name} {y}"
    return str(y)


def resolve_with_evidence_v2(reader_output, evidence_strings, turn_map, provenance, conv_key, question):
    """Resolve relative date expressions with precision preservation."""
    new_output = reader_output
    resolutions = []
    
    # Detect question granularity
    granularity = detect_question_granularity(question)
    
    expressions = detect_relative_expression(reader_output)
    if not expressions:
        return new_output, resolutions
    
    for expr_type, start, end in expressions:
        expr_text = reader_output[start:end]
        
        # Search evidence turns for this expression
        best_match = None
        best_score = 0
        
        for ev_str in evidence_strings:
            parts = ev_str.split(" | ", 2)
            if len(parts) < 3:
                continue
            local_id = parts[0]
            ev_text = parts[2]
            
            dia_id = turn_map.get(local_id, "")
            if not dia_id or (conv_key, dia_id) not in provenance:
                continue
            
            ev = provenance[(conv_key, dia_id)]
            if not ev.timestamp_parsed:
                continue
            
            # Check if this evidence turn contains the expression
            if expr_text.lower() in ev_text.lower():
                # Score by keyword overlap
                q_kw = extract_keywords(question)
                ev_kw = extract_keywords(ev_text)
                score = len(q_kw & ev_kw)
                if score > best_score:
                    best_score = score
                    best_match = ev
        
        if best_match:
            resolved = resolve_relative(expr_type, best_match.timestamp_parsed, expr_text)
            if resolved:
                # Apply granularity
                resolved_str = format_with_granularity(resolved, granularity)
                new_output = new_output.replace(expr_text, resolved_str)
                resolutions.append({
                    "expression": expr_text,
                    "resolved_value": resolved_str,
                    "source_turn_id": best_match.turn_id,
                    "anchor_timestamp": best_match.timestamp,
                    "status": "RESOLVED",
                    "granularity": granularity,
                })
            else:
                resolutions.append({
                    "expression": expr_text,
                    "resolved_value": None,
                    "source_turn_id": best_match.turn_id,
                    "anchor_timestamp": best_match.timestamp,
                    "status": "UNSUPPORTED_EXPRESSION",
                })
        else:
            resolutions.append({
                "expression": expr_text,
                "resolved_value": None,
                "source_turn_id": None,
                "anchor_timestamp": None,
                "status": "UNAVAILABLE",
            })
    
    return new_output, resolutions


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    print("=" * 78)
    print("M30-29 v2: Hardened Temporal Resolution (Fixed)")
    print("=" * 78)
    print()
    
    locomo_path = PROJECT_ROOT / "datasets/external/locomo10.json"
    bucketA_path = PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json"
    intervention_path = PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json"
    m30_3_path = PROJECT_ROOT / "results/m30_3_evaluation_results.json"
    
    bucketA = json.loads(bucketA_path.read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucketA["cases"]}
    conversations = bucketA["conversations"]
    
    intervention = json.loads(intervention_path.read_text(encoding="utf-8"))
    targets = sorted([r["question_id"] for r in intervention["rows"] if r.get("group") == "TARGET"])
    
    m30_3 = json.loads(m30_3_path.read_text(encoding="utf-8"))
    m30_3_results = {r["question_id"]: r for r in m30_3["results"]}
    
    print(f"TARGET cases: {len(targets)}")
    print()
    
    # Build provenance
    print("--- Building Temporal Provenance ---")
    provenance = build_temporal_provenance(locomo_path)
    print(f"Built provenance for {len(provenance)} turns")
    print()
    
    # Load M30-28 baseline
    m28 = json.loads((PROJECT_ROOT / "results/m30_28/results_fixed.json").read_text(encoding="utf-8"))
    m28_correct = m28["treatment_correct"]
    
    # Baseline
    grader = OfficialLongMemEvalGrader()
    baseline_correct = 0
    baseline_results = {}
    for qid in targets:
        r = m30_3_results.get(qid, {})
        correct = r.get("scorer_correct", False)
        baseline_results[qid] = {
            "reader_text": r.get("reader_text", ""),
            "gold": r.get("gold", ""),
            "correct": correct,
        }
        if correct:
            baseline_correct += 1
    
    print(f"Baseline (M30-3): {baseline_correct}/{len(targets)} correct")
    print(f"M30-28 Treatment B: {m28_correct}/{len(targets)} correct")
    print()
    
    # Treatment v2
    treatment_correct = 0
    treatment_results = {}
    recovered_vs_m28 = 0
    regressed_vs_m28 = 0
    
    for qid in targets:
        case = cases.get(qid, {})
        if not case:
            continue
        
        question = case.get("question", "")
        gold = str(case.get("gold_answer", ""))
        reader_text = baseline_results[qid]["reader_text"]
        conv_key = case.get("source_conversation", "")
        
        # Use improved evidence selection
        evidence_turns, turn_map = select_evidence_turns_v2(
            question, gold, conversations, conv_key
        )
        
        new_output, resolutions = resolve_with_evidence_v2(
            reader_text, evidence_turns, turn_map, provenance, conv_key, question
        )
        
        # Score
        if new_output != reader_text:
            try:
                qtype_map = {1: "multi-session", 2: "temporal-reasoning", 3: "single-session-user", 4: "single-session-assistant", 5: "abstention"}
                qtype_name = qtype_map.get(case.get("category", 3), "single-session-user")
                result = grader.grade(qid, qtype_name, new_output, gold)
                treatment_correct_flag = result.is_correct
            except Exception as e:
                print(f"    ERROR grading {qid}: {e}")
                treatment_correct_flag = False
        else:
            treatment_correct_flag = baseline_results[qid]["correct"]
        
        m28_flag = m28["results"].get(qid, {}).get("treatment_correct", False)
        
        treatment_results[qid] = {
            "reader_text": reader_text,
            "treatment_output": new_output,
            "gold": gold,
            "baseline_correct": baseline_results[qid]["correct"],
            "m28_correct": m28_flag,
            "treatment_correct": treatment_correct_flag,
            "resolutions": resolutions,
            "evidence_turns": evidence_turns,
            "turn_map": turn_map,
        }
        
        if treatment_correct_flag:
            treatment_correct += 1
        if not m28_flag and treatment_correct_flag:
            recovered_vs_m28 += 1
        elif m28_flag and not treatment_correct_flag:
            regressed_vs_m28 += 1
    
    print(f"M30-29 v2 Treatment: {treatment_correct}/{len(targets)} correct")
    print(f"Recovered vs M30-28: {recovered_vs_m28}")
    print(f"Regressed vs M30-28: {regressed_vs_m28}")
    print()
    
    # Relative-date cases
    print("--- Relative-date cases ---")
    rel_count = 0
    rel_correct = 0
    for qid in targets:
        reader_text = baseline_results[qid]["reader_text"]
        if detect_relative_expression(reader_text):
            rel_count += 1
            r = treatment_results[qid]
            if r["treatment_correct"]:
                rel_correct += 1
            print(f"  {qid}:")
            print(f"    Reader:     {r['reader_text']}")
            print(f"    Treatment:  {r['treatment_output']}")
            print(f"    Gold:       {r['gold']}")
            print(f"    Baseline:   {'✓' if r['baseline_correct'] else '✗'}")
            print(f"    M30-28:     {'✓' if r['m28_correct'] else '✗'}")
            print(f"    M30-29 v2:  {'✓' if r['treatment_correct'] else '✗'}")
            for res in r.get("resolutions", []):
                print(f"      {res['expression']} → {res.get('resolved_value', 'N/A')} ({res['status']})")
                print(f"        turn: {res.get('source_turn_id', 'N/A')}")
                print(f"        timestamp: {res.get('anchor_timestamp', 'N/A')}")
                print(f"        granularity: {res.get('granularity', 'N/A')}")
            print()
    
    print(f"Relative-date: {rel_correct}/{rel_count} correct")
    print()
    
    # Save
    output = {
        "baseline_correct": baseline_correct,
        "m28_correct": m28_correct,
        "treatment_correct": treatment_correct,
        "recovered_vs_m28": recovered_vs_m28,
        "regressed_vs_m28": regressed_vs_m28,
        "total_cases": len(targets),
        "results": treatment_results,
    }
    output_path = PROJECT_ROOT / "results/m30_29/results_v2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
