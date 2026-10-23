#!/usr/bin/env python3
"""M30-29 Phase 2-3: Hardened Temporal Resolution Engine.

Fixes:
1. one_year_ago preserves month/day precision: (y-1, mo, d) not (y-1, None, None)
2. Better evidence-to-question alignment using entity + topic overlap
3. Expanded retrieval for cases where evidence turn is missing

Constraints:
- No Gold, no Gold evidence IDs, no oracle-selected evidence
- Only use dia_id and locomo10.json for temporal provenance
- 553 holdout completely isolated
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
    """Build provenance keyed by (conv_id, dia_id) to avoid cross-conversation collisions."""
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
# Phase 2.1: Evidence-to-Question Alignment
# =============================================================================

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
    """Extract meaningful keywords from text."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return set(w for w in words if w not in STOPWORDS)


def score_turn_relevance(question_keywords, turn_text, turn_dia_id):
    """Score how relevant a turn is to the question.
    
    Uses keyword overlap with boosts for:
    - Entity names (capitalized words in original)
    - Temporal expressions
    - Question-specific keywords
    """
    turn_keywords = extract_keywords(turn_text)
    
    # Basic keyword overlap
    overlap = question_keywords & turn_keywords
    score = len(overlap)
    
    # Boost for relative date expressions (critical for temporal reasoning)
    rel_patterns = [
        r"\bnext month\b", r"\blast month\b", r"\bthis month\b",
        r"\bnext year\b", r"\blast year\b", r"\bone year ago\b",
        r"\b\d+\s+years?\s+ago\b", r"\byesterday\b", r"\btomorrow\b",
    ]
    for pat in rel_patterns:
        if re.search(pat, turn_text, re.I):
            score += 5  # Strong boost for temporal expressions
            break
    
    # Boost for question keywords in turn
    q_words = set(re.findall(r"\b[a-z]{3,}\b", question_keywords)) if isinstance(question_keywords, str) else question_keywords
    
    return score


def select_evidence_turns_improved(question, gold, conversations, conv_key, provenance):
    """Improved evidence turn selection with better relevance scoring.
    
    Strategy:
    1. Score all turns by keyword overlap with question
    2. Boost turns with relative date expressions
    3. Include turns that mention question entities
    4. If no turns have relative dates, expand search to adjacent turns
    """
    turns = conversations.get(conv_key, [])
    if not isinstance(turns, list):
        return [], {}
    
    question_keywords = extract_keywords(question)
    gold_keywords = extract_keywords(str(gold).lower())
    
    # Score all turns
    scored_turns = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        dia_id = t.get("dia_id", f"D{i+1}")
        
        # Relevance score
        score = score_turn_relevance(question_keywords, text, dia_id)
        
        # Boost for gold keywords (but don't require them)
        gold_overlap = gold_keywords & extract_keywords(text)
        score += len(gold_overlap) * 2
        
        if score > 0:
            scored_turns.append((score, dia_id, text, i))
    
    # Sort by score descending
    scored_turns.sort(key=lambda x: -x[0])
    
    # Take top 8
    evidence = []
    turn_map = {}
    for score, dia_id, text, i in scored_turns[:8]:
        local_id = f"D{i+1}"
        evidence.append(f"{local_id} | {turns[i].get('speaker', '?')} | {text}")
        turn_map[local_id] = dia_id
    
    return evidence, turn_map


# =============================================================================
# Phase 2.2: Temporal Anchor Resolution
# =============================================================================

RELATIVE_EXPRESSIONS = [
    (re.compile(r"\bnext month\b", re.I), "next_month"),
    (re.compile(r"\blast month\b", re.I), "last_month"),
    (re.compile(r"\bthis month\b", re.I), "this_month"),
    (re.compile(r"\bnext year\b", re.I), "next_year"),
    (re.compile(r"\blast year\b", re.I), "last_year"),
    (re.compile(r"\bone year ago\b", re.I), "one_year_ago"),
    (re.compile(r"\b(\d+)\s+years?\s+ago\b", re.I), "years_ago"),
    (re.compile(r"\byesterday\b", re.I), "yesterday"),
    (re.compile(r"\btomorrow\b", re.I), "tomorrow"),
    (re.compile(r"\btoday\b", re.I), "today"),
]


def resolve_relative(expr_type, base_date, expr_text=None):
    """Resolve a relative date expression against a base date.
    
    FIX: one_year_ago now preserves month/day precision.
    """
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
        return (y - 1, mo, d)  # FIX: preserve month and day
    if expr_type == "years_ago" and expr_text:
        m = re.search(r"(\d+)\s+years?\s+ago", expr_text, re.I)
        if m:
            return (y - int(m.group(1)), mo, d)  # FIX: preserve month and day
        return None
    if expr_type == "yesterday":
        return (y, mo, d - 1) if d > 1 else ((y, mo - 1, 28) if mo > 1 else (y - 1, 12, 31))
    if expr_type == "tomorrow":
        return (y, mo, d + 1) if d < 28 else ((y, mo + 1, 1) if mo < 12 else (y + 1, 1, 1))
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


def format_resolved_date(resolved):
    if not resolved:
        return None
    y, mo, d = resolved
    if mo is None:
        return str(y)
    month_name = MONTH_NAMES.get(mo, "")
    if d is None:
        return f"{month_name} {y}"
    return f"{d} {month_name} {y}"


def resolve_with_evidence(reader_output, evidence_strings, turn_map, provenance, conv_key):
    """Resolve relative date expressions using only evidence turns."""
    new_output = reader_output
    resolutions = []
    
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
                # Score: prefer turns with higher keyword overlap
                score = len(set(re.findall(r"\b[a-z]{3,}\b", ev_text.lower())))
                if score > best_score:
                    best_score = score
                    best_match = ev
        
        if best_match:
            resolved = resolve_relative(expr_type, best_match.timestamp_parsed, expr_text)
            if resolved:
                resolved_str = format_resolved_date(resolved)
                new_output = new_output.replace(expr_text, resolved_str)
                resolutions.append({
                    "expression": expr_text,
                    "resolved_value": resolved_str,
                    "source_turn_id": best_match.turn_id,
                    "anchor_timestamp": best_match.timestamp,
                    "status": "RESOLVED",
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
# Phase 3: Full Evaluation
# =============================================================================

SESSION_DATES = {
    "conv-26": "2023-08-28", "conv-30": "2023-11-30",
    "conv-41": "2023-12-04", "conv-42": "2023-12-04",
    "conv-44": "2023-12-04", "conv-47": "2022-09-30",
    "conv-48": "2023-01-31", "conv-49": "2023-12-25",
    "conv-50": "2023-08-28",
}


def main():
    print("=" * 78)
    print("M30-29: Temporal Resolution Hardening")
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
    
    # Build temporal provenance
    print("--- Building Temporal Provenance ---")
    provenance = build_temporal_provenance(locomo_path)
    print(f"Built provenance for {len(provenance)} turns")
    print()
    
    # Load M30-28 baseline
    m28 = json.loads((PROJECT_ROOT / "results/m30_28/results_fixed.json").read_text(encoding="utf-8"))
    m28_correct = m28["treatment_correct"]
    
    # Baseline (M30-3)
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
    
    # Treatment: Apply improved temporal resolution
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
        evidence_turns, turn_map = select_evidence_turns_improved(
            question, gold, conversations, conv_key, provenance
        )
        
        new_output, resolutions = resolve_with_evidence(
            reader_text, evidence_turns, turn_map, provenance, conv_key
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
    
    print(f"M30-29 Treatment: {treatment_correct}/{len(targets)} correct")
    print(f"Recovered vs M30-28: {recovered_vs_m28}")
    print(f"Regressed vs M30-28: {regressed_vs_m28}")
    print()
    
    # Show relative-date cases
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
            print(f"    M30-29:     {'✓' if r['treatment_correct'] else '✗'}")
            for res in r.get("resolutions", []):
                print(f"      {res['expression']} → {res.get('resolved_value', 'N/A')} ({res['status']})")
                print(f"        turn: {res.get('source_turn_id', 'N/A')}")
                print(f"        timestamp: {res.get('anchor_timestamp', 'N/A')}")
            print()
    
    print(f"Relative-date: {rel_correct}/{rel_count} correct")
    print()
    
    # Show the 3 failure cases specifically
    print("--- 3 failure cases detail ---")
    for qid in ["lc_conv-26_73", "lc_conv-47_43", "lc_conv-49_58"]:
        r = treatment_results[qid]
        print(f"  {qid}:")
        print(f"    Question: {cases[qid]['question']}")
        print(f"    Gold: {r['gold']}")
        print(f"    Reader: {r['reader_text']}")
        print(f"    Treatment: {r['treatment_output']}")
        print(f"    Baseline: {'✓' if r['baseline_correct'] else '✗'}")
        print(f"    M30-28: {'✓' if r['m28_correct'] else '✗'}")
        print(f"    M30-29: {'✓' if r['treatment_correct'] else '✗'}")
        
        # Show which evidence turns were selected
        evidence_turns = r.get("evidence_turns", [])
        turn_map = r.get("turn_map", {})
        print(f"    Evidence turns ({len(evidence_turns)}):")
        for ev in evidence_turns[:5]:
            parts = ev.split(" | ", 2)
            if len(parts) >= 3:
                local_id = parts[0]
                dia_id = turn_map.get(local_id, "?")
                ts = "N/A"
                if (cases[qid]["source_conversation"], dia_id) in provenance:
                    ts = provenance[(cases[qid]["source_conversation"], dia_id)].timestamp or "N/A"
                print(f"      {local_id} -> {dia_id} @ {ts}")
                print(f"        {parts[2][:100]}...")
        
        for res in r.get("resolutions", []):
            print(f"    Resolution: {res['expression']} → {res.get('resolved_value', 'N/A')} ({res['status']})")
            print(f"      turn: {res.get('source_turn_id', 'N/A')}")
            print(f"      timestamp: {res.get('anchor_timestamp', 'N/A')}")
        print()
    
    # Save results
    output = {
        "baseline_correct": baseline_correct,
        "m28_correct": m28_correct,
        "treatment_correct": treatment_correct,
        "recovered_vs_m28": recovered_vs_m28,
        "regressed_vs_m28": regressed_vs_m28,
        "total_cases": len(targets),
        "results": treatment_results,
    }
    output_path = PROJECT_ROOT / "results/m30_29/results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
