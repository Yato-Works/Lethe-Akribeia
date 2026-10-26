#!/usr/bin/env python3
"""M30-27 Phase 2-3: Non-oracle temporal resolution experiment.

Phase 2: Implement a deterministic temporal resolver that uses ONLY
information available in the production pipeline's context (timestamps
in turn headers like '[D23:23 on 1:32 pm on 6 January, 2024]').

Phase 3: Apply the resolver to the 8 relative-date cases and measure
whether non-oracle resolution improves accuracy.

Constraints:
- No Gold, no Gold evidence IDs, no oracle-selected evidence.
- Only use timestamps present in the production pipeline's context.
- If a reference datetime cannot be uniquely determined, mark as AMBIGUOUS.
- Do not modify production code, Gold, scorer, or benchmark data.
- 553 holdout completely isolated.
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCOMO = PROJECT_ROOT / "datasets/external/locomo10.json"
CACHE = PROJECT_ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
INTERVENTION = PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json"
M30_3 = PROJECT_ROOT / "results/m30_3_evaluation_results.json"

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}

# Relative date patterns
RELATIVE_PATTERNS = [
    (re.compile(r"\bnext month\b", re.I), "next_month"),
    (re.compile(r"\blast month\b", re.I), "last_month"),
    (re.compile(r"\bthis month\b", re.I), "this_month"),
    (re.compile(r"\bnext year\b", re.I), "next_year"),
    (re.compile(r"\blast year\b", re.I), "last_year"),
    (re.compile(r"\bone year ago\b", re.I), "one_year_ago"),
    (re.compile(r"\b(\d+)\s+years?\s+ago\b", re.I), "years_ago"),
    (re.compile(r"\byesterday\b", re.I), "yesterday"),
    (re.compile(r"\btomorrow\b", re.I), "tomorrow"),
]

def parse_timestamp(ts_str):
    """Parse '1:32 pm on 6 January, 2024' -> (2024, 1, 6)"""
    if not ts_str:
        return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", ts_str)
    if m and m.group(2).lower() in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None

def resolve_relative(expr_type, base_date):
    """Resolve a relative date expression against a base date."""
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
    if expr_type == "last_year" or expr_type == "one_year_ago":
        return (y - 1, None, None)
    if expr_type == "years_ago":
        # Extract the number from the expression
        m = re.search(r"(\d+)\s+years?\s+ago", expr_type, re.I)
        if m:
            return (y - int(m.group(1)), None, None)
        return None
    if expr_type == "yesterday":
        return (y, mo, d - 1) if d > 1 else ((y, mo - 1, 28) if mo > 1 else (y - 1, 12, 31))
    if expr_type == "tomorrow":
        return (y, mo, d + 1) if d < 28 else ((y, mo + 1, 1) if mo < 12 else (y + 1, 1, 1))
    return None

def extract_timestamps_from_context(context_text):
    """Extract all timestamps from context text.
    
    Returns list of (dia_id, timestamp_str, parsed_date) tuples.
    """
    results = []
    # Pattern: [D23:23 on 1:32 pm on 6 January, 2024]
    pattern = re.compile(r"\[(D\d+:\d+)\s+on\s+([^\]]+)\]")
    for m in pattern.finditer(context_text):
        dia_id = m.group(1)
        ts_str = m.group(2)
        parsed = parse_timestamp(ts_str)
        if parsed:
            results.append((dia_id, ts_str, parsed))
    return results

def find_question_relevant_turn(question, context_text, timestamps):
    """Find the turn most relevant to the question.
    
    Uses simple keyword matching between question and turn text.
    Returns the timestamp of the most relevant turn, or None.
    """
    if not timestamps:
        return None
    
    # Extract question keywords (simple approach)
    q_words = set(re.findall(r"\b[a-z]{3,}\b", question.lower()))
    q_words -= {"the", "and", "for", "are", "was", "were", "did", "does", "has", "have", "had", "will", "would", "could", "should", "can", "may", "might", "must", "this", "that", "these", "those", "when", "where", "what", "which", "who", "whom", "whose", "why", "how", "not", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "than", "too", "very", "just", "also", "now", "then", "here", "there", "once", "again", "further", "else", "ever", "never", "always", "often", "sometimes", "usually", "already", "yet", "still", "even", "back", "away", "down", "off", "over", "under", "above", "below", "between", "through", "during", "before", "after", "about", "into", "onto", "upon", "out", "up", "far", "near", "inside", "outside", "behind", "beyond", "across", "around", "along", "among", "against", "within", "without", "toward", "towards", "upon", "per", "via", "like", "unlike", "despite", "except", "regarding", "concerning", "according", "due", "owing", "thanks", "regardless", "irrespective", "notwithstanding", "whether", "either", "neither", "both", "whether", "whatever", "whenever", "wherever", "however", "whichever", "whoever", "whomever", "whosever", "whatsoever", "wheresoever", "whensoever", "howsoever", "whosoever", "whomsoever", "whosesoever", "whatsoever", "wheresoever", "whensoever", "howsoever", "whosoever", "whomsoever", "whosesoever"}
    
    # Score each turn by keyword overlap
    best_score = 0
    best_ts = None
    
    for dia_id, ts_str, parsed in timestamps:
        # Find the turn text associated with this dia_id
        # Pattern: [D23:23 on ...] Speaker: text
        turn_pattern = re.compile(re.escape(f"[{dia_id} on {ts_str}]") + r"\s*[^:]+:\s*([^\n]+)")
        m = turn_pattern.search(context_text)
        if m:
            turn_text = m.group(1).lower()
            t_words = set(re.findall(r"\b[a-z]{3,}\b", turn_text))
            score = len(q_words & t_words)
            if score > best_score:
                best_score = score
                best_ts = (dia_id, ts_str, parsed)
    
    return best_ts

def main():
    print("=" * 78)
    print("M30-27 Phase 2-3: Non-oracle temporal resolution")
    print("=" * 78)
    print()
    
    # Load data
    locomo = json.loads(LOCOMO.read_text(encoding="utf-8"))
    locomo_by_id = {c["sample_id"]: c for c in locomo}
    
    cache = {}
    with open(CACHE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row.get("qid", "")] = row.get("context", "")
    
    # Build QID conversion map: lc_conv-26_63 -> conv-26-qa-063
    def to_cache_qid(qid):
        # lc_conv-26_63 -> conv-26-qa-063
        m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
        if m:
            return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}"
        return qid
    
    intervention = json.loads(INTERVENTION.read_text(encoding="utf-8"))
    targets = sorted([r["question_id"] for r in intervention["rows"] if r.get("group") == "TARGET"])
    
    m30_3 = json.loads(M30_3.read_text(encoding="utf-8"))
    m30_3_results = {r["question_id"]: r for r in m30_3["results"]}
    
    # Find the 8 relative-date cases
    rel_cases = []
    for qid in targets:
        r = m30_3_results.get(qid, {})
        out = r.get("reader_text", "").lower()
        for pattern, expr_type in RELATIVE_PATTERNS:
            if pattern.search(out):
                rel_cases.append((qid, r.get("reader_text", ""), r.get("gold", ""), expr_type))
                break
    
    print(f"Relative-date cases: {len(rel_cases)}")
    print()
    
    # Phase 2: For each case, try to resolve the relative date
    # using ONLY the production pipeline's context (with timestamps)
    print("--- Phase 2: Non-oracle resolution ---")
    print()
    
    resolved = 0
    ambiguous = 0
    unavailable = 0
    
    for qid, reader_output, gold, expr_type in rel_cases:
        conv_id = f"conv-{qid.split('_')[0].replace('lc_conv-', '')}"
        question = m30_3_results.get(qid, {}).get("question", "")
        
        # Get the production pipeline's context (with timestamps)
        context = cache.get(to_cache_qid(qid), "")
        if not context:
            print(f"  {qid}: NO CONTEXT in cache")
            unavailable += 1
            continue
        
        # Extract timestamps from context
        timestamps = extract_timestamps_from_context(context)
        if not timestamps:
            print(f"  {qid}: NO TIMESTAMPS in context")
            unavailable += 1
            continue
        
        # Find the question-relevant turn
        relevant = find_question_relevant_turn(question, context, timestamps)
        if not relevant:
            print(f"  {qid}: CANNOT IDENTIFY RELEVANT TURN")
            ambiguous += 1
            continue
        
        dia_id, ts_str, base_date = relevant
        
        # Resolve the relative date
        result = resolve_relative(expr_type, base_date)
        if result:
            ry, rmo, rd = result
            if rmo:
                result_str = f"{ry}-{rmo:02d}" if not rd else f"{ry}-{rmo:02d}-{rd:02d}"
            else:
                result_str = str(ry)
            print(f"  {qid}: '{reader_output}' -> {result_str} (via {dia_id} @ {ts_str})")
            resolved += 1
        else:
            print(f"  {qid}: CANNOT RESOLVE '{expr_type}'")
            ambiguous += 1
    
    print()
    print(f"Resolved: {resolved}/{len(rel_cases)}")
    print(f"Ambiguous: {ambiguous}/{len(rel_cases)}")
    print(f"Unavailable: {unavailable}/{len(rel_cases)}")
    print()
    
    # Phase 3: Compare baseline vs treatment
    print("--- Phase 3: Baseline vs Treatment ---")
    print()
    
    # Baseline: M30-3's scorer results
    baseline_correct = 0
    for qid, _, _, _ in rel_cases:
        r = m30_3_results.get(qid, {})
        if r.get("scorer_correct"):
            baseline_correct += 1
    
    print(f"Baseline (M30-3 scorer): {baseline_correct}/{len(rel_cases)} correct")
    print()
    
    # Treatment: Apply non-oracle resolution
    # For each resolved case, check if the resolved date matches Gold
    treatment_correct = 0
    recovered = 0
    regressed = 0
    
    for qid, reader_output, gold, expr_type in rel_cases:
        conv_id = f"conv-{qid.split('_')[0].replace('lc-conv-', '')}"
        question = m30_3_results.get(qid, {}).get("question", "")
        context = cache.get(to_cache_qid(qid), "")
        
        if not context:
            continue
        
        timestamps = extract_timestamps_from_context(context)
        if not timestamps:
            continue
        
        relevant = find_question_relevant_turn(question, context, timestamps)
        if not relevant:
            continue
        
        dia_id, ts_str, base_date = relevant
        result = resolve_relative(expr_type, base_date)
        
        if not result:
            continue
        
        ry, rmo, rd = result
        
        # Check if resolved date matches Gold
        gold_str = str(gold).lower()
        match = False
        
        # Check if year appears in Gold
        if str(ry) not in gold_str:
            match = False
        elif rmo is None:
            # Year-only resolution: match if year appears in Gold
            match = True
        else:
            # Year+month resolution: check if month name appears in Gold
            month_names = [m for m, i in MONTHS.items() if i == rmo]
            if month_names:
                match = month_names[0] in gold_str
            else:
                match = False
        
        # Check baseline
        baseline_match = m30_3_results.get(qid, {}).get("scorer_correct", False)
        
        if match:
            treatment_correct += 1
            if not baseline_match:
                recovered += 1
                date_str = f"{ry}-{rmo:02d}" if rmo is not None else str(ry)
                print(f"  RECOVERED: {qid}: '{reader_output}' -> {date_str} (Gold: {gold})")
        elif baseline_match:
            regressed += 1
            date_str = f"{ry}-{rmo:02d}" if rmo is not None else str(ry)
            print(f"  REGRESSED: {qid}: '{reader_output}' -> {date_str} (Gold: {gold})")
    
    print()
    print(f"Treatment correct: {treatment_correct}/{len(rel_cases)}")
    print(f"Recovered: {recovered}")
    print(f"Regressed: {regressed}")
    print()
    
    # Summary
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Relative-date cases: {len(rel_cases)}")
    print(f"Baseline correct: {baseline_correct}/{len(rel_cases)}")
    print(f"Treatment correct: {treatment_correct}/{len(rel_cases)}")
    print(f"Recovered: {recovered}")
    print(f"Regressed: {regressed}")
    print(f"Net change: {treatment_correct - baseline_correct}")
    print()
    print("NOTE: This experiment uses the production pipeline's context")
    print("(with timestamps), NOT M30-3's normal pipeline input.")
    print("M30-3's normal pipeline does NOT include timestamps.")
    print()
    print("If treatment shows improvement, the fix is to include timestamps")
    print("in M30-3's normal pipeline input (i.e., use locomo_adapter.py")
    print("instead of extract_evidence_turns + v4_bucketA_all.json).")

if __name__ == "__main__":
    main()
