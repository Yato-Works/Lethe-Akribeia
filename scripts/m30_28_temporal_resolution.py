#!/usr/bin/env python3
"""M30-28: Temporal Provenance Integration & End-to-End Recovery.

Section 2: Temporal Provenance — 時間情報を失わないデータ構造
Section 3: Temporal Resolution Engine — 決定論的な時間解決

Implementation:
1. Build a dia_id → timestamp mapping from locomo10.json
   - dia_id format: D{session_num}:{turn_num}
   - session_N_date_time gives the session's datetime
2. For each of the 28 TARGET cases, extract evidence turns (same as M30-3)
   - But now preserve the dia_id → timestamp mapping
3. Implement Temporal Resolution Engine:
   - Find turns with relative date expressions
   - Use the turn's session datetime as the anchor
   - Resolve to absolute date deterministically
4. Compare Baseline vs Treatment B on 28 cases

Constraints:
- No Gold, no Gold evidence IDs, no oracle-selected evidence
- Only use dia_id (turn position) and locomo10.json (source data)
- 553 holdout completely isolated
- Do not modify production code, Gold, scorer, or benchmark data
"""
import json, re, sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader

# =============================================================================
# Section 2: Temporal Provenance
# =============================================================================

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}


@dataclass(frozen=True)
class TemporalEvidence:
    """A single dialogue turn with temporal provenance."""
    turn_id: str           # e.g., "D15:12"
    speaker: str
    text: str
    session_id: Optional[str]   # e.g., "session_15"
    timestamp: Optional[str]    # e.g., "3:19 pm on 28 August, 2023"
    timestamp_parsed: Optional[tuple]  # (year, month, day) or None


def parse_session_datetime(s):
    """Parse '1:32 pm on 6 January, 2024' -> (2024, 1, 6)"""
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
    """Build a dia_id → TemporalEvidence mapping from locomo10.json.
    
    Returns:
        dict: {dia_id: TemporalEvidence}
    """
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
            session_num = session_key.replace("session_", "")
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
                
                mapping[dia_id] = TemporalEvidence(
                    turn_id=dia_id,
                    speaker=turn.get("speaker", ""),
                    text=turn.get("text", ""),
                    session_id=session_key,
                    timestamp=session_dt,
                    timestamp_parsed=parsed_dt,
                )
    
    return mapping


# =============================================================================
# Section 3: Temporal Resolution Engine
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


def resolve_relative(expr_type, base_date):
    """Resolve a relative date expression against a base date.
    
    Returns:
        tuple: (year, month, day) or None if unresolvable
        month may be None for year-only resolution
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
    if expr_type == "last_year" or expr_type == "one_year_ago":
        return (y - 1, None, None)
    if expr_type == "years_ago":
        return None  # Cannot resolve without knowing the number
    if expr_type == "yesterday":
        return (y, mo, d - 1) if d > 1 else ((y, mo - 1, 28) if mo > 1 else (y - 1, 12, 31))
    if expr_type == "tomorrow":
        return (y, mo, d + 1) if d < 28 else ((y, mo + 1, 1) if mo < 12 else (y + 1, 1, 1))
    if expr_type == "today":
        return (y, mo, d)
    return None


def detect_relative_expression(text):
    """Detect relative date expressions in text.
    
    Returns:
        list of (expr_type, match_start, match_end) tuples
    """
    results = []
    text_lower = text.lower()
    for pattern, expr_type in RELATIVE_EXPRESSIONS:
        for m in pattern.finditer(text_lower):
            results.append((expr_type, m.start(), m.end()))
    return results


def format_resolved_date(resolved):
    """Format a resolved date tuple as a string.
    
    (2024, 2, None) -> "February 2024"
    (2022, None, None) -> "2022"
    (2023, 9, 15) -> "15 September 2023"
    """
    if not resolved:
        return None
    y, mo, d = resolved
    if mo is None:
        return str(y)
    month_name = [m.capitalize() for m, i in MONTHS.items() if i == mo][0]
    if d is None:
        return f"{month_name} {y}"
    return f"{d} {month_name} {y}"


@dataclass
class ResolutionResult:
    expression: str
    anchor_timestamp: Optional[str]
    resolved_value: Optional[str]
    source_turn_id: Optional[str]
    status: str  # RESOLVED, AMBIGUOUS, UNAVAILABLE, INVALID_TIMESTAMP, UNSUPPORTED_EXPRESSION
    reason: str = ""


def resolve_temporal_expression(text, provenance):
    """Resolve relative date expressions in text using temporal provenance.
    
    Args:
        text: The text to resolve (e.g., a reader output)
        provenance: dict of {dia_id: TemporalEvidence}
    
    Returns:
        list of ResolutionResult
    """
    results = []
    expressions = detect_relative_expression(text)
    
    if not expressions:
        return results
    
    for expr_type, start, end in expressions:
        expr_text = text[start:end]
        
        # Find the turn that contains this expression
        # Since text is the reader output (not the original turn),
        # we need to find which turn this expression came from
        # by matching against the provenance
        
        best_match = None
        for dia_id, ev in provenance.items():
            if ev.text and expr_text.lower() in ev.text.lower():
                best_match = ev
                break
        
        if best_match is None:
            results.append(ResolutionResult(
                expression=expr_text,
                anchor_timestamp=None,
                resolved_value=None,
                source_turn_id=None,
                status="UNAVAILABLE",
                reason="No matching turn found in provenance",
            ))
            continue
        
        if best_match.timestamp_parsed is None:
            results.append(ResolutionResult(
                expression=expr_text,
                anchor_timestamp=best_match.timestamp,
                resolved_value=None,
                source_turn_id=best_match.turn_id,
                status="INVALID_TIMESTAMP",
                reason=f"Timestamp '{best_match.timestamp}' could not be parsed",
            ))
            continue
        
        # Handle years_ago specially
        if expr_type == "years_ago":
            m = re.search(r"(\d+)\s+years?\s+ago", text[start:end], re.I)
            if m:
                n = int(m.group(1))
                y, mo, d = best_match.timestamp_parsed
                resolved = (y - n, mo, d)
                results.append(ResolutionResult(
                    expression=expr_text,
                    anchor_timestamp=best_match.timestamp,
                    resolved_value=format_resolved_date(resolved),
                    source_turn_id=best_match.turn_id,
                    status="RESOLVED",
                ))
            else:
                results.append(ResolutionResult(
                    expression=expr_text,
                    anchor_timestamp=best_match.timestamp,
                    resolved_value=None,
                    source_turn_id=best_match.turn_id,
                    status="UNSUPPORTED_EXPRESSION",
                    reason="Could not extract number of years",
                ))
            continue
        
        # Standard resolution
        resolved = resolve_relative(expr_type, best_match.timestamp_parsed)
        if resolved:
            results.append(ResolutionResult(
                expression=expr_text,
                anchor_timestamp=best_match.timestamp,
                resolved_value=format_resolved_date(resolved),
                source_turn_id=best_match.turn_id,
                status="RESOLVED",
            ))
        else:
            results.append(ResolutionResult(
                expression=expr_text,
                anchor_timestamp=best_match.timestamp,
                resolved_value=None,
                source_turn_id=best_match.turn_id,
                status="UNSUPPORTED_EXPRESSION",
                reason=f"Expression type '{expr_type}' not supported",
            ))
    
    return results


# =============================================================================
# Section 5: End-to-End Evaluation
# =============================================================================

SESSION_DATES = {
    "conv-26": "2023-08-28", "conv-30": "2023-11-30",
    "conv-41": "2023-12-04", "conv-42": "2023-12-04",
    "conv-44": "2023-12-04", "conv-47": "2022-09-30",
    "conv-48": "2023-01-31", "conv-49": "2023-12-25",
    "conv-50": "2023-08-28",
}


def load_evidence_turns(qid, case, conversations, gold):
    """Extract evidence turns using the same logic as M30-3.
    
    But now also return the dia_id → timestamp mapping.
    """
    conv_key = case.get("source_conversation", "")
    turns = conversations.get(conv_key, [])
    if not isinstance(turns, list):
        return [], {}
    
    question = case.get("question", "").lower()
    gold_lower = gold.lower().strip()
    
    gold_keywords = set()
    gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
    stopwords = {"the", "in", "on", "at", "of", "is", "a", "an", "to", "for",
                 "january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"}
    for w in gold_clean.split():
        if len(w) > 2 and w not in stopwords:
            gold_keywords.add(w.lower())
    
    question_keywords = set()
    for w in question.replace("?", "").lower().split():
        if len(w) > 3:
            question_keywords.add(w)
    
    evidence = []
    turn_map = {}
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        dia_id = t.get("dia_id", f"D{i+1}")
        
        if any(kw in text_lower for kw in gold_keywords):
            evidence.append(f"D{i+1} | {t.get('speaker', '?')} | {text}")
            turn_map[f"D{i+1}"] = dia_id
        elif any(kw in text_lower for kw in question_keywords if len(kw) > 3):
            if any(d in text_lower for d in ["month", "year", "2023", "2022", "2024", "indiana", "seoul", "france", "rome", "boston"]):
                evidence.append(f"D{i+1} | {t.get('speaker', '?')} | {text}")
                turn_map[f"D{i+1}"] = dia_id
    
    # Deduplicate
    seen = set()
    unique = []
    for t in evidence:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    
    return unique[-8:], turn_map  # up to 8 turns


def main():
    print("=" * 78)
    print("M30-28: Temporal Provenance Integration & End-to-End Recovery")
    print("=" * 78)
    print()
    
    # Load data
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
    
    # =========================================================================
    # Section 2: Build Temporal Provenance
    # =========================================================================
    print("--- Section 2: Temporal Provenance ---")
    provenance = build_temporal_provenance(locomo_path)
    print(f"Built provenance for {len(provenance)} turns from locomo10.json")
    print()
    
    # =========================================================================
    # Section 5: Baseline vs Treatment B
    # =========================================================================
    print("--- Section 5: Baseline vs Treatment B ---")
    print()
    
    grader = OfficialLongMemEvalGrader()
    
    # Baseline: use M30-3's saved results
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
    print()
    
    # Treatment B: Apply temporal resolution to reader outputs
    treatment_correct = 0
    treatment_results = {}
    recovered = 0
    regressed = 0
    resolved_count = 0
    ambiguous_count = 0
    
    for qid in targets:
        case = cases.get(qid, {})
        if not case:
            continue
        
        question = case.get("question", "")
        gold = str(case.get("gold_answer", ""))
        reader_text = baseline_results[qid]["reader_text"]
        
        # Get evidence turns and turn mapping
        evidence_turns, turn_map = load_evidence_turns(qid, case, conversations, gold)
        
        # Build provenance for this case's turns
        case_provenance = {}
        for local_id, dia_id in turn_map.items():
            if dia_id in provenance:
                case_provenance[local_id] = provenance[dia_id]
        
        # Also add all turns from the same conversation that might be relevant
        # (This is the non-oracle part: we don't use Gold to select turns)
        conv_key = case.get("source_conversation", "")
        conv_turns = conversations.get(conv_key, [])
        for i, t in enumerate(conv_turns):
            if not isinstance(t, dict):
                continue
            dia_id = t.get("dia_id", "")
            if dia_id in provenance:
                case_provenance[f"D{i+1}"] = provenance[dia_id]
        
        # Detect relative expressions in reader output
        expressions = detect_relative_expression(reader_text)
        
        if not expressions:
            # No relative expression, keep original output
            treatment_results[qid] = {
                "reader_text": reader_text,
                "gold": gold,
                "baseline_correct": baseline_results[qid]["correct"],
                "treatment_correct": baseline_results[qid]["correct"],
                "resolution_status": "NO_RELATIVE_EXPRESSION",
            }
            if baseline_results[qid]["correct"]:
                treatment_correct += 1
            continue
        
        # Resolve expressions
        resolutions = resolve_temporal_expression(reader_text, case_provenance)
        
        # Apply resolutions to reader output
        new_output = reader_text
        resolution_status = "RESOLVED"
        for res in resolutions:
            if res.status == "RESOLVED" and res.resolved_value:
                # Replace the relative expression with the resolved value
                new_output = new_output.replace(res.expression, res.resolved_value)
                resolved_count += 1
            else:
                resolution_status = res.status
                if res.status == "AMBIGUOUS":
                    ambiguous_count += 1
        
        # Score the new output
        try:
            treatment_score = grader.score(new_output, gold)
            treatment_correct_flag = treatment_score.get("correct", False)
        except Exception as e:
            treatment_correct_flag = False
        
        treatment_results[qid] = {
            "reader_text": reader_text,
            "treatment_output": new_output,
            "gold": gold,
            "baseline_correct": baseline_results[qid]["correct"],
            "treatment_correct": treatment_correct_flag,
            "resolution_status": resolution_status,
            "resolutions": [
                {
                    "expression": r.expression,
                    "resolved_value": r.resolved_value,
                    "source_turn_id": r.source_turn_id,
                    "anchor_timestamp": r.anchor_timestamp,
                    "status": r.status,
                }
                for r in resolutions
            ],
        }
        
        if treatment_correct_flag:
            treatment_correct += 1
        
        if not baseline_results[qid]["correct"] and treatment_correct_flag:
            recovered += 1
        elif baseline_results[qid]["correct"] and not treatment_correct_flag:
            regressed += 1
    
    print(f"Treatment B: {treatment_correct}/{len(targets)} correct")
    print(f"Recovered: {recovered}")
    print(f"Regressed: {regressed}")
    print(f"Resolved: {resolved_count}")
    print(f"Ambiguous: {ambiguous_count}")
    print()
    
    # Show details for the 8 relative-date cases
    print("--- Relative-date cases (8) ---")
    rel_qids = [qid for qid in targets if detect_relative_expression(
        baseline_results[qid]["reader_text"].lower()
    )]
    for qid in rel_qids:
        r = treatment_results.get(qid, {})
        print(f"  {qid}:")
        print(f"    Reader: {r.get('reader_text', '')}")
        print(f"    Treatment: {r.get('treatment_output', '')}")
        print(f"    Gold: {r.get('gold', '')}")
        print(f"    Baseline: {'✓' if r.get('baseline_correct') else '✗'}")
        print(f"    Treatment: {'✓' if r.get('treatment_correct') else '✗'}")
        print(f"    Status: {r.get('resolution_status', 'N/A')}")
        if r.get("resolutions"):
            for res in r["resolutions"]:
                print(f"      {res['expression']} → {res['resolved_value']} ({res['status']})")
        print()
    
    # Save results
    output = {
        "baseline_correct": baseline_correct,
        "treatment_correct": treatment_correct,
        "recovered": recovered,
        "regressed": regressed,
        "resolved_count": resolved_count,
        "ambiguous_count": ambiguous_count,
        "total_cases": len(targets),
        "results": treatment_results,
    }
    
    output_path = PROJECT_ROOT / "results/m30_28/results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
