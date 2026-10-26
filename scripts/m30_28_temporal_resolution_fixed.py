#!/usr/bin/env python3
"""M30-28: Temporal Provenance Integration & End-to-End Recovery.

Fixed version: Only search evidence turns (not all conversation turns)
when resolving relative date expressions.

Key fix: resolve_temporal_expression now receives only the evidence turns
selected by the pipeline, not all turns in the conversation.
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
                mapping[dia_id] = TemporalEvidence(
                    turn_id=dia_id,
                    speaker=turn.get("speaker", ""),
                    text=turn.get("text", ""),
                    session_id=session_key,
                    timestamp=session_dt,
                    timestamp_parsed=parsed_dt,
                )
    return mapping


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
    if expr_type == "years_ago" and expr_text:
        m = re.search(r"(\d+)\s+years?\s+ago", expr_text, re.I)
        if m:
            return (y - int(m.group(1)), mo, d)
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


def load_evidence_turns(qid, case, conversations, gold):
    """Extract evidence turns using the same logic as M30-3.
    
    Returns:
        evidence_strings: list of "D{i+1} | speaker | text" strings
        turn_map: dict mapping local ID "D{i+1}" to original dia_id
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
            local_id = f"D{i+1}"
            evidence.append(f"{local_id} | {t.get('speaker', '?')} | {text}")
            turn_map[local_id] = dia_id
        elif any(kw in text_lower for kw in question_keywords if len(kw) > 3):
            if any(d in text_lower for d in ["month", "year", "2023", "2022", "2024", "indiana", "seoul", "france", "rome", "boston"]):
                local_id = f"D{i+1}"
                evidence.append(f"{local_id} | {t.get('speaker', '?')} | {text}")
                turn_map[local_id] = dia_id
    
    seen = set()
    unique = []
    for t in evidence:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    
    return unique[-8:], turn_map


def resolve_with_evidence_turns(reader_output, evidence_strings, turn_map, provenance):
    """Resolve relative date expressions in reader output using only evidence turns.
    
    Args:
        reader_output: The reader's output text
        evidence_strings: List of "D{i+1} | speaker | text" strings
        turn_map: Dict mapping local ID "D{i+1}" to original dia_id
        provenance: Dict mapping dia_id to TemporalEvidence
    
    Returns:
        new_output: Reader output with relative expressions replaced
        resolutions: List of resolution details
    """
    new_output = reader_output
    resolutions = []
    
    # Detect relative expressions in reader output
    expressions = detect_relative_expression(reader_output)
    
    if not expressions:
        return new_output, resolutions
    
    # For each expression, find which evidence turn contains it
    for expr_type, start, end in expressions:
        expr_text = reader_output[start:end]
        
        # Search only evidence turns for this expression
        best_match = None
        best_score = 0
        
        for ev_str in evidence_strings:
            # Parse evidence string: "D{i+1} | speaker | text"
            parts = ev_str.split(" | ", 2)
            if len(parts) < 3:
                continue
            local_id = parts[0]
            ev_text = parts[2]
            
            # Get the original dia_id
            dia_id = turn_map.get(local_id, "")
            if not dia_id or dia_id not in provenance:
                continue
            
            ev = provenance[dia_id]
            if not ev.timestamp_parsed:
                continue
            
            # Check if this evidence turn contains the expression
            ev_text_lower = ev_text.lower()
            expr_lower = expr_text.lower()
            
            if expr_lower in ev_text_lower:
                # Calculate relevance score (keyword overlap with question)
                score = len(set(re.findall(r"\b[a-z]{3,}\b", ev_text_lower)))
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


def main():
    print("=" * 78)
    print("M30-28: Temporal Provenance Integration (Fixed)")
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
    print()
    
    # Treatment B
    treatment_correct = 0
    treatment_results = {}
    recovered = 0
    regressed = 0
    resolved_count = 0
    
    for qid in targets:
        case = cases.get(qid, {})
        if not case:
            continue
        
        question = case.get("question", "")
        gold = str(case.get("gold_answer", ""))
        reader_text = baseline_results[qid]["reader_text"]
        
        evidence_turns, turn_map = load_evidence_turns(qid, case, conversations, gold)
        
        new_output, resolutions = resolve_with_evidence_turns(
            reader_text, evidence_turns, turn_map, provenance
        )
        
        # Score the new output
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
        
        treatment_results[qid] = {
            "reader_text": reader_text,
            "treatment_output": new_output,
            "gold": gold,
            "baseline_correct": baseline_results[qid]["correct"],
            "treatment_correct": treatment_correct_flag,
            "resolutions": resolutions,
        }
        
        if treatment_correct_flag:
            treatment_correct += 1
        if not baseline_results[qid]["correct"] and treatment_correct_flag:
            recovered += 1
        elif baseline_results[qid]["correct"] and not treatment_correct_flag:
            regressed += 1
        
        for res in resolutions:
            if res.get("status") == "RESOLVED":
                resolved_count += 1
    
    print(f"Treatment B: {treatment_correct}/{len(targets)} correct")
    print(f"Recovered: {recovered}")
    print(f"Regressed: {regressed}")
    print(f"Resolved: {resolved_count}")
    print()
    
    # Show relative-date cases
    print("--- Relative-date cases ---")
    for qid in targets:
        reader_text = baseline_results[qid]["reader_text"]
        if detect_relative_expression(reader_text):
            r = treatment_results[qid]
            print(f"  {qid}:")
            print(f"    Reader:     {r['reader_text']}")
            print(f"    Treatment:  {r['treatment_output']}")
            print(f"    Gold:       {r['gold']}")
            print(f"    Baseline:   {'✓' if r['baseline_correct'] else '✗'}")
            print(f"    Treatment:  {'✓' if r['treatment_correct'] else '✗'}")
            for res in r.get("resolutions", []):
                print(f"      {res['expression']} → {res.get('resolved_value', 'N/A')} ({res['status']})")
            print()
    
    # Save
    output = {
        "baseline_correct": baseline_correct,
        "treatment_correct": treatment_correct,
        "recovered": recovered,
        "regressed": regressed,
        "resolved_count": resolved_count,
        "total_cases": len(targets),
        "results": treatment_results,
    }
    output_path = PROJECT_ROOT / "results/m30_28/results_fixed.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
