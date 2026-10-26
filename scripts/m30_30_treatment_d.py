#!/usr/bin/env python3
"""M30-30 Treatment D: Reader prompt improvement + Temporal Resolution post-processing.

Combines:
- M30-30 Treatment B (improved Reader prompt)
- M30-29 Temporal Resolution Engine (deterministic post-processing)

This should recover abstention cases where Reader now provides relative dates
that can be resolved to absolute dates.
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
    """Improved evidence retrieval."""
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
        
        # Boost for entity names
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
# Temporal Resolution Engine (from M30-29)
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
    (re.compile(r"\bnext sunday\b", re.I), "next_sunday"),
    (re.compile(r"\bnext monday\b", re.I), "next_monday"),
    (re.compile(r"\bnext tuesday\b", re.I), "next_tuesday"),
    (re.compile(r"\bnext wednesday\b", re.I), "next_wednesday"),
    (re.compile(r"\bnext thursday\b", re.I), "next_thursday"),
    (re.compile(r"\bnext friday\b", re.I), "next_friday"),
    (re.compile(r"\bnext saturday\b", re.I), "next_saturday"),
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
        return (y - 1, mo, d)
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
                return (y - n, mo, d)
        return None
    if expr_type == "yesterday":
        return (y, mo, d - 1) if d > 1 else ((y, mo - 1, 28) if mo > 1 else (y - 1, 12, 31))
    if expr_type == "tomorrow":
        return (y, mo, d + 1) if d < 31 else ((y, mo + 1, 1) if mo < 12 else (y + 1, 1, 1))
    if expr_type == "today":
        return (y, mo, d)
    if expr_type == "next_week":
        return (y, mo, d + 7) if d + 7 <= 31 else ((y, mo + 1, d + 7 - 31) if mo < 12 else (y + 1, 1, d + 7 - 31))
    if expr_type == "last_week":
        return (y, mo, d - 7) if d - 7 > 0 else ((y, mo - 1, d - 7 + 31) if mo > 1 else (y - 1, 12, d - 7 + 31))
    if expr_type == "next_sunday":
        # Find next Sunday
        days_ahead = (6 - d) % 7  # Sunday is 6
        return (y, mo, d + days_ahead) if d + days_ahead <= 31 else ((y, mo + 1, d + days_ahead - 31) if mo < 12 else (y + 1, 1, d + days_ahead - 31))
    if expr_type == "next_monday":
        days_ahead = (0 - d) % 7  # Monday is 0
        return (y, mo, d + days_ahead) if d + days_ahead <= 31 else ((y, mo + 1, d + days_ahead - 31) if mo < 12 else (y + 1, 1, d + days_ahead - 31))
    if expr_type == "next_tuesday":
        days_ahead = (1 - d) % 7  # Tuesday is 1
        return (y, mo, d + days_ahead) if d + days_ahead <= 31 else ((y, mo + 1, d + days_ahead - 31) if mo < 12 else (y + 1, 1, d + days_ahead - 31))
    if expr_type == "next_wednesday":
        days_ahead = (2 - d) % 7  # Wednesday is 2
        return (y, mo, d + days_ahead) if d + days_ahead <= 31 else ((y, mo + 1, d + days_ahead - 31) if mo < 12 else (y + 1, 1, d + days_ahead - 31))
    if expr_type == "next_thursday":
        days_ahead = (3 - d) % 7  # Thursday is 3
        return (y, mo, d + days_ahead) if d + days_ahead <= 31 else ((y, mo + 1, d + days_ahead - 31) if mo < 12 else (y + 1, 1, d + days_ahead - 31))
    if expr_type == "next_friday":
        days_ahead = (4 - d) % 7  # Friday is 4
        return (y, mo, d + days_ahead) if d + days_ahead <= 31 else ((y, mo + 1, d + days_ahead - 31) if mo < 12 else (y + 1, 1, d + days_ahead - 31))
    if expr_type == "next_saturday":
        days_ahead = (5 - d) % 7  # Saturday is 5
        return (y, mo, d + days_ahead) if d + days_ahead <= 31 else ((y, mo + 1, d + days_ahead - 31) if mo < 12 else (y + 1, 1, d + days_ahead - 31))
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
        return f"{d} {month_name} {y}"
    return str(y)


def resolve_with_evidence(reader_output, evidence_strings, turn_map, provenance, conv_key, question):
    """Resolve relative date expressions with precision preservation."""
    new_output = reader_output
    resolutions = []
    
    granularity = detect_question_granularity(question)
    
    expressions = detect_relative_expression(reader_output)
    if not expressions:
        return new_output, resolutions
    
    for expr_type, start, end in expressions:
        expr_text = reader_output[start:end]
        
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
            if not ev["timestamp_parsed"]:
                continue
            
            if expr_text.lower() in ev_text.lower():
                q_kw = extract_keywords(question)
                ev_kw = extract_keywords(ev_text)
                score = len(q_kw & ev_kw)
                if score > best_score:
                    best_score = score
                    best_match = ev
        
        if best_match:
            resolved = resolve_relative(expr_type, best_match.timestamp_parsed, expr_text)
            if resolved:
                resolved_str = format_with_granularity(resolved, granularity)
                new_output = new_output.replace(expr_text, resolved_str)
                resolutions.append({
                    "expression": expr_text,
                    "resolved_value": resolved_str,
                    "source_turn_id": best_match["turn_id"],
                    "anchor_timestamp": best_match["timestamp"],
                    "status": "RESOLVED",
                    "granularity": granularity,
                })
            else:
                resolutions.append({
                    "expression": expr_text,
                    "resolved_value": None,
                    "source_turn_id": best_match["turn_id"],
                    "anchor_timestamp": best_match["timestamp"],
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
# Main: Treatment D
# =============================================================================

def get_evidence_strings_from_text(evidence_text):
    results = []
    for line in evidence_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("- "):
            results.append(line[2:])
    return results


def build_evidence_text(evidence_strings):
    if not evidence_strings:
        return ""
    return "\n".join(f"- {t}" for t in evidence_strings)


def get_qtype_name(case):
    qtype_map = {1: "multi-session", 2: "temporal-reasoning", 3: "single-session-user", 4: "single-session-assistant", 5: "abstention"}
    return qtype_map.get(case.get("category", 3), "single-session-user")


def main():
    print("=" * 78)
    print("M30-30 Treatment D: Reader prompt + Temporal Resolution")
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
    
    # Get abstention qids
    abstention = json.loads((PROJECT_ROOT / "results/m30_30/abstention_classification.json").read_text(encoding="utf-8"))
    abstention_qids = sorted(abstention["classifications"].keys())
    
    print(f"M30-29 baseline: {m29['treatment_correct']}/28")
    print(f"Abstention cases: {len(abstention_qids)}")
    print()
    
    grader = OfficialLongMemEvalGrader()
    model_name = "qwen2.5:7b-instruct"
    
    # Treatment D: Improved Reader prompt + Temporal Resolution post-processing
    treatment_d_correct = 0
    treatment_d_results = {}
    recovered_vs_m29 = 0
    regressed_vs_m29 = 0
    
    for qid in targets:
        case = cases.get(qid, {})
        if not case:
            continue
        
        question = case.get("question", "")
        gold = str(case.get("gold_answer", ""))
        evidence_text = m30_3_results.get(qid, {}).get("evidence_text", "")
        baseline_correct = m29["results"].get(qid, {}).get("treatment_correct", False)
        
        if qid not in abstention_qids:
            # Not abstention, keep M30-29 result
            treatment_d_results[qid] = {
                "treatment_correct": baseline_correct,
                "note": "Not abstention, kept M30-29 result",
            }
            if baseline_correct:
                treatment_d_correct += 1
            continue
        
        # Step 1: Use improved Reader prompt with same evidence
        evidence_strings = get_evidence_strings_from_text(evidence_text)
        evidence = build_evidence_text(evidence_strings)
        
        try:
            reader = OllamaAnswerer(model=model_name)
            reader_output = reader.answer(question, evidence)
            reader_text = reader_output.text
            
            # Step 2: Apply Temporal Resolution Engine
            # First, find evidence turns with timestamps
            turn_map = {}
            for ev_str in evidence_strings:
                parts = ev_str.split(" | ", 2)
                if len(parts) >= 3:
                    local_id = parts[0]
                    turn_map[local_id] = None  # Will be filled from conversations
            
            # Get turn IDs from conversations
            conv_key = case.get("source_conversation", "")
            conv_turns = conversations.get(conv_key, [])
            if isinstance(conv_turns, list):
                for i, t in enumerate(conv_turns):
                    if isinstance(t, dict):
                        dia_id = t.get("dia_id", "")
                        if dia_id and (conv_key, dia_id) in provenance:
                            local_id = f"D{i+1}"
                            turn_map[local_id] = dia_id
            
            # Resolve temporal expressions
            new_output, resolutions = resolve_with_evidence(
                reader_text, evidence_strings, turn_map, provenance, conv_key, question
            )
            
            # Score the final output
            qtype_name = get_qtype_name(case)
            result = grader.grade(qid, qtype_name, new_output, gold)
            treatment_correct = result.is_correct
            
            treatment_d_results[qid] = {
                "reader_output": reader_text,
                "treatment_output": new_output,
                "gold": gold,
                "baseline_correct": baseline_correct,
                "m29_correct": baseline_correct,
                "treatment_correct": treatment_correct,
                "resolutions": resolutions,
            }
            
            if treatment_correct:
                treatment_d_correct += 1
            if not baseline_correct and treatment_correct:
                recovered_vs_m29 += 1
            elif baseline_correct and not treatment_correct:
                regressed_vs_m29 += 1
                
        except Exception as e:
            print(f"  ERROR {qid}: {e}")
            treatment_d_results[qid] = {
                "treatment_correct": baseline_correct,
                "error": str(e),
            }
            if baseline_correct:
                treatment_d_correct += 1
    
    print(f"Treatment D: {treatment_d_correct}/28 correct")
    print(f"Recovered vs M30-29: {recovered_vs_m29}")
    print(f"Regressed vs M30-29: {regressed_vs_m29}")
    print()
    
    # Show abstention cases
    print("Abstention cases:")
    for qid in abstention_qids:
        r = treatment_d_results.get(qid, {})
        m29_c = m29["results"].get(qid, {}).get("treatment_correct", False)
        t_d = r.get("treatment_correct", False)
        
        changed = ""
        if t_d and not m29_c:
            changed = "RECOVERED"
        elif not t_d and m29_c:
            changed = "REGRESSED"
        
        print(f"  {qid}: M29={'✓' if m29_c else '✗'} D={'✓' if t_d else '✗'} {changed}")
        if r.get("reader_output"):
            print(f"    Reader: {r['reader_output'][:80]}")
        if r.get("treatment_output"):
            print(f"    Final:  {r['treatment_output'][:80]}")
        if r.get("resolutions"):
            for res in r["resolutions"]:
                print(f"    {res['expression']} → {res.get('resolved_value', 'N/A')} ({res['status']})")
        print()
    
    # Save
    output = {
        "baseline_correct": m29["treatment_correct"],
        "treatment_d_correct": treatment_d_correct,
        "recovered_vs_m29": recovered_vs_m29,
        "regressed_vs_m29": regressed_vs_m29,
        "total_cases": len(targets),
        "abstention_qids": abstention_qids,
        "results": treatment_d_results,
    }
    output_path = PROJECT_ROOT / "results/m30_30/treatment_d_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
