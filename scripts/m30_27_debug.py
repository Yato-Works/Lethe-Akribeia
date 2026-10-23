#!/usr/bin/env python3
"""Debug M30-27 Phase 2-3 Gold matching logic."""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
M30_3 = PROJECT_ROOT / "results/m30_3_evaluation_results.json"
CACHE = PROJECT_ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}

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
    if not ts_str:
        return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", ts_str)
    if m and m.group(2).lower() in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None

def resolve_relative(expr_type, base_date):
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
        m = re.search(r"(\d+)\s+years?\s+ago", expr_type, re.I)
        if m:
            return (y - int(m.group(1)), None, None)
        return None
    return None

def extract_timestamps(context_text):
    results = []
    pattern = re.compile(r"\[(D\d+:\d+)\s+on\s+([^\]]+)\]")
    for m in pattern.finditer(context_text):
        parsed = parse_timestamp(m.group(2))
        if parsed:
            results.append((m.group(1), m.group(2), parsed))
    return results

def find_relevant_turn(question, context_text, timestamps):
    if not timestamps:
        return None
    stop = {"the","and","for","are","was","were","did","does","has","have","had",
            "will","would","could","should","can","may","might","must","this",
            "that","these","those","when","where","what","which","who","whom",
            "whose","why","how","not","all","any","both","each","few","more",
            "most","other","some","such","only","own","same","than","too","very",
            "just","also","now","then","here","there","once","again","further",
            "else","ever","never","always","often","sometimes","usually","already",
            "yet","still","even","back","away","down","off","over","under","above",
            "below","between","through","during","before","after","about","into",
            "onto","upon","out","up","far","near","inside","outside","behind",
            "beyond","across","around","along","among","against","within","without",
            "toward","towards","per","via","like","unlike","despite","except",
            "regarding","concerning","according","due","owing","thanks","regardless",
            "irrespective","notwithstanding","whether","either","neither","both",
            "whatever","whenever","wherever","however","whichever","whoever",
            "whomever","whosever","whatsoever","wheresoever","whensoever",
            "howsoever","whosoever","whomsoever","whosesoever"}
    q_words = set(re.findall(r"\b[a-z]{3,}\b", question.lower())) - stop
    best_score, best_ts = 0, None
    for dia_id, ts_str, parsed in timestamps:
        pat = re.compile(re.escape(f"[{dia_id} on {ts_str}]") + r"\s*[^:]+:\s*([^\n]+)")
        m = pat.search(context_text)
        if m:
            t_words = set(re.findall(r"\b[a-z]{3,}\b", m.group(1).lower()))
            score = len(q_words & t_words)
            if score > best_score:
                best_score, best_ts = score, (dia_id, ts_str, parsed)
    return best_ts

# Load
m30_3 = json.loads(M30_3.read_text(encoding="utf-8"))
m30_3_results = {r["question_id"]: r for r in m30_3["results"]}

cache = {}
with open(CACHE, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        cache[row.get("qid", "")] = row.get("context", "")

def to_cache_qid(qid):
    m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
    if m:
        return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}"
    return qid

# Find the 8 cases
rel_cases = []
for qid, r in m30_3_results.items():
    out = r.get("reader_text", "").lower()
    for pattern, expr_type in RELATIVE_PATTERNS:
        if pattern.search(out):
            rel_cases.append((qid, r.get("reader_text", ""), r.get("gold", ""), expr_type))
            break

print(f"Relative-date cases: {len(rel_cases)}")
print()

for qid, reader_output, gold, expr_type in rel_cases:
    question = m30_3_results.get(qid, {}).get("question", "")
    context = cache.get(to_cache_qid(qid), "")

    if not context:
        print(f"{qid}: NO CONTEXT")
        continue

    timestamps = extract_timestamps(context)
    relevant = find_relevant_turn(question, context, timestamps)

    if not relevant:
        print(f"{qid}: NO RELEVANT TURN")
        continue

    dia_id, ts_str, base_date = relevant
    result = resolve_relative(expr_type, base_date)

    if not result:
        print(f"{qid}: CANNOT RESOLVE")
        continue

    ry, rmo, rd = result

    # Gold matching
    gold_str = str(gold).lower()
    match = False

    if str(ry) not in gold_str:
        match = False
    elif rmo is None:
        match = True
    else:
        month_names = [m for m, i in MONTHS.items() if i == rmo]
        if month_names:
            match = month_names[0] in gold_str
        else:
            match = False

    baseline = m30_3_results.get(qid, {}).get("scorer_correct", False)

    print(f"{qid}:")
    print(f"  question: {question}")
    print(f"  reader: {reader_output}")
    print(f"  gold: {repr(gold)} -> gold_str: {repr(gold_str)}")
    print(f"  resolved: ry={ry}, rmo={rmo}")
    print(f"  base_date: {base_date} (via {dia_id} @ {ts_str})")
    if rmo:
        mn = [m for m, i in MONTHS.items() if i == rmo]
        print(f"  month_names: {mn}, in gold: {mn[0] in gold_str if mn else 'N/A'}")
    print(f"  match: {match}")
    print(f"  baseline_correct: {baseline}")
    print()
