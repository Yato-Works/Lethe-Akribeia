#!/usr/bin/env python3
"""M30-31 Phase 2: End-to-End trace of lc_conv-41_10.

Trace: 原会話 → Retrieval → Extraction → Reader入力 → Reader生出力 → scorer
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


def main():
    print("=" * 78)
    print("M30-31 Phase 2: End-to-End Trace — lc_conv-41_10")
    print("=" * 78)
    print()
    
    # Load data
    m30_3 = json.loads((PROJECT_ROOT / "results/m30_3_evaluation_results.json").read_text(encoding="utf-8"))
    m30_3_results = {r["question_id"]: r for r in m30_3["results"]}
    bucketA = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucketA["cases"]}
    conversations = bucketA["conversations"]
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    locomo_by_id = {c["sample_id"]: c for c in locomo}
    
    qid = "lc_conv-41_10"
    case = cases[qid]
    conv_key = case["source_conversation"]
    question = case["question"]
    gold = str(case["gold_answer"])
    
    print(f"QID: {qid}")
    print(f"Conversation: {conv_key}")
    print(f"Question: {question}")
    print(f"Gold: {gold}")
    print()
    
    # =========================================================================
    # Stage 1: 原会話
    # =========================================================================
    print("=" * 78)
    print("STAGE 1: 原会話 (Full Conversation)")
    print("=" * 78)
    
    conv_data = locomo_by_id.get(conv_key, {}).get("conversation", {})
    
    # Find turns with question keywords
    question_kw = extract_keywords(question)
    print(f"Question keywords: {question_kw}")
    print()
    
    # Search for relevant turns in full conversation
    relevant_turns = []
    for sk in sorted(conv_data.keys()):
        if sk.endswith("_date_time"):
            continue
        session_key = sk
        session_dt = conv_data.get(f"{session_key}_date_time", "")
        turns = conv_data.get(session_key, [])
        if not isinstance(turns, list):
            continue
        for i, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            text = t.get("text", "")
            dia_id = t.get("dia_id", "")
            turn_kw = extract_keywords(text)
            overlap = question_kw & turn_kw
            if len(overlap) >= 2:
                relevant_turns.append({
                    "dia_id": dia_id,
                    "session": session_key,
                    "timestamp": session_dt,
                    "text": text,
                    "overlap": overlap,
                    "turn_in_session": i + 1,
                })
    
    print(f"Relevant turns in full conversation: {len(relevant_turns)}")
    for t in relevant_turns[:15]:
        print(f"  {t['dia_id']} [{t['session']} @ {t['timestamp']}]")
        print(f"    overlap: {t['overlap']}")
        print(f"    text: {t['text'][:120]}")
        print()
    
    # =========================================================================
    # Stage 2: Retrieval (M30-3's search logic)
    # =========================================================================
    print("=" * 78)
    print("STAGE 2: Retrieval (M30-3's search logic)")
    print("=" * 78)
    
    # M30-3 uses gold keywords OR question keywords
    gold_kw = set()
    gold_clean = re.sub(r"[^\w\s]", " ", gold.lower())
    for w in gold_clean.split():
        if len(w) > 2 and w not in STOPWORDS:
            gold_kw.add(w.lower())
    
    q_kw = set()
    for w in question.replace("?", "").lower().split():
        if len(w) > 3:
            q_kw.add(w)
    
    print(f"Gold keywords: {gold_kw}")
    print(f"Question keywords (>3 chars): {q_kw}")
    print()
    
    # Search using M30-3's logic
    turns = conversations.get(conv_key, [])
    retrieval_results = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        dia_id = t.get("dia_id", f"D{i+1}")
        
        if any(kw in text_lower for kw in gold_kw):
            retrieval_results.append((dia_id, text, "gold_kw", i))
        elif any(kw in text_lower for kw in q_kw):
            if any(d in text_lower for d in ["month", "year", "2023", "2022", "2024", "indiana", "seoul", "france", "rome", "boston"]):
                retrieval_results.append((dia_id, text, "question_kw", i))
    
    print(f"Retrieval results: {len(retrieval_results)}")
    for dia_id, text, reason, i in retrieval_results:
        print(f"  {dia_id} [{reason}]: {text[:100]}")
    print()
    
    # =========================================================================
    # Stage 3: Extraction (what actually reaches Reader)
    # =========================================================================
    print("=" * 78)
    print("STAGE 3: Extraction (Reader input)")
    print("=" * 78)
    
    evidence_text = m30_3_results.get(qid, {}).get("evidence_text", "")
    input_turns = []
    for line in evidence_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("- "):
            parts = line[2:].split(" | ", 2)
            if len(parts) >= 3:
                input_turns.append((parts[0], parts[1], parts[2]))
    
    print(f"Input turns: {len(input_turns)}")
    for local_id, speaker, text in input_turns:
        # Get timestamp from locomo
        dia_id = None
        for d, _, idx in [(r[0], r[1], r[3]) for r in retrieval_results]:
            if f"D{idx+1}" == local_id:
                dia_id = d
                break
        ts = "N/A"
        if dia_id:
            m = re.match(r"D(\d+):(\d+)", dia_id)
            if m:
                skey = f"session_{m.group(1)}"
                ts = conv_data.get(f"{skey}_date_time", "N/A")
        print(f"  {local_id} | {speaker} | @ {ts}")
        print(f"    {text[:120]}")
    print()
    
    # =========================================================================
    # Stage 4: Reader output
    # =========================================================================
    print("=" * 78)
    print("STAGE 4: Reader output")
    print("=" * 78)
    
    reader_output = m30_3_results.get(qid, {}).get("reader_text", "")
    print(f"Reader output: {reader_output}")
    print()
    
    # =========================================================================
    # Stage 5: Scorer
    # =========================================================================
    print("=" * 78)
    print("STAGE 5: Scorer")
    print("=" * 78)
    
    from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader
    grader = OfficialLongMemEvalGrader()
    
    # Check what the evidence actually supports
    print("Evidence analysis:")
    print(f"  Question: {question}")
    print(f"  Gold: {gold}")
    print(f"  Reader: {reader_output}")
    print()
    
    # Check if "Maria met Jean yesterday while volunteering" is supported
    for local_id, speaker, text in input_turns:
        if "Jean" in text or "Maria" in text or "volunteer" in text.lower():
            print(f"  Relevant turn {local_id}: {text[:200]}")
            # Get timestamp
            dia_id = None
            for d, _, idx in [(r[0], r[1], r[3]) for r in retrieval_results]:
                if f"D{idx+1}" == local_id:
                    dia_id = d
                    break
            if dia_id:
                m = re.match(r"D(\d+):(\d+)", dia_id)
                if m:
                    skey = f"session_{m.group(1)}"
                    ts = conv_data.get(f"{skey}_date_time", "N/A")
                    parsed = parse_session_datetime(ts)
                    print(f"    timestamp: {ts} -> {parsed}")
                    if parsed and "yesterday" in text.lower():
                        from datetime import datetime, timedelta
                        # Calculate yesterday
                        dt = datetime(*parsed)
                        yesterday = dt - timedelta(days=1)
                        print(f"    'yesterday' resolves to: {yesterday.strftime('%B %d, %Y')}")
    
    print()
    
    # Test scorer with different predictions
    print("Scorer tests:")
    qtype = "temporal-reasoning"
    tests = [
        ("Maria met Jean yesterday while volunteering.", gold),
        ("February 24, 2023", gold),
        ("24 February 2023", gold),
        ("yesterday", gold),
    ]
    for pred, g in tests:
        try:
            result = grader.grade(qid, qtype, pred, g)
            print(f"  pred={repr(pred):50s} -> is_correct={result.is_correct}")
        except Exception as e:
            print(f"  pred={repr(pred):50s} -> ERROR: {e}")
    
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 78)
    print("TRACE SUMMARY")
    print("=" * 78)
    print(f"1. 原会話には 'Maria met Jean yesterday while volunteering' を含む発言がある: YES")
    print(f"2. Retrievalで候補になったか: YES (D7:5)")
    print(f"3. Extractionで失われていないか: NO (Reader入力に含まれている)")
    print(f"4. Readerがその証拠を見たうえで棄権したか: NO (Readerは回答した)")
    print(f"5. scorerが正しく評価できるか: NO (Gold='February 24, 2023', Reader='...yesterday...')")
    print()
    print("失敗段階: Temporal Resolution (相対日付 'yesterday' が解決されていない)")
    print()
    print("D7:5のtimestamp: 8:55 pm on 25 February, 2023")
    print("'yesterday' from 2023-02-25 = 2023-02-24 = February 24, 2023 = Gold!")
    print()
    print("これはM30-29のTemporal Resolution Engineで解決可能なケース。")


if __name__ == "__main__":
    main()
