#!/usr/bin/env python3
"""M30-29 Phase 1: Trace the 3 failure cases from M30-28 logs."""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}

def parse_session_datetime(s):
    if not s:
        return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", s)
    if m and m.group(2).lower() in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None

# Load M30-28 results
m28 = json.loads((PROJECT_ROOT / "results/m30_28/results_fixed.json").read_text(encoding="utf-8"))

# Load locomo10
locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
locomo_by_id = {c["sample_id"]: c for c in locomo}

# Load bucketA
bucketA = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
cases = {c["question_id"]: c for c in bucketA["cases"]}
conversations = bucketA["conversations"]

# M30-3 results
m30_3 = json.loads((PROJECT_ROOT / "results/m30_3_evaluation_results.json").read_text(encoding="utf-8"))
m30_3_results = {r["question_id"]: r for r in m30_3["results"]}

targets = ["lc_conv-26_73", "lc_conv-47_43", "lc_conv-49_58"]

for qid in targets:
    case = cases[qid]
    conv_key = case["source_conversation"]
    question = case["question"]
    gold = str(case["gold_answer"])
    
    print("=" * 78)
    print(f"{qid}")
    print("=" * 78)
    print(f"Question: {question}")
    print(f"Gold: {gold}")
    print()
    
    # M30-28 treatment result
    m28_r = m28["results"].get(qid, {})
    print(f"M30-28 Treatment output: {m28_r.get('treatment_output', 'N/A')}")
    print(f"M30-28 Baseline correct: {m28_r.get('baseline_correct', 'N/A')}")
    print(f"M30-28 Treatment correct: {m28_r.get('treatment_correct', 'N/A')}")
    print()
    
    # Show resolutions
    resolutions = m28_r.get("resolutions", [])
    print(f"M30-28 Resolutions ({len(resolutions)}):")
    for res in resolutions:
        print(f"  {res.get('expression', 'N/A')} -> {res.get('resolved_value', 'N/A')} ({res.get('status', 'N/A')})")
        print(f"    source_turn_id: {res.get('source_turn_id', 'N/A')}")
        print(f"    anchor_timestamp: {res.get('anchor_timestamp', 'N/A')}")
    print()
    
    # Find the evidence turns selected by M30-28's load_evidence_turns
    # (replicate the logic)
    turns = conversations.get(conv_key, [])
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
    
    evidence_turns = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        dia_id = t.get("dia_id", f"D{i+1}")
        
        if any(kw in text_lower for kw in gold_keywords):
            evidence_turns.append((dia_id, text, "gold_keyword"))
        elif any(kw in text_lower for kw in question_keywords if len(kw) > 3):
            if any(d in text_lower for d in ["month", "year", "2023", "2022", "2024", "indiana", "seoul", "france", "rome", "boston"]):
                evidence_turns.append((dia_id, text, "question_keyword"))
    
    print(f"Evidence turns selected ({len(evidence_turns)}):")
    for dia_id, text, reason in evidence_turns[:10]:
        # Get session datetime from locomo
        conv_data = locomo_by_id.get(conv_key, {}).get("conversation", {})
        m = re.match(r"D(\d+):(\d+)", dia_id)
        session_dt = "N/A"
        if m:
            skey = f"session_{m.group(1)}"
            session_dt = conv_data.get(f"{skey}_date_time", "N/A")
        print(f"  {dia_id} [{reason}] @ {session_dt}")
        print(f"    text: {text[:120]}...")
    print()
    
    # For lc_conv-26_73: check if D15:12 is in evidence
    if qid == "lc_conv-26_73":
        print("--- lc_conv-26_73 specific analysis ---")
        # Check if D15:12 is in evidence
        d15_12_in_evidence = any(d == "D15:12" for d, _, _ in evidence_turns)
        print(f"D15:12 in evidence: {d15_12_in_evidence}")
        
        # Check what D15:12's timestamp is
        conv_data = locomo_by_id.get(conv_key, {}).get("conversation", {})
        skey = "session_15"
        s15_dt = conv_data.get(f"{skey}_date_time", "N/A")
        print(f"D15:12 session_15 datetime: {s15_dt}")
        
        # Check if D15:12 text contains "last month"
        s15_turns = conv_data.get(skey, [])
        if s15_turns:
            for t in s15_turns:
                if isinstance(t, dict) and t.get("dia_id") == "D15:12":
                    print(f"D15:12 text: {t.get('text', '')[:200]}")
                    break
        
        # Check what the M30-28 treatment selected
        if resolutions:
            src = resolutions[0].get("source_turn_id", "N/A")
            print(f"M30-28 selected turn: {src}")
            if src != "N/A":
                m2 = re.match(r"D(\d+):(\d+)", src)
                if m:
                    skey2 = f"session_{m.group(1)}"
                    s_dt = conv_data.get(f"{skey2}_date_time", "N/A")
                    print(f"  timestamp: {s_dt}")
        print()
    
    # For lc_conv-47_43
    if qid == "lc_conv-47_43":
        print("--- lc_conv-47_43 specific analysis ---")
        # Check if D20:17 is in evidence
        d20_17_in_evidence = any(d == "D20:17" for d, _, _ in evidence_turns)
        print(f"D20:17 in evidence: {d20_17_in_evidence}")
        
        conv_data = locomo_by_id.get(conv_key, {}).get("conversation", {})
        skey = "session_20"
        s20_dt = conv_data.get(f"{skey}_date_time", "N/A")
        print(f"D20:17 session_20 datetime: {s20_dt}")
        
        s20_turns = conv_data.get(skey, [])
        if s20_turns:
            for t in s20_turns:
                if isinstance(t, dict) and t.get("dia_id") == "D20:17":
                    print(f"D20:17 text: {t.get('text', '')[:200]}")
                    break
        
        if resolutions:
            src = resolutions[0].get("source_turn_id", "N/A")
            print(f"M30-28 selected turn: {src}")
            if src != "N/A":
                m2 = re.match(r"D(\d+):(\d+)", src)
                if m:
                    skey2 = f"session_{m.group(1)}"
                    s_dt = conv_data.get(f"{skey2}_date_time", "N/A")
                    print(f"  timestamp: {s_dt}")
        print()
    
    # For lc_conv-49_58
    if qid == "lc_conv-49_58":
        print("--- lc_conv-49_58 specific analysis ---")
        # Check if D12:2 is in evidence
        d12_2_in_evidence = any(d == "D12:2" for d, _, _ in evidence_turns)
        print(f"D12:2 in evidence: {d12_2_in_evidence}")
        
        conv_data = locomo_by_id.get(conv_key, {}).get("conversation", {})
        skey = "session_12"
        s12_dt = conv_data.get(f"{skey}_date_time", "N/A")
        print(f"D12:2 session_12 datetime: {s12_dt}")
        
        s12_turns = conv_data.get(skey, [])
        if s12_turns:
            for t in s12_turns:
                if isinstance(t, dict) and t.get("dia_id") == "D12:2":
                    print(f"D12:2 text: {t.get('text', '')[:200]}")
                    break
        
        if resolutions:
            src = resolutions[0].get("source_turn_id", "N/A")
            print(f"M30-28 selected turn: {src}")
            if src != "N/A":
                m2 = re.match(r"D(\d+):(\d+)", src)
                if m:
                    skey2 = f"session_{m.group(1)}"
                    s_dt = conv_data.get(f"{skey2}_date_time", "N/A")
                    print(f"  timestamp: {s_dt}")
        print()
