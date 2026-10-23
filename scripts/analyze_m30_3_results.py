#!/usr/bin/env python3
"""Analyze M30-3 results in detail"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader

def main():
    with open("results/m30_3_evaluation_results.json", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data["results"]
    grader = OfficialLongMemEvalGrader()
    
    # Categorize
    dont_know = []
    near_miss_relative = []
    other_wrong = []
    
    for r in results:
        if r["scorer_correct"]:
            continue
        reader_text = r["reader_text"]
        gold = r["gold"]
        qid = r["question_id"]
        
        if "don't know" in reader_text.lower():
            dont_know.append(r)
        elif any(phrase in reader_text.lower() for phrase in ["next month", "last month", "this month", "last year", "next year", "year ago", "years ago"]):
            near_miss_relative.append(r)
        else:
            other_wrong.append(r)
    
    print("=== M30-3 Detailed Analysis ===")
    print(f"Total: 28, Correct: 2, Wrong: 26")
    print(f"  'I don't know': {len(dont_know)}")
    print(f"  Near-miss relative dates: {len(near_miss_relative)}")
    print(f"  Other wrong: {len(other_wrong)}")
    
    print("\n--- Near-miss relative dates ---")
    for r in near_miss_relative:
        print(f"  {r['question_id']}: reader='{r['reader_text']}' | gold='{r['gold']}'")
        # Check what the scorer does
        ref_date = None
        conv_key = r["question_id"].split("_")[0] + "_" + r["question_id"].split("_")[1]
        # Get ref_date from the evidence mapping
        # Actually, let's just check the normalization
        reader_text = r["reader_text"]
        gold = r["gold"]
        
        # Simulate what the scorer does
        # We need the ref_date... let's get it from the SESSION_DATES
        SESSION_DATES = {
            "conv-26": "2023-08-28",
            "conv-30": "2023-11-30",
            "conv-41": "2023-12-04",
            "conv-42": "2023-12-04",
            "conv-44": "2023-12-04",
            "conv-47": "2022-09-30",
            "conv-48": "2023-01-31",
            "conv-49": "2023-12-25",
            "conv-50": "2023-08-28",
        }
        conv_key = r["question_id"].rsplit("_", 1)[0]
        ref_date = SESSION_DATES.get(conv_key)
        
        if ref_date:
            norm_pred = grader._normalize_relative_date(reader_text, ref_date)
            norm_gold = grader._normalize_relative_date(gold, ref_date)
            print(f"    ref_date={ref_date} | norm_pred='{norm_pred}' | norm_gold='{norm_gold}'")
            
            # Check _normalize_text
            pred_norm = grader._normalize_text(norm_pred)
            gold_norm = grader._normalize_text(norm_gold)
            print(f"    _normalize_text: pred='{pred_norm}' | gold='{gold_norm}'")
            
            pred_num = grader._normalize_number(pred_norm)
            gold_num = grader._normalize_number(gold_norm)
            print(f"    _normalize_number: pred='{pred_num}' | gold='{gold_num}'")
    
    print("\n--- 'I don't know' cases ---")
    for r in dont_know:
        print(f"  {r['question_id']}: gold='{r['gold']}'")
        # Check if evidence contains the answer
        evidence = r["evidence_text"]
        gold = r["gold"].lower()
        # Simple check: does any word from gold appear in evidence?
        gold_words = [w for w in re.sub(r"[^\w\s]", " ", gold).split() if len(w) > 2]
        found = [w for w in gold_words if w.lower() in evidence.lower()]
        print(f"    gold words: {gold_words}")
        print(f"    found in evidence: {found}")
        if not found:
            print(f"    → Evidence likely MISSING answer keywords")
        else:
            print(f"    → Evidence contains keywords but reader still said 'I don't know'")

if __name__ == "__main__":
    main()