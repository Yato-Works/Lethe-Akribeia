#!/usr/bin/env python3
"""Deep analysis of M30-3 near-miss cases"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader

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

def main():
    with open("results/m30_3_evaluation_results.json", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data["results"]
    grader = OfficialLongMemEvalGrader()
    
    print("=== Near-miss relative date analysis ===\n")
    
    near_miss = [
        "lc_conv-26_73", "lc_conv-47_43", "lc_conv-48_32",
        "lc_conv-49_58", "lc_conv-49_79", "lc_conv-50_69"
    ]
    
    for qid in near_miss:
        r = next(x for x in results if x["question_id"] == qid)
        conv_key = qid.rsplit("_", 1)[0]
        ref_date = SESSION_DATES.get(conv_key)
        
        reader_text = r["reader_text"]
        gold = r["gold"]
        
        print(f"--- {qid} ---")
        print(f"  Reader: '{reader_text}'")
        print(f"  Gold:   '{gold}'")
        print(f"  Ref date: {ref_date}")
        
        # Normalize
        norm_pred = grader._normalize_relative_date(reader_text, ref_date)
        norm_gold = grader._normalize_relative_date(gold, ref_date)
        print(f"  Normalized pred: '{norm_pred}'")
        print(f"  Normalized gold: '{norm_gold}'")
        
        # Text normalization
        pred_norm = grader._normalize_text(norm_pred)
        gold_norm = grader._normalize_text(norm_gold)
        print(f"  _normalize_text pred: '{pred_norm}'")
        print(f"  _normalize_text gold: '{gold_norm}'")
        
        # Number normalization
        pred_num = grader._normalize_number(pred_norm)
        gold_num = grader._normalize_number(gold_norm)
        print(f"  _normalize_number pred: '{pred_num}'")
        print(f"  _normalize_number gold: '{gold_num}'")
        
        # Check match
        is_correct = (pred_num == gold_num) or (pred_norm == gold_norm)
        print(f"  Match: {is_correct}")
        
        # Check evidence
        evidence = r["evidence_text"]
        gold_words = [w for w in re.sub(r"[^\w\s]", " ", gold.lower()).split() if len(w) > 2]
        found = [w for w in gold_words if w.lower() in evidence.lower()]
        print(f"  Gold words in evidence: {found}")
        
        # Check what the evidence says about the date
        print(f"  Evidence (first 200 chars): {evidence[:200]}")
        print()

if __name__ == "__main__":
    main()