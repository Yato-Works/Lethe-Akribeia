#!/usr/bin/env python3
"""
M30-4: 20件の"I don't know"ケースを詳細分類

各ケースについて:
1. 証拠に含まれる相対日付表現を特定
2. goldが絶対日付かどうか確認
3. 基準日を渡せば解決できるか推定
4. 質問の意図と証拠の十分性を評価
"""
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

def get_ref_date(qid):
    """lc_conv-26_63 -> conv-26 -> SESSION_DATES"""
    # lc_conv-26_63 -> conv-26
    parts = qid.split("_")
    if len(parts) >= 2:
        conv_part = parts[0] + "_" + parts[1]  # lc_conv-26
        conv_key = conv_part.replace("lc_", "")  # conv-26
        return SESSION_DATES.get(conv_key)
    return None

def classify_evidence_date(evidence_text, ref_date, gold, question):
    """
    証拠内の日付表現を分類する
    """
    evidence_lower = evidence_text.lower()
    
    # 相対日付表現を検出
    relative_patterns = {
        "next month": r"\bnext month\b",
        "last month": r"\blast month\b",
        "this month": r"\bthis month\b",
        "last year": r"\blast year\b",
        "next year": r"\bnext year\b",
        "one year ago": r"\b(one|a)\s+year\s+ago\b",
        "X years ago": r"\b\d+\s+years?\s+ago\b",
        "yesterday": r"\byesterday\b",
        "today": r"\btoday\b",
        "tomorrow": r"\btomorrow\b",
        "last week": r"\blast week\b",
        "next week": r"\bnext week\b",
    }
    
    detected_relative = []
    for rtype, pattern in relative_patterns.items():
        if re.search(pattern, evidence_lower):
            detected_relative.append(rtype)
    
    # 絶対日付表現を検出
    absolute_patterns = [
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b20\d{2}\b",
    ]
    has_absolute = any(re.search(p, evidence_lower) for p in absolute_patterns)
    
    # goldが絶対日付か相対日付か
    gold_lower = gold.lower().strip()
    gold_is_absolute = bool(re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|20\d{2})\b", gold_lower))
    
    # 基準日を渡せば解決できるか？
    can_resolve = False
    difficulty = "impossible"
    
    if detected_relative and ref_date:
        # 相対日付表現があれば、基準日から計算可能
        # ただし、証拠内の相対日付がgoldと一致するか確認
        
        grader = OfficialLongMemEvalGrader()
        
        # 証拠テキスト全体を基準日で正規化
        norm_evidence = grader._normalize_relative_date(evidence_text, ref_date)
        
        # 正規化後にgoldが証拠に含まれるか
        gold_norm = grader._normalize_relative_date(gold, ref_date)
        gold_norm_clean = re.sub(r"[^\w\s]", " ", gold_norm.lower()).strip()
        norm_evidence_clean = re.sub(r"[^\w\s]", " ", norm_evidence.lower()).strip()
        
        gold_words = [w for w in gold_norm_clean.split() if len(w) > 2]
        evidence_words = set(norm_evidence_clean.split())
        
        if all(w in evidence_words for w in gold_words):
            can_resolve = True
            difficulty = "easy"
        elif any(w in evidence_words for w in gold_words):
            can_resolve = True
            difficulty = "medium"
    
    # 証拠に絶対日付が含まれていて、それがgoldと一致する場合
    if has_absolute and gold_is_absolute:
        gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
        gold_words = set(w for w in gold_clean.split() if len(w) > 2)
        evidence_words = set(re.sub(r"[^\w\s]", " ", evidence_lower).split())
        if gold_words & evidence_words:
            can_resolve = True
            difficulty = "easy"
    
    return {
        "detected_relative_dates": detected_relative,
        "has_absolute_date": has_absolute,
        "gold_is_absolute": gold_is_absolute,
        "can_resolve_with_ref_date": can_resolve,
        "resolution_difficulty": difficulty,
    }

def main():
    with open("results/m30_3_evaluation_results.json", encoding="utf-8") as f:
        data = json.load(f)
    
    # "I don't know" ケースのみ
    dont_know_cases = [r for r in data["results"] if "don't know" in r["reader_text"].lower()]
    
    print(f"=== M30-4: {len(dont_know_cases)}件の'I don't know'ケース分類 ===\n")
    
    can_resolve_count = 0
    cannot_resolve_count = 0
    
    for r in dont_know_cases:
        qid = r["question_id"]
        ref_date = get_ref_date(qid)
        gold = r["gold"]
        evidence_text = r["evidence_text"]
        question = r["question"]
        
        classification = classify_evidence_date(evidence_text, ref_date, gold, question)
        
        print(f"--- {qid} ---")
        print(f"  質問: {question}")
        print(f"  Gold: {gold}")
        print(f"  基準日: {ref_date}")
        print(f"  証拠内の相対日付: {classification['detected_relative_dates']}")
        print(f"  証拠内の絶対日付: {classification['has_absolute_date']}")
        print(f"  Goldが絶対日付: {classification['gold_is_absolute']}")
        print(f"  基準日で解決可能: {classification['can_resolve_with_ref_date']}")
        print(f"  解決難易度: {classification['resolution_difficulty']}")
        
        # 証拠テキストの最初の300文字
        print(f"  証拠(先頭300文字): {evidence_text[:300]}")
        print()
        
        if classification["can_resolve_with_ref_date"]:
            can_resolve_count += 1
        else:
            cannot_resolve_count += 1
    
    print(f"\n=== サマリー ===")
    print(f"  基準日で解決可能: {can_resolve_count}件")
    print(f"  基準日では解決不可: {cannot_resolve_count}件")
    
    # 保存
    with open("results/m30_4_classification.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_dont_know": len(dont_know_cases),
            "can_resolve_with_ref_date": can_resolve_count,
            "cannot_resolve": cannot_resolve_count,
            "cases": [
                {
                    "question_id": r["question_id"],
                    "question": r["question"],
                    "gold": r["gold"],
                    "reference_date": get_ref_date(r["question_id"]),
                    "evidence_text_preview": r["evidence_text"][:500],
                    "classification": classify_evidence_date(
                        r["evidence_text"],
                        get_ref_date(r["question_id"]),
                        r["gold"],
                        r["question"]
                    )
                }
                for r in dont_know_cases
            ]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"分類結果を results/m30_4_classification.json に保存しました")

if __name__ == "__main__":
    main()