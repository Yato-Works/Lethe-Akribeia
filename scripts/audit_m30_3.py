#!/usr/bin/env python3
"""
M30-3 失敗26件の詳細監査

各失敗ケースについて:
1. 元の会話に答えを導ける情報があるか
2. 抽出された証拠にその情報が残っているか
3. Readerの生出力は何か
4. goldと予測値の比較
5. 基準日と正規化結果
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

def load_all_cases(bucketA_path):
    with open(bucketA_path, encoding="utf-8") as f:
        data = json.load(f)
    return {c["question_id"]: c for c in data["cases"]}

def load_conversations(bucketA_path):
    with open(bucketA_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["conversations"]

def find_answer_in_conversation(case, conversations, gold, question):
    """
    元の会話全体から、goldの答えを導ける情報を探す
    """
    conv_key = case.get("source_conversation", "")
    turns = conversations.get(conv_key, [])
    if not isinstance(turns, list):
        return []
    
    gold_lower = gold.lower().strip()
    gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
    gold_words = [w for w in gold_clean.split() if len(w) > 2]
    
    # 月名や年などのキーワード
    date_keywords = set()
    for w in gold_words:
        if w in ["january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"]:
            date_keywords.add(w)
        elif w.isdigit() and len(w) == 4:
            date_keywords.add(w)
        elif len(w) > 3:
            date_keywords.add(w)
    
    # 会話全体から関連する発話を探す
    relevant_turns = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        
        # キーワードマッチ
        if any(kw in text_lower for kw in date_keywords):
            relevant_turns.append({
                "turn_id": f"D{i+1}",
                "speaker": t.get("speaker", "?"),
                "text": text[:200]
            })
    
    return relevant_turns

def check_evidence_completeness(case, evidence_text, gold, question):
    """
    抽出された証拠に、答えを導ける情報が含まれているか確認
    """
    gold_lower = gold.lower().strip()
    gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
    gold_words = [w for w in gold_clean.split() if len(w) > 2]
    
    evidence_lower = evidence_text.lower()
    
    # 各キーワードが証拠に含まれているか
    keyword_presence = {}
    for w in gold_words:
        keyword_presence[w] = w in evidence_lower
    
    # 相対日付表現があるか
    relative_date_patterns = [
        r"\bnext month\b", r"\blast month\b", r"\bthis month\b",
        r"\blast year\b", r"\bnext year\b", r"\b\d+ years? ago\b",
        r"\bone year ago\b"
    ]
    has_relative_date = any(re.search(p, evidence_lower) for p in relative_date_patterns)
    
    # 具体的な日付表現があるか
    date_patterns = [
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b20\d{2}\b"
    ]
    has_specific_date = any(re.search(p, evidence_lower) for p in date_patterns)
    
    return {
        "keyword_presence": keyword_presence,
        "has_relative_date": has_relative_date,
        "has_specific_date": has_specific_date,
        "all_keywords_present": all(keyword_presence.values()) if keyword_presence else False
    }

def main():
    # データ読み込み
    bucketA_path = Path("benchmark_results/evidence_audit/v4_bucketA_all.json")
    m25_path = Path("benchmark_results/evidence_audit/v4_intervention_temporal.json")
    
    all_cases = load_all_cases(bucketA_path)
    conversations = load_conversations(bucketA_path)
    
    with open(m25_path) as f:
        m25 = json.load(f)
    target_qids = [r["question_id"] for r in m25["rows"] if r.get("group") == "TARGET"]
    
    with open("results/m30_3_evaluation_results.json", encoding="utf-8") as f:
        eval_results = json.load(f)
    
    grader = OfficialLongMemEvalGrader()
    
    # 失敗ケースのみ
    failed_cases = [r for r in eval_results["results"] if not r["scorer_correct"]]
    
    print(f"=== M30-3 失敗{len(failed_cases)}件の詳細監査 ===\n")
    
    audit_results = []
    
    for r in failed_cases:
        qid = r["question_id"]
        case = all_cases.get(qid)
        if not case:
            continue
        
        question = case.get("question", "")
        gold = case.get("gold_answer", "")
        conv_key = case.get("source_conversation", "")
        ref_date = SESSION_DATES.get(conv_key)
        reader_text = r["reader_text"]
        evidence_text = r["evidence_text"]
        
        print(f"\n{'='*60}")
        print(f"QID: {qid}")
        print(f"質問: {question}")
        print(f"Gold: {gold}")
        print(f"Reader出力: '{reader_text}'")
        print(f"基準日: {ref_date}")
        
        # 1. 元の会話に答えを導ける情報があるか
        relevant_turns = find_answer_in_conversation(case, conversations, gold, question)
        print(f"\n[1] 元の会話内の関連発話: {len(relevant_turns)}件")
        for t in relevant_turns[:3]:
            print(f"    {t['turn_id']} | {t['speaker']}: {t['text'][:100]}")
        
        # 2. 抽出された証拠の完全性
        completeness = check_evidence_completeness(case, evidence_text, gold, question)
        print(f"\n[2] 証拠の完全性:")
        print(f"    キーワード存在: {completeness['keyword_presence']}")
        print(f"    相対日付表現: {completeness['has_relative_date']}")
        print(f"    具体日付表現: {completeness['has_specific_date']}")
        print(f"    全キーワード存在: {completeness['all_keywords_present']}")
        
        # 3. Readerの生出力の分類
        if "don't know" in reader_text.lower():
            reader_category = "abstain"
        elif any(phrase in reader_text.lower() for phrase in ["next month", "last month", "this month", "last year", "next year", "year ago"]):
            reader_category = "relative_date"
        else:
            reader_category = "other"
        print(f"\n[3] Reader出力分類: {reader_category}")
        
        # 4. scorerの比較
        if ref_date:
            norm_pred = grader._normalize_relative_date(reader_text, ref_date)
            norm_gold = grader._normalize_relative_date(gold, ref_date)
        else:
            norm_pred = reader_text
            norm_gold = gold
        
        pred_norm = grader._normalize_text(norm_pred)
        gold_norm = grader._normalize_text(norm_gold)
        pred_num = grader._normalize_number(pred_norm)
        gold_num = grader._normalize_number(gold_norm)
        
        print(f"\n[4] Scorer比較:")
        print(f"    正規化後 pred: '{norm_pred}'")
        print(f"    正規化后 gold: '{norm_gold}'")
        print(f"    _normalize_text pred: '{pred_norm}'")
        print(f"    _normalize_text gold: '{gold_norm}'")
        print(f"    _normalize_number pred: '{pred_num}'")
        print(f"    _normalize_number gold: '{gold_num}'")
        
        # 5. 原因分類
        if reader_category == "abstain":
            if not completeness["all_keywords_present"] and not completeness["has_relative_date"]:
                cause = "evidence_missing"  # 証拠に情報がない
            else:
                cause = "reader_abstain_despite_evidence"  # 証拠にあるのに拒否
        elif reader_category == "relative_date":
            if ref_date is None:
                cause = "no_reference_date"
            elif norm_pred == norm_gold:
                cause = "scorer_normalization_error"
            else:
                cause = "reader_wrong_date"
        else:
            cause = "other"
        
        print(f"\n[5] 原因分類: {cause}")
        
        audit_results.append({
            "question_id": qid,
            "question": question,
            "gold": gold,
            "reader_text": reader_text,
            "reference_date": ref_date,
            "relevant_turns_count": len(relevant_turns),
            "evidence_completeness": completeness,
            "reader_category": reader_category,
            "normalized_pred": norm_pred,
            "normalized_gold": norm_gold,
            "cause": cause
        })
    
    # サマリー
    print(f"\n\n{'='*60}")
    print("=== 原因別サマリー ===")
    cause_counts = {}
    for a in audit_results:
        cause_counts[a["cause"]] = cause_counts.get(a["cause"], 0) + 1
    for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
        print(f"  {cause}: {count}件")
    
    # 保存
    with open("results/m30_3_audit.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_failed": len(failed_cases),
            "cause_summary": cause_counts,
            "details": audit_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n監査結果を results/m30_3_audit.json に保存しました")

if __name__ == "__main__":
    main()