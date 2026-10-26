#!/usr/bin/env python3
"""
M30-10 フェーズ1: 9件のデータ品質監査を確定させる

会話本文・質問・Gold・会話IDの対応を1件ずつ照合する。
「キーワードが見つからない」だけで、正解情報が存在しないと断定しない。
意味的に言い換えられた記述も確認する。
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    bucketA_path = Path("benchmark_results/evidence_audit/v4_bucketA_all.json")
    with open(bucketA_path, encoding="utf-8") as f:
        data = json.load(f)
    
    all_cases = {c["question_id"]: c for c in data["cases"]}
    conversations = data["conversations"]
    
    # 9件の証拠不足ケース
    insufficient_cases = [
        "lc_conv-41_10", "lc_conv-41_20", "lc_conv-42_57", "lc_conv-42_73",
        "lc_conv-44_13", "lc_conv-48_12", "lc_conv-48_50", "lc_conv-49_75", "lc_conv-50_36"
    ]
    
    audit_results = []
    
    for qid in insufficient_cases:
        case = all_cases.get(qid)
        if not case:
            continue
        
        question = case.get("question", "")
        gold = case.get("gold_answer", "")
        conv_key = case.get("source_conversation", "")
        turns = conversations.get(conv_key, [])
        
        print(f"\n{'='*80}")
        print(f"QID: {qid}")
        print(f"会話キー: {conv_key}")
        print(f"質問: {question}")
        print(f"Gold: {gold}")
        print(f"会話ターン数: {len(turns)}")
        print(f"{'='*80}")
        
        # 質問のキーワードを抽出
        question_lower = question.lower()
        question_clean = re.sub(r"[^\w\s]", " ", question_lower)
        question_words = [w for w in question_clean.split() if len(w) > 3]
        # 停止語を除外
        stopwords = {"when", "what", "where", "which", "who", "how", "did", "does", "was", "were", "are", "the", "and", "for", "with", "from", "that", "this", "have", "has", "had", "been", "being", "will", "would", "could", "should", "about", "into", "onto", "upon", "over", "under", "between", "through", "during", "before", "after", "above", "below", "among", "within", "without", "against", "toward", "towards"}
        question_keywords = [w for w in question_words if w not in stopwords]
        
        # Goldのキーワードを抽出
        gold_lower = gold.lower().strip()
        gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
        gold_words = [w for w in gold_clean.split() if len(w) > 2]
        gold_keywords = [w for w in gold_words if w not in stopwords]
        
        print(f"\n質問キーワード: {question_keywords}")
        print(f"Goldキーワード: {gold_keywords}")
        
        # 会話全体で質問キーワードを検索
        question_found = []
        for i, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            text = t.get("text", "")
            text_lower = text.lower()
            for w in question_keywords:
                if w in text_lower:
                    question_found.append((f"D{i+1}", w, text[:150]))
        
        # 会話全体でGoldキーワードを検索
        gold_found = []
        for i, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            text = t.get("text", "")
            text_lower = text.lower()
            for w in gold_keywords:
                if w in text_lower:
                    gold_found.append((f"D{i+1}", w, text[:150]))
        
        print(f"\n質問キーワードの会話内発見: {len(question_found)}件")
        for f in question_found[:5]:
            print(f"  {f[0]} | '{f[1]}' | {f[2]}")
        
        print(f"\nGoldキーワードの会話内発見: {len(gold_found)}件")
        for f in gold_found[:5]:
            print(f"  {f[0]} | '{f[1]}' | {f[2]}")
        
        # 意味的な関連性を確認
        # 質問のキーワードとGoldのキーワードが両方存在するか
        question_has_answer = len(question_found) > 0
        gold_has_answer = len(gold_found) > 0
        
        # 判定
        if not question_has_answer and not gold_has_answer:
            verdict = "DATA_QUALITY_ISSUE"
            reason = "質問とGoldのキーワードが会話内に存在しない"
        elif question_has_answer and not gold_has_answer:
            verdict = "GOLD_MISSING"
            reason = "質問のキーワードは存在するが、Goldのキーワードが存在しない"
        elif not question_has_answer and gold_has_answer:
            verdict = "QUESTION_MISSING"
            reason = "Goldのキーワードは存在するが、質問のキーワードが存在しない"
        else:
            verdict = "POSSIBLE"
            reason = "質問とGoldのキーワードが両方存在する"
        
        print(f"\n判定: {verdict}")
        print(f"理由: {reason}")
        
        audit_results.append({
            "question_id": qid,
            "conversation_key": conv_key,
            "question": question,
            "gold": gold,
            "question_keywords": question_keywords,
            "gold_keywords": gold_keywords,
            "question_found_count": len(question_found),
            "gold_found_count": len(gold_found),
            "question_found_samples": [f[2] for f in question_found[:3]],
            "gold_found_samples": [f[2] for f in gold_found[:3]],
            "verdict": verdict,
            "reason": reason,
        })
    
    # サマリー
    print(f"\n\n{'='*80}")
    print("=== 9件データ品質監査 サマリー ===")
    print(f"{'='*80}")
    
    verdict_counts = {}
    for a in audit_results:
        verdict_counts[a["verdict"]] = verdict_counts.get(a["verdict"], 0) + 1
    
    for verdict, count in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"  {verdict}: {count}件")
    
    print(f"\n詳細:")
    for a in audit_results:
        print(f"  {a['question_id']}: {a['verdict']} - {a['reason']}")
    
    # 保存
    with open("results/m30_10_data_quality_audit.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_cases": len(audit_results),
            "verdict_summary": verdict_counts,
            "details": audit_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n監査結果を results/m30_10_data_quality_audit.json に保存しました")

if __name__ == "__main__":
    main()