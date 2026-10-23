#!/usr/bin/env python3
"""
M30-8: 9件の元会話を個別監査

「会話全体のキーワード検索でGoldに対応する情報を発見できなかった」9件について、
実際に会話を読んで、答えが推論できるかどうか確認する。
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
        print(f"質問: {question}")
        print(f"Gold: {gold}")
        print(f"会話キー: {conv_key}")
        print(f"会話ターン数: {len(turns)}")
        print(f"{'='*80}")
        
        # 会話全体を表示（最大50ターン）
        for i, t in enumerate(turns[:50]):
            if not isinstance(t, dict):
                continue
            speaker = t.get("speaker", "?")
            text = t.get("text", "")
            print(f"  D{i+1} | {speaker} | {text[:150]}")
        
        if len(turns) > 50:
            print(f"  ... (残り{len(turns)-50}ターン)")
        
        # goldのキーワードを抽出
        gold_lower = gold.lower().strip()
        gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
        gold_words = [w for w in gold_clean.split() if len(w) > 2]
        
        # 会話全体でキーワード検索
        found = []
        for i, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            text = t.get("text", "")
            text_lower = text.lower()
            for w in gold_words:
                if w in text_lower:
                    found.append((f"D{i+1}", w, text[:100]))
        
        print(f"\n  Goldキーワード: {gold_words}")
        print(f"  会話全体での発見: {len(found)}件")
        for f in found:
            print(f"    {f[0]} | '{f[1]}' | {f[2]}")
        
        # 関連する表現を探す
        related_patterns = []
        if "2022" in gold or "2023" in gold or "2024" in gold:
            related_patterns.extend([r"\b20\d{2}\b", r"\byear\b", r"\bago\b", r"\blast\b", r"\bnext\b", r"\bthis\b"])
        if any(m in gold_lower for m in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
            related_patterns.extend([r"\bmonth\b", r"\bjanuary\b", r"\bfebruary\b", r"\bmarch\b", r"\bapril\b", r"\bmay\b", r"\bjune\b", r"\bjuly\b", r"\baugust\b", r"\bseptember\b", r"\boctober\b", r"\bnovember\b", r"\bdecember\b"])
        
        related_found = []
        for i, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            text = t.get("text", "")
            text_lower = text.lower()
            for p in related_patterns:
                if re.search(p, text_lower):
                    related_found.append((f"D{i+1}", p, text[:100]))
                    break
        
        print(f"  関連表現の発見: {len(related_found)}件")
        for f in related_found[:5]:
            print(f"    {f[0]} | {f[1]} | {f[2]}")

if __name__ == "__main__":
    main()