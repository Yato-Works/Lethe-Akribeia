#!/usr/bin/env python3
"""
M30-10 Phase 2: 9件の意味的証拠・Gold出所監査

Goldの文字列検索だけでなく、同義表現、言い換え、相対日付、前後の発言、
間接的な記述も調べる。Goldを裏付ける具体的な会話の引用箇所を記録する。
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def build_semantic_patterns(gold, question):
    """Goldと質問から意味的な検索パターンを構築"""
    patterns = {
        "exact": [],
        "synonyms": [],
        "relative_dates": [],
        "partial": [],
        "contextual": [],
    }
    
    gold_lower = gold.lower().strip()
    question_lower = question.lower().strip()
    
    # 月名の略称と同義語
    month_map = {
        "january": ["jan", "january", "1月"],
        "february": ["feb", "february", "2月"],
        "march": ["mar", "march", "3月"],
        "april": ["apr", "april", "4月"],
        "may": ["may", "5月"],
        "june": ["jun", "june", "6月"],
        "july": ["jul", "july", "7月"],
        "august": ["aug", "august", "8月"],
        "september": ["sep", "sept", "september", "9月"],
        "october": ["oct", "october", "10月"],
        "november": ["nov", "november", "11月"],
        "december": ["dec", "december", "12月"],
    }
    
    # Goldから月名を抽出
    gold_months = []
    for month, aliases in month_map.items():
        for alias in aliases:
            if alias in gold_lower:
                gold_months.append(month)
                patterns["exact"].extend(aliases)
                break
    
    # Goldから年を抽出
    year_match = re.search(r"\b(20\d{2})\b", gold)
    if year_match:
        year = year_match.group(1)
        patterns["exact"].append(year)
        # 相対年表現
        patterns["relative_dates"].extend([
            f"{year}",
            f"in {year}",
            f"during {year}",
        ])
    
    # Goldから日を抽出
    day_match = re.search(r"\b(\d{1,2})\b", gold)
    if day_match:
        day = day_match.group(1)
        patterns["exact"].append(day)
        # 日付形式
        if gold_months:
            month_num = list(month_map.keys()).index(gold_months[0]) + 1
            day_int = int(day)
            patterns["exact"].extend([
                f"{month_num}/{day}",
                f"{month_num:02d}/{day}",
                f"{month_num}/{day_int:02d}",
                f"{month_num:02d}/{day_int:02d}",
            ])
    
    # 質問からキーワードを抽出
    question_clean = re.sub(r"[^\w\s]", " ", question_lower)
    question_words = [w for w in question_clean.split() if len(w) > 3]
    stopwords = {"when", "what", "where", "which", "who", "how", "did", "does", "was", "were", "are", "the", "and", "for", "with", "from", "that", "this", "have", "has", "had", "been", "being", "will", "would", "could", "should", "about", "into", "onto", "upon", "over", "under", "between", "through", "during", "before", "after", "above", "below", "among", "within", "without", "against", "toward", "towards"}
    question_keywords = [w for w in question_words if w not in stopwords]
    
    # 質問のキーワードの同義語
    synonym_map = {
        "meet": ["met", "meeting", "introduce", "introduced", "encounter", "encountered"],
        "visit": ["visited", "visiting", "trip", "travel", "traveling", "went", "gone"],
        "move": ["moved", "moving", "relocate", "relocated", "relocation"],
        "start": ["started", "starting", "begin", "began", "begun", "commence", "commenced"],
        "announce": ["announced", "announcing", "share", "shared", "sharing", "tell", "told"],
        "release": ["released", "releasing", "launch", "launched", "launching", "publish", "published"],
        "receive": ["received", "receiving", "get", "got", "getting"],
        "plan": ["planned", "planning", "schedule", "scheduled", "arrange", "arranged"],
        "adopt": ["adopted", "adopting"],
        "lose": ["lost", "losing", "loss"],
        "launch": ["launched", "launching", "release", "released"],
    }
    
    for w in question_keywords:
        patterns["contextual"].append(w)
        if w in synonym_map:
            patterns["synonyms"].extend(synonym_map[w])
    
    # 重複を除去
    for key in patterns:
        patterns[key] = list(set(patterns[key]))
    
    return patterns, question_keywords

def search_conversation_semantic(turns, patterns, question_keywords, gold):
    """会話全体を意味的に検索"""
    results = {
        "exact_matches": [],
        "synonym_matches": [],
        "relative_date_matches": [],
        "contextual_matches": [],
        "partial_matches": [],
    }
    
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        
        # 完全一致
        for p in patterns["exact"]:
            if p in text_lower:
                results["exact_matches"].append({
                    "turn": f"D{i+1}",
                    "pattern": p,
                    "text": text[:200]
                })
        
        # 同義語
        for p in patterns["synonyms"]:
            if p in text_lower:
                results["synonym_matches"].append({
                    "turn": f"D{i+1}",
                    "pattern": p,
                    "text": text[:200]
                })
        
        # 相対日付
        for p in patterns["relative_dates"]:
            if p in text_lower:
                results["relative_date_matches"].append({
                    "turn": f"D{i+1}",
                    "pattern": p,
                    "text": text[:200]
                })
        
        # コンテキスト（質問キーワード）
        for p in question_keywords:
            if p in text_lower:
                results["contextual_matches"].append({
                    "turn": f"D{i+1}",
                    "pattern": p,
                    "text": text[:200]
                })
        
        # 部分一致（Goldの一部が含まれる）
        gold_parts = [w for w in re.sub(r"[^\w\s]", " ", gold.lower()).split() if len(w) > 2]
        for p in gold_parts:
            if p in text_lower:
                results["partial_matches"].append({
                    "turn": f"D{i+1}",
                    "pattern": p,
                    "text": text[:200]
                })
    
    # 重複を除去
    for key in results:
        seen = set()
        unique = []
        for r in results[key]:
            k = (r["turn"], r["pattern"])
            if k not in seen:
                seen.add(k)
                unique.append(r)
        results[key] = unique
    
    return results

def classify_evidence(semantic_results, gold, question):
    """証拠の強さに基づいて分類"""
    exact_count = len(semantic_results["exact_matches"])
    synonym_count = len(semantic_results["synonym_matches"])
    relative_count = len(semantic_results["relative_date_matches"])
    contextual_count = len(semantic_results["contextual_matches"])
    partial_count = len(semantic_results["partial_matches"])
    
    # 判定ロジック
    if exact_count > 0:
        # Goldの完全一致が見つかった
        return "SUPPORTED", f"Goldの完全一致が{exact_count}件見つかった"
    elif relative_count > 0 and contextual_count > 0:
        # 相対日付とコンテキストの両方が見つかった
        return "PARAPHRASE_SUPPORTED", f"相対日付表現が{relative_count}件、コンテキストが{contextual_count}件見つかった"
    elif synonym_count > 0 and contextual_count > 0:
        # 同義語とコンテキストの両方が見つかった
        return "PARAPHRASE_SUPPORTED", f"同義表現が{synonym_count}件、コンテキストが{contextual_count}件見つかった"
    elif contextual_count > 0 and partial_count > 0:
        # コンテキストと部分一致が見つかった
        return "INCONCLUSIVE", f"コンテキストが{contextual_count}件、部分一致が{partial_count}件見つかったが、Goldを直接支持しない"
    elif contextual_count > 0:
        # コンテキストのみ
        return "INCONCLUSIVE", f"コンテキストのみ{contextual_count}件。Goldを直接支持する証拠なし"
    else:
        return "UNSUPPORTED", "Goldを支持する証拠が見つからなかった"

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
        
        # 意味的な検索パターンを構築
        patterns, question_keywords = build_semantic_patterns(gold, question)
        
        print(f"\n検索パターン:")
        print(f"  完全一致: {patterns['exact']}")
        print(f"  同義語: {patterns['synonyms']}")
        print(f"  相対日付: {patterns['relative_dates']}")
        print(f"  コンテキスト: {patterns['contextual']}")
        
        # 会話全体を意味的に検索
        semantic_results = search_conversation_semantic(turns, patterns, question_keywords, gold)
        
        print(f"\n検索結果:")
        print(f"  完全一致: {len(semantic_results['exact_matches'])}件")
        for m in semantic_results["exact_matches"][:3]:
            print(f"    {m['turn']} | '{m['pattern']}' | {m['text'][:100]}")
        
        print(f"  同義語: {len(semantic_results['synonym_matches'])}件")
        for m in semantic_results["synonym_matches"][:3]:
            print(f"    {m['turn']} | '{m['pattern']}' | {m['text'][:100]}")
        
        print(f"  相対日付: {len(semantic_results['relative_date_matches'])}件")
        for m in semantic_results["relative_date_matches"][:3]:
            print(f"    {m['turn']} | '{m['pattern']}' | {m['text'][:100]}")
        
        print(f"  コンテキスト: {len(semantic_results['contextual_matches'])}件")
        for m in semantic_results["contextual_matches"][:3]:
            print(f"    {m['turn']} | '{m['pattern']}' | {m['text'][:100]}")
        
        print(f"  部分一致: {len(semantic_results['partial_matches'])}件")
        for m in semantic_results["partial_matches"][:3]:
            print(f"    {m['turn']} | '{m['pattern']}' | {m['text'][:100]}")
        
        # 分類
        verdict, reason = classify_evidence(semantic_results, gold, question)
        
        print(f"\n判定: {verdict}")
        print(f"根拠: {reason}")
        
        # 未確認事項
        unconfirmed = []
        if verdict == "UNSUPPORTED":
            unconfirmed.append("Goldの出所が特定できない")
            unconfirmed.append("データセットの仕様（意図的に回答不能なケースか）が不明")
        elif verdict == "INCONCLUSIVE":
            unconfirmed.append("Goldを直接支持する証拠が見つからない")
            unconfirmed.append("意味的な言い換えの可能性が残る")
        
        audit_results.append({
            "question_id": qid,
            "conversation_key": conv_key,
            "question": question,
            "gold": gold,
            "search_patterns": patterns,
            "semantic_results": {
                "exact_matches_count": len(semantic_results["exact_matches"]),
                "synonym_matches_count": len(semantic_results["synonym_matches"]),
                "relative_date_matches_count": len(semantic_results["relative_date_matches"]),
                "contextual_matches_count": len(semantic_results["contextual_matches"]),
                "partial_matches_count": len(semantic_results["partial_matches"]),
                "exact_matches_samples": semantic_results["exact_matches"][:3],
                "synonym_matches_samples": semantic_results["synonym_matches"][:3],
                "relative_date_matches_samples": semantic_results["relative_date_matches"][:3],
                "contextual_matches_samples": semantic_results["contextual_matches"][:3],
                "partial_matches_samples": semantic_results["partial_matches"][:3],
            },
            "verdict": verdict,
            "reason": reason,
            "unconfirmed": unconfirmed,
        })
    
    # サマリー
    print(f"\n\n{'='*80}")
    print("=== 9件意味的証拠・Gold出所監査 サマリー ===")
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
    with open("results/m30_10_semantic_audit.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_cases": len(audit_results),
            "verdict_summary": verdict_counts,
            "details": audit_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n監査結果を results/m30_10_semantic_audit.json に保存しました")

if __name__ == "__main__":
    main()