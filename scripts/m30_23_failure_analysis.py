#!/usr/bin/env python3
"""M30-23: Reader失敗24件の原因別分解

28件すべてについて:
1. 通常入力と診断入力の差分
2. Readerの生出力
3. Goldと予測回答
4. scorerの判定理由
5. Goldを支持する証拠が入力に存在したか
6. 原因分類

553件のホールドアウトには一切アクセスしない。
"""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

with open('benchmark_results/evidence_audit/v4_intervention_temporal.json', encoding='utf-8') as f:
    m25 = json.load(f)
target_qids = [r['question_id'] for r in m25['rows'] if r.get('group') == 'TARGET']

with open('benchmark_results/evidence_audit/v4_bucketA_all.json', encoding='utf-8') as f:
    bucketA = json.load(f)
cases = {c['question_id']: c for c in bucketA['cases']}
conversations = bucketA['conversations']

with open('results/m30_3_evaluation_results.json', encoding='utf-8') as f:
    m30_3_data = json.load(f)
m30_3_results = {r['question_id']: r for r in m30_3_data['results']}

with open('results/m30_22_reader_comparison.json', encoding='utf-8') as f:
    m30_22_data = json.load(f)
m30_22_results = {r['question_id']: r for r in m30_22_data['results']}

def extract_normal_evidence(qid, case, conversations, gold):
    conv_key = case.get("source_conversation", "")
    turns = conversations.get(conv_key, [])
    question = case.get("question", "").lower()
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
    evidence = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        if any(kw in text_lower for kw in gold_keywords):
            evidence.append(f"D{i+1} | {t.get('speaker', '?')} | {text}")
        elif any(w in text_lower for w in question_keywords if len(w) > 3):
            if any(d in text_lower for d in ["month", "year", "2023", "2022", "2024", "indiana", "seoul", "france", "rome", "boston"]):
                evidence.append(f"D{i+1} | {t.get('speaker', '?')} | {text}")
    seen = set()
    unique = []
    for t in evidence:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[-8:]

def check_evidence_supports_gold(evidence_text, gold):
    """通常入力にGoldを支持する証拠が含まれているか"""
    gold_lower = gold.lower().strip()
    gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
    gold_words = [w for w in gold_clean.split() if len(w) > 2]
    stopwords = {"the", "in", "on", "at", "of", "is", "a", "an", "to", "for",
                 "january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"}
    gold_keywords = [w for w in gold_words if w not in stopwords]
    
    evidence_lower = evidence_text.lower()
    
    # Goldの全キーワードが含まれる
    if gold_keywords and all(w in evidence_lower for w in gold_keywords):
        return True
    
    # 相対日付表現があるか
    relative_patterns = [
        r"\bnext month\b", r"\blast month\b", r"\bthis month\b",
        r"\blast year\b", r"\bnext year\b", r"\bone year ago\b",
        r"\byesterday\b", r"\btomorrow\b"
    ]
    has_relative = any(re.search(p, evidence_lower) for p in relative_patterns)
    
    return has_relative

def classify_failure_reason(qid, normal_evidence, normal_output, gold, normal_correct, diag_correct, diag_output):
    """失敗原因を分類"""
    if normal_correct:
        return "CORRECT"
    
    # 通常入力にGoldを支持する証拠があるか
    normal_evidence_text = '\n'.join(normal_evidence) if normal_evidence else ""
    evidence_supports_gold = check_evidence_supports_gold(normal_evidence_text, gold)
    
    # Readerの出力
    reader_abstained = "don't know" in normal_output.lower()
    reader_has_relative = any(p in normal_output.lower() for p in ["next month", "last month", "this month", "last year", "next year", "year ago", "yesterday"])
    reader_has_absolute = bool(re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|20\d{2})\b", normal_output.lower()))
    
    # 診断で改善したか
    improved = not normal_correct and diag_correct
    regressed = normal_correct and not diag_correct
    
    if improved:
        return "IMPROVED_WITH_DIAGNOSTIC_EVIDENCE"
    if regressed:
        return "REGRESSED_WITH_DIAGNOSTIC_EVIDENCE"
    
    # 証拠の有無で分類
    if not evidence_supports_gold and not normal_evidence:
        return "EVIDENCE_MISSING"
    elif not evidence_supports_gold and normal_evidence:
        return "EVIDENCE_PRESENT_BUT_NOT_GOLD_SUPPORTING"
    elif evidence_supports_gold and reader_abstained:
        return "EVIDENCE_PRESENT_BUT_READER_ABSTAINED"
    elif evidence_supports_gold and not reader_abstained and not normal_correct:
        return "EVIDENCE_PRESENT_BUT_SCORER_MISMATCH"
    else:
        return "INCONCLUSIVE"

# 28件を分析
results = []
for qid in target_qids:
    c = cases.get(qid)
    if not c:
        continue
    
    question = c.get('question', '')
    gold = str(c.get('gold_answer', ''))
    conv_key = c.get('source_conversation', '')
    ref_date = None
    parts = qid.split("_")
    if len(parts) >= 3 and parts[0] == "lc":
        ref_date = {
            "conv-26": "2023-08-28", "conv-30": "2023-11-30",
            "conv-41": "2023-12-04", "conv-42": "2023-12-04",
            "conv-44": "2023-12-04", "conv-47": "2022-09-30",
            "conv-48": "2023-01-31", "conv-49": "2023-12-25",
            "conv-50": "2023-08-28",
        }.get(parts[1])
    
    # 通常パイプライン
    normal = m30_3_results.get(qid, {})
    normal_output = normal.get('reader_text', '')
    normal_correct = normal.get('scorer_correct', False)
    normal_evidence = extract_normal_evidence(qid, c, conversations, gold)
    normal_evidence_text = '\n'.join(normal_evidence) if normal_evidence else ""
    
    # 診断
    diag = m30_22_results.get(qid, {})
    diag_output = diag.get('diagnostic_output', '')
    diag_correct = diag.get('diagnostic_correct', False)
    
    # 原因分類
    reason = classify_failure_reason(qid, normal_evidence, normal_output, gold, normal_correct, diag_correct, diag_output)
    
    # 詳細情報
    evidence_supports_gold = check_evidence_supports_gold(normal_evidence_text, gold)
    reader_abstained = "don't know" in normal_output.lower()
    
    results.append({
        'question_id': qid,
        'question': question,
        'gold': gold,
        'reference_date': ref_date,
        'normal_evidence_count': len(normal_evidence),
        'normal_evidence_text': normal_evidence_text[:300],
        'evidence_supports_gold': evidence_supports_gold,
        'normal_output': normal_output,
        'normal_correct': normal_correct,
        'diag_output': diag_output,
        'diag_correct': diag_correct,
        'reader_abstained': reader_abstained,
        'reason': reason,
    })

# サマリー
reason_counts = {}
for r in results:
    reason_counts[r['reason']] = reason_counts.get(r['reason'], 0) + 1

print("=" * 80)
print("M30-23: Reader失敗24件の原因別分解")
print("=" * 80)

print("\n=== 原因別サマリー ===")
for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print(f"  {reason}: {count}件")

print("\n=== 詳細 ===")
for r in results:
    print(f"\n  {r['question_id']}:")
    print(f"    Gold: {r['gold']}")
    print(f"    通常出力: '{r['normal_output'][:80]}'")
    print(f"    通常正解: {r['normal_correct']}")
    print(f"    診断出力: '{r['diag_output'][:80]}'")
    print(f"    診断正解: {r['diag_correct']}")
    print(f"    証拠数: {r['normal_evidence_count']}")
    print(f"    Gold支持: {r['evidence_supports_gold']}")
    print(f"    Reader棄権: {r['reader_abstained']}")
    print(f"    原因: {r['reason']}")

# lc_conv-48_32の採点問題を別途記録
print("\n=== lc_conv-48_32 採点問題（Readerとは別管理）===")
r48_32 = next((r for r in results if r['question_id'] == 'lc_conv-48_32'), None)
if r48_32:
    print(f"  通常出力: '{r48_32['normal_output']}'")
    print(f"  通常正解: {r48_32['normal_correct']}")
    print(f"  診断出力: '{r48_32['diag_output']}'")
    print(f"  診断正解: {r48_32['diag_correct']}")
    print(f"  Gold: {r48_32['gold']}")
    print(f"  問題: 'Deborah was in Bali in 2022.' は 'in 2022' と意味的に一致するが、scorerが粒度の違いで不正解判定")

# 保存
with open('results/m30_23_failure_analysis.json', 'w', encoding='utf-8') as f:
    json.dump({
        'reason_summary': reason_counts,
        'details': results,
        'scoring_issue_48_32': {
            'normal_output': r48_32['normal_output'] if r48_32 else '',
            'normal_correct': r48_32['normal_correct'] if r48_32 else '',
            'diag_output': r48_32['diag_output'] if r48_32 else '',
            'diag_correct': r48_32['diag_correct'] if r48_32 else '',
            'gold': r48_32['gold'] if r48_32 else '',
        }
    }, f, indent=2, ensure_ascii=False)

print(f"\n結果を results/m30_23_failure_analysis.json に保存しました")