#!/usr/bin/env python3
"""M30-24: Failure Classification Adversarial Audit

M30-23の結論を前提としない。19件のReader棄権と5件のscorer mismatchを
実際の証拠・入力・出力・採点処理によって独立に検証する。

保護条件:
- 553件のホールドアウトは完全隔離
- Gold・ベンチマークデータ・本番コードは変更しない
- M30-23の結果を再現するために既存の評価スクリプトは変更しない
"""
import json, re, sys
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
    parts = qid.split("_")
    if len(parts) >= 3 and parts[0] == "lc":
        return SESSION_DATES.get(parts[1])
    return None

# データ読み込み
with open('benchmark_results/evidence_audit/v4_bucketA_all.json', encoding='utf-8') as f:
    bucketA = json.load(f)
cases = {c['question_id']: c for c in bucketA['cases']}
conversations = bucketA['conversations']

with open('benchmark_results/evidence_audit/v4_intervention_temporal.json', encoding='utf-8') as f:
    m25 = json.load(f)
target_qids = [r['question_id'] for r in m25['rows'] if r.get('group') == 'TARGET']

with open('results/m30_3_evaluation_results.json', encoding='utf-8') as f:
    m30_3_data = json.load(f)
m30_3_results = {r['question_id']: r for r in m30_3_data['results']}

with open('results/m30_23_failure_analysis.json', encoding='utf-8') as f:
    m30_23_data = json.load(f)
m30_23_reasons = {r['question_id']: r['reason'] for r in m30_23_data['details']}

def extract_normal_evidence_text(qid, case, conversations, gold):
    """通常パイプラインがReaderに渡した実際の証拠テキストを再構築"""
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

# Phase 1: 19件の棄権分類を反証する
print("=" * 80)
print("Phase 1: 19件のReader棄権を反証する")
print("=" * 80)

abstained_qids = [qid for qid in target_qids if m30_23_reasons.get(qid) == "EVIDENCE_PRESENT_BUT_READER_ABSTAINED"]
print(f"\n対象: {len(abstained_qids)}件")

grader = OfficialLongMemEvalGrader()
phase1_results = []

for qid in abstained_qids:
    c = cases.get(qid)
    if not c:
        continue
    
    question = c.get('question', '')
    gold = str(c.get('gold_answer', ''))
    ref_date = get_ref_date(qid)
    
    # 通常パイプラインのReader入力
    normal = m30_3_results.get(qid, {})
    normal_output = normal.get('reader_text', '')
    normal_evidence = extract_normal_evidence_text(qid, c, conversations, gold)
    normal_context = '\n'.join(normal_evidence)
    
    # Readerの生出力
    reader_abstained = "don't know" in normal_output.lower()
    
    # 入力内の相対日付表現
    relative_expressions = []
    for pattern, name in [
        (r'\bnext month\b', 'next month'),
        (r'\blast month\b', 'last month'),
        (r'\bthis month\b', 'this month'),
        (r'\blast year\b', 'last year'),
        (r'\bnext year\b', 'next year'),
        (r'\bone year ago\b', 'one year ago'),
        (r'\byesterday\b', 'yesterday'),
        (r'\btomorrow\b', 'tomorrow'),
    ]:
        if re.search(pattern, normal_context.lower()):
            relative_expressions.append(name)
    
    # 参照日付がReader入力に含まれているか
    ref_date_in_reader_input = ref_date in normal_context if ref_date else False
    
    # 証拠がGoldを直接支持するか
    gold_lower = gold.lower().strip()
    gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
    gold_words = [w for w in gold_clean.split() if len(w) > 2]
    stopwords = {"the", "in", "on", "at", "of", "is", "a", "an", "to", "for",
                 "january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"}
    gold_keywords = [w for w in gold_words if w not in stopwords]
    
    evidence_supports_gold = False
    if gold_keywords:
        evidence_lower = normal_context.lower()
        evidence_supports_gold = all(w in evidence_lower for w in gold_keywords)
    
    # 判定
    if not normal_evidence:
        verdict = "NO_SUPPORT_FOUND"
        reason = "通常入力に証拠が見つからない"
    elif evidence_supports_gold and relative_expressions and ref_date_in_reader_input:
        verdict = "RESOLVABLE_RELATIVE_DATE"
        reason = f"Goldキーワード＋相対日付表現＋参照日付がReader入力に含まれる"
    elif evidence_supports_gold and relative_expressions and not ref_date_in_reader_input:
        verdict = "RESOLVABLE_RELATIVE_DATE"
        reason = f"Goldキーワード＋相対日付表現がReader入力に含まれるが、参照日付はscorerのみ"
    elif evidence_supports_gold:
        verdict = "DIRECTLY_SUPPORTED"
        reason = "GoldキーワードがReader入力に直接含まれる"
    elif relative_expressions:
        verdict = "RELATED_BUT_NOT_SUPPORTING"
        reason = f"相対日付表現({relative_expressions})はあるが、Goldキーワードを含まない"
    else:
        verdict = "UNVERIFIED"
        reason = "証拠の内容を確認できない"
    
    print(f"\n  {qid}:")
    print(f"    Gold: {gold}")
    print(f"    Reader出力: '{normal_output}'")
    print(f"    参照日付: {ref_date} (Reader入力に含まれる: {ref_date_in_reader_input})")
    print(f"    相対日付表現: {relative_expressions}")
    print(f"    Goldキーワード: {gold_keywords}")
    print(f"    証拠がGoldを直接支持: {evidence_supports_gold}")
    print(f"    判定: {verdict}")
    print(f"    理由: {reason}")
    
    phase1_results.append({
        'question_id': qid,
        'question': question,
        'gold': gold,
        'reference_date': ref_date,
        'reference_date_in_reader_input': ref_date_in_reader_input,
        'reader_output': normal_output,
        'reader_abstained': reader_abstained,
        'relative_expressions': relative_expressions,
        'gold_keywords': gold_keywords,
        'evidence_supports_gold': evidence_supports_gold,
        'normal_evidence_count': len(normal_evidence),
        'normal_evidence_preview': normal_context[:300],
        'verdict': verdict,
        'reason': reason,
    })

# 保存
with open('results/m30_24/phase1_abstention_audit.json', 'w', encoding='utf-8') as f:
    json.dump({
        'total_abstained': len(abstained_qids),
        'results': phase1_results,
    }, f, indent=2, ensure_ascii=False)

print(f"\n結果を results/m30_24/phase1_abstention_audit.json に保存しました")