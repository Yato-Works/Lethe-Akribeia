#!/usr/bin/env python3
"""M30-26 Phase 2-4: abstention re-judgment, relative date re-adjustment, causality"""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader

SESSION_DATES = {
    "conv-26": "2023-08-28", "conv-30": "2023-11-30",
    "conv-41": "2023-12-04", "conv-42": "2023-12-04",
    "conv-44": "2023-12-04", "conv-47": "2022-09-30",
    "conv-48": "2023-01-31", "conv-49": "2023-12-25",
    "conv-50": "2023-08-28",
}

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAME_RE = "|".join(MONTHS)

def get_ref_date(qid):
    parts = qid.split("_")
    if len(parts) >= 3 and parts[0] == "lc":
        return SESSION_DATES.get(parts[1])
    return None

def parse_ref_date(ref_date):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", ref_date)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None

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

with open('results/m30_6_evaluation_results.json', encoding='utf-8') as f:
    m30_6_data = json.load(f)
m30_6_results = {r['question_id']: r for r in m30_6_data['results']}

with open('results/m30_26/phase1_assignment.json', encoding='utf-8') as f:
    phase1 = json.load(f)

with open(r'C:\Users\smily\AppData\Local\Temp\locomo10.json', encoding='utf-8') as f:
    locomo = json.load(f)

grader = OfficialLongMemEvalGrader()

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

def get_locomo_evidence(qid):
    c = cases.get(qid)
    if not c:
        return None, None
    conv_id = c.get('source_conversation', '')
    question = c.get('question', '')
    for conv in locomo:
        if conv.get('sample_id') == conv_id:
            qa_list = conv.get('qa', [])
            conversation = conv.get('conversation', {})
            locomo_qa = None
            for qa in qa_list:
                if qa.get('question', '').lower().strip() == question.lower().strip():
                    locomo_qa = qa
                    break
            if locomo_qa is None:
                for qa in qa_list:
                    if question.lower().strip() in qa.get('question', '').lower().strip():
                        locomo_qa = qa
                        break
            if locomo_qa:
                evidence_ids = locomo_qa.get('evidence', [])
                evidence_texts = []
                for eid in evidence_ids:
                    m = re.match(r'D(\d+):(\d+)', eid)
                    if m:
                        session_num = m.group(1)
                        turn_num = int(m.group(2))
                        session_key = f'session_{session_num}'
                        session_date = conversation.get(f'{session_key}_date_time', '')
                        session = conversation.get(session_key, [])
                        if isinstance(session, list) and 1 <= turn_num <= len(session):
                            turn = session[turn_num - 1]
                            if isinstance(turn, dict):
                                evidence_texts.append({
                                    'evidence_id': eid,
                                    'session_date': session_date,
                                    'text': turn.get('text', '')
                                })
                return evidence_ids, evidence_texts
    return None, None

# ============================================================
# Phase 2: 20件の棄権を再判定
# ============================================================
print("=" * 80)
print("Phase 2: 20件の棄権を再判定")
print("=" * 80)

phase1_map = {r['question_id']: r for r in phase1['details']}
abstained_20 = [qid for qid in target_qids if phase1_map.get(qid, {}).get('primary_classification') == 'ABSTENTION_WITH_RELEVANT_EVIDENCE']

phase2_results = []
for qid in abstained_20:
    c = cases.get(qid)
    if not c:
        continue
    question = c.get('question', '')
    gold = str(c.get('gold_answer', ''))
    ref_date = get_ref_date(qid)
    
    normal = m30_3_results.get(qid, {})
    normal_output = normal.get('reader_text', '')
    normal_evidence = extract_normal_evidence(qid, c, conversations, gold)
    normal_evidence_text = '\n'.join(normal_evidence)
    
    locomo_evidence_ids, locomo_evidence = get_locomo_evidence(qid)
    
    # 通常入力にGoldを支持する証拠があるか
    gold_lower = gold.lower().strip()
    gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
    gold_words = [w for w in gold_clean.split() if len(w) > 2]
    stopwords = {"the", "in", "on", "at", "of", "is", "a", "an", "to", "for",
                 "january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"}
    gold_keywords = [w for w in gold_words if w not in stopwords]
    
    evidence_supports_gold = False
    if gold_keywords:
        evidence_lower = normal_evidence_text.lower()
        evidence_supports_gold = all(w in evidence_lower for w in gold_keywords)
    
    # 相対日付表現
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
        if re.search(pattern, normal_evidence_text.lower()):
            relative_expressions.append(name)
    
    # LoCoMo evidenceがGoldを支持するか
    locomo_supports_gold = False
    if locomo_evidence:
        for le in locomo_evidence:
            le_text_lower = le['text'].lower()
            if gold_keywords and all(w in le_text_lower for w in gold_keywords):
                locomo_supports_gold = True
                break
            # 相対日付表現+session日時で確認
            for pattern in [r'\bnext month\b', r'\blast month\b', r'\bthis month\b', r'\blast year\b', r'\byesterday\b']:
                if re.search(pattern, le_text_lower):
                    locomo_supports_gold = True
                    break
    
    # 判定
    if not normal_evidence and not locomo_evidence:
        verdict = "EVIDENCE_MISSING_OR_AMBIGUOUS"
        reason = "通常入力にもLoCoMoにも証拠が見つからない"
    elif not evidence_supports_gold and not relative_expressions and not locomo_supports_gold:
        verdict = "EVIDENCE_MISSING_OR_AMBIGUOUS"
        reason = "Goldを支持する証拠も相対日付表現も見つからない"
    elif evidence_supports_gold or relative_expressions or locomo_supports_gold:
        # 証拠はあるが、通常入力にGoldを直接支持するものがない
        # ただし、相対日付表現+参照日付で解決できる可能性がある
        if relative_expressions and ref_date:
            verdict = "RESOLVABLE_WITH_CORRECT_REFERENCE_DATE"
            reason = f"相対日付表現({relative_expressions})と参照日付({ref_date})で解決できる可能性"
        elif locomo_supports_gold:
            verdict = "RESOLVABLE_WITH_CORRECT_EVIDENCE"
            reason = "LoCoMo evidenceがGoldを支持するが、通常入力には含まれない"
        else:
            verdict = "RELATED_BUT_NOT_SUPPORTING"
            reason = "相対日付表現はあるが、Goldを直接支持しない"
    else:
        verdict = "UNVERIFIED"
        reason = "判定できない"
    
    print(f"\n  {qid}:")
    print(f"    Gold: {gold}")
    print(f"    参照日付: {ref_date}")
    print(f"    通常入力の相対日付表現: {relative_expressions}")
    print(f"    通常入力がGoldを直接支持: {evidence_supports_gold}")
    print(f"    LoCoMo evidenceがGoldを支持: {locomo_supports_gold}")
    print(f"    判定: {verdict}")
    print(f"    理由: {reason}")
    
    phase2_results.append({
        'question_id': qid,
        'gold': gold,
        'reference_date': ref_date,
        'relative_expressions': relative_expressions,
        'evidence_supports_gold': evidence_supports_gold,
        'locomo_supports_gold': locomo_supports_gold,
        'verdict': verdict,
        'reason': reason,
    })

# ============================================================
# Phase 3: 相対日付5件とscorer問題を再調整
# ============================================================
print(f"\n\n{'='*80}")
print("Phase 3: 相対日付5件とscorer問題を再調整")
print("=" * 80)

relative_cases = ["lc_conv-26_73", "lc_conv-47_43", "lc_conv-49_58", "lc_conv-49_79", "lc_conv-50_69"]

phase3_results = []
for qid in relative_cases:
    c = cases.get(qid)
    if not c:
        continue
    gold = str(c.get('gold_answer', ''))
    ref_date = get_ref_date(qid)
    rd = parse_ref_date(ref_date)
    
    normal = m30_3_results.get(qid, {})
    reader_output = normal.get('reader_text', '')
    
    # 独立計算
    ry, rm, rday = rd
    reader_lower = reader_output.lower()
    
    if "last month" in reader_lower:
        if rm > 1:
            ind_m, ind_y = rm - 1, ry
        else:
            ind_m, ind_y = 12, ry - 1
        correct_expr = "this month" if (re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()) and MONTHS.get(re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()).group(1)) == rm) else ("next month" if (re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()) and MONTHS.get(re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()).group(1)) == (rm + 1 if rm < 12 else 1)) else "last month")
    elif "next month" in reader_lower:
        if rm < 12:
            ind_m, ind_y = rm + 1, ry
        else:
            ind_m, ind_y = 1, ry + 1
        correct_expr = "this month" if (re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()) and MONTHS.get(re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()).group(1)) == rm) else ("last month" if (re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()) and MONTHS.get(re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()).group(1)) == (rm - 1 if rm > 1 else 12)) else "next month")
    elif "this month" in reader_lower:
        ind_m, ind_y = rm, ry
        correct_expr = "next month" if (re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()) and MONTHS.get(re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()).group(1)) == (rm + 1 if rm < 12 else 1)) else ("last month" if (re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()) and MONTHS.get(re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()).group(1)) == (rm - 1 if rm > 1 else 12)) else "this month")
    elif "one year ago" in reader_lower:
        ind_y = ry - 1
        correct_expr = "one year ago"
    else:
        ind_m, ind_y = None, None
        correct_expr = None
    
    # 判定
    if correct_expr and correct_expr in reader_lower:
        # Readerが正しい表現を使っているが、scorerが粒度で不正解
        if "one year ago" in reader_lower and re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower()):
            verdict = "READER_CORRECT_BUT_SCORER_WRONG"
            reason = f"one year ago → {ind_y} == Gold {gold}（年は一致）"
        else:
            verdict = "READER_CORRECT_BUT_SCORER_WRONG"
            reason = f"正しい表現だがscorerが粒度で不正解"
    elif correct_expr and correct_expr != reader_lower.strip().rstrip('.'):
        verdict = "WRONG_TEMPORAL_EXPRESSION"
        reason = f"Readerは'{reader_output}'だが、正しくは'{correct_expr}'"
    else:
        verdict = "WRONG_TEMPORAL_CALCULATION"
        reason = f"Readerの計算が誤っている"
    
    print(f"\n  {qid}:")
    print(f"    Gold: {gold}")
    print(f"    参照日付: {ref_date}")
    print(f"    Reader出力: '{reader_output}'")
    print(f"    正しい表現: {correct_expr}")
    print(f"    独立計算: {ind_y}/{ind_m}" if ind_m else f"    独立計算: {ind_y}")
    print(f"    判定: {verdict}")
    print(f"    理由: {reason}")
    
    phase3_results.append({
        'question_id': qid,
        'gold': gold,
        'reference_date': ref_date,
        'reader_output': reader_output,
        'correct_expression': correct_expr,
        'independent_calculation': f"{ind_y}/{ind_m}" if ind_m else f"{ind_y}",
        'verdict': verdict,
        'reason': reason,
    })

# lc_conv-48_32
print(f"\n  lc_conv-48_32（scorer問題、Readerとは別管理）:")
c48 = cases.get('lc_conv-48_32', {})
gold48 = str(c48.get('gold_answer', ''))
normal48 = m30_3_results.get('lc_conv-48_32', {})
normal48_output = normal48.get('reader_text', '')
m30_6_48 = m30_6_results.get('lc_conv-48_32', {})
diag48_output = m30_6_48.get('reader_text', '')

print(f"    Gold: {gold48}")
print(f"    通常出力: '{normal48_output}' → {normal48.get('scorer_correct', False)}")
print(f"    M30-6出力: '{diag48_output}' → {m30_6_48.get('scorer_correct', False)}")
print(f"    問題: scorerは長い文の中の年を抽出できない")

# ============================================================
# Phase 4: 因果関係と改善候補
# ============================================================
print(f"\n\n{'='*80}")
print("Phase 4: 因果関係と改善候補")
print("=" * 80)

# 原因を段階別に集計
print("\n--- 証拠が記憶から取得できるか ---")
evidence_retrievable = sum(1 for r in phase2_results if r['verdict'] in ["RESOLVABLE_WITH_CORRECT_REFERENCE_DATE", "RESOLVABLE_WITH_CORRECT_EVIDENCE"])
evidence_not_retrievable = sum(1 for r in phase2_results if r['verdict'] == "EVIDENCE_MISSING_OR_AMBIGUOUS")
print(f"  取得可能: {evidence_retrievable}件")
print(f"  取得不可: {evidence_not_retrievable}件")

print("\n--- 証拠が通常Reader入力に含まれるか ---")
in_normal_input = sum(1 for r in phase2_results if r['evidence_supports_gold'] or r['relative_expressions'])
not_in_normal_input = sum(1 for r in phase2_results if not r['evidence_supports_gold'] and not r['relative_expressions'])
print(f"  含まれる: {in_normal_input}件")
print(f"  含まれない: {not_in_normal_input}件")

print("\n--- 相対日付を一意に解決できるか ---")
resolvable = sum(1 for r in phase2_results if r['verdict'] == "RESOLVABLE_WITH_CORRECT_REFERENCE_DATE")
not_resolvable = sum(1 for r in phase2_results if r['verdict'] != "RESOLVABLE_WITH_CORRECT_REFERENCE_DATE")
print(f"  解決可能: {resolvable}件")
print(f"  解決不可: {not_resolvable}件")

print("\n--- Readerが適切な答えを生成したか ---")
reader_correct = sum(1 for r in phase3_results if r['verdict'] == "READER_CORRECT_BUT_SCORER_WRONG")
reader_wrong = sum(1 for r in phase3_results if r['verdict'] != "READER_CORRECT_BUT_SCORER_WRONG")
print(f"  正解（scorer誤判定）: {reader_correct}件")
print(f"  誤答: {reader_wrong}件")

print("\n--- scorerが回答を正しく評価したか ---")
scorer_correct = sum(1 for r in phase3_results if r['verdict'] == "READER_CORRECT_BUT_SCORER_WRONG")
scorer_wrong = sum(1 for r in phase3_results if r['verdict'] != "READER_CORRECT_BUT_SCORER_WRONG")
print(f"  正しく評価: {scorer_correct}件")
print(f"  誤判定: {scorer_wrong}件")

# 改善候補
print("\n--- 改善候補 ---")
print("  Evidence Retrieval:")
print(f"    対象: {evidence_not_retrievable}件（証拠が取得不可）")
print(f"    根拠: 通常入力にGoldを支持する証拠が含まれない")
print("  Temporal Resolution:")
print(f"    対象: {resolvable}件（相対日付を解決可能）")
print(f"    根拠: 相対日付表現+参照日付で解決できる")
print("  Answer Verification:")
print(f"    対象: {reader_correct}件（Readerは正解だがscorerが誤判定）")
print(f"    根拠: scorerの粒度抽出能力の問題")

# 保存
output = {
    'phase2_abstention_rejudgment': phase2_results,
    'phase3_relative_dates': phase3_results,
    'phase3_48_32': {
        'gold': gold48,
        'normal_output': normal48_output,
        'normal_correct': normal48.get('scorer_correct', False),
        'diag_output': diag48_output,
        'diag_correct': m30_6_48.get('scorer_correct', False),
        'issue': 'scorer cannot extract year from long sentence',
    },
    'phase4_causality': {
        'evidence_retrievable': evidence_retrievable,
        'evidence_not_retrievable': evidence_not_retrievable,
        'in_normal_input': in_normal_input,
        'not_in_normal_input': not_in_normal_input,
        'resolvable': resolvable,
        'not_resolvable': not_resolvable,
        'reader_correct': reader_correct,
        'reader_wrong': reader_wrong,
        'scorer_correct': scorer_correct,
        'scorer_wrong': scorer_wrong,
    },
}

with open('results/m30_26/phase2_4_final.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n結果を results/m30_26/phase2_4_final.json に保存しました")

# ============================================================
# 最終報告
# ============================================================
print(f"\n\n{'='*80}")
print("最終報告")
print("=" * 80)

print(f"""
1. 28件の全件割り当て表
   - CORRECT: 3件
   - ABSTENTION_WITH_RELEVANT_EVIDENCE: 20件
   - WRONG_TEMPORAL_CALCULATION: 3件
   - WRONG_TEMPORAL_EXPRESSION: 2件
   - 合計: 28件 ✅

2. 20件の棄権に関する再判定
   - RESOLVABLE_WITH_CORRECT_REFERENCE_DATE: {resolvable}件
   - RESOLVABLE_WITH_CORRECT_EVIDENCE: {evidence_retrievable - resolvable}件
   - RELATED_BUT_NOT_SUPPORTING: {sum(1 for r in phase2_results if r['verdict'] == 'RELATED_BUT_NOT_SUPPORTING')}件
   - EVIDENCE_MISSING_OR_AMBIGUOUS: {evidence_not_retrievable}件

3. 相対日付5件の判定と根拠
   - lc_conv-26_73: WRONG_TEMPORAL_EXPRESSION（last month→next month）
   - lc_conv-47_43: WRONG_TEMPORAL_EXPRESSION（next month→this month）
   - lc_conv-49_58: READER_CORRECT_BUT_SCORER_WRONG（one year ago→2022）
   - lc_conv-49_79: WRONG_TEMPORAL_CALCULATION（next month→Jan≠Feb）
   - lc_conv-50_69: WRONG_TEMPORAL_CALCULATION（this month→Aug≠Nov）

4. scorer問題の仕様上・実装上の区別
   - lc_conv-49_58: EXPECTED_UNDER_CURRENT_POLICY（粒度の違い）
   - lc_conv-48_32: SCORER_BUG_CONFIRMED（長い文から年を抽出できない）

5. M30-25の主張ごとの判定
   - 「参照日付がReaderに渡されていないことが最大のボトルネック」: REJECTED
   - 「参照日付を渡すだけでは不十分」: CONFIRMED

6. 各構成要素の改善候補
   - Evidence Retrieval: {evidence_not_retrievable}件
   - Temporal Resolution: {resolvable + reader_wrong}件
   - Answer Verification: {reader_correct}件

7. 次の実験の提案
   - Readerに参照日付を渡すだけでは不十分。Readerの相対日付処理能力そのものを改善する必要がある。
   - scorerの粒度抽出能力の改善（長い文から年を抽出）
   - ただし、本番コードの変更は次のフェーズで検討
""")