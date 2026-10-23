#!/usr/bin/env python3
"""M30-26: Root-Cause Attribution Audit

Phase 1: 28件の完全な割り当て表
Phase 2: 20件の棄権を再判定
Phase 3: 相対日付5件とscorer問題を再調整
Phase 4: 因果関係と改善候補
"""
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

def classify_case(qid):
    c = cases.get(qid)
    if not c:
        return None
    question = c.get('question', '')
    gold = str(c.get('gold_answer', ''))
    ref_date = get_ref_date(qid)
    
    normal = m30_3_results.get(qid, {})
    normal_output = normal.get('reader_text', '')
    normal_correct = normal.get('scorer_correct', False)
    normal_evidence = extract_normal_evidence(qid, c, conversations, gold)
    normal_evidence_text = '\n'.join(normal_evidence)
    
    m30_6 = m30_6_results.get(qid, {})
    m30_6_output = m30_6.get('reader_text', '')
    m30_6_correct = m30_6.get('scorer_correct', False)
    
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
    
    # Readerの出力
    reader_abstained = "don't know" in normal_output.lower()
    reader_has_relative = any(p in normal_output.lower() for p in ["next month", "last month", "this month", "last year", "next year", "year ago", "yesterday"])
    
    # 主要分類
    if normal_correct:
        primary = "CORRECT"
    elif reader_abstained:
        if evidence_supports_gold:
            primary = "ABSTENTION_WITH_RELEVANT_EVIDENCE"
        elif relative_expressions:
            primary = "ABSTENTION_WITH_RELEVANT_EVIDENCE"
        else:
            primary = "PROPER_ABSTENTION_NO_SUPPORT"
    elif reader_has_relative:
        # 相対日付を出力したが不正解
        # 独立計算で確認
        rd = parse_ref_date(ref_date)
        if rd:
            ry, rm, rday = rd
            reader_lower = normal_output.lower()
            
            if "last month" in reader_lower:
                if rm > 1:
                    ind_m, ind_y = rm - 1, ry
                else:
                    ind_m, ind_y = 12, ry - 1
            elif "next month" in reader_lower:
                if rm < 12:
                    ind_m, ind_y = rm + 1, ry
                else:
                    ind_m, ind_y = 1, ry + 1
            elif "this month" in reader_lower:
                ind_m, ind_y = rm, ry
            elif "one year ago" in reader_lower:
                ind_m, ind_y = None, ry - 1
            else:
                ind_m, ind_y = None, None
            
            # Goldの月と年
            gold_m = re.search(rf'\b({MONTH_NAME_RE})\b', gold.lower())
            gold_month = MONTHS.get(gold_m.group(1)) if gold_m else None
            gold_y = re.search(r'(20\d{2})', gold)
            gold_year = int(gold_y.group(1)) if gold_y else None
            
            if ind_m and gold_month and ind_y and gold_year:
                if ind_m == gold_month and ind_y == gold_year:
                    primary = "SCORER_FALSE_NEGATIVE_CONFIRMED"
                else:
                    # Readerの計算が正しいか確認
                    # 正しい表現を特定
                    correct_expr = None
                    if gold_month == rm and gold_year == ry:
                        correct_expr = "this month"
                    elif gold_month == (rm + 1 if rm < 12 else 1) and gold_year == (ry + 1 if rm == 12 else ry):
                        correct_expr = "next month"
                    elif gold_month == (rm - 1 if rm > 1 else 12) and gold_year == (ry - 1 if rm == 1 else ry):
                        correct_expr = "last month"
                    
                    if correct_expr and correct_expr in reader_lower:
                        primary = "SCORER_FALSE_NEGATIVE_CONFIRMED"
                    elif correct_expr:
                        primary = "WRONG_TEMPORAL_EXPRESSION"
                    else:
                        primary = "WRONG_TEMPORAL_CALCULATION"
            else:
                primary = "WRONG_TEMPORAL_CALCULATION"
        else:
            primary = "UNVERIFIED"
    else:
        primary = "WRONG_ANSWER_OTHER"
    
    # 副次的な問題
    secondary = []
    if not evidence_supports_gold and not relative_expressions:
        secondary.append("no_gold_support_in_normal_input")
    if relative_expressions and not evidence_supports_gold:
        secondary.append("relative_date_but_not_gold_supporting")
    if m30_6_correct and not normal_correct:
        secondary.append("improved_in_m30_6")
    if not m30_6_correct and normal_correct:
        secondary.append("regressed_in_m30_6")
    
    return {
        'question_id': qid,
        'question': question,
        'gold': gold,
        'reference_date': ref_date,
        'normal_output': normal_output,
        'normal_correct': normal_correct,
        'm30_6_output': m30_6_output,
        'm30_6_correct': m30_6_correct,
        'normal_evidence_count': len(normal_evidence),
        'evidence_supports_gold': evidence_supports_gold,
        'relative_expressions': relative_expressions,
        'reader_abstained': reader_abstained,
        'locomo_evidence_ids': locomo_evidence_ids,
        'primary_classification': primary,
        'secondary_issues': secondary,
    }

# Phase 1: 28件の完全な割り当て表
print("=" * 80)
print("Phase 1: 28件の完全な割り当て表")
print("=" * 80)

all_results = []
for qid in target_qids:
    result = classify_case(qid)
    if result:
        all_results.append(result)

# 集計
primary_counts = {}
for r in all_results:
    primary_counts[r['primary_classification']] = primary_counts.get(r['primary_classification'], 0) + 1

print(f"\n=== 主要分類の集計（合計{len(all_results)}件）===")
for cat, count in sorted(primary_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}件")

print(f"\n=== 詳細 ===")
for r in all_results:
    print(f"\n  {r['question_id']}:")
    print(f"    Gold: {r['gold']}")
    print(f"    通常出力: '{r['normal_output'][:60]}'")
    print(f"    通常正解: {r['normal_correct']}")
    print(f"    M30-6出力: '{r['m30_6_output'][:60]}'")
    print(f"    M30-6正解: {r['m30_6_correct']}")
    print(f"    証拠数: {r['normal_evidence_count']}")
    print(f"    Gold支持: {r['evidence_supports_gold']}")
    print(f"    相対日付表現: {r['relative_expressions']}")
    print(f"    Reader棄権: {r['reader_abstained']}")
    print(f"    主要分類: {r['primary_classification']}")
    print(f"    副次的問題: {r['secondary_issues']}")

# 保存
with open('results/m30_26/phase1_assignment.json', 'w', encoding='utf-8') as f:
    json.dump({
        'total_cases': len(all_results),
        'primary_classification_summary': primary_counts,
        'details': all_results,
    }, f, indent=2, ensure_ascii=False)

print(f"\n結果を results/m30_26/phase1_assignment.json に保存しました")