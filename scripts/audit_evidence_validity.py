#!/usr/bin/env python3
"""M30-19 Phase 2: 28件のEvidence妥当性監査（精密版）"""
import json, re

with open(r'C:\Users\smily\AppData\Local\Temp\locomo10.json', encoding='utf-8') as f:
    locomo = json.load(f)

with open('benchmark_results/evidence_audit/v4_intervention_temporal.json', encoding='utf-8') as f:
    m25 = json.load(f)
target_qids = [r['question_id'] for r in m25['rows'] if r.get('group') == 'TARGET']

with open('benchmark_results/evidence_audit/v4_bucketA_all.json', encoding='utf-8') as f:
    bucketA = json.load(f)
cases = {c['question_id']: c for c in bucketA['cases']}
conversations = bucketA['conversations']

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}

def parse_date(date_str):
    if not date_str:
        return None
    m = re.search(r'(\d{1,2})\s+(\w+),\s*(\d{4})', date_str)
    if m:
        day = int(m.group(1))
        month_name = m.group(2).lower()
        year = int(m.group(3))
        month = MONTHS.get(month_name, 0)
        if month:
            return (year, month, day)
    return None

def resolve_relative_date(text, session_date):
    sd = parse_date(session_date)
    if not sd:
        return None
    year, month, day = sd
    text_lower = text.lower()
    
    if 'next month' in text_lower:
        new_month = month + 1 if month < 12 else 1
        new_year = year + 1 if month == 12 else year
        return (new_year, new_month, None)
    if 'last month' in text_lower:
        new_month = month - 1 if month > 1 else 12
        new_year = year - 1 if month == 1 else year
        return (new_year, new_month, None)
    if 'this month' in text_lower:
        return (year, month, None)
    if 'last year' in text_lower:
        return (year - 1, None, None)
    if 'next year' in text_lower:
        return (year + 1, None, None)
    if re.search(r'\b(one|a)\s+year\s+ago\b', text_lower):
        return (year - 1, None, None)
    m = re.search(r'\b(\d+)\s+years?\s+ago\b', text_lower)
    if m:
        return (year - int(m.group(1)), None, None)
    if 'yesterday' in text_lower:
        return (year, month, day - 1) if day > 1 else (year, month - 1, 28) if month > 1 else (year - 1, 12, 31)
    if 'tomorrow' in text_lower:
        return (year, month, day + 1) if day < 28 else (year, month + 1, 1) if month < 12 else (year + 1, 1, 1)
    return None

def check_gold_support(evidence_text, gold, session_date):
    gold_lower = gold.lower().strip()
    text_lower = evidence_text.lower()
    
    if gold_lower in text_lower:
        return 'DIRECT', 'Goldと完全一致'
    
    gold_clean = re.sub(r'[^\w\s]', ' ', gold_lower)
    gold_words = [w for w in gold_clean.split() if len(w) > 2]
    if gold_words and all(w in text_lower for w in gold_words):
        return 'DIRECT', 'Goldの全キーワードが含まれる'
    
    resolved = resolve_relative_date(evidence_text, session_date)
    if resolved:
        ry, rm, rd = resolved
        if re.fullmatch(r'\s*(?:in\s+)?(?:19|20)\d{2}\s*', gold_lower):
            gold_year = re.search(r'(19|20)\d{2}', gold_lower).group(0)
            if str(ry) == gold_year:
                return 'DERIVED', f'相対日付解決: {ry} == Gold {gold_year}'
        m = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s*,?\s*((?:19|20)\d{2})\b', gold_lower)
        if m and rm:
            gold_month = m.group(1)
            gold_year = m.group(2)
            if rm == MONTHS.get(gold_month, 0) and str(ry) == gold_year:
                return 'DERIVED', f'相対日付解決: {ry}/{rm} == Gold {gold_month} {gold_year}'
    
    if gold_words:
        partial = [w for w in gold_words if w in text_lower]
        if partial:
            return 'PARTIAL', f'部分一致: {partial}'
    
    return 'NO_SUPPORT', 'Goldを支持しない'

results = []
for qid in target_qids:
    c = cases.get(qid)
    if not c:
        continue
    conv_id = c.get('source_conversation', '')
    question = c.get('question', '')
    gold = str(c.get('gold_answer', ''))

    locomo_evidence = []
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
                for eid in locomo_qa.get('evidence', []):
                    m = re.match(r'D(\d+):(\d+)', eid)
                    if m:
                        session_num = m.group(1)
                        turn_num = int(m.group(2))
                        session_key = f'session_{session_num}'
                        session = conversation.get(session_key, [])
                        session_date = conversation.get(f'session_{session_num}_date_time', '')
                        if isinstance(session, list) and 1 <= turn_num <= len(session):
                            turn = session[turn_num - 1]
                            if isinstance(turn, dict):
                                locomo_evidence.append({
                                    'evidence_id': eid,
                                    'session_date': session_date,
                                    'text': turn.get('text', '')
                                })
            break

    evidence_checks = []
    for ev in locomo_evidence:
        support_type, reason = check_gold_support(ev['text'], gold, ev['session_date'])
        evidence_checks.append({
            'evidence_id': ev['evidence_id'],
            'session_date': ev['session_date'],
            'text_preview': ev['text'][:100],
            'support_type': support_type,
            'reason': reason,
        })

    has_direct = any(e['support_type'] == 'DIRECT' for e in evidence_checks)
    has_derived = any(e['support_type'] == 'DERIVED' for e in evidence_checks)
    has_partial = any(e['support_type'] == 'PARTIAL' for e in evidence_checks)
    
    if has_direct or has_derived:
        verdict = 'EVIDENCE_PRESERVED'
    elif has_partial:
        verdict = 'ALTERNATIVE_SUPPORT_PRESENT'
    else:
        verdict = 'QUERY_OR_RETRIEVAL_FAILURE'

    results.append({
        'question_id': qid,
        'gold': gold,
        'evidence_checks': evidence_checks,
        'verdict': verdict,
    })

verdict_counts = {}
for r in results:
    verdict_counts[r['verdict']] = verdict_counts.get(r['verdict'], 0) + 1

print('=== 28件のEvidence妥当性監査（精密版）===')
for v, c in sorted(verdict_counts.items(), key=lambda x: -x[1]):
    print(f'  {v}: {c}件')

print('\n詳細:')
for r in results:
    print(f"\n  {r['question_id']}:")
    print(f"    Gold: {r['gold']}")
    for ec in r['evidence_checks']:
        print(f"    {ec['evidence_id']} | {ec['support_type']} | {ec['reason']}")
    print(f"    判定: {r['verdict']}")

with open('results/m30_19_evidence_validity_v2.json', 'w', encoding='utf-8') as f:
    json.dump({'verdict_summary': verdict_counts, 'details': results}, f, indent=2, ensure_ascii=False)
print('\n結果を results/m30_19_evidence_validity_v2.json に保存しました')