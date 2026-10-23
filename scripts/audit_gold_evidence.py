#!/usr/bin/env python3
"""M30-20: 28件のGoldとEvidenceを独立監査"""
import json, re
from pathlib import Path

with open('benchmark_results/evidence_audit/v4_intervention_temporal.json', encoding='utf-8') as f:
    m25 = json.load(f)
target_qids = [r['question_id'] for r in m25['rows'] if r.get('group') == 'TARGET']

with open('benchmark_results/evidence_audit/v4_bucketA_all.json', encoding='utf-8') as f:
    bucketA = json.load(f)
cases = {c['question_id']: c for c in bucketA['cases']}

with open(r'C:\Users\smily\AppData\Local\Temp\locomo10.json', encoding='utf-8') as f:
    locomo = json.load(f)

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
        if day > 1:
            return (year, month, day - 1)
        elif month > 1:
            return (year, month - 1, 28)
        else:
            return (year - 1, 12, 31)
    if 'tomorrow' in text_lower:
        if day < 28:
            return (year, month, day + 1)
        elif month < 12:
            return (year, month + 1, 1)
        else:
            return (year + 1, 1, 1)
    return None

def get_session_date(conversation, session_num):
    return conversation.get(f'session_{session_num}_date_time', '')

def get_all_turns(conversation):
    all_turns = []
    session_keys = [k for k in conversation.keys() if k.startswith('session_') and k[8:].isdigit()]
    for key in sorted(session_keys, key=lambda x: int(x.split('_')[1])):
        session_num = key.split('_')[1]
        session_date = get_session_date(conversation, session_num)
        turns = conversation.get(key, [])
        if isinstance(turns, list):
            for i, turn in enumerate(turns):
                if isinstance(turn, dict):
                    all_turns.append({
                        'session': session_num,
                        'turn': i + 1,
                        'dia_id': f'D{session_num}:{i+1}',
                        'date': session_date,
                        'speaker': turn.get('speaker', ''),
                        'text': turn.get('text', '')
                    })
    return all_turns

def check_gold_support(text, gold, session_date):
    gold_lower = gold.lower().strip()
    text_lower = text.lower()
    
    if gold_lower in text_lower:
        return True, 'direct'
    
    gold_clean = re.sub(r'[^\w\s]', ' ', gold_lower)
    gold_words = [w for w in gold_clean.split() if len(w) > 2]
    if gold_words and all(w in text_lower for w in gold_words):
        return True, 'all_keywords'
    
    resolved = resolve_relative_date(text, session_date)
    if resolved:
        ry, rm, rd = resolved
        if re.fullmatch(r'\s*(?:in\s+)?(?:19|20)\d{2}\s*', gold_lower):
            gold_year = re.search(r'(19|20)\d{2}', gold_lower).group(0)
            if str(ry) == gold_year:
                return True, 'relative_year'
        m = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s*,?\s*((?:19|20)\d{2})\b', gold_lower)
        if m and rm:
            gold_month = m.group(1)
            gold_year = m.group(2)
            if rm == MONTHS.get(gold_month, 0) and str(ry) == gold_year:
                return True, 'relative_month_year'
    
    return False, 'no_support'

results = []
for qid in target_qids:
    c = cases.get(qid)
    if not c:
        continue
    conv_id = c.get('source_conversation', '')
    question = c.get('question', '')
    gold = str(c.get('gold_answer', ''))
    
    locomo_conv = None
    locomo_qa = None
    for conv in locomo:
        if conv.get('sample_id') == conv_id:
            locomo_conv = conv
            qa_list = conv.get('qa', [])
            for qa in qa_list:
                if qa.get('question', '').lower().strip() == question.lower().strip():
                    locomo_qa = qa
                    break
            if locomo_qa is None:
                for qa in qa_list:
                    if question.lower().strip() in qa.get('question', '').lower().strip():
                        locomo_qa = qa
                        break
            break
    
    if not locomo_conv or not locomo_qa:
        results.append({
            'question_id': qid,
            'question': question,
            'gold': gold,
            'verdict': 'INCONCLUSIVE',
            'reason': 'LoCoMo QA not found',
            'annotated_evidence': [],
            'supporting_turns': [],
        })
        continue
    
    annotated_evidence_ids = locomo_qa.get('evidence', [])
    all_turns = get_all_turns(locomo_conv.get('conversation', {}))
    
    annotated_evidence = []
    for eid in annotated_evidence_ids:
        m = re.match(r'D(\d+):(\d+)', eid)
        if m:
            session_num = m.group(1)
            turn_num = int(m.group(2))
            session_date = get_session_date(locomo_conv.get('conversation', {}), session_num)
            session = locomo_conv.get('conversation', {}).get(f'session_{session_num}', [])
            if isinstance(session, list) and 1 <= turn_num <= len(session):
                turn = session[turn_num - 1]
                if isinstance(turn, dict):
                    text = turn.get('text', '')
                    supports, support_type = check_gold_support(text, gold, session_date)
                    annotated_evidence.append({
                        'evidence_id': eid,
                        'dia_id': turn.get('dia_id', ''),
                        'date': session_date,
                        'text': text[:200],
                        'supports_gold': supports,
                        'support_type': support_type,
                    })
    
    supporting_turns = []
    for turn in all_turns:
        supports, support_type = check_gold_support(turn['text'], gold, turn['date'])
        if supports:
            supporting_turns.append({
                'dia_id': turn['dia_id'],
                'session': turn['session'],
                'date': turn['date'],
                'speaker': turn['speaker'],
                'text': turn['text'][:200],
                'support_type': support_type,
            })
    
    has_annotated_support = any(e['supports_gold'] for e in annotated_evidence)
    has_other_support = len(supporting_turns) > 0
    
    if has_annotated_support:
        verdict = 'GOLD_SUPPORTED_BY_ANNOTATED_EVIDENCE'
        reason = f'注釈EvidenceがGoldを支持（{sum(1 for e in annotated_evidence if e["supports_gold"])}/{len(annotated_evidence)}件）'
    elif has_other_support:
        verdict = 'GOLD_SUPPORTED_BY_OTHER_TURN'
        reason = f'注釈EvidenceはGoldを支持しないが、別ターンが支持（{len(supporting_turns)}件）'
    else:
        verdict = 'GOLD_UNSUPPORTED_OR_AMBIGUOUS'
        reason = '会話全体からGoldを支持するターンが見つからない'
    
    results.append({
        'question_id': qid,
        'question': question,
        'gold': gold,
        'verdict': verdict,
        'reason': reason,
        'annotated_evidence': annotated_evidence,
        'supporting_turns': supporting_turns[:5],
        'total_supporting_turns': len(supporting_turns),
    })

verdict_counts = {}
for r in results:
    verdict_counts[r['verdict']] = verdict_counts.get(r['verdict'], 0) + 1

print(f'=== 28件のGold & Evidence 独立監査 ===')
for v, c in sorted(verdict_counts.items(), key=lambda x: -x[1]):
    print(f'  {v}: {c}件')

print(f'\n詳細:')
for r in results:
    print(f"\n  {r['question_id']}:")
    print(f"    Gold: {r['gold']}")
    print(f"    判定: {r['verdict']}")
    print(f"    理由: {r['reason']}")
    if r['annotated_evidence']:
        for ae in r['annotated_evidence']:
            print(f"    注釈: {ae['evidence_id']} | supports={ae['supports_gold']} | {ae['support_type']}")
            print(f"      {ae['text'][:100]}")
    if r.get('supporting_turns'):
        for st in r['supporting_turns'][:3]:
            print(f"    支持: {st['dia_id']} | {st['support_type']} | {st['text'][:100]}")

with open('results/m30_20_gold_evidence_audit.json', 'w', encoding='utf-8') as f:
    json.dump({
        'verdict_summary': verdict_counts,
        'details': results,
    }, f, indent=2, ensure_ascii=False)
print(f'\n結果を results/m30_20_gold_evidence_audit.json に保存しました')