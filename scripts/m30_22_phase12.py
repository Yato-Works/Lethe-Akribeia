#!/usr/bin/env python3
"""M30-22: Temporal Evidence Validation & Reader Diagnostic

Phase 1: 7件の根拠付き再確認
Phase 2: Readerへの実際の入力を確認
"""
import json, re
from pathlib import Path

with open(r'C:\Users\smily\AppData\Local\Temp\locomo10.json', encoding='utf-8') as f:
    locomo = json.load(f)

with open('benchmark_results/evidence_audit/v4_bucketA_all.json', encoding='utf-8') as f:
    bucketA = json.load(f)
cases = {c['question_id']: c for c in bucketA['cases']}
conversations = bucketA['conversations']

with open('benchmark_results/evidence_audit/v4_intervention_temporal.json', encoding='utf-8') as f:
    m25 = json.load(f)
target_qids = [r['question_id'] for r in m25['rows'] if r.get('group') == 'TARGET']

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAME_RE = "|".join(MONTHS)

def parse_date(date_str):
    if not date_str:
        return None
    m = re.search(r'(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})', date_str)
    if m:
        month = MONTHS.get(m.group(2).lower())
        if month:
            return (int(m.group(3)), month, int(m.group(1)))
    return None

def extract_turns_by_ids(locomo, conv_id, evidence_ids):
    """LoCoMoのevidence IDからターン本文を取得"""
    for conv in locomo:
        if conv.get('sample_id') == conv_id:
            conversation = conv.get('conversation', {})
            results = []
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
                            results.append({
                                'evidence_id': eid,
                                'dia_id': turn.get('dia_id', ''),
                                'session_date': session_date,
                                'speaker': turn.get('speaker', ''),
                                'text': turn.get('text', '')
                            })
            return results
    return []

def extract_normal_evidence(qid, case, conversations, gold):
    """M30-3の通常抽出ロジックを再現"""
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

# Phase 1: 7件の根拠付き再確認
print("=" * 80)
print("Phase 1: 7件の根拠付き再確認")
print("=" * 80)

unresolved_7 = [
    "lc_conv-30_7", "lc_conv-42_73", "lc_conv-47_48", "lc_conv-47_65",
    "lc_conv-48_50", "lc_conv-49_59", "lc_conv-50_36"
]

phase1_results = []
for qid in unresolved_7 + ["lc_conv-48_12"]:
    c = cases.get(qid)
    if not c:
        continue
    conv_id = c.get('source_conversation', '')
    question = c.get('question', '')
    gold = str(c.get('gold_answer', ''))
    
    # LoCoMoのQAとevidenceを取得
    locomo_qa = None
    for conv in locomo:
        if conv.get('sample_id') == conv_id:
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
    
    evidence_ids = locomo_qa.get('evidence', []) if locomo_qa else []
    evidence_turns = extract_turns_by_ids(locomo, conv_id, evidence_ids)
    
    # 通常の抽出結果
    normal_evidence = extract_normal_evidence(qid, c, conversations, gold)
    
    # 判定
    print(f"\n{'='*60}")
    print(f"QID: {qid}")
    print(f"質問: {question}")
    print(f"Gold: {gold}")
    print(f"Evidence IDs: {evidence_ids}")
    print(f"通常抽出: {len(normal_evidence)}件")
    
    if evidence_turns:
        for et in evidence_turns:
            print(f"\n  {et['evidence_id']} | {et['speaker']} | {et['session_date']}")
            print(f"  本文: {et['text'][:300]}")
            
            # 支持タイプの判定
            text_lower = et['text'].lower()
            session_date = et['session_date']
            
            # 明示的日付の確認
            explicit_date = None
            m = re.search(rf'\b(\d{{1,2}})\s+({MONTH_NAME_RE})\s+(\d{{4}})', text_lower)
            if m:
                explicit_date = (int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
            else:
                m = re.search(rf'\b({MONTH_NAME_RE})\s+(\d{{1,2}})\s+(\d{{4}})', text_lower)
                if m:
                    explicit_date = (int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
            
            # Gold日付
            gold_date = parse_date(gold)
            
            # 相対日付表現
            has_relative = bool(re.search(r'\b(next month|last month|this month|last year|next year|one year ago|yesterday|tomorrow)\b', text_lower))
            
            # 判定
            if explicit_date and gold_date and explicit_date == gold_date:
                support_type = 'EXPLICIT_SUPPORT'
                reason = f'明示的日付 {explicit_date} == Gold {gold_date}'
            elif explicit_date and gold_date and explicit_date[0] == gold_date[0] and explicit_date[1] == gold_date[1] and (gold_date[2] is None or explicit_date[2] == gold_date[2]):
                support_type = 'TEMPORAL_SUPPORT'
                reason = f'月+年一致: {explicit_date} ~= Gold {gold_date}'
            elif has_relative:
                # 相対日付を解決
                sd = parse_date(session_date)
                if sd:
                    sy, sm, sd_day = sd
                    resolved = None
                    if 'next month' in text_lower:
                        nm = sm + 1 if sm < 12 else 1
                        ny = sy + 1 if sm == 12 else sy
                        resolved = (ny, nm, None)
                    elif 'last month' in text_lower:
                        nm = sm - 1 if sm > 1 else 12
                        ny = sy - 1 if sm == 1 else sy
                        resolved = (ny, nm, None)
                    elif 'this month' in text_lower:
                        resolved = (sy, sm, None)
                    elif 'last year' in text_lower:
                        resolved = (sy - 1, None, None)
                    elif 'one year ago' in text_lower:
                        resolved = (sy - 1, None, None)
                    elif 'yesterday' in text_lower:
                        resolved = (sy, sm, sd_day - 1)
                    elif 'tomorrow' in text_lower:
                        resolved = (sy, sm, sd_day + 1)
                    
                    if resolved and gold_date:
                        if resolved[0] == gold_date[0] and (gold_date[1] is None or resolved[1] == gold_date[1]) and (gold_date[2] is None or resolved[2] == gold_date[2]):
                            support_type = 'TEMPORAL_SUPPORT'
                            reason = f'相対日付解決: {resolved} == Gold {gold_date} (session={session_date})'
                        else:
                            support_type = 'AMBIGUOUS'
                            reason = f'相対日付解決: {resolved} != Gold {gold_date}'
                    else:
                        support_type = 'AMBIGUOUS'
                        reason = f'相対日付表現あり: {has_relative} (session={session_date})'
                else:
                    support_type = 'AMBIGUOUS'
                    reason = f'相対日付表現あり: {has_relative} (session不明)'
            elif 'fort wayne' in text_lower and 'indiana' in gold.lower():
                support_type = 'WORLD_KNOWLEDGE_REQUIRED'
                reason = 'Fort Wayne → Indiana は地理的推論'
            else:
                # 別ターンを検索
                all_turns_in_conv = conversations.get(conv_id, [])
                found_other = False
                for i, t in enumerate(all_turns_in_conv):
                    if isinstance(t, dict):
                        text = t.get('text', '').lower()
                        gold_words = [w for w in re.sub(r'[^\w\s]', ' ', gold.lower()).split() if len(w) > 2]
                        if gold_words and all(w in text for w in gold_words):
                            found_other = True
                            break
                
                if found_other:
                    support_type = 'UNSUPPORTED_BUT_OTHER_EXISTS'
                    reason = '注釈EvidenceはGoldを支持しないが、別ターンが支持する'
                else:
                    support_type = 'UNSUPPORTED'
                    reason = '注釈Evidenceも別ターンもGoldを支持しない'
            
            print(f"  判定: {support_type}")
            print(f"  理由: {reason}")
    else:
        print("  Evidence: 見つからない")
    
    phase1_results.append({
        'question_id': qid,
        'question': question,
        'gold': gold,
        'evidence_ids': evidence_ids,
        'evidence_turns': evidence_turns,
        'normal_evidence_count': len(normal_evidence),
        'verdict': support_type if evidence_turns else 'INCONCLUSIVE',
        'reason': reason if evidence_turns else 'Evidence not found',
    })

# Phase 2: Readerへの実際の入力を確認
print(f"\n\n{'='*80}")
print("Phase 2: Readerへの実際の入力を確認")
print("=" * 80)

# M30-3のReader実行コードを確認
print("\n--- 通常パイプラインのReader入力 ---")
print("スクリプト: scripts/run_m30_3_evaluation.py")
print("Reader呼び出し: reader.answer(question, context)")
print("context = evidence_text + '\\n\\n' + question")
print("reference_dateはReaderに渡されない（scorerにのみ渡される）")

# M30-6のReader実行コードを確認
print("\n--- M30-6のReader入力 ---")
print("スクリプト: scripts/run_m30_6_evaluation.py")
print("contextに 'Reference date: {ref_date}' を追加")
print("相対日付の解釈指示を追加")

# 通常抽出が空の場合の確認
print("\n--- 通常抽出が空のケース ---")
for qid in unresolved_7:
    c = cases.get(qid)
    if not c:
        continue
    gold = str(c.get('gold_answer', ''))
    normal_evidence = extract_normal_evidence(qid, c, conversations, gold)
    print(f"  {qid}: {len(normal_evidence)}件")

# 保存
with open('results/m30_22_phase1_phase2.json', 'w', encoding='utf-8') as f:
    json.dump({
        'phase1_results': phase1_results,
        'phase2_notes': {
            'normal_pipeline': 'reference_dateはReaderに渡されない',
            'm30_6': 'Reference date + 相対日付解釈指示を追加',
            'empty_extraction_cases': [qid for qid in unresolved_7 if len(extract_normal_evidence(qid, cases.get(qid), conversations, str(cases.get(qid, {}).get('gold_answer', '')))) == 0],
        }
    }, f, indent=2, ensure_ascii=False)

print(f"\n結果を results/m30_22_phase1_phase2.json に保存しました")
