#!/usr/bin/env python3
"""M30-22 Phase 3: Reader比較評価

A. 通常パイプライン評価（M30-3の結果を再利用）
B. 根拠付き診断評価（正しいevidence本文 + セッション日時を明示）

553件のホールドアウトには一切アクセスしない。
"""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.llm import OllamaAnswerer
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

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}

def get_ref_date(qid):
    # lc_conv-26_63 -> conv-26
    parts = qid.split("_")
    if len(parts) >= 3 and parts[0] == "lc":
        return SESSION_DATES.get(parts[1])
    return None

with open(r'C:\Users\smily\AppData\Local\Temp\locomo10.json', encoding='utf-8') as f:
    locomo = json.load(f)

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

def get_locomo_evidence_text(qid):
    """LoCoMoのevidence本文とセッション日時を取得"""
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
                evidence_texts = []
                for eid in locomo_qa.get('evidence', []):
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
                                evidence_texts.append(f"[{eid} | {session_date}] {turn.get('speaker')}: {turn.get('text', '')}")
                return evidence_texts, locomo_qa.get('evidence', [])
    return None, None

# Phase 3: Reader比較評価
grader = OfficialLongMemEvalGrader()
reader = OllamaAnswerer(model="qwen2.5:7b-instruct")

print("=" * 80)
print("Phase 3: Reader比較評価")
print("=" * 80)

results = []
for qid in target_qids:
    c = cases.get(qid)
    if not c:
        continue
    question = c.get('question', '')
    gold = str(c.get('gold_answer', ''))
    ref_date = get_ref_date(qid)
    
    # A. 通常パイプライン（M30-3の結果を再利用）
    m30_3 = m30_3_results.get(qid, {})
    normal_output = m30_3.get('reader_text', '')
    normal_correct = m30_3.get('scorer_correct', False)
    
    # B. 根拠付き診断評価
    locomo_evidence_texts, evidence_ids = get_locomo_evidence_text(qid)
    
    if locomo_evidence_texts:
        evidence_text = '\n'.join(f'- {t}' for t in locomo_evidence_texts)
        if ref_date:
            context = f"Reference date: {ref_date}\n\n{evidence_text}\n\n{question}"
        else:
            context = f"{evidence_text}\n\n{question}"
    else:
        # LoCoMo evidenceが使えない場合は通常抽出を使う
        normal_evidence = extract_normal_evidence(qid, c, conversations, gold)
        evidence_text = '\n'.join(f'- {t}' for t in normal_evidence)
        context = f"{evidence_text}\n\n{question}"
    
    try:
        response = reader.answer(question, context)
        diagnostic_output = response.text.strip()
    except Exception as exc:
        diagnostic_output = f"ERROR: {exc}"
    
    diag_result = grader.grade(qid, "temporal-reasoning", diagnostic_output, gold, reference_date=ref_date)
    diag_correct = diag_result.is_correct
    
    change = "improved" if diag_correct and not normal_correct else ("regressed" if not diag_correct and normal_correct else "unchanged")
    
    print(f"\n[{qid}]")
    print(f"  Gold: {gold}")
    print(f"  A. 通常: '{normal_output[:60]}' → {normal_correct}")
    print(f"  B. 診断: '{diagnostic_output[:60]}' → {diag_correct}")
    print(f"  変化: {change}")
    
    results.append({
        'question_id': qid,
        'gold': gold,
        'normal_output': normal_output,
        'normal_correct': normal_correct,
        'diagnostic_output': diagnostic_output,
        'diagnostic_correct': diag_correct,
        'change': change,
        'evidence_ids': evidence_ids,
    })

# サマリー
improved = sum(1 for r in results if r['change'] == 'improved')
regressed = sum(1 for r in results if r['change'] == 'regressed')
unchanged_correct = sum(1 for r in results if r['normal_correct'] and r['diagnostic_correct'])
unchanged_wrong = sum(1 for r in results if not r['normal_correct'] and not r['diagnostic_correct'])

print(f"\n=== サマリー ===")
print(f"  改善（誤→正）: {improved}件")
print(f"  悪化（正→誤）: {regressed}件")
print(f"  不変（正→正）: {unchanged_correct}件")
print(f"  不変（誤→誤）: {unchanged_wrong}件")

# 保存
with open('results/m30_22_reader_comparison.json', 'w', encoding='utf-8') as f:
    json.dump({
        'summary': {
            'improved': improved,
            'regressed': regressed,
            'unchanged_correct': unchanged_correct,
            'unchanged_wrong': unchanged_wrong,
        },
        'results': results,
    }, f, indent=2, ensure_ascii=False)

print(f"\n結果を results/m30_22_reader_comparison.json に保存しました")
