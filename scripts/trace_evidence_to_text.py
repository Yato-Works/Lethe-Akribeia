#!/usr/bin/env python3
"""
M30-17: 28件のEvidence IDを本文まで追跡する

1. 元データ：Evidence IDに対応する原文とGoldとの関係を確認
2. 変換の各段階：原文が入力、選択済みターン、中間表現、最終コンテキストのどこまで残っているか確認
3. 欠落箇所の確定：最初に証拠が消えた段階を特定する
4. Reader再評価：原文の根拠が正しく渡る状態で回答を評価する
"""
import json
import re
import sys
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

def get_ref_date(qid):
    parts = qid.split("_")
    if len(parts) >= 2:
        conv_part = parts[0] + "_" + parts[1]
        conv_key = conv_part.replace("lc_", "")
        return SESSION_DATES.get(conv_key)
    return None

def load_all_cases(bucketA_path):
    with open(bucketA_path, encoding="utf-8") as f:
        data = json.load(f)
    return {c["question_id"]: c for c in data["cases"]}

def load_conversations(bucketA_path):
    with open(bucketA_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["conversations"]

def extract_evidence_turns(qid, case, conversations, gold):
    """通常の証拠抽出ロジック"""
    conv_key = case.get("source_conversation", "")
    turns = conversations.get(conv_key, [])
    if not isinstance(turns, list):
        return []
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
        elif any(kw in text_lower for kw in question_keywords if len(kw) > 3):
            if any(d in text_lower for d in ["month", "year", "2023", "2022", "2024", "indiana", "seoul", "france", "rome", "boston"]):
                evidence.append(f"D{i+1} | {t.get('speaker', '?')} | {text}")
    seen = set()
    unique = []
    for t in evidence:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[-8:]

def oracle_evidence_text(turns):
    if not turns:
        return ""
    return "\n".join(f"- {t}" for t in turns)

def find_evidence_in_locomo(locomo, conv_id, evidence_ids):
    """LoCoMoデータからevidence IDに対応する本文を検索"""
    for conv in locomo:
        if conv.get('sample_id') == conv_id:
            conversation = conv.get('conversation', {})
            results = []
            for eid in evidence_ids:
                # D7:5 形式を解析
                m = re.match(r'D(\d+):(\d+)', eid)
                if m:
                    session_num = m.group(1)
                    turn_num = m.group(2)
                    
                    # session_X を検索
                    session_key = f'session_{session_num}'
                    session = conversation.get(session_key, [])
                    
                    if isinstance(session, list):
                        # turn_num は 1始まり
                        turn_idx = int(turn_num) - 1
                        if 0 <= turn_idx < len(session):
                            turn = session[turn_idx]
                            if isinstance(turn, dict):
                                results.append({
                                    'evidence_id': eid,
                                    'dia_id': turn.get('dia_id', ''),
                                    'speaker': turn.get('speaker', '?'),
                                    'text': turn.get('text', '')
                                })
            return results
    return []

def main():
    bucketA_path = Path("benchmark_results/evidence_audit/v4_bucketA_all.json")
    m25_path = Path("benchmark_results/evidence_audit/v4_intervention_temporal.json")
    all_cases = load_all_cases(bucketA_path)
    conversations = load_conversations(bucketA_path)
    with open(m25_path) as f:
        m25 = json.load(f)
    target_qids = [r["question_id"] for r in m25["rows"] if r.get("group") == "TARGET"]
    assert len(target_qids) == 28

    # LoCoMoデータ
    with open(r'C:\Users\smily\AppData\Local\Temp\locomo10.json', encoding='utf-8') as f:
        locomo = json.load(f)

    grader = OfficialLongMemEvalGrader()
    reader = OllamaAnswerer(model="qwen2.5:7b-instruct")

    results = []
    for qid in target_qids:
        case = all_cases.get(qid)
        if not case:
            continue

        question = case.get("question", "")
        gold = str(case.get("gold_answer", ""))
        conv_id = case.get("source_conversation", "")
        ref_date = get_ref_date(qid)

        # LoCoMoのQAを検索
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
        locomo_evidence = find_evidence_in_locomo(locomo, conv_id, evidence_ids)

        # 通常の証拠抽出
        normal_evidence = extract_evidence_turns(qid, case, conversations, gold)
        normal_evidence_text = oracle_evidence_text(normal_evidence)

        # 欠落箇所の特定
        # LoCoMoのevidenceが通常の抽出に含まれているか
        locomo_evidence_texts = [e['text'] for e in locomo_evidence]
        normal_evidence_lower = normal_evidence_text.lower()
        
        missing_in_extraction = []
        for le in locomo_evidence:
            # LoCoMoのevidence本文が通常の抽出に含まれているか
            le_text_lower = le['text'].lower()
            # 部分一致で確認
            if le_text_lower[:50] not in normal_evidence_lower:
                missing_in_extraction.append(le)

        # 判定
        if not locomo_evidence:
            verdict = "EVIDENCE_NOT_FOUND_IN_LOCOMO"
        elif not missing_in_extraction:
            verdict = "EVIDENCE_PRESENT_IN_EXTRACTION"
        else:
            verdict = "EVIDENCE_MISSING_IN_EXTRACTION"

        results.append({
            "question_id": qid,
            "question": question,
            "gold": gold,
            "reference_date": ref_date,
            "locomo_evidence_ids": evidence_ids,
            "locomo_evidence": locomo_evidence,
            "normal_evidence_count": len(normal_evidence),
            "normal_evidence": normal_evidence,
            "missing_in_extraction": missing_in_extraction,
            "verdict": verdict,
        })

    # サマリー
    print(f"=== {len(results)}件のEvidence追跡結果 ===\n")

    verdict_counts = {}
    for r in results:
        verdict_counts[r['verdict']] = verdict_counts.get(r['verdict'], 0) + 1

    for verdict, count in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"  {verdict}: {count}件")

    print(f"\n詳細:")
    for r in results:
        print(f"\n  {r['question_id']}:")
        print(f"    Gold: {r['gold']}")
        print(f"    LoCoMo evidence_ids: {r['locomo_evidence_ids']}")
        for le in r['locomo_evidence']:
            print(f"      {le['evidence_id']} | {le['speaker']} | {le['text'][:100]}")
        print(f"    通常抽出: {r['normal_evidence_count']}件")
        print(f"    欠落: {len(r['missing_in_extraction'])}件")
        print(f"    判定: {r['verdict']}")

    # 保存
    with open("results/m30_17_evidence_trace.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_cases": len(results),
            "verdict_summary": verdict_counts,
            "details": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n結果を results/m30_17_evidence_trace.json に保存しました")

if __name__ == "__main__":
    main()