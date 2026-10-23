#!/usr/bin/env python3
"""
M30-6: プロンプトに相対日付解釈指示を追加して再評価

変更点: Readerのプロンプトに相対日付の解釈指示を追加
変更前: M30-5の結果（results/m30_5_evaluation_results.json）
変更後: このスクリプトの結果（results/m30_6_evaluation_results.json）

scorerは変更しない。証拠テキスト、質問、モデル、採点基準はM30-5と同じ。
"""
import json
import sys
import re
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

def classify_reader_output(reader_text, gold, ref_date, evidence_text):
    """
    Readerの出力を3段階で分類:
    - date_conversion_success: 相対日付を正しい絶対日付に変換できた
    - evidence_citation_or_abstain: 証拠を返しただけ、または回答を拒否した
    - wrong_conversion: 絶対日付を出したが、基準日や証拠と一致しない
    """
    reader_lower = reader_text.lower().strip()
    gold_lower = gold.lower().strip()
    
    # 棄権
    if "don't know" in reader_lower or "i don't know" in reader_lower:
        return "evidence_citation_or_abstain"
    
    # 証拠の引用（Readerの出力が証拠文の一部と完全一致するか）
    evidence_lower = evidence_text.lower()
    if reader_lower in evidence_lower:
        return "evidence_citation_or_abstain"
    
    # 日付変換の成功判定
    # Readerの出力がgoldと一致するか（scorerで判定）
    grader = OfficialLongMemEvalGrader()
    result = grader.grade("temp", "temporal-reasoning", reader_text, gold, reference_date=ref_date)
    if result.is_correct:
        return "date_conversion_success"
    
    # 誤変換: 絶対日付を出したがgoldと一致しない
    # 月名や年が含まれているか
    has_date = bool(re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|20\d{2})\b", reader_lower))
    if has_date:
        return "wrong_conversion"
    
    # 相対日付のまま
    has_relative = bool(re.search(r"\b(next month|last month|this month|last year|next year|one year ago|years ago)\b", reader_lower))
    if has_relative:
        return "wrong_conversion"  # 相対日付のまま = 変換失敗
    
    return "other"

def main():
    # 変更前の結果を読み込み
    with open("results/m30_5_evaluation_results.json", encoding="utf-8") as f:
        before_data = json.load(f)
    before_results = {r["question_id"]: r for r in before_data["results"]}
    
    # データ読み込み
    bucketA_path = Path("benchmark_results/evidence_audit/v4_bucketA_all.json")
    m25_path = Path("benchmark_results/evidence_audit/v4_intervention_temporal.json")
    all_cases = load_all_cases(bucketA_path)
    conversations = load_conversations(bucketA_path)
    with open(m25_path) as f:
        m25 = json.load(f)
    target_qids = [r["question_id"] for r in m25["rows"] if r.get("group") == "TARGET"]
    assert len(target_qids) == 28
    
    grader = OfficialLongMemEvalGrader()
    reader = OllamaAnswerer(model="qwen2.5:7b-instruct")
    
    results = []
    summary = {"correct": 0, "total": 0, "generation_failure": 0, "scorer_mismatch": 0, "evidence_problem": 0}
    
    for qid in target_qids:
        case = all_cases.get(qid)
        if not case:
            continue
        
        question = case.get("question", "")
        gold = case.get("gold_answer", "")
        conv_key = case.get("source_conversation", "")
        ref_date = get_ref_date(qid)
        
        evidence_turns = extract_evidence_turns(qid, case, conversations, gold)
        evidence_text = oracle_evidence_text(evidence_turns)
        
        # M30-6の変更点: プロンプトに相対日付の解釈指示を追加
        if ref_date:
            context = (
                f"Reference date: {ref_date}\n\n"
                f"When the question requires interpreting a relative date, "
                f"use the reference date and the supplied evidence to resolve it.\n"
                f"Convert relative dates to absolute dates when the evidence supports the conversion. "
                f"Do not invent missing facts.\n"
                f"If the evidence is insufficient, abstain.\n\n"
                f"{evidence_text}\n\n{question}"
            )
        else:
            context = f"{evidence_text}\n\n{question}"
        
        try:
            response = reader.answer(question, context)
            reader_text = response.text.strip()
        except Exception as exc:
            reader_text = f"ERROR: {exc}"
            summary["generation_failure"] += 1
            results.append({
                "question_id": qid,
                "model": "qwen2.5:7b-instruct",
                "question": question,
                "gold": gold,
                "reference_date": ref_date,
                "evidence_text": evidence_text,
                "reader_text": reader_text,
                "reader_correct": False,
                "scorer_correct": False,
                "verdict": "generation_failure",
                "output_category": "generation_failure",
                "note": str(exc)
            })
            continue
        
        # scorerで採点（M30-5と同じscorer、修正しない）
        result = grader.grade(qid, "temporal-reasoning", reader_text, gold, reference_date=ref_date)
        scorer_correct = result.is_correct
        
        gold_lower = gold.lower().strip()
        reader_lower = reader_text.lower().strip()
        reader_correct = gold_lower in reader_lower or reader_lower in gold_lower
        
        # 3段階分類
        output_category = classify_reader_output(reader_text, gold, ref_date, evidence_text)
        
        if scorer_correct:
            summary["correct"] += 1
            verdict = "correct"
        else:
            if reader_correct and not scorer_correct:
                summary["scorer_mismatch"] += 1
                verdict = "scorer_mismatch"
            else:
                summary["evidence_problem"] += 1
                verdict = "evidence_or_processing"
        
        summary["total"] += 1
        results.append({
            "question_id": qid,
            "model": "qwen2.5:7b-instruct",
            "question": question,
            "gold": gold,
            "reference_date": ref_date,
            "evidence_text": evidence_text,
            "reader_text": reader_text,
            "reader_correct": reader_correct,
            "scorer_correct": scorer_correct,
            "verdict": verdict,
            "output_category": output_category,
            "note": ""
        })
        
        # 変更前後の比較
        before = before_results.get(qid, {})
        before_correct = before.get("scorer_correct", False)
        change = "improved" if scorer_correct and not before_correct else ("regressed" if not scorer_correct and before_correct else "unchanged")
        print(f"[{qid}] ref={ref_date} | reader='{reader_text[:50]}' | gold='{gold[:30]}' | correct={scorer_correct} | category={output_category} | change={change}")
    
    # 変更前後の比較サマリー
    improved = sum(1 for r in results if r["scorer_correct"] and not before_results.get(r["question_id"], {}).get("scorer_correct", False))
    regressed = sum(1 for r in results if not r["scorer_correct"] and before_results.get(r["question_id"], {}).get("scorer_correct", False))
    unchanged_correct = sum(1 for r in results if r["scorer_correct"] and before_results.get(r["question_id"], {}).get("scorer_correct", False))
    unchanged_wrong = sum(1 for r in results if not r["scorer_correct"] and not before_results.get(r["question_id"], {}).get("scorer_correct", False))
    
    # 3段階分類のサマリー
    category_counts = {}
    for r in results:
        cat = r.get("output_category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    accuracy = summary["correct"] / summary["total"] if summary["total"] > 0 else 0.0
    
    print(f"\n=== M30-6 評価サマリー ===")
    print(f"総件数: {summary['total']}")
    print(f"正解: {summary['correct']} ({accuracy:.1%})")
    print(f"生成失敗: {summary['generation_failure']}")
    print(f"scorer不一致: {summary['scorer_mismatch']}")
    print(f"証拠/処理問題: {summary['evidence_problem']}")
    print(f"\n変更前後比較:")
    print(f"  改善（誤→正）: {improved}件")
    print(f"  悪化（正→誤）: {regressed}件")
    print(f"  不変（正→正）: {unchanged_correct}件")
    print(f"  不変（誤→誤）: {unchanged_wrong}件")
    print(f"\n3段階分類:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}件")
    
    # 保存
    with open("results/m30_6_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "accuracy": accuracy,
            "change_comparison": {
                "improved": improved,
                "regressed": regressed,
                "unchanged_correct": unchanged_correct,
                "unchanged_wrong": unchanged_wrong,
            },
            "output_category_summary": category_counts,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果を results/m30_6_evaluation_results.json に保存しました")

if __name__ == "__main__":
    main()