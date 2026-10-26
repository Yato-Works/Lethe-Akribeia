#!/usr/bin/env python3
"""
M30-11: Reader入力・証拠・基準日のトレース監査

対象4件: lc_conv-41_10, lc_conv-41_20, lc_conv-48_12, lc_conv-50_36
実装変更なし。トレース監査のみ。
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
    """M30-3と同じ証拠抽出ロジック"""
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

def trace_case(qid, all_cases, conversations, grader, m30_3_results, m30_6_results):
    """1件の全パイプラインをトレース"""
    case = all_cases.get(qid)
    if not case:
        return None
    
    question = case.get("question", "")
    gold = case.get("gold_answer", "")
    conv_key = case.get("source_conversation", "")
    ref_date = get_ref_date(qid)
    turns = conversations.get(conv_key, [])
    
    print(f"\n{'='*80}")
    print(f"QID: {qid}")
    print(f"{'='*80}")
    
    # 1. 元の質問とGold
    print(f"\n[1] 元の質問とGold")
    print(f"  質問: {question}")
    print(f"  Gold: {gold}")
    
    # 2. 元会話の該当箇所と前後の文脈
    print(f"\n[2] 元会話の該当箇所と前後の文脈")
    print(f"  会話キー: {conv_key}")
    print(f"  会話ターン数: {len(turns)}")
    
    # Goldのキーワードで会話を検索
    gold_lower = gold.lower().strip()
    gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
    gold_words = [w for w in gold_clean.split() if len(w) > 2]
    
    # 質問のキーワードで会話を検索
    question_lower = question.lower()
    question_clean = re.sub(r"[^\w\s]", " ", question_lower)
    question_words = [w for w in question_clean.split() if len(w) > 3]
    
    # 関連する発話を検索
    relevant_turns = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        for w in gold_words + question_words:
            if w in text_lower:
                relevant_turns.append((i, t))
                break
    
    print(f"  関連発話数: {len(relevant_turns)}件")
    for i, t in relevant_turns[:5]:
        print(f"    D{i+1} | {t.get('speaker', '?')} | {t.get('text', '')[:150]}")
    
    # 3. Goldを支持する具体的な証拠
    print(f"\n[3] Goldを支持する具体的な証拠")
    
    # 意味的な検索
    # 月名の略称
    month_map = {
        "january": ["jan", "january"],
        "february": ["feb", "february"],
        "march": ["mar", "march"],
        "april": ["apr", "april"],
        "may": ["may"],
        "june": ["jun", "june"],
        "july": ["jul", "july"],
        "august": ["aug", "august"],
        "september": ["sep", "sept", "september"],
        "october": ["oct", "october"],
        "november": ["nov", "november"],
        "december": ["dec", "december"],
    }
    
    # Goldから月名を抽出
    gold_months = []
    for month, aliases in month_map.items():
        for alias in aliases:
            if alias in gold_lower:
                gold_months.append(month)
                break
    
    # Goldから年を抽出
    year_match = re.search(r"\b(20\d{2})\b", gold)
    gold_year = year_match.group(1) if year_match else None
    
    # Goldから日を抽出
    day_match = re.search(r"\b(\d{1,2})\b", gold)
    gold_day = day_match.group(1) if day_match else None
    
    # 会話内でGoldの要素を検索
    evidence_for_gold = []
    if gold_months:
        for i, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            text = t.get("text", "")
            text_lower = text.lower()
            for month in gold_months:
                for alias in month_map[month]:
                    if alias in text_lower:
                        evidence_for_gold.append((f"D{i+1}", alias, text[:150]))
                        break
    
    if gold_year:
        for i, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            text = t.get("text", "")
            if gold_year in text.lower():
                evidence_for_gold.append((f"D{i+1}", gold_year, text[:150]))
    
    if gold_day:
        for i, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            text = t.get("text", "")
            if gold_day in text.lower():
                evidence_for_gold.append((f"D{i+1}", gold_day, text[:150]))
    
    # 重複を除去
    seen = set()
    unique_evidence = []
    for e in evidence_for_gold:
        k = (e[0], e[1])
        if k not in seen:
            seen.add(k)
            unique_evidence.append(e)
    
    print(f"  Gold要素: 月={gold_months}, 年={gold_year}, 日={gold_day}")
    print(f"  会話内での発見: {len(unique_evidence)}件")
    for e in unique_evidence[:5]:
        print(f"    {e[0]} | '{e[1]}' | {e[2]}")
    
    # 4. reference_dateの取得元と実際の値
    print(f"\n[4] reference_dateの取得元と実際の値")
    print(f"  取得元: SESSION_DATES[\"{conv_key}\"]")
    print(f"  値: {ref_date}")
    
    # 5. Readerへ渡された最終的な質問・証拠・コンテキスト
    print(f"\n[5] Readerへ渡された最終的な質問・証拠・コンテキスト")
    
    # M30-6のコンテキストを再構築
    evidence_turns = extract_evidence_turns(qid, case, conversations, gold)
    evidence_text = oracle_evidence_text(evidence_turns)
    
    # M30-6のプロンプト
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
    
    print(f"  証拠ターン数: {len(evidence_turns)}")
    print(f"  証拠テキスト（先頭500文字）:")
    print(f"    {evidence_text[:500]}")
    print(f"  コンテキスト（先頭500文字）:")
    print(f"    {context[:500]}")
    
    # 6. Readerの生出力
    print(f"\n[6] Readerの生出力")
    
    # M30-3の結果
    m30_3_result = next((r for r in m30_3_results if r["question_id"] == qid), None)
    m30_6_result = next((r for r in m30_6_results if r["question_id"] == qid), None)
    
    if m30_3_result:
        print(f"  M30-3（基準日なし）: '{m30_3_result['reader_text']}'")
    if m30_6_result:
        print(f"  M30-6（基準日あり）: '{m30_6_result['reader_text']}'")
    
    # 7. Scorerへ渡された回答と採点結果
    print(f"\n[7] Scorerへ渡された回答と採点結果")
    
    if m30_6_result:
        reader_text = m30_6_result["reader_text"]
        result = grader.grade(qid, "temporal-reasoning", reader_text, gold, reference_date=ref_date)
        
        # 正規化の過程
        if ref_date:
            norm_pred = grader._normalize_relative_date(reader_text, ref_date)
            norm_gold = grader._normalize_relative_date(gold, ref_date)
        else:
            norm_pred = reader_text
            norm_gold = gold
        
        pred_norm = grader._normalize_text(norm_pred)
        gold_norm = grader._normalize_text(norm_gold)
        pred_num = grader._normalize_number(pred_norm)
        gold_num = grader._normalize_number(gold_norm)
        
        print(f"  Reader出力: '{reader_text}'")
        print(f"  Gold: '{gold}'")
        print(f"  reference_date: {ref_date}")
        print(f"  _normalize_relative_date(pred): '{norm_pred}'")
        print(f"  _normalize_relative_date(gold): '{norm_gold}'")
        print(f"  _normalize_text(pred): '{pred_norm}'")
        print(f"  _normalize_text(gold): '{gold_norm}'")
        print(f"  _normalize_number(pred): '{pred_num}'")
        print(f"  _normalize_number(gold): '{gold_num}'")
        print(f"  採点結果: {result.is_correct}")
        print(f"  details: {result.details}")
    
    # 判定
    print(f"\n[判定]")
    
    # Goldの全要素が会話内に存在するか
    gold_elements = []
    if gold_months:
        gold_elements.append("月")
    if gold_year:
        gold_elements.append("年")
    if gold_day:
        gold_elements.append("日")
    
    evidence_elements = set()
    for e in unique_evidence:
        evidence_elements.add(e[1])
    
    all_elements_present = True
    if gold_months and not any(m in [e[1] for e in unique_evidence] for m in gold_months):
        all_elements_present = False
    if gold_year and gold_year not in evidence_elements:
        all_elements_present = False
    if gold_day and gold_day not in evidence_elements:
        all_elements_present = False
    
    # Readerが証拠を受け取っているか
    reader_received_evidence = len(evidence_turns) > 0
    
    # Readerが正しく回答できたか
    reader_correct = m30_6_result and m30_6_result["scorer_correct"]
    
    if all_elements_present and reader_correct:
        verdict = "SUPPORTED_AND_CORRECT"
    elif all_elements_present and not reader_correct:
        verdict = "SUPPORTED_BUT_WRONG_ANSWER"
    elif not all_elements_present and not reader_correct:
        verdict = "EVIDENCE_INSUFFICIENT"
    else:
        verdict = "INCONCLUSIVE"
    
    # 証拠の受け取りと回答能力を分離
    evidence_issue = not all_elements_present
    reader_issue = all_elements_present and not reader_correct
    
    print(f"  Gold要素: {gold_elements}")
    print(f"  証拠内発見: {evidence_elements}")
    print(f"  Gold全要素の証拠内存在: {all_elements_present}")
    print(f"  Readerが証拠を受け取った: {reader_received_evidence}")
    print(f"  Readerが正しく回答した: {reader_correct}")
    print(f"  証拠の問題: {evidence_issue}")
    print(f"  Readerの問題: {reader_issue}")
    print(f"  最終判定: {verdict}")
    
    return {
        "question_id": qid,
        "question": question,
        "gold": gold,
        "reference_date": ref_date,
        "evidence_turns_count": len(evidence_turns),
        "evidence_text_preview": evidence_text[:500],
        "reader_output_m30_6": m30_6_result["reader_text"] if m30_6_result else None,
        "reader_output_m30_3": m30_3_result["reader_text"] if m30_3_result else None,
        "scorer_correct": m30_6_result["scorer_correct"] if m30_6_result else None,
        "gold_elements": gold_elements,
        "evidence_elements": list(evidence_elements),
        "all_elements_present": all_elements_present,
        "reader_received_evidence": reader_received_evidence,
        "reader_correct": reader_correct,
        "evidence_issue": evidence_issue,
        "reader_issue": reader_issue,
        "verdict": verdict,
    }

def main():
    bucketA_path = Path("benchmark_results/evidence_audit/v4_bucketA_all.json")
    all_cases = load_all_cases(bucketA_path)
    conversations = load_conversations(bucketA_path)
    
    with open("results/m30_3_evaluation_results.json", encoding="utf-8") as f:
        m30_3_data = json.load(f)
    with open("results/m30_6_evaluation_results.json", encoding="utf-8") as f:
        m30_6_data = json.load(f)
    
    grader = OfficialLongMemEvalGrader()
    
    target_cases = ["lc_conv-41_10", "lc_conv-41_20", "lc_conv-48_12", "lc_conv-50_36"]
    
    audit_results = []
    for qid in target_cases:
        result = trace_case(qid, all_cases, conversations, grader, m30_3_data["results"], m30_6_data["results"])
        if result:
            audit_results.append(result)
    
    # サマリー
    print(f"\n\n{'='*80}")
    print("=== M30-11 トレース監査 サマリー ===")
    print(f"{'='*80}")
    
    for a in audit_results:
        print(f"\n  {a['question_id']}:")
        print(f"    Gold: {a['gold']}")
        print(f"    reference_date: {a['reference_date']}")
        print(f"    Gold要素: {a['gold_elements']}")
        print(f"    証拠内発見: {a['evidence_elements']}")
        print(f"    Reader出力: '{a['reader_output_m30_6']}'")
        print(f"    採点結果: {a['scorer_correct']}")
        print(f"    証拠の問題: {a['evidence_issue']}")
        print(f"    Readerの問題: {a['reader_issue']}")
        print(f"    最終判定: {a['verdict']}")
    
    # 保存
    with open("results/m30_11_trace_audit.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_cases": len(audit_results),
            "details": audit_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n監査結果を results/m30_11_trace_audit.json に保存しました")

if __name__ == "__main__":
    main()