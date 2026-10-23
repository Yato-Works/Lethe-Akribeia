#!/usr/bin/env python3
"""
M30-12: 証拠欠落地点の特定

Phase A: 元会話全体を確認し、Goldを支持する情報が存在するか判定
Phase B: パイプラインの追跡（どの段階で情報が失われたか）
Phase C: Readerの切り分け（支持証拠を明示的に含めた診断用コンテキスト）
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

MONTH_MAP = {
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

def search_full_conversation(turns, gold, question):
    """元会話全体からGoldの要素を検索"""
    gold_lower = gold.lower().strip()
    gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
    gold_words = [w for w in gold_clean.split() if len(w) > 2]
    
    # 月名を抽出
    gold_months = []
    for month, aliases in MONTH_MAP.items():
        for alias in aliases:
            if alias in gold_lower:
                gold_months.append(month)
                break
    
    # 年を抽出
    year_match = re.search(r"\b(20\d{2})\b", gold)
    gold_year = year_match.group(1) if year_match else None
    
    # 日を抽出
    day_match = re.search(r"\b(\d{1,2})\b", gold)
    gold_day = day_match.group(1) if day_match else None
    
    # 質問のキーワード
    question_lower = question.lower()
    question_clean = re.sub(r"[^\w\s]", " ", question_lower)
    question_words = [w for w in question_clean.split() if len(w) > 3]
    stopwords = {"when", "what", "where", "which", "who", "how", "did", "does", "was", "were", "are", "the", "and", "for", "with", "from", "that", "this", "have", "has", "had", "been", "being", "will", "would", "could", "should", "about", "into", "onto", "upon", "over", "under", "between", "through", "during", "before", "after", "above", "below", "among", "within", "without", "against", "toward", "towards"}
    question_keywords = [w for w in question_words if w not in stopwords]
    
    # 会話全体を検索
    results = {
        "gold_month_evidence": [],
        "gold_year_evidence": [],
        "gold_day_evidence": [],
        "question_keyword_evidence": [],
        "all_gold_elements_found": False,
        "gold_elements": {
            "months": gold_months,
            "year": gold_year,
            "day": gold_day,
        }
    }
    
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        
        # 月名で検索
        for month in gold_months:
            for alias in MONTH_MAP[month]:
                if alias in text_lower:
                    results["gold_month_evidence"].append({
                        "turn": f"D{i+1}",
                        "pattern": alias,
                        "text": text[:200]
                    })
                    break
        
        # 年で検索
        if gold_year and gold_year in text_lower:
            results["gold_year_evidence"].append({
                "turn": f"D{i+1}",
                "pattern": gold_year,
                "text": text[:200]
            })
        
        # 日で検索
        if gold_day and gold_day in text_lower:
            results["gold_day_evidence"].append({
                "turn": f"D{i+1}",
                "pattern": gold_day,
                "text": text[:200]
            })
        
        # 質問キーワードで検索
        for w in question_keywords:
            if w in text_lower:
                results["question_keyword_evidence"].append({
                    "turn": f"D{i+1}",
                    "pattern": w,
                    "text": text[:200]
                })
    
    # 重複を除去
    for key in results:
        if isinstance(results[key], list):
            seen = set()
            unique = []
            for r in results[key]:
                k = (r["turn"], r["pattern"])
                if k not in seen:
                    seen.add(k)
                    unique.append(r)
            results[key] = unique
    
    # Gold全要素の有無を判定
    has_month = len(results["gold_month_evidence"]) > 0 if gold_months else True
    has_year = len(results["gold_year_evidence"]) > 0 if gold_year else True
    has_day = len(results["gold_day_evidence"]) > 0 if gold_day else True
    
    results["all_gold_elements_found"] = has_month and has_year and has_day
    
    return results

def build_diagnostic_context(turns, gold, question, ref_date, support_evidence):
    """支持証拠を明示的に含めた診断用コンテキストを構築"""
    # 支持証拠のターンを抽出
    support_turn_ids = set()
    for e in support_evidence:
        turn_id = e["turn"]  # "D123"
        turn_num = int(turn_id[1:])  # 123
        support_turn_ids.add(turn_num)
    
    # 前後のターンも含める（文脈のため）
    context_turn_ids = set()
    for tid in support_turn_ids:
        for offset in [-2, -1, 0, 1, 2]:
            context_turn_ids.add(tid + offset)
    
    # 証拠テキストを構築
    evidence_lines = []
    for tid in sorted(context_turn_ids):
        if 1 <= tid <= len(turns):
            t = turns[tid - 1]
            if isinstance(t, dict):
                speaker = t.get("speaker", "?")
                text = t.get("text", "")
                evidence_lines.append(f"D{tid} | {speaker} | {text}")
    
    evidence_text = "\n".join(evidence_lines)
    
    # 診断用コンテキスト
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
    
    return context, evidence_text

def main():
    bucketA_path = Path("benchmark_results/evidence_audit/v4_bucketA_all.json")
    all_cases = load_all_cases(bucketA_path)
    conversations = load_conversations(bucketA_path)
    
    with open("results/m30_6_evaluation_results.json", encoding="utf-8") as f:
        m30_6_data = json.load(f)
    
    grader = OfficialLongMemEvalGrader()
    reader = OllamaAnswerer(model="qwen2.5:7b-instruct")
    
    target_cases = ["lc_conv-41_10", "lc_conv-41_20", "lc_conv-48_12", "lc_conv-50_36"]
    
    audit_results = []
    
    for qid in target_cases:
        case = all_cases.get(qid)
        if not case:
            continue
        
        question = case.get("question", "")
        gold = case.get("gold_answer", "")
        conv_key = case.get("source_conversation", "")
        ref_date = get_ref_date(qid)
        turns = conversations.get(conv_key, [])
        
        print(f"\n{'='*80}")
        print(f"QID: {qid}")
        print(f"{'='*80}")
        
        # Phase A: 元会話全体の確認
        print(f"\n--- Phase A: 元会話全体の確認 ---")
        
        full_search = search_full_conversation(turns, gold, question)
        
        print(f"  Gold要素: {full_search['gold_elements']}")
        print(f"  月名証拠: {len(full_search['gold_month_evidence'])}件")
        for e in full_search["gold_month_evidence"][:3]:
            print(f"    {e['turn']} | '{e['pattern']}' | {e['text'][:100]}")
        print(f"  年証拠: {len(full_search['gold_year_evidence'])}件")
        for e in full_search["gold_year_evidence"][:3]:
            print(f"    {e['turn']} | '{e['pattern']}' | {e['text'][:100]}")
        print(f"  日証拠: {len(full_search['gold_day_evidence'])}件")
        for e in full_search["gold_day_evidence"][:3]:
            print(f"    {e['turn']} | '{e['pattern']}' | {e['text'][:100]}")
        print(f"  質問キーワード証拠: {len(full_search['question_keyword_evidence'])}件")
        print(f"  Gold全要素発見: {full_search['all_gold_elements_found']}")
        
        # Phase B: パイプラインの追跡
        print(f"\n--- Phase B: パイプラインの追跡 ---")
        
        # 通常の証拠抽出
        normal_evidence = extract_evidence_turns(qid, case, conversations, gold)
        normal_evidence_text = "\n".join(f"- {t}" for t in normal_evidence)
        
        print(f"  通常抽出: {len(normal_evidence)}件")
        for e in normal_evidence:
            print(f"    {e[:150]}")
        
        # 支持証拠を特定
        support_evidence = []
        if full_search["all_gold_elements_found"]:
            # Goldの全要素が存在する場合、それらの証拠を統合
            support_evidence = (
                full_search["gold_month_evidence"][:2] +
                full_search["gold_year_evidence"][:2] +
                full_search["gold_day_evidence"][:2]
            )
        
        # 欠落地点の特定
        missing_in_extraction = []
        if full_search["all_gold_elements_found"]:
            # 支持証拠が通常の抽出に含まれているか確認
            support_turn_ids = set(e["turn"] for e in support_evidence)
            normal_turn_ids = set()
            for e in normal_evidence:
                # "D123 | ..." から "D123" を抽出
                match = re.match(r"(D\d+)", e)
                if match:
                    normal_turn_ids.add(match.group(1))
            
            missing_in_extraction = support_turn_ids - normal_turn_ids
        
        print(f"  支持証拠: {len(support_evidence)}件")
        print(f"  通常抽出に欠落: {len(missing_in_extraction)}件")
        if missing_in_extraction:
            print(f"    欠落ターン: {missing_in_extraction}")
        
        # Phase C: Readerの切り分け
        print(f"\n--- Phase C: Readerの切り分け ---")
        
        # 通常のコンテキスト
        if ref_date:
            normal_context = (
                f"Reference date: {ref_date}\n\n"
                f"When the question requires interpreting a relative date, "
                f"use the reference date and the supplied evidence to resolve it.\n"
                f"Convert relative dates to absolute dates when the evidence supports the conversion. "
                f"Do not invent missing facts.\n"
                f"If the evidence is insufficient, abstain.\n\n"
                f"{normal_evidence_text}\n\n{question}"
            )
        else:
            normal_context = f"{normal_evidence_text}\n\n{question}"
        
        # 通常のReader出力
        m30_6_result = next((r for r in m30_6_data["results"] if r["question_id"] == qid), None)
        normal_reader_output = m30_6_result["reader_text"] if m30_6_result else None
        
        print(f"  通常Reader出力: '{normal_reader_output}'")
        
        # 診断用コンテキスト（支持証拠を明示的に含める）
        diagnostic_reader_output = None
        diagnostic_context = None
        
        if full_search["all_gold_elements_found"] and support_evidence:
            diagnostic_context, diagnostic_evidence_text = build_diagnostic_context(
                turns, gold, question, ref_date, support_evidence
            )
            
            # 診断用Reader実行
            try:
                response = reader.answer(question, diagnostic_context)
                diagnostic_reader_output = response.text.strip()
                print(f"  診断用Reader出力: '{diagnostic_reader_output}'")
                
                # 診断用出力をscorerで採点
                diag_result = grader.grade(qid, "temporal-reasoning", diagnostic_reader_output, gold, reference_date=ref_date)
                print(f"  診断用採点結果: {diag_result.is_correct}")
            except Exception as exc:
                print(f"  診断用Reader実行エラー: {exc}")
                diagnostic_reader_output = f"ERROR: {exc}"
        else:
            print(f"  診断用コンテキスト: 作成跳过（Gold全要素が会話に存在しない）")
        
        # 判定
        if not full_search["all_gold_elements_found"]:
            verdict = "NOT_IN_CONVERSATION"
            missing_location = "N/A（会話に証拠なし）"
        elif missing_in_extraction:
            verdict = "LOST_IN_EXTRACTION"
            missing_location = f"証拠抽出段階で欠落: {missing_in_extraction}"
        elif diagnostic_reader_output and "don't know" not in diagnostic_reader_output.lower():
            verdict = "READER_CAN_ANSWER_WITH_EVIDENCE"
            missing_location = "Readerの問題（証拠はあるが回答できない）"
        else:
            verdict = "INCONCLUSIVE"
            missing_location = "不明"
        
        print(f"\n  判定: {verdict}")
        print(f"  欠落地点: {missing_location}")
        
        audit_results.append({
            "question_id": qid,
            "question": question,
            "gold": gold,
            "reference_date": ref_date,
            "phase_a": {
                "gold_elements": full_search["gold_elements"],
                "gold_month_evidence": full_search["gold_month_evidence"][:3],
                "gold_year_evidence": full_search["gold_year_evidence"][:3],
                "gold_day_evidence": full_search["gold_day_evidence"][:3],
                "question_keyword_evidence_count": len(full_search["question_keyword_evidence"]),
                "all_gold_elements_found": full_search["all_gold_elements_found"],
            },
            "phase_b": {
                "normal_evidence_count": len(normal_evidence),
                "normal_evidence": normal_evidence,
                "support_evidence": support_evidence,
                "missing_in_extraction": list(missing_in_extraction),
            },
            "phase_c": {
                "normal_reader_output": normal_reader_output,
                "diagnostic_reader_output": diagnostic_reader_output,
                "diagnostic_context": diagnostic_context,
            },
            "verdict": verdict,
            "missing_location": missing_location,
        })
    
    # サマリー
    print(f"\n\n{'='*80}")
    print("=== M30-12 証拠欠落地点特定 サマリー ===")
    print(f"{'='*80}")
    
    for a in audit_results:
        print(f"\n  {a['question_id']}:")
        print(f"    Gold: {a['gold']}")
        print(f"    Phase A: Gold全要素発見={a['phase_a']['all_gold_elements_found']}")
        print(f"    Phase B: 通常抽出={a['phase_b']['normal_evidence_count']}件, 欠落={len(a['phase_b']['missing_in_extraction'])}件")
        print(f"    Phase C: 通常出力='{a['phase_c']['normal_reader_output']}'")
        print(f"    Phase C: 診断用出力='{a['phase_c']['diagnostic_reader_output']}'")
        print(f"    判定: {a['verdict']}")
        print(f"    欠落地点: {a['missing_location']}")
    
    # 保存
    with open("results/m30_12_missing_location.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_cases": len(audit_results),
            "details": audit_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n監査結果を results/m30_12_missing_location.json に保存しました")

if __name__ == "__main__":
    main()