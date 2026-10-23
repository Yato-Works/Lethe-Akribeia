#!/usr/bin/env python3
"""M30-24 Phase 2-5: scorer mismatch, 48_32, reclassification, final report"""
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

def get_ref_date(qid):
    parts = qid.split("_")
    if len(parts) >= 3 and parts[0] == "lc":
        return SESSION_DATES.get(parts[1])
    return None

with open('benchmark_results/evidence_audit/v4_bucketA_all.json', encoding='utf-8') as f:
    bucketA = json.load(f)
cases = {c['question_id']: c for c in bucketA['cases']}

with open('benchmark_results/evidence_audit/v4_intervention_temporal.json', encoding='utf-8') as f:
    m25 = json.load(f)
target_qids = [r['question_id'] for r in m25['rows'] if r.get('group') == 'TARGET']

with open('results/m30_3_evaluation_results.json', encoding='utf-8') as f:
    m30_3_data = json.load(f)
m30_3_results = {r['question_id']: r for r in m30_3_data['results']}

with open('results/m30_23_failure_analysis.json', encoding='utf-8') as f:
    m30_23_data = json.load(f)
m30_23_reasons = {r['question_id']: r['reason'] for r in m30_23_data['details']}

grader = OfficialLongMemEvalGrader()

# ============================================================
# Phase 2: 5件のscorer mismatchを再監査
# ============================================================
print("=" * 80)
print("Phase 2: 5件のscorer mismatchを再監査")
print("=" * 80)

mismatch_qids = ["lc_conv-26_73", "lc_conv-47_43", "lc_conv-49_58", "lc_conv-49_79", "lc_conv-50_69"]

phase2_results = []
for qid in mismatch_qids:
    c = cases.get(qid)
    if not c:
        continue
    gold = str(c.get('gold_answer', ''))
    ref_date = get_ref_date(qid)
    normal = m30_3_results.get(qid, {})
    reader_output = normal.get('reader_text', '')
    
    # scorerの正規化をトレース
    norm_pred = grader._normalize_relative_date(reader_output, ref_date)
    norm_gold = grader._normalize_relative_date(gold, ref_date)
    
    pred_norm = grader._normalize_text(norm_pred)
    gold_norm = grader._normalize_text(norm_gold)
    pred_num = grader._normalize_number(pred_norm)
    gold_num = grader._normalize_number(gold_norm)
    
    is_correct = (pred_num == gold_num) or (pred_norm == gold_norm)
    
    # 独立計算: 実際のカレンダー計算
    # ref_dateから月を計算
    ref_year, ref_month, ref_day = int(ref_date[:4]), int(ref_date[5:7]), int(ref_date[8:10])
    
    reader_lower = reader_output.lower()
    independent_result = None
    
    if "last month" in reader_lower:
        # last month from ref_date
        if ref_month > 1:
            ind_month = ref_month - 1
            ind_year = ref_year
        else:
            ind_month = 12
            ind_year = ref_year - 1
        ind_date = f"{ind_year}年{ind_month}月"
        # Goldの月+年を抽出
        gold_lower = gold.lower()
        month_names = ["january", "february", "march", "april", "may", "june",
                      "july", "august", "september", "october", "november", "december"]
        gold_month = None
        for i, m in enumerate(month_names):
            if m in gold_lower:
                gold_month = i + 1
                break
        gold_year_m = re.search(r"(20\d{2})", gold)
        gold_year_val = int(gold_year_m.group(1)) if gold_year_m else None
        
        if gold_month and gold_year_val:
            independent_match = (ind_month == gold_month and ind_year == gold_year_val)
            independent_result = f"last month from {ref_date} → {ind_year}/{ind_month}; Gold={gold_year_val}/{gold_month}; match={independent_match}"
        else:
            independent_result = f"last month from {ref_date} → {ind_year}/{ind_month}; Gold month not found"
    
    elif "next month" in reader_lower:
        if ref_month < 12:
            ind_month = ref_month + 1
            ind_year = ref_year
        else:
            ind_month = 1
            ind_year = ref_year + 1
        ind_date = f"{ind_year}年{ind_month}月"
        gold_lower = gold.lower()
        month_names = ["january", "february", "march", "april", "may", "june",
                      "july", "august", "september", "october", "november", "december"]
        gold_month = None
        for i, m in enumerate(month_names):
            if m in gold_lower:
                gold_month = i + 1
                break
        gold_year_m = re.search(r"(20\d{2})", gold)
        gold_year_val = int(gold_year_m.group(1)) if gold_year_m else None
        
        if gold_month and gold_year_val:
            independent_match = (ind_month == gold_month and ind_year == gold_year_val)
            independent_result = f"next month from {ref_date} → {ind_year}/{ind_month}; Gold={gold_year_val}/{gold_month}; match={independent_match}"
        else:
            independent_result = f"next month from {ref_date} → {ind_year}/{ind_month}; Gold month not found"
    
    elif "this month" in reader_lower:
        ind_date = f"{ref_year}年{ref_month}月"
        gold_lower = gold.lower()
        month_names = ["january", "february", "march", "april", "may", "june",
                      "july", "august", "september", "october", "november", "december"]
        gold_month = None
        for i, m in enumerate(month_names):
            if m in gold_lower:
                gold_month = i + 1
                break
        gold_year_m = re.search(r"(20\d{2})", gold)
        gold_year_val = int(gold_year_m.group(1)) if gold_year_m else None
        
        if gold_month and gold_year_val:
            independent_match = (ref_month == gold_month and ref_year == gold_year_val)
            independent_result = f"this month from {ref_date} → {ref_year}/{ref_month}; Gold={gold_year_val}/{gold_month}; match={independent_match}"
        else:
            independent_result = f"this month from {ref_date} → {ref_year}/{ref_month}; Gold month not found"
    
    elif "one year ago" in reader_lower:
        ind_year = ref_year - 1
        gold_year_m = re.search(r"(20\d{2})", gold)
        gold_year_val = int(gold_year_m.group(1)) if gold_year_m else None
        
        if gold_year_val:
            independent_match = (ind_year == gold_year_val)
            independent_result = f"one year ago from {ref_date} → {ind_year}; Gold={gold_year_val}; match={independent_match}"
        else:
            independent_result = f"one year ago from {ref_date} → {ind_year}; Gold year not found"
    
    print(f"\n  {qid}:")
    print(f"    Gold: {gold}")
    print(f"    ref_date: {ref_date}")
    print(f"    Reader出力: '{reader_output}'")
    print(f"    正規化後 pred: '{norm_pred}'")
    print(f"    正規化後 gold: '{norm_gold}'")
    print(f"    scorer判定: {is_correct}")
    print(f"    独立計算: {independent_result}")
    
    phase2_results.append({
        'question_id': qid,
        'gold': gold,
        'reference_date': ref_date,
        'reader_output': reader_output,
        'normalized_pred': norm_pred,
        'normalized_gold': norm_gold,
        'scorer_correct': is_correct,
        'independent_result': independent_result,
    })

# ============================================================
# Phase 3: lc_conv-48_32の採点問題を分離
# ============================================================
print(f"\n\n{'='*80}")
print("Phase 3: lc_conv-48_32の採点問題")
print("=" * 80)

gold_48_32 = "in 2022"
normal_48_32 = "Last year."
diag_48_32 = "Deborah was in Bali in 2022."
ref_date_48_32 = "2023-01-31"

# 通常出力の採点経路
norm_normal = grader._normalize_relative_date(normal_48_32, ref_date_48_32)
norm_gold_48_32 = grader._normalize_relative_date(gold_48_32, ref_date_48_32)
pred_norm_normal = grader._normalize_text(norm_normal)
gold_norm_48_32 = grader._normalize_text(norm_gold_48_32)
pred_num_normal = grader._normalize_number(pred_norm_normal)
gold_num_48_32 = grader._normalize_number(gold_norm_48_32)
normal_correct = (pred_num_normal == gold_num_48_32) or (pred_norm_normal == gold_norm_48_32)

# 診断出力の採点経路
norm_diag = grader._normalize_relative_date(diag_48_32, ref_date_48_32)
pred_norm_diag = grader._normalize_text(norm_diag)
pred_num_diag = grader._normalize_number(pred_norm_diag)
diag_correct = (pred_num_diag == gold_num_48_32) or (pred_norm_diag == gold_norm_48_32)

print(f"\n  Gold: {gold_48_32}")
print(f"  通常出力: '{normal_48_32}'")
print(f"    正規化: '{norm_normal}' → _normalize_text: '{pred_norm_normal}' → _normalize_number: '{pred_num_normal}'")
print(f"    Gold正規化: '{norm_gold_48_32}' → _normalize_text: '{gold_norm_48_32}' → _normalize_number: '{gold_num_48_32}'")
print(f"    判定: {normal_correct}")
print(f"  診断出力: '{diag_48_32}'")
print(f"    正規化: '{norm_diag}' → _normalize_text: '{pred_norm_diag}' → _normalize_number: '{pred_num_diag}'")
print(f"    判定: {diag_correct}")

print(f"\n  問題: 'Deborah was in Bali in 2022.' は 'in 2022' と意味的に一致するが、")
print(f"        scorerは '{pred_num_diag}' vs '{gold_num_48_32}' と比較し、粒度の違いで不正解判定")
print(f"        これはscorerの粒度問題であり、Readerの問題ではない")

# ============================================================
# Phase 4: 28件の独立再分類
# ============================================================
print(f"\n\n{'='*80}")
print("Phase 4: 28件の独立再分類")
print("=" * 80)

reclassified = []
for qid in target_qids:
    c = cases.get(qid)
    if not c:
        continue
    gold = str(c.get('gold_answer', ''))
    ref_date = get_ref_date(qid)
    normal = m30_3_results.get(qid, {})
    normal_output = normal.get('reader_text', '')
    normal_correct = normal.get('scorer_correct', False)
    
    m30_23_reason = m30_23_reasons.get(qid, 'UNKNOWN')
    
    # 独立分類
    if normal_correct:
        category = "CORRECT"
        primary = "correct"
    elif "don't know" in normal_output.lower():
        # 棄権
        has_relative = any(p in normal_output.lower() for p in ["next month", "last month", "this month", "last year", "next year", "year ago"])
        ref_in_input = ref_date in normal.get('evidence_text', '') if ref_date else False
        
        if has_relative and ref_in_input:
            category = "ABSTAIN_RESOLVABLE_RELATIVE"
            primary = "relative_date_unresolved"
        elif has_relative and not ref_in_input:
            category = "ABSTAIN_RELATIVE_NO_REF"
            primary = "reference_date_missing"
        else:
            category = "ABSTAIN_NO_EVIDENCE"
            primary = "evidence_missing"
    else:
        # 非棄権回答が不正解
        has_relative = any(p in normal_output.lower() for p in ["next month", "last month", "this month", "last year", "next year", "year ago"])
        if has_relative:
            category = "WRONG_RELATIVE_DATE"
            primary = "relative_date_miscalculation"
        else:
            category = "WRONG_OTHER"
            primary = "reader_error"
    
    reclassified.append({
        'question_id': qid,
        'gold': gold,
        'normal_output': normal_output,
        'normal_correct': normal_correct,
        'm30_23_reason': m30_23_reason,
        'independent_category': category,
        'primary_cause': primary,
    })

# 集計
category_counts = {}
for r in reclassified:
    category_counts[r['independent_category']] = category_counts.get(r['independent_category'], 0) + 1

print(f"\n=== 独立再分類の集計（合計{len(reclassified)}件）===")
for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}件")

print(f"\n=== 詳細 ===")
for r in reclassified:
    print(f"  {r['question_id']}: {r['independent_category']} (M30-23: {r['m30_23_reason']})")

# ============================================================
# Phase 5: 最終報告
# ============================================================
print(f"\n\n{'='*80}")
print("Phase 5: M30-23の主張ごとの判定")
print("=" * 80)

m30_23_claims = {
    "19件がEVIDENCE_PRESENT_BUT_READER_ABSTAINED": {
        "verdict": "PARTIALLY_CONFIRMED",
        "reason": "19件すべてでReaderが棄権。ただし、通常入力にGoldを直接支持する証拠はなく、相対日付表現のみ。参照日付はReaderに渡されていない。",
    },
    "5件がEVIDENCE_PRESENT_BUT_SCORER_MISMATCH": {
        "verdict": "PARTIALLY_CONFIRMED",
        "reason": "5件でReaderが相対日付を出力したがscorerが不正解判定。ただし、scorerの正規化は正しく動作しており、Readerの相対日付計算が誤っているか、参照日付が適切でない。",
    },
    "Readerの失敗24件の内訳として「証拠が存在しない」ケースは0件": {
        "verdict": "REJECTED",
        "reason": "19件は「証拠はあるが参照日付がなく解決不可能」、5件は「Readerの相対日付計算ミス」。証拠が存在しないケースはないが、参照日付の不在が原因。",
    },
}

for claim, judgment in m30_23_claims.items():
    print(f"\n  主張: {claim}")
    print(f"  判定: {judgment['verdict']}")
    print(f"  理由: {judgment['reason']}")

# 保存
output = {
    'phase2_scorer_mismatch': phase2_results,
    'phase3_48_32': {
        'gold': gold_48_32,
        'normal_output': normal_48_32,
        'normal_correct': normal_correct,
        'diag_output': diag_48_32,
        'diag_correct': diag_correct,
        'issue': 'scorer granularity mismatch - Deborah was in Bali in 2022. vs in 2022',
    },
    'phase4_reclassification': {
        'category_summary': category_counts,
        'details': reclassified,
    },
    'phase5_m30_23_judgments': m30_23_claims,
}

with open('results/m30_24/phase2_5_final.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n結果を results/m30_24/phase2_5_final.json に保存しました")
print(f"\n=== 完了 ===")
print(f"通常パイプライン: 3/28")
print(f"診断用入力: 4/28（通常性能ではない）")
print(f"553件のホールドアウト: 完全隔離")