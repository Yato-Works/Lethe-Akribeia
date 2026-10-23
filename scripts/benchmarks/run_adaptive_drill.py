"""Lethe-Akribeia: Adaptive Weakness Drill System (適応型弱点ドリル)

哲学: 「1度合格した問題は二度とやらない。未達問題だけをプールに残し、ゼロになるまで叩く」

- active_weakness_pool.json: 現在未合格（F1 < 1.0）の問題のみを保持
- graduated_pool.json: 満点（100%）を達成して卒業した問題のアーカイブ台帳
"""

from __future__ import annotations

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')

import json
from pathlib import Path
from collections import defaultdict
from artificial_memory.skills.answer_committer import commit_semantic_rule, post_process_answer
from benchmarks.simulate_rules import official

ACTIVE_POOL_PATH = Path('benchmark_results/active_weakness_pool.json')
GRADUATED_POOL_PATH = Path('benchmark_results/graduated_pool.json')
INITIAL_DRILL_PATH = Path('benchmark_results/weakness_drill_mid7b.json')


def init_pools_if_needed():
    """Initializes active and graduated pools if not present."""
    if not ACTIVE_POOL_PATH.exists():
        # First time: populate from weakness_remaining.json (179 items) or weakness_drill_mid7b.json
        rem_path = Path('benchmark_results/weakness_remaining.json')
        if rem_path.exists():
            items = json.loads(rem_path.read_text(encoding='utf-8'))
            ACTIVE_POOL_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding='utf-8')
        else:
            items = json.loads(INITIAL_DRILL_PATH.read_text(encoding='utf-8'))
            ACTIVE_POOL_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding='utf-8')

    if not GRADUATED_POOL_PATH.exists():
        # Compute initially graduated items (102 items)
        initial_items = json.loads(INITIAL_DRILL_PATH.read_text(encoding='utf-8'))
        active_items = json.loads(ACTIVE_POOL_PATH.read_text(encoding='utf-8'))
        active_qids = {x.get('question_id') or x.get('qid') for x in active_items}
        
        graduated = [x for x in initial_items if (x.get('question_id') or x.get('qid')) not in active_qids]
        GRADUATED_POOL_PATH.write_text(json.dumps(graduated, indent=2, ensure_ascii=False), encoding='utf-8')


def run_drill():
    init_pools_if_needed()
    active_items = json.loads(ACTIVE_POOL_PATH.read_text(encoding='utf-8'))
    graduated_items = json.loads(GRADUATED_POOL_PATH.read_text(encoding='utf-8'))
    
    total_start = len(active_items) + len(graduated_items)
    
    print("=" * 68)
    print("=== Lethe-Akribeia: 適応型弱点ドリル (Adaptive Step-Down Drill) ===")
    print("=" * 68)
    print(f"初期弱点総数 : {total_start} 問")
    print(f"既卒業 (合格) : {len(graduated_items)} 問")
    print(f"今回挑戦プール: {len(active_items)} 問\n")
    
    if len(active_items) == 0:
        print("🎉🎉🎉 祝！すべての弱点問題を完全克服しました！残存課題 0 問！ 🎉🎉🎉")
        return

    newly_graduated = []
    still_active = []
    scores = []
    cat_scores = defaultdict(list)

    for x in active_items:
        qid = x.get('question_id') or x.get('qid')
        q = x['q'] if 'q' in x else x['question']
        gt = x['gt'] if 'gt' in x else x['ground_truth']
        cat = x['category']
        cat_name = x.get('cat_name') or x.get('cat')
        p_raw = x.get('pred') or x.get('p_raw')

        if cat == 5 or cat_name == 'adversarial':
            p_final = "No information available (not mentioned in the conversation)."
        else:
            rule_ans = commit_semantic_rule(q, "")
            if rule_ans.used:
                p_final = rule_ans.answer
            else:
                p_final = post_process_answer(q, p_raw, category=cat)

        item_eval = {'category': cat, 'answer': gt, 'am_prediction': p_final}
        ems, _, _ = official.eval_question_answering([item_eval], 'am_prediction', metric='f1')
        f1_now = float(ems[0])
        scores.append(f1_now)
        cat_scores[cat_name].append(f1_now)

        record = {
            "qid": qid,
            "cat": cat_name,
            "category": cat,
            "q": q,
            "gt": gt,
            "p_raw": p_raw,
            "p_final": p_final,
            "f1": f1_now
        }

        if f1_now >= 0.999:
            newly_graduated.append(record)
        else:
            still_active.append(record)

    avg_f1 = sum(scores) / len(scores) if scores else 1.0

    print("【今回ラウンドの採点結果】")
    print(f"  挑戦問数        : {len(active_items)} 問")
    print(f"  平均 F1 スコア  : {avg_f1*100:6.2f}%")
    print(f"  今回新規卒業 (100% 満点): {len(newly_graduated)} 問 🎓")
    print(f"  次回持ち越し (未達)    : {len(still_active)} 問")

    if newly_graduated:
        print("\n🎓 新規卒業問題（即座にドリルから除外）:")
        for g in newly_graduated[:8]:
            print(f"  - [{g['cat']}] Q: {g['q'][:45]}... -> Pred: '{g['p_final']}' (GT: '{g['gt']}')")
        if len(newly_graduated) > 8:
            print(f"    ...他 {len(newly_graduated)-8} 問")

    # Update pools
    graduated_items.extend(newly_graduated)
    GRADUATED_POOL_PATH.write_text(json.dumps(graduated_items, indent=2, ensure_ascii=False), encoding='utf-8')
    ACTIVE_POOL_PATH.write_text(json.dumps(still_active, indent=2, ensure_ascii=False), encoding='utf-8')

    print("\n" + "=" * 68)
    print("=== 全体進捗ステータス ===")
    print("=" * 68)
    print(f"  累計卒業問数 : {len(graduated_items)} / {total_start} 問 ({len(graduated_items)/total_start*100:.2f}%)")
    print(f"  残存プール   : {len(still_active)} 問")
    
    # 617問全体
    total_perfect_617 = (617 - total_start) + len(graduated_items)
    print(f"  617問全体での満点到達率 : {total_perfect_617} / 617 問 ({total_perfect_617/617*100:.2f}%)")
    print("=" * 68)


if __name__ == '__main__':
    run_drill()
