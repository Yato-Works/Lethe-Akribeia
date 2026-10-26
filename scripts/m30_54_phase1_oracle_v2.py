#!/usr/bin/env python3
"""M30-54 Phase 1 (v2): Oracle Recall & Evidence Delivery Audit.

Simplified approach: check if the gold evidence turn (by Dxx:yy ID) appears
in the IR records and in the compiled context. No manual phrase annotation needed.
"""
import json, re, sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer
from artificial_memory.core.ir.memory_types import ApexMemoryUnit

scorer = UniversalEvidenceScorer()
TURN_RE = re.compile(r"(?:\[SPEAKER:[^\]]+\]\s*)?\[(D\d+:\d+)\s+on\s+([^\]]+)\](.*)")

# Gold evidence turn IDs — manually verified by reading the conversation context
# These are the turn IDs where the answer-relevant content appears
GOLD_TURNS = {
    "lc_conv-26_63": "D15:12",  # "talent show for the kids next month"
    "lc_conv-26_72": "D17:13",  # adoption mention
    "lc_conv-26_73": "D15:1",   # LGBTQ+ youth center
    "lc_conv-30_1": "D1:1",     # community meeting date
    "lc_conv-30_7": "D2:2",     # 29 January 2023
    "lc_conv-41_10": "D7:5",    # Maria met Jean
    "lc_conv-41_20": "D3:15",   # road trip to Pacific Northwest
    "lc_conv-41_24": "D5:2",    # job offer / promotion
    "lc_conv-42_57": "D26:19",  # Joanna tomorrow (5 November) - recipes
    "lc_conv-42_73": "D25:2",   # Indiana
    "lc_conv-44_13": "D1:3",    # Audrey
    "lc_conv-47_43": "D20:18",  # siblings next month (September 2022)
    "lc_conv-47_47": "D3:2",    # September 3, 2022
    "lc_conv-47_48": "D1:3",    # baseball game
    "lc_conv-47_65": "D31:1",   # currently on road trip
    "lc_conv-48_12": "D2:7",    # yesterday (26 January 2023) appreciation letter
    "lc_conv-48_32": "D1:1",    # 2022
    "lc_conv-48_50": "D7:19",   # yoga
    "lc_conv-49_47": "D2:2",    # July 2023
    "lc_conv-49_58": "D1:1",    # 2022
    "lc_conv-49_59": "D1:7",    # kayaking
    "lc_conv-49_75": "D2:2",    # January 5, 2024
    "lc_conv-49_79": "D4:2",    # February 2024
    "lc_conv-50_36": "D8:1",    # album yesterday (8 June 2023)
    "lc_conv-50_52": "D24:17",  # Tokyo next month (November 2023)
    "lc_conv-50_55": "D2:1",    # October 24, 2023
    "lc_conv-50_65": "D1:2",    # September 9, 2023
    "lc_conv-50_69": "D2:5",    # November 2023
}


def main():
    prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
    baseline = {r["qid"]: r for r in prod["rows"]}

    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    # Build conv_num -> idx map
    conv_to_idx = {}
    for idx, sid in enumerate(sids):
        conv_num = int(sid.split("-")[1])
        conv_to_idx[conv_num] = idx

    audit = {}
    retrieval_hits = 0
    delivery_hits = 0
    reader_correct_given_evidence = 0
    evidence_sufficient = 0

    for qid in sorted(baseline.keys()):
        bp = baseline[qid]
        m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
        if not m: continue
        conv_num = int(m.group(1))
        qa_num = int(m.group(2))

        # Load conversation
        idx = conv_to_idx.get(conv_num)
        if idx is None:
            continue
        turns, questions, ir_records = adapter.load_conversation(idx)

        # Find question
        question = None
        for q in questions:
            # question_id format: "conv-26-qa-063"
            parts = q.question_id.split("-")
            if len(parts) >= 4:
                q_conv = int(parts[1])
                q_qa = int(parts[3])
                if q_conv == conv_num and q_qa == qa_num:
                    question = q
                    break

        if question is None:
            audit[qid] = {"error": "Question not found"}
            continue

        gold = bp["gold"]
        cat = question.category
        raw_pred = bp.get("predicted", "")
        reader_correct = LoCoMoAdapter.score_binary(cat, gold, raw_pred)

        # Compile context
        pcc = adapter.compiler.compile(question.question, ir_records) if adapter.compiler else None
        ctx = pcc.context_text if pcc else ""

        # Parse compiled context turns
        compiled_turns = set()
        for m2 in TURN_RE.finditer(ctx):
            compiled_turns.add(m2.group(1))

        # Check gold turn presence
        gold_turn = GOLD_TURNS.get(qid, "")
        
        # A. Retrieval Recall — check if gold turn appears in any IR record's raw_content
        retrieval_hit = False
        if gold_turn:
            for rec in ir_records:
                content = rec.raw_content if hasattr(rec, 'raw_content') else str(rec)
                if gold_turn in content:
                    retrieval_hit = True
                    break

        # B. Evidence Delivery — check if gold turn is in compiled context
        evidence_delivered = gold_turn in compiled_turns if gold_turn else False

        # C. Reader conditional accuracy
        if evidence_delivered:
            evidence_sufficient += 1
            if reader_correct:
                reader_correct_given_evidence += 1

        if retrieval_hit:
            retrieval_hits += 1
        if evidence_delivered:
            delivery_hits += 1

        audit[qid] = {
            "question": question.question[:100],
            "gold": gold[:50],
            "reader_output": raw_pred[:80],
            "reader_correct": reader_correct,
            "gold_evidence_turn": gold_turn,
            "evidence_retrieved": retrieval_hit,
            "evidence_delivered": evidence_delivered,
            "compiled_turns_count": len(compiled_turns),
        }

    total = len(audit)
    print("M30-54 Phase 1: Oracle Recall & Evidence Delivery Audit (v2)")
    print("="*70)
    print(f"Total cases: {total}")
    print()
    print("A. RETRIEVAL RECALL (EvidenceScore rank)")
    print(f"   Gold turn in IR records: {retrieval_hits}/{total}")
    print(f"   Percentage: {retrieval_hits/total*100:.1f}%")
    print()
    print("B. EVIDENCE DELIVERY RECALL (post-compile to Reader)")
    print(f"   Gold turn in compiled context: {delivery_hits}/{total}")
    print(f"   Percentage: {delivery_hits/total*100:.1f}%")
    print()
    print("C. READER CONDITIONAL ACCURACY (given sufficient evidence)")
    print(f"   Cases with sufficient evidence: {evidence_sufficient}")
    print(f"   Reader correct given sufficient evidence: {reader_correct_given_evidence}/{evidence_sufficient}")
    if evidence_sufficient > 0:
        print(f"   Percentage: {reader_correct_given_evidence/evidence_sufficient*100:.1f}%")
    print()

    print("D. WRONG CASE BREAKDOWN")
    for qid in sorted(audit.keys()):
        a = audit.get(qid, {})
        if a.get("reader_correct"):
            continue
        print(f"  {qid}: gold={a['gold'][:25]} | retrieved={a.get('evidence_retrieved')} delivered={a.get('evidence_delivered')}")
        if not a.get("evidence_delivered", True):
            print(f"    → ENGINE/EVIDENCE_FAILURE: gold turn {a.get('gold_evidence_turn')} NOT in context")

    out_dir = PROJECT_ROOT / "results/m30_54"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase1_oracle_audit.json").write_text(json.dumps({
        "summary": {
            "total": total,
            "retrieval_hits": retrieval_hits,
            "delivery_hits": delivery_hits,
            "reader_correct_given_evidence": reader_correct_given_evidence,
            "evidence_sufficient_count": evidence_sufficient,
        },
        "per_qid": audit,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'phase1_oracle_audit.json'}")


if __name__ == "__main__":
    main()
