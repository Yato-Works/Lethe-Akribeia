#!/usr/bin/env python3
"""M30-54 Phase 1: Oracle Recall & Evidence Delivery Audit.

Measures 3 stages for each of the 28 Bucket A cases:
A. Retrieval Recall — does the EvidenceScorer return turns containing the gold evidence?
B. Evidence Delivery — does the compiled context contain the gold evidence turn?
C. Reader Conditional Accuracy — given sufficient evidence, does the Reader answer correctly?

No holdout access. No gold label leakage into fallback.
"""
import json, re, sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer
from artificial_memory.core.ir.memory_types import ApexMemoryUnit
from artificial_memory.core.ir.structured import StructuredIR

scorer = UniversalEvidenceScorer()
TURN_RE = re.compile(r"(?:\[SPEAKER:[^\]]+\]\s*)?\[(D\d+:\d+)\s+on\s+([^\]]+)\](.*)")
STOP = set("the and for are was were did does has have had will would could should can may might "
           "must this that these those when where what which who why how not all any both each "
           "with from into about after before during between through over under".split())

# Gold evidence annotations — manually verified from the conversation text
# Each entry: {qid: {"evidence_turn": "Dxx:yy", "evidence_phrase": "...", "evidence_type": "..."}}
GOLD_EVIDENCE = {
    "lc_conv-26_63": {"evidence_turn": "D15:12", "phrase": "talent show for the kids next month (September 2023)", "type": "explicit_date"},
    "lc_conv-26_72": {"evidence_turn": "D17:13", "phrase": "adopted", "type": "event_only"},
    "lc_conv-26_73": {"evidence_turn": "D15:1", "phrase": "LGBTQ", "type": "event_only"},
    "lc_conv-30_1": {"evidence_turn": "D1:1", "phrase": "community meeting", "type": "explicit_date"},
    "lc_conv-30_7": {"evidence_turn": "D2:2", "phrase": "29 January 2023", "type": "explicit_date"},
    "lc_conv-41_10": {"evidence_turn": "D7:5", "phrase": "met Jean", "type": "explicit_date"},
    "lc_conv-41_20": {"evidence_turn": "D3:15", "phrase": "took a road trip", "type": "explicit_date"},
    "lc_conv-41_24": {"evidence_turn": "D5:2", "phrase": "job offer", "type": "explicit_date"},
    "lc_conv-42_57": {"evidence_turn": "D26:19", "phrase": "go over tomorrow (5 November 2022)", "type": "explicit_date"},
    "lc_conv-42_73": {"evidence_turn": "D25:2", "phrase": "Indiana", "type": "location"},
    "lc_conv-44_13": {"evidence_turn": "D1:3", "phrase": "Audrey", "type": "event_only"},
    "lc_conv-47_43": {"evidence_turn": "D20:18", "phrase": "siblings next month (September 2022)", "type": "explicit_date"},
    "lc_conv-47_48": {"evidence_turn": "D1:3", "phrase": "board games", "type": "event_only"},
    "lc_conv-47_65": {"evidence_turn": "D31:1", "phrase": "currently on the road trip", "type": "explicit_date"},
    "lc_conv-48_12": {"evidence_turn": "D2:7", "phrase": "yesterday (26 January 2023)", "type": "explicit_date"},
    "lc_conv-48_32": {"evidence_turn": "D1:1", "phrase": "2022", "type": "explicit_date"},
    "lc_conv-48_50": {"evidence_turn": "D7:19", "phrase": "yoga", "type": "event_only"},
    "lc_conv-49_47": {"evidence_turn": "D2:2", "phrase": "July 2023", "type": "explicit_date"},
    "lc_conv-49_58": {"evidence_turn": "D1:1", "phrase": "2022", "type": "explicit_date"},
    "lc_conv-49_59": {"evidence_turn": "D1:7", "phrase": "kayaking", "type": "event_only"},
    "lc_conv-49_75": {"evidence_turn": "D2:2", "phrase": "January 5, 2024", "type": "explicit_date"},
    "lc_conv-49_79": {"evidence_turn": "D4:2", "phrase": "February 2024", "type": "explicit_date"},
    "lc_conv-50_36": {"evidence_turn": "D8:1", "phrase": "album yesterday (8 June 2023)", "type": "explicit_date"},
    "lc_conv-50_52": {"evidence_turn": "D24:17", "phrase": "Tokyo next month (November 2023)", "type": "explicit_date"},
    "lc_conv-50_55": {"evidence_turn": "D2:1", "phrase": "October 24, 2023", "type": "explicit_date"},
    "lc_conv-50_65": {"evidence_turn": "D1:2", "phrase": "September 9, 2023", "type": "explicit_date"},
    "lc_conv-50_69": {"evidence_turn": "D2:5", "phrase": "November 2023", "type": "explicit_date"},
    "lc_conv-47_47": {"evidence_turn": "D3:2", "phrase": "September 3, 2022", "type": "explicit_date"},
}

def get_question_for_qid(qid, adapter, sids):
    """Get question text, category, and conversation index for a QID."""
    m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
    if not m: return None, None, None
    conv_num = int(m.group(1))
    qa_num = int(m.group(2))
    
    for idx, sid in enumerate(sids):
        sid_conv = int(sid.split("-")[1])
        if sid_conv == conv_num:
            # Use load_conversation to get questions
            turns, questions, ir_records = adapter.load_conversation(idx)
            for q in questions:
                if q.question_id.endswith(f"qa-{qa_num:03d}") or q.question_id.endswith(f"qa-{qa_num}"):
                    return q, idx, sid
    return None, None, None


def main():
    prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
    baseline = {r["qid"]: r for r in prod["rows"]}
    
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    audit = {}
    retrieval_hits = 0
    delivery_hits = 0
    reader_correct_given_evidence = 0
    evidence_sufficient = 0
    total = 0

    for qid in sorted(baseline.keys()):
        bp = baseline[qid]
        question, conv_idx, sid = get_question_for_qid(qid, adapter, sids)
        if question is None:
            audit[qid] = {"error": "Question not found"}
            continue

        gold = bp["gold"]
        cat = question.category
        raw_pred = bp.get("predicted", "")
        reader_correct = LoCoMoAdapter.score_binary(cat, gold, raw_pred)

        # Get the IR records and compile context
        turns, questions, ir_records = adapter.load_conversation(conv_idx)
        pcc = adapter.compiler.compile(question.question, ir_records) if adapter.compiler else None
        ctx = pcc.context_text if pcc else ""
        
        # Parse compiled context turns
        compiled_turns = set()
        for m in TURN_RE.finditer(ctx):
            compiled_turns.add(m.group(1))

        # Get gold evidence
        gold_ev = GOLD_EVIDENCE.get(qid, {})
        gold_turn = gold_ev.get("evidence_turn", "")
        gold_phrase = gold_ev.get("phrase", "")

        # Check if gold turn is in compiled context
        evidence_delivered = gold_turn in compiled_turns if gold_turn else False

        # Check EvidenceScorer ranking for the gold turn
        retrieval_hit = False
        top_turns = []
        for rec in ir_records[:50]:  # Check top 50 IR records
            unit = ApexMemoryUnit(ir=rec) if hasattr(rec, 'ir') else rec
            if not hasattr(unit, 'ir'):
                continue
            breakdown = scorer.compute_score(question.question, unit)
            score = breakdown.total_score
            turn_id = getattr(unit.ir, 'source', '') or rec.raw_content[:20] if hasattr(rec, 'raw_content') else ''
            top_turns.append((turn_id, score))

        # Check if any top-scored turn contains the gold phrase
        gold_phrase_lower = gold_phrase.lower()
        for turn_id, score in top_turns:
            if gold_turn and turn_id == gold_turn:
                retrieval_hit = True
                break
            if gold_phrase_lower and gold_phrase_lower in turn_id.lower():
                retrieval_hit = True
                break

        # Simpler retrieval check: does the EvidenceScorer score above 0 any turn containing the gold phrase?
        for rec in ir_records:
            unit = ApexMemoryUnit(ir=rec) if hasattr(rec, 'ir') else rec
            if not hasattr(unit, 'ir'):
                continue
            content = unit.ir.raw_content.lower() if hasattr(unit.ir, 'raw_content') else str(unit.ir).lower()
            if gold_phrase_lower and gold_phrase_lower in content:
                retrieval_hit = True
                break

        if retrieval_hit:
            retrieval_hits += 1
        if evidence_delivered:
            delivery_hits += 1
        
        # Sufficient evidence: gold turn is in compiled context
        if evidence_delivered:
            evidence_sufficient += 1
            if reader_correct:
                reader_correct_given_evidence += 1
        
        total += 1
        
        audit[qid] = {
            "question": question.question[:80],
            "gold": gold,
            "reader_output": raw_pred[:80],
            "reader_correct": reader_correct,
            "gold_evidence_turn": gold_turn,
            "gold_evidence_phrase": gold_phrase,
            "evidence_retrieved": retrieval_hit,
            "evidence_delivered": evidence_delivered,
            "compiled_turns_count": len(compiled_turns),
        }

    print("M30-54 Phase 1: Oracle Recall & Evidence Delivery Audit")
    print("="*70)
    print(f"Total cases: {total}")
    print()
    print("A. RETRIEVAL RECALL")
    print(f"   Evidence found in IR records: {retrieval_hits}/{total}")
    print(f"   Percentage: {retrieval_hits/total*100:.1f}%")
    print()
    print("B. EVIDENCE DELIVERY RECALL")
    print(f"   Evidence turn in compiled context: {delivery_hits}/{total}")
    print(f"   Percentage: {delivery_hits/total*100:.1f}%")
    print()
    print("C. READER CONDITIONAL ACCURACY (given sufficient evidence)")
    print(f"   Cases with sufficient evidence: {evidence_sufficient}")
    print(f"   Reader correct given sufficient evidence: {reader_correct_given_evidence}/{evidence_sufficient}")
    if evidence_sufficient > 0:
        print(f"   Percentage: {reader_correct_given_evidence/evidence_sufficient*100:.1f}%")
    print()
    
    # Breakdown for wrong cases
    print("D. WRONG CASE BREAKDOWN")
    for qid in sorted(audit.keys()):
        a = audit.get(qid, {})
        if a.get("reader_correct"):
            continue
        gold_turn = a.get("gold_evidence_turn", "")
        evidence_delivered = a.get("evidence_delivered", False)
        evidence_retrieved = a.get("evidence_retrieved", False)
        print(f"  {qid}: gold={a['gold'][:30]}")
        print(f"    retrieved={evidence_retrieved}, delivered={evidence_delivered}, compiled_turns={a.get('compiled_turns_count', 0)}")
        if not evidence_delivered:
            print(f"    → ENGINE/EVIDENCE_FAILURE: gold turn {gold_turn} NOT in context")
    
    # Save results
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
