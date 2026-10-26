#!/usr/bin/env python3
"""M30-46 Phase 3+4: Context-ORDER experiment (the one allowed change).

Baseline : production context order (as compiled).
Treatment: same turns, reordered so higher-scoring turns come first, using
           ONLY the pipeline's own relevance signal — no gold, no evidence IDs.

Then both arms run through the real LoCoMoAdapter with the SAME reader.

Also runs an ORACLE diagnostic (gold-evidence-first) SEPARATELY, clearly
labelled as a position-sensitivity probe, never as a product candidate.
"""
import sys, json, re, hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer
from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader
from artificial_memory.skills.answer_committer import commit_answer

LINE_RE = re.compile(r"^\[SPEAKER:([^\]]+)\]\s*\[(D\d+:\d+) on ([^\]]+)\](.*)$")


def split_context(ctx):
    """Split a compiled context into (preamble_lines, turn_blocks)."""
    blocks = []
    preamble = []
    for ln in ctx.split("\n"):
        m = LINE_RE.match(ln)
        if m:
            blocks.append({"speaker": m.group(1), "dia": m.group(2),
                           "date": m.group(3), "body": m.group(4), "raw": ln})
        else:
            preamble.append(ln)
    return preamble, blocks


def reorder_treatment(ctx, query):
    """Reorder turn blocks by the pipeline's own lexical relevance to the query.
    No gold, no evidence IDs — only query<->text overlap, which the pipeline
    already computes internally."""
    preamble, blocks = split_context(ctx)
    if not blocks:
        return ctx, 0
    qk = set(w for w in re.findall(r"\b[a-z]{4,}\b", query.lower()))
    def score(b):
        tk = set(w for w in re.findall(r"\b[a-z]{4,}\b", b["raw"].lower()))
        return len(qk & tk)
    # stable sort: higher relevance first, original order preserved on ties
    idx = list(range(len(blocks)))
    idx.sort(key=lambda i: (-score(blocks[i]), i))
    new_blocks = [blocks[i] for i in idx]
    moved = sum(1 for a, b in zip(blocks, new_blocks) if a is not b)
    out = "\n".join(preamble + [b["raw"] for b in new_blocks])
    return out, moved


def reorder_oracle(ctx, gold_ids):
    """ORACLE diagnostic only: gold evidence first. NEVER a product candidate."""
    preamble, blocks = split_context(ctx)
    if not blocks:
        return ctx
    gold = [b for b in blocks if b["dia"] in gold_ids]
    rest = [b for b in blocks if b["dia"] not in gold_ids]
    return "\n".join(preamble + [b["raw"] for b in gold + rest])


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    gold_ev = {}
    for conv in locomo:
        sid = conv.get("sample_id", "")
        for qa in conv.get("qa", []):
            gold_ev[(sid, qa.get("question", "").strip().lower())] = qa.get("evidence", []) or []

    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")

    def to_conv(q):
        m = re.match(r"lc_conv-(\d+)_(\d+)", q)
        return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}"
    tmap = {to_conv(q): q for q in targets}

    reader = OllamaAnswerer(model="qwen2.5:7b-instruct")
    grader = OfficialLongMemEvalGrader()
    qtype = {1:"multi-session",2:"temporal-reasoning",3:"single-session-user",
             4:"single-session-assistant",5:"abstention"}

    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    def eval_ctx(q, ctx, qid):
        c = commit_answer(q.question, ctx, category=q.category)
        if c.used:
            pred = c.answer
        else:
            pred = reader.answer(q.question, ctx).text.strip()
        ok = grader.grade(qid, qtype.get(q.category, "single-session-user"),
                          str(pred), q.ground_truth).is_correct
        return str(pred), bool(ok)

    base_rows, treat_rows, oracle_rows = [], [], []
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            if q.question_id not in tmap:
                continue
            qid = tmap[q.question_id]
            pcc = adapter.compiler.compile(q.question, ir_records)
            base_ctx = pcc.context_text
            treat_ctx, moved = reorder_treatment(base_ctx, q.question)
            gold = gold_ev.get((sid, q.question.strip().lower()), [])
            orc_ctx = reorder_oracle(base_ctx, set(gold))

            bp, bok = eval_ctx(q, base_ctx, qid)
            tp, tok = eval_ctx(q, treat_ctx, qid)
            op, ook = eval_ctx(q, orc_ctx, qid)

            base_rows.append({"qid": qid, "pred": bp, "correct": bok,
                              "ctx_hash": hashlib.sha256(base_ctx.encode()).hexdigest()[:12]})
            treat_rows.append({"qid": qid, "pred": tp, "correct": tok, "moved": moved,
                               "ctx_hash": hashlib.sha256(treat_ctx.encode()).hexdigest()[:12]})
            oracle_rows.append({"qid": qid, "pred": op, "correct": ook})
            print(f"  {qid:16s} base={'OK' if bok else '--'} treat={'OK' if tok else '--'} "
                  f"oracle={'OK' if ook else '--'} moved={moved}")

    def summ(rows, label):
        n = sum(1 for r in rows if r["correct"])
        print(f"{label}: {n}/{len(rows)}")
        return n

    print()
    print("=" * 78)
    bn = summ(base_rows, "Baseline (production order) ")
    tn = summ(treat_rows, "Treatment (relevance-first)")
    on = summ(oracle_rows, "ORACLE (gold-first, diagnostic only)")
    print("=" * 78)

    # pair
    bm = {r["qid"]: r for r in base_rows}
    rec = reg = unch = 0
    for r in treat_rows:
        b = bm[r["qid"]]
        if r["correct"] and not b["correct"]: rec += 1
        elif b["correct"] and not r["correct"]: reg += 1
        else: unch += 1
    print(f"  Treatment vs Baseline: recovered={rec} regressed={reg} unchanged={unch} (sum={rec+reg+unch})")
    print(f"  contexts changed: {sum(1 for r in treat_rows if r['moved']>0)}/28")
    print()

    out = PROJECT_ROOT / "results/m30_46"
    out.mkdir(parents=True, exist_ok=True)
    (out / "order_experiment.json").write_text(json.dumps({
        "baseline_correct": bn, "treatment_correct": tn, "oracle_correct": on,
        "recovered": rec, "regressed": reg, "unchanged": unch,
        "base_rows": base_rows, "treat_rows": treat_rows, "oracle_rows": oracle_rows,
        "note": "oracle is a diagnostic position probe, NOT a product candidate",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'order_experiment.json'}")


if __name__ == "__main__":
    main()
