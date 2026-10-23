#!/usr/bin/env python3
"""M30-46 Phase 3+4 (CORRECTED): order experiment via context_override.

The previous attempt hand-rolled the answer path (commit_answer + reader only),
which dropped evaluate_question's guards/verification and produced 7/28 instead
of the production 13/28. That is the exact trap M30-43 identified.

Correct method: keep the WHOLE production evaluate_question, and inject only the
reordered context through adapter.context_override (the documented A/B hook).

Baseline  = production compiled context
Treatment = same turns, relevance-first order (pipeline's own lexical signal)
Oracle    = gold-first (DIAGNOSTIC ONLY, never a product candidate)
"""
import sys, json, re, hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter, CachedContext
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

LINE_RE = re.compile(r"^\[SPEAKER:([^\]]+)\]\s*\[(D\d+:\d+) on ([^\]]+)\](.*)$")


def split_blocks(ctx):
    blocks, preamble = [], []
    for ln in ctx.split("\n"):
        m = LINE_RE.match(ln)
        if m:
            blocks.append({"dia": m.group(2), "raw": ln})
        else:
            preamble.append(ln)
    return preamble, blocks


def reorder(ctx, query, gold_ids=None):
    preamble, blocks = split_blocks(ctx)
    if not blocks:
        return ctx, 0
    if gold_ids is not None:
        order = [b for b in blocks if b["dia"] in gold_ids] + \
                [b for b in blocks if b["dia"] not in gold_ids]
    else:
        qk = set(w for w in re.findall(r"\b[a-z]{4,}\b", query.lower()))
        def sc(b):
            return len(qk & set(re.findall(r"\b[a-z]{4,}\b", b["raw"].lower())))
        idx = sorted(range(len(blocks)), key=lambda i: (-sc(blocks[i]), i))
        order = [blocks[i] for i in idx]
    moved = sum(1 for a, b in zip(blocks, order) if a is not b)
    return "\n".join(preamble + [b["raw"] for b in order]), moved


def to_conv(q):
    m = re.match(r"lc_conv-(\d+)_(\d+)", q)
    return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}"


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    gold_ev = {}
    for conv in locomo:
        sid = conv.get("sample_id", "")
        for qa in conv.get("qa", []):
            gold_ev[(sid, qa.get("question", "").strip().lower())] = qa.get("evidence", []) or []

    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")
    tmap = {to_conv(q): q for q in targets}

    answerer = OllamaAnswerer(model="qwen2.5:7b-instruct")

    # Precompute contexts
    adapter0 = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
    ctxs = {}
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter0.load_conversation(idx)
        for q in questions:
            if q.question_id not in tmap:
                continue
            pcc = adapter0.compiler.compile(q.question, ir_records)
            ctxs[q.question_id] = (pcc.context_text, sid)

    def run(arm):
        adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
        override = {}
        meta = {}
        for cqid, (ctx, sid) in ctxs.items():
            q = next(x for x in [None] if False)  # placeholder
        # need question text; reload
        rows = []
        for idx, sid in enumerate(sample_ids):
            turns, questions, ir_records = adapter.load_conversation(idx)
            for q in questions:
                if q.question_id not in tmap:
                    continue
                base_ctx = ctxs[q.question_id][0]
                gold = gold_ev.get((sid, q.question.strip().lower()), [])
                if arm == "baseline":
                    new_ctx, moved = base_ctx, 0
                elif arm == "treatment":
                    new_ctx, moved = reorder(base_ctx, q.question)
                else:
                    new_ctx, moved = reorder(base_ctx, q.question, set(gold))
                override[q.question_id] = CachedContext(context_text=new_ctx)
        adapter.context_override = override
        for idx, sid in enumerate(sample_ids):
            turns, questions, ir_records = adapter.load_conversation(idx)
            for q in questions:
                if q.question_id not in tmap:
                    continue
                res = adapter.evaluate_question(q, turns, ir_records, answerer)
                rows.append({"qid": tmap[q.question_id], "pred": str(res.predicted_answer),
                             "correct": bool(res.is_correct)})
        return rows

    print("=" * 78)
    print("M30-46 Phase 3+4 (corrected): order experiment via context_override")
    print("=" * 78)
    print()

    base = run("baseline")
    bn = sum(1 for r in base if r["correct"])
    print(f"Baseline (production order):    {bn}/28")
    treat = run("treatment")
    tn = sum(1 for r in treat if r["correct"])
    print(f"Treatment (relevance-first):    {tn}/28")
    orc = run("oracle")
    on = sum(1 for r in orc if r["correct"])
    print(f"ORACLE (gold-first, diagnostic): {on}/28")
    print()

    bm = {r["qid"]: r for r in base}
    rec = reg = unch = 0
    for r in treat:
        b = bm[r["qid"]]
        if r["correct"] and not b["correct"]: rec += 1
        elif b["correct"] and not r["correct"]: reg += 1
        else: unch += 1
    print(f"Treatment vs Baseline: recovered={rec} regressed={reg} unchanged={unch} (sum={rec+reg+unch})")
    bm2 = {r["qid"]: r for r in base}
    rec2 = reg2 = 0
    for r in orc:
        b = bm2[r["qid"]]
        if r["correct"] and not b["correct"]: rec2 += 1
        elif b["correct"] and not r["correct"]: reg2 += 1
    print(f"Oracle vs Baseline:    recovered={rec2} regressed={reg2}")
    print()

    out = PROJECT_ROOT / "results/m30_46"
    out.mkdir(parents=True, exist_ok=True)
    (out / "order_experiment_v2.json").write_text(json.dumps({
        "baseline_correct": bn, "treatment_correct": tn, "oracle_correct": on,
        "recovered": rec, "regressed": reg, "unchanged": unch,
        "oracle_recovered": rec2, "oracle_regressed": reg2,
        "base_rows": base, "treat_rows": treat, "oracle_rows": orc,
        "note": "full evaluate_question via context_override; oracle is diagnostic only",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'order_experiment_v2.json'}")


if __name__ == "__main__":
    main()
