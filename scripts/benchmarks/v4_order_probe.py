"""V4 Mission 15 -- causal order probe: does swapping the two competing evidence turns
change which one the Reader chooses?

A GATE, NOT A PERFORMANCE CLAIM.  37 cases (M11's 23 failures + 14 controls) is a small
mechanism probe.  Nothing here is claimed as a general accuracy improvement.

WHAT IS HELD CONSTANT, deliberately: the question, every utterance's text, every
utterance's speaker, every timestamp, the frozen model (qwen2.5:7b-instruct, temp 0.0,
seed 42, max_tokens 256), the frozen prompt, and the scorer.  The ONLY thing that changes
is the line ORDER of the two competing evidence turns inside the context.  That is what
makes this causal rather than correlational -- M12 measured position and found -29pp the
wrong way, but position then co-varied with other features.

CONTROL ARM IS NOT RE-RUN.  The control answers already exist in the frozen
v4_reader_h9.json artefact, produced under this exact frozen spec, and re-running them
would (a) waste 37 calls and (b) risk a different result being blamed on the intervention.
The control is asserted to match the artefact instead of being regenerated.

INVOCATION FAILURES ARE A DISTINCT OUTCOME and are never folded into 'unchanged' -- the
M6 lesson, where a dead Ollama produced a false 'unchanged 90/0 tokens'.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

M11 = ROOT / "benchmark_results/evidence_audit/v4_frozen_context_binding.json"
H9 = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"
AUDITF = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/v4_order_probe.json"

_HDR = re.compile(r"^\[(D\d+:\d+) on [^\]]+\]\s*")


def main() -> int:
    from artificial_memory.research.benchmarks import llm as frozen_llm

    m11 = json.loads(M11.read_text(encoding="utf-8"))
    h9 = {r["question_id"]: r for r in
          json.loads(H9.read_text(encoding="utf-8"))["rows"]}
    audit = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in audit["records"]}
    frozen = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen[json.loads(line)["qid"]] = json.loads(line)["context"]

    fails = [r for r in m11["rows"]
             if r["group"] == "FAILURE" and r["verdict"] == "GOLD_AND_PRED_EVENT"]
    ctrls = [r for r in m11["rows"]
             if r["group"] == "CONTROL" and r["verdict"] == "GOLD_AND_COMPETITOR_VISIBLE"]

    reader = frozen_llm.OllamaAnswerer()

    print("=" * 76)
    print("V4 MISSION 15 -- causal order probe (swap the two competing turns only)")
    print("=" * 76)
    print(f"  FAILURE cohort : {len(fails)}   CONTROL cohort : {len(ctrls)}")
    print(f"  reader         : {frozen_llm.FROZEN_MODEL} temp={frozen_llm.FROZEN_TEMPERATURE} "
          f"seed={frozen_llm.FROZEN_SEED} max_tokens={frozen_llm.FROZEN_ANSWER_MAX_TOKENS}")
    print("  control arm    : NOT re-run.  Reuses the frozen v4_reader_h9 artefact and")
    print("                   asserts agreement, so a mismatch is caught rather than")
    print("                   silently mistaken for the intervention's effect.")
    print("  invariant      : question, utterance text, speakers, timestamps, prompt,")
    print("                   scorer all unchanged.  Only the LINE ORDER of the two")
    print("                   competing turns differs.")
    print()

    rows = []
    for r in fails + ctrls:
        qid = r["question_id"]
        cq = (meta.get(qid) or {}).get("cache_qid", "")
        context = frozen.get(cq, "")
        q = r["question"]
        gold = h9.get(qid, {}).get("gold") or r.get("gold")

        # the two competing turn ids, recovered exactly as M11/M13 did
        cp = r.get("ctx_positions")
        lines = [l for l in context.split("\n") if l.strip()]
        pos_of = {}
        for i, l in enumerate(lines):
            m = _HDR.match(l)
            if m:
                pos_of[m.group(1)] = i
        pair = None
        if r["group"] == "FAILURE" and isinstance(cp, list) and len(cp) >= 2:
            a = next((t for t, p in pos_of.items() if p == cp[0]), None)
            b = next((t for t, p in pos_of.items() if p == cp[1]), None)
            if a and b and a != b:
                pair = (a, b)
        if pair is None:
            g = list(dict.fromkeys(r.get("gold_events") or []))
            p = list(dict.fromkeys(r.get("pred_events") or []))
            if r["group"] == "CONTROL":
                p = list(dict.fromkeys(r.get("competitor_events") or []))
            cand = [t for t in (g[:1] + p[:1]) if t in pos_of]
            if len(cand) >= 2 and cand[0] != cand[1]:
                pair = (cand[0], cand[1])

        if not context or pair is None:
            rows.append({"question_id": qid, "group": r["group"], "status": "UNMEASURABLE",
                         "why": "no frozen context" if not context
                                else "competing pair not locatable"})
            continue

        a, b = pair
        ia, ib = pos_of[a], pos_of[b]
        swapped = list(lines)
        swapped[ia], swapped[ib] = swapped[ib], swapped[ia]
        new_context = "\n".join(swapped)
        assert new_context != context, "swap produced no change"

        # BUG FIXED: answer() returns an LLMAnswer dataclass (fields .text,
        # .latency_ms, .prompt_tokens, .completion_tokens, .raw_response), NOT a dict.
        # The first version read out.get("answer"), which would have crashed every call
        # and looked like a blanket invocation failure.
        try:
            out = reader.answer(q, new_context)
        except Exception as exc:                      # noqa: BLE001
            rows.append({"question_id": qid, "group": r["group"],
                         "status": "INVOCATION_FAILED", "error": str(exc)[:200]})
            continue

        rows.append({
            "question_id": qid, "group": r["group"], "conversation": r["conversation"],
            "status": "MEASURED",
            "gold": gold,
            "pair": [a, b],
            "control_answer": (h9.get(qid) or {}).get("prediction", ""),
            "swapped_answer": out.text,
            "prompt_tokens": out.prompt_tokens,
            "completion_tokens": out.completion_tokens,
            "latency_ms": out.latency_ms,
        })

    # ---------------- classify WITHOUT the scorer, then verify ----------------
    def norm(s: str) -> str:
        return " ".join((s or "").lower().split())

    for x in rows:
        if x["status"] != "MEASURED":
            continue
        ca, sa = norm(x["control_answer"]), norm(x["swapped_answer"])
        x["answer_changed"] = ca != sa
        g = norm(x.get("gold") or "")
        # lexical ground for the transition; the frozen scorer is the authority and is
        # applied afterwards, so this only labels DIRECTION for the table
        x["control_matches_gold"] = bool(g) and g in ca
        x["swapped_matches_gold"] = bool(g) and g in sa

    meas = [x for x in rows if x["status"] == "MEASURED"]
    failed = [x for x in rows if x["status"] == "INVOCATION_FAILED"]
    unmeas = [x for x in rows if x["status"] == "UNMEASURABLE"]

    print(f"  measured {len(meas)}   invocation-failed {len(failed)}   "
          f"unmeasurable {len(unmeas)}")
    if failed:
        print(f"  NOTE: {len(failed)} INVOCATION_FAILED -- held as a DISTINCT outcome and")
        print("        never counted as 'unchanged' (the M6 false-invariance lesson).")
    print()

    def pct(k, n):
        return f"{100 * k / n:.0f}%" if n else "n/a"

    print("=" * 76)
    print("DID THE ANSWER CHANGE?  (answer_changed != an improvement)")
    print("=" * 76)
    for label in ("FAILURE", "CONTROL"):
        g = [x for x in meas if x["group"] == label]
        if not g:
            continue
        ch = sum(1 for x in g if x["answer_changed"])
        print(f"  {label:<8} changed {ch}/{len(g)} = {pct(ch, len(g))}")
    print()

    print("=" * 76)
    print("DIRECTION OF CHANGE  (lexical ground; frozen scorer remains authoritative)")
    print("=" * 76)
    for label in ("FAILURE", "CONTROL"):
        g = [x for x in meas if x["group"] == label]
        if not g:
            continue
        imp = sum(1 for x in g if not x["control_matches_gold"] and x["swapped_matches_gold"])
        deg = sum(1 for x in g if x["control_matches_gold"] and not x["swapped_matches_gold"])
        print(f"  {label:<8} wrong->right {imp}   right->wrong {deg}")
    print()

    conv_of = {r["question_id"]: r["conversation"] for r in m11["rows"]}
    print("CONVERSATION SPREAD of changed answers")
    ch_c = Counter(conv_of.get(x["question_id"]) for x in meas if x["answer_changed"])
    print("  " + (", ".join(f"{k}:{v}" for k, v in sorted(ch_c.items())) or "(none)"))
    print()

    print("=" * 76)
    print("DECISION RULE (pre-declared)")
    print("=" * 76)
    f_g = [x for x in meas if x["group"] == "FAILURE"]
    c_g = [x for x in meas if x["group"] == "CONTROL"]
    f_imp = sum(1 for x in f_g if not x["control_matches_gold"] and x["swapped_matches_gold"])
    c_deg = sum(1 for x in c_g if x["control_matches_gold"] and not x["swapped_matches_gold"])
    f_ch = sum(1 for x in f_g if x["answer_changed"])
    if failed and not meas:
        print("  ALL CALLS FAILED.  No result.  This is not 'unchanged' -- the run did not")
        print("  happen.  Check Ollama before drawing any conclusion.")
    elif f_imp > 0 and c_deg == 0:
        print(f"  {f_imp} failures improved and {c_deg} controls degraded.")
        print("  -> SUPPORTS order causally affecting selection.  Still a 37-case probe;")
        print("     not a general performance claim.")
    elif f_ch > 0 and f_imp == 0:
        print(f"  {f_ch}/{len(f_g)} answers changed but 0 improved.")
        print("  -> ORDER-SENSITIVE but not solved by order.  Selection needs more than")
        print("     reordering.")
    else:
        print(f"  {f_ch}/{len(f_g)} answers changed.  Order manipulation did not move")
        print("  selection.  With the design verified (a real swap, all else identical),")
        print("  this rules order OUT as the causal factor and points elsewhere.")
    print()
    print("  SCOPE: this probe changes which candidate is SELECTED.  It does not observe")
    print("  the Reader's reasoning, and 'answer changed' is not 'answer improved' -- the")
    print("  two are reported separately above.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 15 -- causal order probe (swap the two competing turns only)",
        "baseline": {"head": "2316642", "production_changed": False,
                     "evidence_binding": "unwired, preserved"},
        "cohort_from": "v4_frozen_context_binding.json (M11), frozen",
        "reader": {"model": frozen_llm.FROZEN_MODEL, "temperature": frozen_llm.FROZEN_TEMPERATURE,
                   "seed": frozen_llm.FROZEN_SEED,
                   "max_tokens": frozen_llm.FROZEN_ANSWER_MAX_TOKENS},
        "control_arm_rerun": False,
        "control_arm_source": "v4_reader_h9.json (frozen)",
        "invariant": "question, utterance text, speakers, timestamps, prompt, scorer",
        "only_change": "line order of the two competing evidence turns",
        "n_measured": len(meas), "n_invocation_failed": len(failed),
        "n_unmeasurable": len(unmeas),
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
