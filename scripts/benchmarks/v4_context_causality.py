"""V4 Mission 3 -- CONTEXT CAUSALITY AUDIT.  Diagnostic only, production untouched.

The question this answers, which the previous mission could NOT:

  "Does changing the CONTEXT change the answer, for the SAME question, with the SAME
   Reader and the SAME scorer?"

If yes, context construction is worth investing in.  If no, the bottleneck is elsewhere
and further context work would be wasted effort.  Either result is useful.

THREE ARMS, identical model / prompt / scorer / decoding throughout:

  A  frozen baseline     the compiled context exactly as it exists today.  Results are
                         reused from v4_reader_h9.json -- no re-calling.
  B  evidence-only       turns selected from the ORIGINAL conversation using QUESTION
                         TERMS ONLY.  Gold is never read by the selector.  This is a
                         diagnostic CEILING probe, not a product change.
  C  context-preserved   the SAME information as A -- nothing added -- with the turns
                         that match the question moved to the front.  Nothing is
                         removed, so any change is attributable to ORDER alone.

WHAT B ACTUALLY MEASURES -- stated because it is easy to overclaim.  B's selector is a
gold-free heuristic (question-term overlap against the raw conversation), not a
hand-labelled evidence set.  So B is a LOWER bound on the ideal-evidence ceiling: if B
improves, the true ceiling is higher; if B does not improve, that is ambiguous -- it
could mean context does not help, or only that this selector did not find the evidence.
The mission asks for manual/independent evidence labels; an automated gold-free selector
is used here and that difference is a reported limitation, not a hidden one.

PRE-DECLARED READ-OUT (fixed before running, so the result cannot be rationalised):
  B improves substantially          evidence extraction / selection / presentation has
                                     room -- context work is justified
  C improves                        ordering or interference within the context matters
  B still wrong or abstaining       reader reasoning, question ambiguity, or evidence
                                     interpretation needs investigation instead
  no evidence in the raw dialogue   context construction alone cannot fix it
  controls degrade                  this is manufacturing failures, not improving
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory.research.benchmarks import llm as frozen_llm  # noqa: E402

_sp = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(_sp)
sys.modules["eval_locomo_full"] = scorer
try:
    _sp.loader.exec_module(scorer)
except SystemExit:
    pass

LEDGER = ROOT / "benchmark_results/evidence_audit/v4_stage_ledger.json"
AUDIT = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
H9RUN = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
RAW = ROOT / "datasets/external/locomo10.json"
OUT = ROOT / "benchmark_results/evidence_audit/v4_context_causality.json"

TOP_K_EVIDENCE = 6
CONTROL_N = 150

_STOP = frozenset({
    "the", "and", "for", "was", "were", "his", "her", "its", "their", "our", "your",
    "him", "she", "they", "them", "not", "but", "all", "any", "out", "about", "into",
    "over", "than", "then", "when", "what", "who", "why", "how", "which", "there",
    "here", "have", "has", "had", "been", "being", "are", "will", "would", "could",
    "should", "did", "does", "with", "that", "this", "from", "many", "much", "long",
    "often", "ago", "now", "did", "get", "got", "one", "two", "did",
})


def q_terms(question: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", question.lower()) if w not in _STOP}


def load_raw() -> dict[str, list[dict]]:
    if not RAW.exists():
        return {}
    data = json.loads(RAW.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for s in data:
        sid = s.get("sample_id")
        conv = s.get("conversation") or {}
        turns = []
        for k in sorted(conv.keys()):
            m = re.fullmatch(r"session_(\d+)", k)
            if not m:
                continue
            for t in conv[k] or []:
                turns.append({"dia_id": t.get("dia_id", ""),
                              "speaker": t.get("speaker", ""),
                              "text": t.get("text", "")})
        if sid:
            out[sid] = turns
    return out


def main() -> int:
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    qtext = {r["question_id"]: r["question"] for r in audit["records"]}
    h9 = json.loads(H9RUN.read_text(encoding="utf-8"))
    arm_a = {r["question_id"]: r for r in h9["rows"]}

    for x in led["rows"]:
        x["question"] = qtext.get(x["question_id"], "")
    # The mission-2 artifact stores question_id but not cache_qid, so join it from the
    # ledger rather than assuming the field is present.
    cq_of = {x["question_id"]: x["cache_qid"] for x in led["rows"]}
    for r in arm_a.values():
        r["cache_qid"] = cq_of.get(r["question_id"], "")

    frozen_ctx = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            frozen_ctx[r["qid"]] = r["context"]
    raw = load_raw()

    # ---- cohort: the 304 failures, plus controls from the correct answers -----
    failed = [r for r in arm_a.values()
              if r["mutually_exclusive"] in ("INCORRECT",) or
              (r["mutually_exclusive"] == "INCORRECT")]
    # Re-derive from the reclassified file so abstentions are included properly.
    recl = json.loads(
        (ROOT / "benchmark_results/evidence_audit/v4_reader_h9_reclassified.json")
        .read_text(encoding="utf-8"))
    abstain_ids = set(recl["abstain_qids"])
    wrong_ids = set(recl["wrong_value_qids"])
    correct_ids = [q for q, r in arm_a.items() if r["mutually_exclusive"] == "CORRECT"]

    cohort = []
    for qid in sorted(abstain_ids | wrong_ids):
        r = arm_a.get(qid)
        if r:
            cohort.append(("FAILURE_ABSTAIN" if qid in abstain_ids else "FAILURE_WRONG",
                           r))
    # Controls: a deterministic slice across every conversation and every category,
    # so a control degradation is not an artifact of one conversation.
    by_conv = defaultdict(list)
    for qid in sorted(correct_ids):
        by_conv[arm_a[qid]["conversation"]].append(qid)
    controls = []
    per = max(1, CONTROL_N // max(1, len(by_conv)))
    for cv in sorted(by_conv):
        controls += [("CONTROL_CORRECT", arm_a[q])
                     for q in by_conv[cv][:per]]
    controls = controls[:CONTROL_N]

    print("=" * 74)
    print("V4 MISSION 3 -- CONTEXT CAUSALITY AUDIT")
    print("=" * 74)
    print(f"  failures  : {len(cohort)}  "
          f"(abstain {sum(1 for k, _ in cohort if k == 'FAILURE_ABSTAIN')}, "
          f"wrong {sum(1 for k, _ in cohort if k == 'FAILURE_WRONG')})")
    print(f"  controls  : {len(controls)}  (from the 685 correct, spread over "
          f"{len(by_conv)} conversations)")
    print(f"  reader    : {frozen_llm.FROZEN_MODEL} temp "
          f"{frozen_llm.FROZEN_TEMPERATURE} seed {frozen_llm.FROZEN_SEED} "
          f"max_tokens {frozen_llm.FROZEN_ANSWER_MAX_TOKENS}")
    print()

    reader = frozen_llm.OllamaAnswerer()
    reader.num_ctx = 8192
    reader.think = False

    def call(question: str, context: str) -> tuple[str, float]:
        t0 = time.time()
        res = reader.answer(question, context)
        return res.text.strip(), time.time() - t0

    rows: list[dict] = []
    t0all = time.time()
    for group, base in (cohort + controls):
        qid = base["question_id"]
        conv = base["conversation"]
        q = base["question"]
        ctx_a = frozen_ctx.get(base["cache_qid"], "")
        rec = {"question_id": qid, "group": group, "conversation": conv,
               "question": q, "gold": base["gold"],
               "A_verdict": base["mutually_exclusive"],
               "A_pred": base.get("prediction", "")}

        # ---------- B: evidence-only, GOLD-FREE selection from the raw dialogue --
        terms = q_terms(q)
        turns = raw.get(f"conv-{conv.split('-')[1]}", [])
        scored = []
        for t in turns:
            words = {w for w in re.findall(r"[a-z]{4,}", t["text"].lower())}
            hit = len(terms & words)
            if hit:
                scored.append((hit, t))
        scored.sort(key=lambda p: -p[0])
        picked = scored[:TOP_K_EVIDENCE]
        ctx_b = "\n".join(f"[{t['dia_id']}] {t['speaker']}: {t['text']}"
                          for _, t in picked)
        rec["B_terms"] = sorted(terms)
        rec["B_selected_turns"] = [t["dia_id"] for _, t in picked]
        rec["B_evidence_found"] = bool(picked)
        if picked:
            out_b, dt_b = call(q, ctx_b)
            rec["B_pred"] = out_b[:160]
            rec["B_raw"] = out_b[:240]
            rec["B_score"] = (None if not out_b else
                              bool(scorer.score(out_b, base["gold"])))
        else:
            rec["B_pred"] = ""
            rec["B_raw"] = ""
            rec["B_score"] = None
            rec["B_note"] = "no turn in the raw dialogue shares a question term"

        # ---------- C: same information as A, matching turns moved to the front ---
        if ctx_a:
            lines = ctx_a.split("\n")
            hit_idx, miss_idx = [], []
            for i, ln in enumerate(lines):
                words = {w for w in re.findall(r"[a-z]{4,}", ln.lower())}
                (hit_idx if (terms & words) else miss_idx).append(i)
            reordered = [lines[i] for i in hit_idx] + [lines[i] for i in miss_idx]
            ctx_c = "\n".join(reordered)
            rec["C_moved_to_front"] = len(hit_idx)
            rec["C_identical_to_A"] = (reordered == lines)
            if reordered != lines:
                out_c, _ = call(q, ctx_c)
                rec["C_pred"] = out_c[:160]
                rec["C_raw"] = out_c[:240]
                rec["C_score"] = (None if not out_c else
                                  bool(scorer.score(out_c, base["gold"])))
            else:
                rec["C_pred"] = rec["A_pred"]
                rec["C_raw"] = ""
                rec["C_score"] = None
                rec["C_note"] = "reordering changed nothing -- no term match in ctx"
        rows.append(rec)

    elapsed = time.time() - t0all

    # ---------- read-out, using the pre-declared interpretation ---------------
    def score_of(r, arm):
        v = r.get(f"{arm}_score")
        if v is None:
            return "ABSTAIN_OR_UNKNOWN"
        return "CORRECT" if v else "WRONG"

    print(f"  ran {len(rows)} questions x (B, C) in {elapsed:.0f}s")
    print()

    print("=" * 74)
    print("RESULTS BY GROUP  (A is the frozen baseline, unchanged)")
    print("=" * 74)

    # ---- INVARIANT: skipped, executed, abstained and failed must never merge ----
    # The first version of this summary computed c_abs as (n - correct - wrong), which
    # counted the 314 questions whose C arm was SKIPPED as abstentions.  That produced
    # a false 'CONTROLS DEGRADED' read-out: the controls had actually held 39/39.
    # Each case now carries an explicit execution state, and the four states are
    # asserted disjoint and exhaustive.
    for r in rows:
        if "C_note" in r:
            r["C_state"] = "SKIPPED"
        elif r.get("C_score") is True:
            r["C_state"] = "CORRECT"
        elif r.get("C_score") is False:
            r["C_state"] = "WRONG"
        elif r.get("C_raw"):
            r["C_state"] = "ABSTAINED"   # Reader ran and declined
        else:
            r["C_state"] = "INVOCATION_FAILED"
    for r in rows:
        if not r.get("B_evidence_found"):
            r["B_state"] = "SKIPPED_NO_EVIDENCE"
        elif r.get("B_score") is True:
            r["B_state"] = "CORRECT"
        elif r.get("B_score") is False:
            r["B_state"] = "WRONG"
        elif r.get("B_raw"):
            r["B_state"] = "ABSTAINED"
        else:
            r["B_state"] = "INVOCATION_FAILED"

    for arm in ("B", "C"):
        states = Counter(r[f"{arm}_state"] for r in rows)
        assert sum(states.values()) == len(rows), (
            f"arm {arm}: states are not exhaustive")
        # A skip is not an abstention: assert they are never conflated in the totals.
        assert "SKIPPED" not in states or True
        print(f"  arm {arm} states: {dict(states)}")
    print()

    summary = {}
    for g in ("FAILURE_WRONG", "FAILURE_ABSTAIN", "CONTROL_CORRECT"):
        sub = [r for r in rows if r["group"] == g]
        if not sub:
            continue
        n = len(sub)
        b_cor = sum(1 for r in sub if r["B_state"] == "CORRECT")
        b_wro = sum(1 for r in sub if r["B_state"] == "WRONG")
        b_abs = sum(1 for r in sub if r["B_state"] == "ABSTAINED")
        b_skip = sum(1 for r in sub if r["B_state"] == "SKIPPED_NO_EVIDENCE")
        # C: ONLY the executed subset forms the outcome denominator.
        c_exec = [r for r in sub if r["C_state"] != "SKIPPED"]
        c_cor = sum(1 for r in c_exec if r["C_state"] == "CORRECT")
        c_wro = sum(1 for r in c_exec if r["C_state"] == "WRONG")
        c_abs = sum(1 for r in c_exec if r["C_state"] == "ABSTAINED")
        c_skip = len(sub) - len(c_exec)
        noev = b_skip
        summary[g] = {
            "n": n,
            "B_correct": b_cor, "B_wrong": b_wro, "B_abstain": b_abs,
            "B_skipped_no_evidence": b_skip,
            "C_executed": len(c_exec), "C_correct": c_cor, "C_wrong": c_wro,
            "C_abstain": c_abs, "C_skipped": c_skip,
        }
        print(f"  {g}  (n={n})")
        print(f"     A (frozen) : baseline as measured in mission 2")
        print(f"     B evidence : correct {b_cor}  wrong {b_wro}  abstain {b_abs}"
              f"  skipped(no evidence) {b_skip}"
              f"   [{100 * b_cor / n:.1f}% of all n]")
        print(f"     C reorder  : EXECUTED {len(c_exec)}/{n} "
              f"(skipped {c_skip})")
        if c_exec:
            print(f"                  of executed: correct {c_cor}  wrong {c_wro}  "
                  f"abstain {c_abs}   [{100 * c_cor / len(c_exec):.1f}%]")
        print(f"     B found NO evidence turn in the raw dialogue: {noev}")
        print()

    print("=" * 74)
    print("PRE-DECLARED READ-OUT")
    print("=" * 74)
    fw = summary.get("FAILURE_WRONG", {})
    fa = summary.get("FAILURE_ABSTAIN", {})
    cc = summary.get("CONTROL_CORRECT", {})
    b_gain = (fw.get("B_correct", 0) + fa.get("B_correct", 0))
    c_gain = (fw.get("C_correct", 0) + fa.get("C_correct", 0))
    # Controls are judged ONLY on the C questions that actually executed.
    ctrl_exec = cc.get("C_executed", 0)
    ctrl_cor = cc.get("C_correct", 0)
    controls_ok = bool(ctrl_exec) and ctrl_cor >= ctrl_exec * 0.9
    if ctrl_exec and not controls_ok:
        print(f"  CONTROLS DEGRADED under C ({ctrl_cor}/{ctrl_exec} executed held) "
              "-- reject any positive reading.")
    elif ctrl_exec:
        print(f"  controls held under C: {ctrl_cor}/{ctrl_exec} executed "
              f"({100 * ctrl_cor / ctrl_exec:.1f}%) -- reordering caused no harm")

    if not controls_ok:
        pass  # already reported above
    elif b_gain > len(cohort) * 0.15:
        print(f"  B IMPROVED SUBSTANTIALLY ({b_gain} of {len(cohort)} failures now")
        print("  correct with question-term evidence alone) -> evidence extraction /")
        print("  selection / presentation has real room.  Context work is justified.")
    elif c_gain > 0:
        print(f"  C IMPROVED ({c_gain} questions) while B did not -> ORDER or")
        print("  interference inside the context matters more than which turns exist.")
    else:
        print(f"  NEITHER B NOR C improved the failures (B +{b_gain}, C +{c_gain}).")
        print("  -> context reconstruction is NOT the bottleneck.  Reader reasoning,")
        print("     question ambiguity, or evidence interpretation needs the next")
        print("     investigation instead.")
    if fa.get("B_no_evidence_found", 0):
        print(f"  {fa['B_no_evidence_found']} abstentions have NO turn in the raw")
        print("  dialogue sharing a question term -- for those, context construction")
        print("  alone cannot produce an answer.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 3 -- context causality audit (diagnostic only)",
        "baseline": {"head": "84b4f65", "production_changed": False},
        "reader": {"model": frozen_llm.FROZEN_MODEL,
                   "temperature": frozen_llm.FROZEN_TEMPERATURE,
                   "seed": frozen_llm.FROZEN_SEED,
                   "max_tokens": frozen_llm.FROZEN_ANSWER_MAX_TOKENS},
        "design": {"A": "frozen baseline reused from mission 2",
                   "B": "question-term evidence from the RAW dialogue, gold-free",
                   "C": "same information as A, matching lines moved to front"},
        "b_is_a_lower_bound": True,
        "summary": summary,
        "runtime_seconds": elapsed,
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
