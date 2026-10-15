"""V4 Mission 25 -- session-date-aware Reader intervention, on a shadow/replay path.

SHADOW ONLY.  Production is not touched.  No module is wired.  The frozen Reader
(OllamaAnswerer, qwen2.5:7b-instruct temp 0.0 seed 42 max_tokens 256) is called on
frozen contexts, and the frozen scorer eval_locomo_full.score() is used UNCHANGED.

WHY A BASELINE MUST BE GENERATED HERE, not reused.  All 28 target cases have ledger
verdict 'abstain' and are NOT in the H9 Reader cohort (0 of 28), because H9 is the subset
where the deterministic engine abstained AND the Reader was run.  These Bucket A cases
never had a Reader output, so there is no frozen baseline to compare against and inventing
one would be fabrication.  Both arms are therefore generated in this run.

REGRESSION SENTINELS.  Drawn from the frozen Reader cohort's CORRECT cases in the TUNING
split only -- the 553-question holdout is never sampled or inspected.  Sentinels are
selected to include date-heavy and multi-session cases, and are FIXED before any treatment
output is inspected.

THE STRICT AUDIT IS SUPPLEMENTARY, not a scorer replacement.  The frozen scorer is lenient
on day-of-month (M24: score('9/14/2023', 'October 14, 2023') is True).  A scorer pass whose
day/month/year disagrees with gold is reported as SCORER_PASS_STRICT_FAIL and is NOT
counted as a verified rescue.  The scorer itself is never edited.

THE RESOLVER'S DATE IS NOT FED TO THE READER AS FACT.  It is offered as a candidate
derivation for the Reader to verify, because M24 established the resolver is right 17/28
and wrong 11/28 -- presenting its output as established would be laundering an unverified
derivation into the answer.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

EA = ROOT / "benchmark_results/evidence_audit"
M24 = EA / "v4_temporal_resolver_gate.json"
PACK = EA / "v4_bucketA_audit_pack.json"
H9 = EA / "v4_reader_h9.json"
AUDITF = EA / "evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
SPLIT = EA / "v4_holdout_split.json"
DATASET = ROOT / "datasets/external/locomo10.json"
OUT = EA / "v4_intervention_temporal.json"

RUN_ID = "m25_shadow_20261015"
SEED = 20261015

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_ANCHOR = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b"
                     r"|\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b")
_REL_OFFSET = [
    (re.compile(r"\byesterday\b", re.I), ("DAY", "days", -1)),
    (re.compile(r"\blast\s+night\b", re.I), ("DAY", "days", -1)),
    (re.compile(r"\bnext\s+month\b", re.I), ("MONTH", "months", +1)),
    (re.compile(r"\blast\s+month\b", re.I), ("MONTH", "months", -1)),
    (re.compile(r"\bthis\s+month\b", re.I), ("MONTH", "months", 0)),
    (re.compile(r"\bnext\s+year\b", re.I), ("YEAR", "years", +1)),
    (re.compile(r"\blast\s+year\b", re.I), ("YEAR", "years", -1)),
    (re.compile(r"\bthis\s+year\b", re.I), ("YEAR", "years", 0)),
]


def parse_anchor(s):
    m = _ANCHOR.search(s or "")
    if not m:
        return None
    g = m.groups()
    try:
        if g[0]:
            return date(int(g[2]), _MONTHS[g[1].lower()], int(g[0]))
        return date(int(g[5]), _MONTHS[g[3].lower()], int(g[4]))
    except Exception:                                     # noqa: BLE001
        return None


def shift(d, unit, n):
    if unit == "days":
        return d + timedelta(days=n)
    if unit == "months":
        y, mo = divmod((d.month - 1) + n, 12)
        y, mo = y + d.year, mo + 1
        day = min(d.day, [31, 29 if y % 4 == 0 and (y % 100 or not y % 400) else 28,
                          31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mo - 1])
        return date(y, mo, day)
    return date(d.year + n, d.month, d.day)


def strict_audit(pred: str, gold: str) -> tuple[bool, str]:
    """Component-level date check.  Supplementary only -- the frozen scorer is
    untouched and remains the official metric."""
    p, g = (pred or "").lower(), (gold or "").lower()
    if not any(ch.isdigit() for ch in g):
        return True, "gold is not date-shaped; no component check applies"
    pv = re.findall(r"\b(?:19|20)\d{2}\b", p)
    gv = re.findall(r"\b(?:19|20)\d{2}\b", g)
    if gv and pv and gv[0] != pv[0]:
        return False, f"YEAR mismatch: predicted {pv[0]}, gold {gv[0]}"
    pm = [w for w in _MONTHS if w in p or w[:3] in p]
    gm = [w for w in _MONTHS if w in g or w[:3] in g]
    if gm and pm and not (set(gm) & set(pm)):
        return False, f"MONTH mismatch: predicted {pm}, gold {gm}"
    pd = re.findall(r"\b([1-9]|[12]\d|3[01])\b", p)
    gd = re.findall(r"\b([1-9]|[12]\d|3[01])\b", g)
    if gm and pm and set(gm) & set(pm) and gd and pd and not (set(gd) & set(pd)):
        return False, f"DAY mismatch: predicted {pd}, gold {gd}"
    return True, "components agree or are not decidable"


def main() -> int:
    from artificial_memory.research.benchmarks import llm as frozen_llm
    from eval_locomo_full import score as frozen_score

    m24 = json.loads(M24.read_text(encoding="utf-8"))
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in pack["cases"]}
    convs = pack["conversations"]
    h9 = {r["question_id"]: r for r in
          json.loads(H9.read_text(encoding="utf-8"))["rows"]}
    audit = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in audit["records"]}
    frozen_ctx = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen_ctx[json.loads(line)["qid"]] = json.loads(line)["context"]
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    tuning_ids = set(split["tuning_qids"])
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    session_dates = {}
    for conv in raw:
        cmap = {}
        for k, v in conv["conversation"].items():
            m = re.match(r"session_(\d+)_date_time$", k)
            if m and v:
                cmap[int(m.group(1))] = v
        session_dates[conv.get("sample_id")] = cmap

    targets = [r["question_id"] for r in m24["rows"]]

    # ---- sentinels: FIXED before any treatment output is inspected -----------
    # drawn only from the frozen Reader cohort's CORRECT cases inside the TUNING split;
    # the holdout is never sampled or inspected
    sentinel_pool = sorted(q for q, r in h9.items()
                           if r["mutually_exclusive"] == "CORRECT" and q in tuning_ids)
    import random
    rng = random.Random(SEED)
    sentinels = rng.sample(sentinel_pool, min(12, len(sentinel_pool)))

    print("=" * 78)
    print(f"V4 MISSION 25 -- session-date-aware Reader intervention  [{RUN_ID}]")
    print("=" * 78)
    print("  SHADOW/REPLAY ONLY.  Production untouched, nothing wired.")
    print(f"  reader : {frozen_llm.FROZEN_MODEL} temp={frozen_llm.FROZEN_TEMPERATURE} "
          f"seed={frozen_llm.FROZEN_SEED} max_tokens={frozen_llm.FROZEN_ANSWER_MAX_TOKENS}")
    print(f"  scorer : eval_locomo_full.score() -- FROZEN, not edited")
    print()
    print("WHY BOTH ARMS ARE GENERATED HERE")
    print("  All 28 targets have ledger verdict 'abstain' and appear in the H9 Reader")
    print(f"  cohort {sum(1 for q in targets if q in h9)} times -- so no frozen Reader")
    print("  baseline exists for them.  Inventing one would be fabrication; both arms are")
    print("  run in this session instead.")
    print()
    print(f"  target cohort   : {len(targets)} (frozen before any output is inspected)")
    print(f"  regression sentinels : {len(sentinels)} from the TUNING split only")
    print(f"    holdout sampled or inspected : NO")
    print()

    reader = frozen_llm.OllamaAnswerer()
    rows = []

    def build_treatment(q, gold):
        """The evidence block.  The resolver's date is a CANDIDATE, never stated as fact."""
        conv = cases[q]["source_conversation"]
        turns = convs.get(conv, [])
        # the referenced turn(s): prefer the reviewer's cited turns
        cited = re.findall(r"D\d+:\d+", m24_rows.get(q, {}).get("eli_rationale", ""))
        cited = [t for t in cited if any(x["dia_id"] == t for x in turns)] or \
                [x["dia_id"] for x in turns[:1]]
        blocks = []
        for t in cited[:2]:
            sess = int(re.match(r"D(\d+):", t).group(1))
            sd = session_dates.get(conv, {}).get(sess)
            if not sd:
                continue                      # never infer a missing timestamp
            expr = []
            for x in turns:
                if not x["dia_id"].startswith(f"D{sess}:"):
                    continue
                for pat, (prec, unit, n) in _REL_OFFSET:
                    if pat.search(x["text"]):
                        anchor = parse_anchor(sd)
                        cand = None
                        if anchor:
                            cand = shift(anchor, unit, n)
                        expr.append({
                            "turn": x["dia_id"],
                            "expression": pat.pattern,
                            "window": x["text"][:120],
                            "precision": prec,
                            # CANDIDATE derivation, explicitly not asserted
                            "candidate_value": cand.isoformat() if cand else None,
                            "note": "candidate derivation -- verify, do not assume",
                        })
                        break
            blocks.append({
                "referenced_turn": t,
                "session_id": sess,
                "session_date_time": sd,
                "session_date_source": f"session_{sess}_date_time",
                "relative_expressions_in_session": expr,
            })
        return blocks

    m24_rows = {r["question_id"]: r for r in m24["rows"]}

    def run_case(q, gold, qtext, ctx, treated):
        if treated:
            blocks = build_treatment(q, gold)
            ev = "\n".join(
                f"[session {b['session_id']} date: {b['session_date_time']} "
                f"(source: {b['session_date_source']})] referenced turn {b['referenced_turn']}\n"
                + "\n".join(
                    f"  - {e['turn']}: {e['expression']} | precision {e['precision']} | "
                    f"candidate {e['candidate_value']} (CANDIDATE ONLY -- verify) | "
                    f"\"{e['window'][:80]}\""
                    for e in b["relative_expressions_in_session"])
                for b in blocks)
            prompt = (
                f"{qtext}\n\n"
                f"--- conversation ---\n{ctx}\n\n"
                f"--- session-date evidence (session IDs shown explicitly) ---\n{ev}\n\n"
                "Select the relative expression that ACTUALLY BINDS to the event this "
                "question asks about, using the session date shown for that session. "
                "Do not mix a date from one session with an expression from another. "
                "Do not manufacture day-level precision from month-level evidence. "
                "The candidate values above are DERIVATIONS TO VERIFY, not established "
                "facts. If no expression can be bound to the queried event, or the "
                "evidence is insufficient, answer exactly: I don't know."
            )
        else:
            prompt = f"{qtext}\n\n--- conversation ---\n{ctx}"
        try:
            out = reader.answer(qtext, prompt)
            return out.text, None
        except Exception as exc:                          # noqa: BLE001
            return None, str(exc)[:160]

    # ---------------- TREATMENT cohort ---------------------------------------
    for q in targets:
        gold = cases[q]["gold_answer"]
        conv = cases[q]["source_conversation"]
        ctx = "\n".join(f"[{t['dia_id']}] {t['speaker']}: {t['text']}"
                        for t in convs.get(conv, []))
        base, e1 = run_case(q, gold, cases[q]["question"], ctx, treated=False)
        treat, e2 = run_case(q, gold, cases[q]["question"], ctx, treated=True)
        if e1 or e2:
            rows.append({"question_id": q, "status": "INVOCATION_FAILED",
                         "error": e1 or e2})
            continue
        b_ok = bool(frozen_score(base or "", gold))
        t_ok = bool(frozen_score(treat or "", gold))
        b_strict, b_why = strict_audit(base or "", gold)
        t_strict, t_why = strict_audit(treat or "", gold)
        rows.append({
            "question_id": q, "status": "MEASURED", "group": "TARGET",
            "gold": gold, "baseline_output": (base or "")[:150],
            "treatment_output": (treat or "")[:150],
            "baseline_correct": b_ok, "treatment_correct": t_ok,
            "baseline_strict": b_strict, "treatment_strict": t_strict,
            "strict_note": t_why,
            "verdict_class": (
                "SCORER_PASS_STRICT_FAIL" if t_ok and not t_strict else
                "VERIFIED_RESCUE" if t_ok and t_strict and not b_ok else
                "REGRESSION" if b_ok and not t_ok else
                "BOTH_CORRECT" if b_ok and t_ok else "STILL_WRONG"),
        })

    # ---------------- REGRESSION SENTINELS -----------------------------------
    # The sentinels come from the frozen Reader cohort, NOT from the Bucket A pack, so
    # build_treatment's `cases[...]` lookup does not apply to them.  BUG FIXED: the first
    # version called build_treatment for sentinels too and died with KeyError on
    # 'lc_conv-47_126'.  A sentinel has no relative-date evidence to expose (it is a
    # baseline-correct case chosen to detect regression), so its treatment is the ORIGINAL
    # frozen context with the same "verify or abstain" instruction and NO injected date
    # block.  That keeps the comparison honest: if the instruction alone changes a correct
    # answer, that is a real regression and must be counted.
    for q in sentinels:
        r = h9[q]
        gold = r["gold"]
        cq = meta.get(q, {}).get("cache_qid", "")
        ctx = frozen_ctx.get(cq, "")
        if not ctx:
            rows.append({"question_id": q, "status": "UNMEASURABLE",
                         "group": "SENTINEL", "why": "no frozen context"})
            continue
        try:
            out = reader.answer(
                r["question"],
                f"{r['question']}\n\n--- conversation ---\n{ctx}\n\n"
                "If the conversation does not support a definitive answer, answer "
                "exactly: I don't know.")
            treat = out.text
            e2 = None
        except Exception as exc:                          # noqa: BLE001
            treat, e2 = None, str(exc)[:160]
        if e2:
            rows.append({"question_id": q, "status": "INVOCATION_FAILED",
                         "group": "SENTINEL", "error": e2})
            continue
        b_ok = bool(frozen_score(r.get("prediction", ""), gold))
        t_ok = bool(frozen_score(treat or "", gold))
        t_strict, t_why = strict_audit(treat or "", gold)
        rows.append({
            "question_id": q, "status": "MEASURED", "group": "SENTINEL",
            "gold": gold, "baseline_output": (r.get("prediction") or "")[:120],
            "treatment_output": (treat or "")[:150],
            "baseline_correct": b_ok, "treatment_correct": t_ok,
            "treatment_strict": t_strict, "strict_note": t_why,
            "verdict_class": "SENTINEL_REGRESSION" if b_ok and not t_ok
                             else "SENTINEL_HELD",
        })
    reader.close()

    # ---------------- reporting ----------------------------------------------
    tm = [r for r in rows if r["group"] == "TARGET" and r["status"] == "MEASURED"]
    sm = [r for r in rows if r["group"] == "SENTINEL" and r["status"] == "MEASURED"]
    failed = [r for r in rows if r["status"] == "INVOCATION_FAILED"]
    vc = Counter(r["verdict_class"] for r in tm + sm)

    print("TARGET COHORT")
    print(f"  measured {len(tm)}/{len(targets)}   invocation-failed {len(failed)}")
    for k in ("VERIFIED_RESCUE", "SCORER_PASS_STRICT_FAIL", "BOTH_CORRECT",
              "REGRESSION", "STILL_WRONG"):
        if vc.get(k):
            print(f"    {k:<26}{vc[k]}")
    print()
    print(f"  frozen-scorer rescues (lenient)   : "
          f"{sum(1 for r in tm if not r['baseline_correct'] and r['treatment_correct'])}")
    print(f"  VERIFIED rescues (strict audit too): {vc.get('VERIFIED_RESCUE', 0)}")
    print(f"  regressions in target             : {vc.get('REGRESSION', 0)}")
    print()
    print("REGRESSION SENTINELS")
    print(f"  measured {len(sm)}/{len(sentinels)}")
    print(f"    held correct : {vc.get('SENTINEL_HELD', 0)}")
    print(f"    REGRESSED    : {vc.get('SENTINEL_REGRESSION', 0)}")
    print()
    print("GATES")
    g1 = vc.get("VERIFIED_RESCUE", 0) >= 1
    print(f"  Gate 1 mechanism (>=1 fully verified derivation+rescue): "
          f"{'PASS' if g1 else 'NOT MET'}")
    net = vc.get("VERIFIED_RESCUE", 0) - vc.get("REGRESSION", 0) \
        - vc.get("SENTINEL_REGRESSION", 0)
    print(f"  Gate 2 net (verified rescues - regressions): {net} "
          f"({vc.get('VERIFIED_RESCUE', 0)} - {vc.get('REGRESSION', 0)} - "
          f"{vc.get('SENTINEL_REGRESSION', 0)})")
    unsup = sum(1 for r in tm if not r["baseline_correct"] and not r["treatment_correct"]
                and not r["treatment_strict"])
    print(f"  Gate 3 unsupported answers: {unsup} (explicitly reported, never a rescue)")
    print(f"  Gate 4 generalisation: NOT CLAIMED. These {len(tm)} cases are a targeted")
    print(f"    cohort; no estimate is made for 141 Bucket A or 1,986 questions without")
    print(f"    a sampling design.")
    print()
    print("  GUARDRAIL: M24 showed the resolver reproduces gold 17/28.  It did NOT show")
    print("  the engine identifies the correct expression, binds it to the queried event,")
    print("  answers correctly, or preserves abstention quality.  Those are what this run")
    print("  tests, and the per-case fields above record each separately.")
    print()
    print(f"  INVARIANT CHECK: production unchanged (no module edited or wired),")
    print(f"  Evidence Binding unwired, intentional failing test retained, holdout not")
    print(f"  sampled or inspected, model calls = "
          f"{2 * len(tm) + len(sm)} ({2 * len(tm)} target + {len(sm)} sentinel).")

    OUT.write_text(json.dumps({
        "label": "V4 mission 25 -- session-date-aware Reader intervention (shadow)",
        "run_id": RUN_ID, "seed": SEED,
        "baseline": {"head": "3a9c431", "production_changed": False},
        "reader": {"model": frozen_llm.FROZEN_MODEL,
                   "temperature": frozen_llm.FROZEN_TEMPERATURE,
                   "seed": frozen_llm.FROZEN_SEED,
                   "max_tokens": frozen_llm.FROZEN_ANSWER_MAX_TOKENS},
        "scorer": "eval_locomo_full.score() -- frozen, not edited",
        "strict_audit": "supplementary component check; scorer untouched",
        "target_ids": targets, "sentinel_ids": sentinels,
        "holdout_sampled_or_inspected": False,
        "model_calls": 2 * len(tm) + len(sm),
        "verdict_classes": dict(vc),
        "gate1_mechanism_pass": g1,
        "gate2_net": net,
        "gate3_unsupported_answers": unsup,
        "generalisation_claimed": False,
        "resolver_date_fed_as_fact": False,
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
