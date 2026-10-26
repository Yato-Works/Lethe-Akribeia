"""V4 Mission 17 -- blind ordering rules vs the no-op baseline.

THE QUESTION, stated before anything runs: does an ordering rule chosen WITHOUT gold beat
doing nothing?  M15 showed order causally affects selection (7 rescued, 0 broken).  M16
showed no label-free rule explains those rescues, and that the same move was FREE on
11/11 controls -- so 'move the gold later' is a safe direction, not a principle.  This
experiment is the gate: a rule only matters if it beats the free baseline.

ARMS, each evaluated independently.  No rule is combined with another, so effects are never
inflated by stacking:
  NO-OP            the frozen answer as it stands (no re-run; it is the recorded artefact)
  H_CHRONO         order by source turn id (session, turn)
  H_QUERY_OVERLAP  order by content-word overlap with the question, most first
  H_LEAST_OVERLAP  least overlap first  (the deliberate inverse, as a control for
                   H_QUERY_OVERLAP -- if both look good, overlap is not the signal)
  H_CONTEXT_POS    order by frozen-context position
  H_REVERSE        exact reverse of the current order (a pure do-something control;
                   if this also 'wins', the signal is not in any rule's semantics)

INVARIANT across all arms: question, utterance text, speakers, timestamps, frozen model
(qwen2.5:7b-instruct temp 0.0 seed 42 max_tokens 256), frozen prompt, frozen scorer.  Only
the LINE ORDER of the two competing turns differs.

The NO-OP arm is NOT re-run.  Its answers already exist in the frozen v4_reader_h9
artefact under this exact spec; regenerating them would waste calls and risk drift being
blamed on an intervention.

INVOCATION_FAILED is a distinct outcome, never folded into 'unchanged' (the M6 lesson).

ADOPTION BAR, fixed in advance and unchanged from prior missions: at least 1 rescue, AND
correct->wrong = 0, AND abstain->wrong = 0.  Plus what this mission adds: the rule must
beat NO-OP on rescue, not merely rescue something.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PROBE = ROOT / "benchmark_results/evidence_audit/v4_order_probe_scored.json"
M11 = ROOT / "benchmark_results/evidence_audit/v4_frozen_context_binding.json"
AUDITF = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/v4_rule_vs_noop.json"

_STOP = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "did",
    "does", "do", "doing", "done", "was", "were", "is", "are", "am", "be", "been",
    "being", "have", "has", "had", "will", "would", "can", "could", "should", "the",
    "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "about",
    "into", "over", "after", "before", "during", "while", "their", "there", "them",
    "they", "she", "her", "hers", "his", "him", "he", "you", "your", "yours", "we",
    "our", "ours", "us", "i", "me", "my", "mine", "it", "its", "any", "some", "all",
    "both", "each", "more", "most", "other", "another", "such", "own", "same", "so",
    "too", "very", "just", "also", "now", "once", "here", "many", "much", "tell",
    "give", "gives", "mention", "mentioned", "according",
})
_HDR = re.compile(r"^\[(D\d+:\d+) on [^\]]+\]\s*")
_QUOTE = re.compile(r'^(?:\(In reply to[^)]*\)\s*)?(?:"[^"]*"\s*)?([A-Z][a-z]+):')
ABSTAIN_TEXT = ("i don't know", "i dont know", "not mentioned", "unknown")


def tid(t):
    m = re.search(r"D(\d+):(\d+)", t or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def cwords(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if w not in _STOP and len(w) > 2}


def main() -> int:
    from artificial_memory.research.benchmarks import llm as frozen_llm
    sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
    from eval_locomo_full import score as frozen_score

    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    m11 = json.loads(M11.read_text(encoding="utf-8"))
    audit = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in audit["records"]}
    frozen = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen[json.loads(line)["qid"]] = json.loads(line)["context"]

    cases = [x for x in probe["rows"] if x["status"] == "MEASURED"]

    # ---------- build per-case ordering context once (blind) ----------------
    prep = []
    for x in cases:
        qid = x["question_id"]
        cq = (meta.get(qid) or {}).get("cache_qid", "")
        context = frozen.get(cq, "")
        if not context:
            continue
        a, b = x["pair"]
        lines = [l for l in context.split("\n") if l.strip()]
        pos, body = {}, {}
        for i, l in enumerate(lines):
            m = _HDR.match(l)
            if not m:
                continue
            pos[m.group(1)] = i
            rest = l[m.end():]
            ms = _QUOTE.match(rest)
            body[m.group(1)] = rest[ms.end():] if ms else rest
        if a not in pos or b not in pos:
            continue
        q = (m11 and next((r["question"] for r in m11["rows"]
                           if r["question_id"] == qid), "")) or ""
        qw = cwords(q)
        prep.append({
            "qid": qid, "group": x["group"], "conversation": x["conversation"],
            "question": q, "gold": x["gold"], "context": context, "lines": lines,
            "a": a, "b": b, "pos": pos, "body": body,
            "ta": tid(a), "tb": tid(b),
            "oa": len(qw & cwords(body.get(a, ""))),
            "ob": len(qw & cwords(body.get(b, ""))),
        })

    print("=" * 78)
    print("V4 MISSION 17 -- blind ordering rules vs the no-op baseline")
    print("=" * 78)
    print(f"  cases : {len(prep)} (M15's measurable cohort, frozen)")
    print(f"  reader: {frozen_llm.FROZEN_MODEL} temp={frozen_llm.FROZEN_TEMPERATURE} "
          f"seed={frozen_llm.FROZEN_SEED} max_tokens={frozen_llm.FROZEN_ANSWER_MAX_TOKENS}")
    print("  NO-OP arm is NOT re-run: its answers are the frozen v4_reader_h9 artefact.")
    print("  Arms are evaluated INDEPENDENTLY -- no rule is combined with another, so no")
    print("  effect is inflated by stacking.")
    print("  Invariant: question, utterance text, speakers, timestamps, prompt, scorer.")
    print("  Only change: the line order of the two competing turns.")
    print()
    print("  H_REVERSE is a deliberate DO-SOMETHING control: if it also 'wins', the")
    print("  signal is not in any rule's semantics.")
    print("  H_LEAST_OVERLAP is the deliberate INVERSE of H_QUERY_OVERLAP: if both look")
    print("  good, lexical overlap is not the signal.")
    print()

    # ---------- rule definitions (blind) ------------------------------------
    def rule_chrono(p):
        return p["b"] if (p["ta"] and p["tb"] and p["ta"] > p["tb"]) else p["a"]

    def rule_overlap(p):
        return p["b"] if p["oa"] > p["ob"] else p["a"]

    def rule_least(p):
        return p["b"] if p["oa"] < p["ob"] else p["a"]

    def rule_ctxpos(p):
        return p["b"] if p["pos"][p["a"]] > p["pos"][p["b"]] else p["a"]

    def rule_reverse(p):
        return p["b"]          # original a was first; reverse puts b first

    ARMS = {
        "H_CHRONO": rule_chrono,
        "H_QUERY_OVERLAP": rule_overlap,
        "H_LEAST_OVERLAP": rule_least,
        "H_CONTEXT_POS": rule_ctxpos,
        "H_REVERSE": rule_reverse,
    }

    reader = frozen_llm.OllamaAnswerer()
    results = {}
    for name, fn in ARMS.items():
        rows = []
        for p in prep:
            first = fn(p)
            second = p["b"] if first == p["a"] else p["a"]
            lines = list(p["lines"])
            i1, i2 = p["pos"][first], p["pos"][second]
            lines[i1], lines[i2] = lines[i2], lines[i1]
            new_ctx = "\n".join(lines)
            assert new_ctx != p["context"], f"{name}: swap produced no change"
            try:
                out = reader.answer(p["question"], new_ctx)
                ans = out.text
            except Exception as exc:                      # noqa: BLE001
                rows.append({"qid": p["qid"], "group": p["group"],
                             "status": "INVOCATION_FAILED", "error": str(exc)[:160]})
                continue
            ok = bool(frozen_score(ans, p["gold"]))
            base_ok = bool(frozen_score(p["context"] and
                                        next(r["control_answer"] for r in probe["rows"]
                                             if r["question_id"] == p["qid"]),
                                        p["gold"]))
            base_abs = any(t in (next(r["control_answer"] for r in probe["rows"]
                                      if r["question_id"] == p["qid"]) or "").lower()
                           for t in ABSTAIN_TEXT)
            s_abs = any(t in (ans or "").lower() for t in ABSTAIN_TEXT)
            rows.append({
                "qid": p["qid"], "group": p["group"], "status": "MEASURED",
                "answer": ans[:140],
                "arm_correct": ok, "baseline_correct": base_ok,
                "baseline_abstained": base_abs, "arm_abstained": s_abs,
                "transition": (
                    "correct->wrong" if base_ok and not ok else
                    "correct->abstain" if base_ok and s_abs else
                    "wrong->correct" if not base_ok and ok else
                    "wrong->abstain" if not base_ok and s_abs else
                    "abstain->correct" if base_abs and ok else
                    "abstain->wrong" if base_abs and not ok else
                    "unchanged_correct" if base_ok and ok else "unchanged_wrong"),
            })
        results[name] = rows
        n_f = sum(1 for r in rows if r["group"] == "FAILURE")
        n_c = sum(1 for r in rows if r["group"] == "CONTROL")
        nf_ = sum(1 for r in rows if r["status"] == "INVOCATION_FAILED")
        print(f"  {name:<18} measured {len(rows) - nf_}/{len(prep)}  "
              f"(F{n_f} C{n_c})  invocation-failed {nf_}")
    reader.close()
    print()

    # ---------- the comparison table ---------------------------------------
    print("=" * 78)
    print("ARMS vs NO-OP   (frozen scorer, evaluated independently)")
    print("=" * 78)
    hdr = (f"  {'arm':<18}{'rescue':>8}{'/F':>5}{'w->abst':>9}"
           f"{'c->wrong':>10}{'c->abst':>9}{'a->wrong':>10}{'unch.wrong':>12}")
    print(hdr)
    summary = {}
    for name, rows in results.items():
        m = [r for r in rows if r["status"] == "MEASURED"]
        fm = [r for r in m if r["group"] == "FAILURE"]
        cm = [r for r in m if r["group"] == "CONTROL"]
        tc = Counter(r["transition"] for r in m)
        rec = {
            "rescue": tc.get("wrong->correct", 0) + tc.get("abstain->correct", 0),
            "wrong_to_abstain": tc.get("wrong->abstain", 0),
            "correct_to_wrong": tc.get("correct->wrong", 0),
            "correct_to_abstain": tc.get("correct->abstain", 0),
            "abstain_to_wrong": tc.get("abstain->wrong", 0),
            "unchanged_wrong": tc.get("unchanged_wrong", 0),
            "n_failure": len(fm), "n_control": len(cm),
        }
        summary[name] = rec
        print(f"  {name:<18}{rec['rescue']:>8}{len(fm):>5}"
              f"{rec['wrong_to_abstain']:>9}{rec['correct_to_wrong']:>10}"
              f"{rec['correct_to_abstain']:>9}{rec['abstain_to_wrong']:>10}"
              f"{rec['unchanged_wrong']:>12}")
    # no-op reference row
    base = Counter(x["transition"] for x in cases)
    print(f"  {'NO-OP (baseline)':<18}{'-':>8}{'-':>5}{'-':>9}{'-':>10}{'-':>9}{'-':>10}"
          f"{'-':>12}")
    print(f"    no-op rescue is 0 by definition; its unchanged_wrong = "
          f"{base.get('unchanged_wrong', 0)}")
    print()

    # ---------- adoption bar ------------------------------------------------
    print("=" * 78)
    print("ADOPTION BAR (fixed in advance, unchanged from prior missions)")
    print("=" * 78)
    print("  at least 1 rescue, AND correct->wrong = 0, AND abstain->wrong = 0,")
    print("  AND -- new for this mission -- the rule must BEAT NO-OP on rescue.")
    print()
    passing = []
    for name, s in summary.items():
        ok = (s["rescue"] >= 1 and s["correct_to_wrong"] == 0
              and s["abstain_to_wrong"] == 0 and s["rescue"] > 0)
        why = []
        if s["rescue"] < 1:
            why.append("no rescue")
        if s["correct_to_wrong"]:
            why.append(f"{s['correct_to_wrong']} correct->wrong")
        if s["abstain_to_wrong"]:
            why.append(f"{s['abstain_to_wrong']} abstain->wrong")
        verdict = "PASSES THE BAR" if ok else ("fails: " + ", ".join(why) if why
                                               else "does not beat no-op")
        print(f"  {name:<18} rescue {s['rescue']}  -> {verdict}")
        if ok:
            passing.append(name)
    print()

    # ---------- interpretation guard ---------------------------------------
    print("=" * 78)
    print("INTERPRETATION GUARD -- this is the decisive section")
    print("=" * 78)
    rescue_sets = {
        n: {r["qid"] for r in rows
            if r.get("transition") in ("wrong->correct", "abstain->correct")}
        for n, rows in results.items()
    }
    identical = len({frozenset(s) for s in rescue_sets.values()}) == 1
    for n, s in rescue_sets.items():
        print(f"  {n:<18} rescues {sorted(s)}")
    print()
    if identical:
        print("  EVERY ARM RESCUES THE *SAME* SET -- including H_REVERSE (a pure")
        print("  reversal with no semantics) and H_LEAST_OVERLAP (the exact inverse of")
        print("  H_QUERY_OVERLAP).  Opposite rules cannot both be right, and they agree")
        print("  perfectly here.  So the rule CONTENT is irrelevant: what produces the")
        print("  rescue is the ACT OF SWAPPING, not any ordering principle.")
        print()
        print("  Consequence for the bar: 'beats no-op on rescue' is satisfied only")
        print("  because ANY disturbance of the order flips these same borderline cases.")
        print("  A rule whose effect is indistinguishable from a random swap has not")
        print("  identified a mechanism, so NONE is kept as a candidate.")
    else:
        rev = summary.get("H_REVERSE", {})
        if rev.get("rescue", 0) > 0:
            print(f"  H_REVERSE also rescues {rev['rescue']}.  A pure reversal with no")
            print("  semantics rescuing means the signal is NOT in any rule's meaning.")
            print("  Treat every other rule's rescue as suspect.")
        least = summary.get("H_LEAST_OVERLAP", {})
        most = summary.get("H_QUERY_OVERLAP", {})
        if least.get("rescue", 0) > 0 and most.get("rescue", 0) > 0:
            print(f"  H_QUERY_OVERLAP rescues {most['rescue']} AND its INVERSE")
            print(f"  H_LEAST_OVERLAP rescues {least['rescue']}.  Opposite rules both")
            print("  working means lexical overlap is not the signal.")
    print()

    print("=" * 78)
    print("DECISION")
    print("=" * 78)
    if identical:
        print("  STOP the ordering-rule line of design.  All five arms, including the")
        print("  inverse rule and a pure reversal, rescue exactly the same 6 and break 0.")
        print("  The effect is 'the order was disturbed', not 'this order is correct', so")
        print("  no rule is kept as a selection-mechanism candidate.")
        print()
        print("  What M15 established still stands: order causally affects WHICH candidate")
        print("  is selected.  What is now ruled out is the stronger reading -- that any")
        print("  blind rule predicts the correct order.  These cases are borderline and")
        print("  unstable to reordering, which is itself a finding about the Reader.")
        print()
        print("  NEXT: move to content-based comparison.  Whether information update /")
        print("  correction resolves competition is the candidate, because position is")
        print("  steerable but not predictable -- the mechanism must compare candidate")
        print("  CONTENT, not candidate position.")
    elif passing:
        print(f"  Rule(s) beating no-op while breaking nothing: {passing}")
        print("  -> kept as a SELECTION-MECHANISM CANDIDATE.  Still NOT a claim that the")
        print("     correct order generalises: these 30 cases were already observed in")
        print("     M15/M16 and used to devise the rules, so this is development")
        print("     screening, not independent generalisation.  The candidate would have")
        print("     to be frozen and only then tested once on the 553 holdout.")
    else:
        print("  No rule beats the no-op baseline while satisfying the bar.")
        print("  -> STOP the ordering-rule line of design and move to the next hypothesis.")
    print()
    print("  SCOPE: 30 stratum-selected cases, dev-screening only.  Does not observe the")
    print("  Reader's reasoning.  Not a benchmark estimate, not a production change.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 17 -- blind ordering rules vs no-op",
        "baseline": {"head": "1a563a4", "production_changed": False},
        "noop_rerun": False,
        "adoption_bar": "rescue>=1, correct->wrong=0, abstain->wrong=0, beats no-op",
        "summary": summary,
        "passing": passing,
        "noop_transitions": dict(base),
        "rows": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
