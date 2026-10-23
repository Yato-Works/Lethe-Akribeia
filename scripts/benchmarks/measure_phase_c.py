"""Phase C baseline -- measure the 67 unmeasured category-4 capabilities, code untouched.

Mission: evaluate the 67 (enumeration 53 + attribute 12 + counting 2) category-4
questions WITHOUT changing production code.  Record the current engine's
correct/wrong/abstain per question, then measure what the EXISTING deterministic
machinery could answer if routed correctly.

Two numbers are kept deliberately separate, because conflating them is how a
capability gets oversold:
  ANSWERABLE   the existing subsystem produces a candidate answer at all
  CORRECT      that candidate passes the strict repaired scorer

A high ANSWERABLE with low CORRECT means the machinery exists but needs routing or
verification.  A low ANSWERABLE means the machinery does not reach these shapes at
all -- which is the honest signal for whether to build.

Subsystems probed, all existing, all read-only:
  aggregation   CountingEngine.resolve_aggregation   (enumeration / set union)
  attribute     EntityAttributeResolver              (single-value attributes)
  identity      IdentityResolver over capitalised surfaces  (entity union)
  counting      CountingEngine.resolve_counting      (cardinality)

Nothing here writes to production paths.  This is a measurement harness.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory_gen.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory_gen.skills.autonomous_engines.counting_engine import (  # noqa: E402
    CountingEngine,
)
from artificial_memory_gen.skills.autonomous_engines.entity_attribute_resolver import (  # noqa: E402
    EntityAttributeResolver,
)
from artificial_memory_gen.skills.autonomous_engines.entity_identity import (  # noqa: E402
    IdentityResolver,
    Mention,
)
from artificial_memory_gen.skills.deterministic_memory_engine import (  # noqa: E402
    DeterministicMemoryEngine,
)

spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = scorer
try:
    spec.loader.exec_module(scorer)
except SystemExit:
    pass

AUDIT = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CAPABILITY = ROOT / "benchmark_results/evidence_audit/engine_capability.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/phase_c_baseline.json"

#: Same shape buckets the capability measurement used, so the cohort is identical.
SHAPE_PATTERNS = {
    "counting": re.compile(r"\bhow many\b|\bcount\b", re.I),
    "preference/attribute": re.compile(
        r"\bwhat .*(?:like|prefer|hobby|hobbies|favorite|favourite)\b", re.I),
    "enumeration": re.compile(
        r"^(?:what|which)\b.*\b(?:things?|items?|names?|books?|movies?|songs?|"
        r"places?|foods?|activities|hobbies|pets?|artists?|bands?)\b", re.I),
    "reason": re.compile(r"\bwhy\b", re.I),
}


def shape_of(question: str) -> str:
    q = question.strip()
    if SHAPE_PATTERNS["counting"].search(q):
        return "counting"
    if SHAPE_PATTERNS["preference/attribute"].search(q):
        return "preference/attribute"
    if SHAPE_PATTERNS["enumeration"].match(q):
        return "enumeration"
    if SHAPE_PATTERNS["reason"].search(q):
        return "reason"
    return "other open-domain"


def probe_subsystems(question: str, context: str, turns) -> dict:
    """Ask each existing subsystem for a candidate, without changing anything."""
    out: dict[str, str | None] = {}
    try:
        a = CountingEngine.resolve_aggregation(question, turns, context)
        out["aggregation"] = a.answer if getattr(a, "used", False) else None
    except Exception as exc:                       # noqa: BLE001
        out["aggregation"] = f"ERR:{type(exc).__name__}"
    try:
        a = EntityAttributeResolver.resolve_entity_attribute(question, turns, context)
        out["attribute"] = a.answer if getattr(a, "used", False) else None
    except Exception as exc:                       # noqa: BLE001
        out["attribute"] = f"ERR:{type(exc).__name__}"
    try:
        a = CountingEngine.resolve_counting(question, turns, context)
        out["counting"] = a.answer if getattr(a, "used", False) else None
    except Exception as exc:                       # noqa: BLE001
        out["counting"] = f"ERR:{type(exc).__name__}"

    # Identity: union every capitalised surface per owner.  This is the module the
    # plural-list cluster needed; here it is measured, not wired in.
    try:
        owner = ""
        m = re.search(r"\b([A-Z][a-z]+)'s\b", question)
        if m:
            owner = m.group(1)
        resolver = IdentityResolver()
        for t in turns:
            for name in re.findall(r"\b(?:named|called)\s+([A-Z][a-z]{2,})\b", t.text):
                resolver.add(Mention(
                    turn_id=t.dia_id, speaker=t.speaker, surface=name,
                    text=t.text, owner=owner or t.speaker, entity_type="entity"))
        names = sorted({e.canonical for e in resolver.entities.values()})
        out["identity_union"] = ", ".join(names) if names else None
    except Exception as exc:                       # noqa: BLE001
        out["identity_union"] = f"ERR:{type(exc).__name__}"
    return out


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    records = audit["records"]
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    # ---- The 67: category 4, NOT_A_DATE, in the three measurable shapes.
    cohort = []
    for r in records:
        if r["category"] != 4 or r["class"] != "NOT_A_DATE":
            continue
        sh = shape_of(r["question"])
        if sh in ("counting", "preference/attribute", "enumeration"):
            cohort.append((r, sh))
    print(f"PHASE C COHORT : {len(cohort)}")
    print(f"  shapes       : {dict(Counter(sh for _, sh in cohort))}")
    print()

    rows: list[dict] = []
    agg = {"current_correct": 0, "current_wrong": 0, "current_abstain": 0}
    per_shape: dict[str, Counter] = {}
    answerable = Counter()
    correct_by_probe = Counter()

    for r, sh in cohort:
        ctx = cache.get(r["cache_qid"]) or {}
        context = ctx.get("context") or ""
        gold = r["gold"]
        row = {
            "question_id": r["question_id"],
            "shape": sh,
            "question": r["question"],
            "gold": gold,
            "gold_is_list": bool(getattr(scorer, "is_list_gold", lambda g: False)(gold)),
        }
        if not context:
            row["current"] = "NO_CONTEXT"
            rows.append(row)
            continue

        turns = parse_turns(context)

        # ---- CURRENT engine, exactly as production routes it.
        ans = DeterministicMemoryEngine.resolve(r["question"], context)
        row["current_answer"] = ans.answer
        row["current_used"] = bool(ans.used)
        row["current_detail"] = (ans.detail or "")[:100]
        if not ans.used:
            row["current"] = "abstain"
            agg["current_abstain"] += 1
        elif scorer.score(ans.answer, gold):
            row["current"] = "correct"
            agg["current_correct"] += 1
        else:
            row["current"] = "wrong"
            agg["current_wrong"] += 1

        # ---- EXISTING SUBSYSTEMS, probed read-only.
        probes = probe_subsystems(r["question"], context, turns)
        row["probes"] = probes
        for name, val in probes.items():
            if val and not str(val).startswith("ERR"):
                answerable[name] += 1
                if scorer.score(str(val), gold):
                    correct_by_probe[name] += 1

        per_shape.setdefault(sh, Counter())[row["current"]] += 1
        rows.append(row)

    print("=" * 74)
    print(f"CURRENT ENGINE ON THE COHORT (n={len(cohort)}, production routing, "
          f"no code changed)")
    print("=" * 74)
    print(f"  correct : {agg['current_correct']}")
    print(f"  wrong   : {agg['current_wrong']}")
    print(f"  abstain : {agg['current_abstain']}")
    print()
    print("  by shape:")
    for sh, c in sorted(per_shape.items()):
        print(f"    {sh:<22} n={sum(c.values()):>3}  correct={c.get('correct', 0):>3} "
              f"wrong={c.get('wrong', 0):>3} abstain={c.get('abstain', 0):>3}")
    print()

    print("=" * 74)
    print("EXISTING SUBSYSTEMS, PROBED SEPARATELY")
    print("=" * 74)
    print(f"  {'subsystem':<20}{'produced a candidate':>22}{'scored CORRECT':>16}")
    for name in ("aggregation", "attribute", "counting", "identity_union"):
        print(f"  {name:<20}{answerable.get(name, 0):>22}{correct_by_probe.get(name, 0):>16}")
    print()
    print("  ANSWERABLE and CORRECT are different numbers on purpose.  A large gap")
    print("  means the machinery exists but needs routing/verification; a small")
    print("  ANSWERABLE means it does not reach these shapes at all.")
    print()

    print("=" * 74)
    print("PER-CASE DETAIL (first 20)")
    print("=" * 74)
    for row in rows[:20]:
        print(f"  {row['question_id']:<18} [{row['shape']:<20}] current={row.get('current')}")
        print(f"    Q    : {row['question'][:62]}")
        print(f"    gold : {str(row['gold'])[:56]!r}")
        print(f"    now  : {str(row.get('current_answer'))[:56]!r}")
        pr = row.get("probes") or {}
        best = [f"{k}={str(v)[:30]}" for k, v in pr.items()
                if v and not str(v).startswith("ERR")]
        if best:
            print(f"    probe: {' | '.join(best)[:120]}")
    if len(rows) > 20:
        print(f"  ... {len(rows) - 20} more in the JSON")

    OUT.write_text(json.dumps({
        "label": f"phase C baseline -- {len(cohort)} category-4 non-date capabilities",
        "cohort_n": len(cohort),
        "denominator_note": "the earlier '67' came from coarse buckets in the "
                            "capability measurement; this strict question-shape "
                            "classification yields 96.  Both are recorded rather "
                            "than one silently replacing the other.",
        "shapes": dict(Counter(sh for _, sh in cohort)),
        "current_engine": agg,
        "per_shape": {k: dict(v) for k, v in per_shape.items()},
        "probe_answerable": dict(answerable),
        "probe_correct": dict(correct_by_probe),
        "scorer": "eval_locomo_full.score (date + day + numeral + month guards)",
        "note": "production code NOT changed.  ANSWERABLE != CORRECT by design.",
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
