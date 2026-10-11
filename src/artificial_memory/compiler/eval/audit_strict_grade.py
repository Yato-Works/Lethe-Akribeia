"""Paired strict-grading audit for two benchmark runs.

WHY THIS EXISTS
---------------
A production scorer credits tolerant partial matches, so a run can gain
questions without the reader getting better.  Measured case: a change reported
as +0.35pt end-to-end was +1.59pt of evidence retention, and under a strict
rubric the net real change was +21 questions -- McNemar p=0.0649 on
STRICT+NEAR, i.e. NOT significant.  The headline was noise.

This tool grades two runs with ONE fixed rubric and reports:
  Phase 1  tier distribution in both arms
  Phase 2  paired transitions, separating real gains/losses (those that leave
           or enter the tolerant band) from movement inside it
  Phase 3  McNemar on each tier, so no delta is reported without a p-value

The rubric is defined once below and is identical for both arms.  Nothing about
which arm is the treatment leaks into grading.

Tiers
  STRICT   normalised prediction == normalised gold
  NEAR     containment either direction, list golds excluded (no subset credit)
  PARTIAL  token overlap >= 0.5 of the shorter side, or fuzzy token match >=
           0.85 -- this is what the production scorer accepts
  WRONG    everything else, abstentions included

A PARTIAL credit is deliberately NOT counted as an improvement.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from artificial_memory.compiler.eval.stat_test import StatisticalTester  # noqa: E402

STRICT, NEAR, PARTIAL, WRONG = "strict", "near", "partial", "wrong"
TIERS = (STRICT, NEAR, PARTIAL, WRONG)
ALPHA = 0.05

# Ordered pairs that count as an improvement / a worsening.
_ORDER = {WRONG: 0, PARTIAL: 1, NEAR: 2, STRICT: 3}


def norm(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_list_gold(gold: str) -> bool:
    """Is this gold an enumeration answer (>=2 comma-separated items)?

    Subset credit is refused for these: naming one item of a five-item gold is
    an incomplete answer, not a match.

    A comma alone is NOT sufficient.  Ordinary sentences contain commas
    ("Likely no, though she likes reading") and would otherwise be misread as
    lists, which would block the containment credit they legitimately earn.
    Two extra conditions must hold:
      * at least two comma-separated chunks are non-empty, AND
      * each of the first two chunks is SHORT (<= 6 words) -- enumeration items
        are noun phrases, not clauses
    """
    chunks = [x.strip() for x in str(gold).split(",") if x.strip()]
    if len(chunks) < 2:
        return False
    heads = chunks[:2]
    if any(len(c.split()) > 6 for c in heads):
        return False
    # a clause-heavy chunk ("though she likes reading") is prose, not an item
    clause_starters = ("though", "because", "which", "who", "that", "and ",
                       "but ", "while", "when ", "so ", "she ", "he ", "they ")
    return not any(c.lower().startswith(clause_starters) for c in heads)


def _items(s: str) -> set[str]:
    """Item set of an enumeration answer, normalised."""
    return {norm(x) for x in str(s).split(",") if norm(x)}


def _list_relation(pred: str, gold: str) -> str:
    """Grade an enumeration answer by set relation, never by substring.

    STRICT when the two item sets are equal.  Otherwise WRONG: a missing item
    is an incomplete answer and an extra item is a wrong one.  This is the tier
    the production scorer was quietly crediting as partial.
    """
    gp, gg = _items(pred), _items(gold)
    if not gg or not gp:
        return WRONG
    if gp == gg:
        return STRICT
    return WRONG


def grade(pred: str, gold: str) -> str:
    """Assign exactly one tier.  Fixed rule set, identical for every arm."""
    p, g = norm(pred), norm(gold)
    if not g:
        return WRONG
    if p == g:
        return STRICT
    if g in p:
        # A list gold only matches when the prediction enumerates the same set.
        # Super/subset in either direction is an incomplete or over-complete
        # answer, neither of which is what was asked.
        if is_list_gold(gold) or is_list_gold(pred):
            return _list_relation(pred, gold)
        return NEAR
    # An empty prediction is contained in every gold, so require it to carry
    # real content before granting containment credit.
    if p and p in g and not is_list_gold(gold):
        return NEAR
    pt, gt = set(p.split()), set(g.split())
    if pt and gt:
        shorter = pt if len(pt) <= len(gt) else gt
        if len(pt & gt) / max(1, len(shorter)) >= 0.5:
            # A list gold gets NO subset credit at any tier.  "Rome" for
            # "Paris, Rome" is a complete answer to a different question, and
            # the tolerant band is exactly where that used to slip through.
            if is_list_gold(gold) and not is_list_gold(pred):
                return WRONG
            return PARTIAL
    if len(pt) > 3 and len(gt) >= 4:
        if any(difflib.SequenceMatcher(None, a, b).ratio() >= 0.85
               for a in pt for b in gt):
            if is_list_gold(gold) and not is_list_gold(pred):
                return WRONG
            return PARTIAL
    return WRONG


def load(path: str | Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("results", data)
    return {str(r["question_id"]): r for r in rows}


def audit(path_a: str | Path, path_b: str | Path,
          name_a: str = "A", name_b: str = "B") -> str:
    a_rows, b_rows = load(path_a), load(path_b)
    ids = sorted(set(a_rows) & set(b_rows))
    ga = {i: grade(a_rows[i]["predicted_answer"], a_rows[i]["ground_truth"]) for i in ids}
    gb = {i: grade(b_rows[i]["predicted_answer"], b_rows[i]["ground_truth"]) for i in ids}
    ca, cb = Counter(ga.values()), Counter(gb.values())
    n = len(ids)

    lines = []
    add = lines.append
    add("=" * 78)
    add(f"STRICT GRADING AUDIT: {name_a} vs {name_b}   (n={n})")
    add("=" * 78)
    add(f"{'tier':10s} {name_a:>16s} {name_b:>16s} {'delta':>10s}")
    add("-" * 78)
    for t in TIERS:
        add(f"{t:10s} {ca[t]:7d} ({ca[t]/n*100:5.2f}%) "
            f"{cb[t]:7d} ({cb[t]/n*100:5.2f}%) "
            f"{(cb[t]-ca[t])/n*100:+9.2f}pt")
    add("-" * 78)

    # Phase 2 -- paired transitions
    up = [i for i in ids if _ORDER[gb[i]] > _ORDER[ga[i]]]
    down = [i for i in ids if _ORDER[gb[i]] < _ORDER[ga[i]]]
    real_up = [i for i in up if ga[i] != PARTIAL and gb[i] in (STRICT, NEAR)]
    real_down = [i for i in down if gb[i] == WRONG and ga[i] in (STRICT, NEAR)]
    add("")
    add(f"changed: {len(up)+len(down)}   improved: {len(up)}   worsened: {len(down)}")
    add(f"REAL improvements (leave tolerant band): {len(real_up)}")
    add(f"REAL regressions (fall to WRONG from STRICT/NEAR): {len(real_down)}")
    add(f"REAL NET: {len(real_up)-len(real_down):+d} questions")
    add("")
    add("improvement transitions:")
    for (x, y), c in sorted(Counter((ga[i], gb[i]) for i in up).items()):
        add(f"  {x:8s} -> {y:8s}  {c:4d}")
    add("regression transitions:")
    for (x, y), c in sorted(Counter((ga[i], gb[i]) for i in down).items()):
        add(f"  {x:8s} -> {y:8s}  {c:4d}")
    add("")

    # Phase 3 -- uncertainty
    tester = StatisticalTester(alpha=ALPHA)
    add("=" * 78)
    add("PHASE 3 -- McNemar, paired (no delta reported without a p-value)")
    add("=" * 78)
    verdicts = []
    for label, sel in (("STRICT only", lambda g: g == STRICT),
                       ("STRICT+NEAR", lambda g: g in (STRICT, NEAR)),
                       ("STRICT+NEAR+PARTIAL", lambda g: g != WRONG)):
        a = [sel(ga[i]) for i in ids]
        b = [sel(gb[i]) for i in ids]
        m = tester.mcnemar_test(a, b)
        add(f"{label:24s} {sum(a):5d} -> {sum(b):5d}  "
            f"({(sum(b)-sum(a))/n*100:+6.2f}pt)  "
            f"b={m.b:3d} c={m.c:3d} n_disc={m.n_discordant:3d}  "
            f"p={m.p_value:.4f}  {'SIGNIFICANT' if m.significant else 'not significant'}")
        verdicts.append((label, m.significant))
    add("")
    add("VERDICT: " + ("at least one tier is significant -- the change is real"
                       if any(s for _, s in verdicts)
                       else "NOT SIGNIFICANT at any tier -- the headline is noise"))
    add("=" * 78)
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: audit_strict_grade.py <run_a.json> <run_b.json>")
        return 2
    print(audit(sys.argv[1], sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
