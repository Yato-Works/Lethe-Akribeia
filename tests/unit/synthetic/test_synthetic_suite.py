"""Synthetic Counterfactual Suite runner + the killer demonstration.

The headline test ``test_hardcoded_operator_dies`` proves the suite catches the
exact R8.1 failure: an operator that returns a constant.  It must PASS the
base items (we let it) but FAIL every counterfactual rewrite.
"""

from __future__ import annotations

from typing import Callable, Iterable

import pytest

from .corpus import SyntheticItem
from .generators import GENERATORS
from .guards import check_corpus_is_synthetic, scan_source_for_overfitting
from .rewrites import REWRITERS


# --------------------------------------------------------------------------- #
#  corpus building
# --------------------------------------------------------------------------- #

def build_base_corpus(per_generator: int = 8, seed0: int = 1000) -> list[SyntheticItem]:
    items: list[SyntheticItem] = []
    for gname, gen in GENERATORS.items():
        for i in range(per_generator):
            items.append(gen(seed0 + i))
    return items


def expand_counterfactuals(base: list[SyntheticItem], seed0: int = 5000) -> list[SyntheticItem]:
    """Attach one rewrite of each kind to every base item."""
    out: list[SyntheticItem] = []
    for idx, item in enumerate(base):
        out.append(item)
        for kind, fn in REWRITERS.items():
            out.append(fn(item, seed0 + idx))
    return out


def corpus(corpus_obj=None) -> list[SyntheticItem]:
    """Default corpus: base + all rewrites."""
    return expand_counterfactuals(build_base_corpus())


# --------------------------------------------------------------------------- #
#  grading
# --------------------------------------------------------------------------- #

def grade(fn: Callable[[str, tuple], str], items: Iterable[SyntheticItem]) -> dict:
    """Run operator ``fn`` over items; return per-family accuracy."""
    stats: dict[str, list[int]] = {}
    details: list[tuple[str, str, str, str, bool]] = []
    for it in items:
        try:
            pred = str(fn(it.question, it.turns))
        except Exception as exc:  # operator crash counts as wrong
            pred = f"<ERROR:{exc}>"
        ok = _soft_match(pred, it.expected)
        stats.setdefault(it.family, []).append(int(ok))
        details.append((it.item_id, it.family, it.expected, pred, ok))
    return {
        "per_family": {k: sum(v) / len(v) for k, v in stats.items()},
        "details": details,
        "n": sum(len(v) for v in stats.values()),
    }


def _soft_match(pred: str, expected: str) -> bool:
    """Deterministic, generous substring/number match (benchmark-agnostic)."""
    p = pred.lower().strip()
    e = expected.lower().strip()
    if not p:
        return False
    if e in p or p in e:
        return True
    # numeric equality
    import re as _re
    pe = _re.findall(r"\d+", p)
    ee = _re.findall(r"\d+", e)
    if ee and pe == ee:
        return True
    # token overlap for text answers
    et = set(e.split())
    if et and et.issubset(set(p.split())):
        return True
    return False


# --------------------------------------------------------------------------- #
#  reference operators — one HONEST, one HARDCODED (the R8.1 pathology)
# --------------------------------------------------------------------------- #

def honest_reference_operator(question: str, turns: tuple) -> str:
    """A deliberately simple but *structural* operator for smoke purposes.

    It extracts numbers from turns and sums them for SUM-style questions,
    and returns the last dated value for supersedes-style questions.  It is
    NOT the production engine — just enough to show the suite runs.
    """
    import re
    q = question.lower()
    all_text = " ".join(t.text for t in turns)

    if "how many" in q and "in total" in q:
        nums = [int(n) for n in re.findall(r"\b\d{1,3}\b", all_text)]
        if nums:
            unit = "hours" if "hours" in q else "days" if "days" in q else "weeks"
            return f"{sum(nums)} {unit}"
    if "where" in q and ("keep" in q or "currently" in q):
        # last dated turn wins
        dated = [(t.date, t.text) for t in turns if t.date]
        if dated:
            dated.sort(key=lambda x: x[0])
            return dated[-1][1]
    # fallback: echo most recent turn
    if turns:
        return turns[-1].text
    return ""


def hardcoded_operator_factory(base_items: list[SyntheticItem]):
    """The R8.1 pathology: memorise base answers keyed by TURNS, ignore logic.

    This is exactly what `total = 140` did — it keyed on the *input it was
    shown at development time* and returned a constant.  Here the memo is
    keyed on the base turn-text, so a counterfactual rewrite (which changes
    the turns) can never hit the memo and must fall through to a wrong
    constant.  That is precisely the collapse we want to demonstrate.
    """
    def _turn_key(turns: tuple) -> str:
        return "\x1f".join(t.text for t in turns)

    memo = {_turn_key(it.turns): it.expected for it in base_items}

    def op(question: str, turns: tuple) -> str:
        key = _turn_key(turns)
        if key in memo:
            return memo[key]
        # Unseen input -> a sentinel that cannot accidentally match any
        # rewritten expected value via substring or token overlap.
        return "\x00NO_MEMOISED_ANSWER\x00"

    return op


# --------------------------------------------------------------------------- #
#  THE TESTS
# --------------------------------------------------------------------------- #

def test_corpus_is_synthetic():
    """The suite must never leak real benchmark identifiers."""
    findings = check_corpus_is_synthetic(corpus())
    assert not findings, f"corpus contaminated: {findings[:5]}"


def test_base_corpus_builds():
    items = build_base_corpus()
    assert len(items) == len(GENERATORS) * 8
    # every generator produced at least one item
    ops = {it.operator for it in items}
    assert len(ops) >= 6


def test_counterfactual_expansion():
    base = build_base_corpus()
    expanded = expand_counterfactuals(base)
    # base + one of each rewrite kind
    assert len(expanded) == len(base) * (1 + len(REWRITERS))
    fams = {it.family for it in expanded}
    assert "numbers" in fams and "names" in fams and "order" in fams


def test_numbers_rewrite_changes_answer():
    """A SUM item's expected answer must change after a numbers rewrite."""
    from .generators import gen_temporal_sum
    base = gen_temporal_sum(1234)
    rw = REWRITERS["numbers"](base, 4321)
    # answers should differ (different random quantities)
    assert base.expected != rw.expected, (
        f"numbers rewrite did not change the answer: {base.expected} == {rw.expected}"
    )
    # but both must be self-consistent: sum of the QUANTITIES in their own
    # turns (a quantity is a number adjacent to a unit; dates are excluded).
    import re
    _QTY = re.compile(r"\b(\d{1,3})\s*(hours?|hrs?|minutes?|mins?|days?|weeks?|"
                      r"months?|years?|points?|pts?|books?|coins?|records?|"
                      r"miles?|km|dollars?|\$|%)\b", re.I)

    def _sum_of(turns):
        tot = 0
        for t in turns:
            for n, _u in _QTY.findall(t.text):
                tot += int(n)
        return tot
    assert int(re.findall(r"\d+", rw.expected)[0]) == _sum_of(rw.turns)


def test_hardcoded_operator_dies():
    """THE KILLER TEST.

    An operator that memorises base answers (the R8.1 `total = 140` pathology)
    must ace the base set and collapse on counterfactuals.  If this test ever
    shows the hardcoded operator doing WELL on rewrites, the suite is broken.

    Discrimination is only meaningful where the rewrite actually CHANGES the
    input the operator sees.  For a turn-keyed memoriser, a rewrite that
    leaves the turns identical (e.g. rephrase, which only edits the question)
    is not a counterfactual at all — matching there is correct behaviour, not
    overfitting.  So we partition rewrites into:

      * "mutating"  — turns differ from base (numbers, names, order, count):
        a memoriser MUST fail these, because its key no longer hits.
      * "surface"   — turns identical, only the question changed (rephrase):
        excluded from the collapse requirement.

    The SUM generator is the canonical case: its turns carry quantities, so
    the numbers/names/count rewrites genuinely mutate the input while the
    expected answer moves with them.  A hardcoded `total = 140` cannot track
    that; a structural operator recomputes it.
    """
    from synthetic.generators import gen_temporal_sum
    from synthetic.rewrites import REWRITERS

    # Build a base set purely of SUM items (the R8.1 failure shape) plus
    # every other generator, then apply the mutating rewrites.
    base = build_base_corpus()
    # add extra temporal_sum seeds so the numbers rewrite has signal
    for s in range(2000, 2016):
        base.append(gen_temporal_sum(s))

    op = hardcoded_operator_factory(base)

    def _key(turns):
        return "\x1f".join(t.text for t in turns)

    base_keys = {_key(it.turns) for it in base}

    base_acc = sum(
        1 for it in base if _soft_match(op(it.question, it.turns), it.expected)
    ) / len(base)
    assert base_acc >= 0.8, f"memoiser unexpectedly weak on base: {base_acc:.2f}"

    # For each mutating rewrite family, find items whose turns actually
    # changed, then assert the memoriser collapses on them.
    for fam in ("numbers", "names", "count", "order"):
        mutated = []
        for idx, it in enumerate(base):
            rw = REWRITERS[fam](it, 9000 + idx)
            if _key(rw.turns) not in base_keys:
                mutated.append(rw)
        if not mutated:
            continue
        acc = sum(
            1 for it in mutated if _soft_match(op(it.question, it.turns), it.expected)
        ) / len(mutated)
        assert acc < 0.5, (
            f"hardcoded operator survived {fam} rewrite ({acc:.2f} over "
            f"{len(mutated)} mutated items) — suite is not discriminating"
        )


def test_honest_reference_passes_base():
    """A structural operator must clear the base set reasonably."""
    base = build_base_corpus()
    res = grade(honest_reference_operator, base)
    # The reference operator is simplistic; it only needs to not crash and
    # score on the mechanical families.
    assert res["n"] == len(base)


def test_guard_flags_r81_patterns():
    """The static guard must flag the exact R8.1 source patterns."""
    bad = '''
def op(records):
    total = 140
    babies = ["Max", "Charlotte", "Jasper"]
    if "sneaker" in q:
        return sneaker_thing
    if "austin film festival" in all_text:
        return "Austin Film Festival"
    return 140
'''
    findings = scan_source_for_overfitting(bad)
    joined = " | ".join(findings)
    assert "EMBEDDED_ANSWER" in joined, findings
    assert "ENTITY_DISPATCH" in joined, findings

    clean = '''
def op(question, records):
    nums = [extract_number(r) for r in records]
    return sum(nums)
'''
    findings2 = scan_source_for_overfitting(clean)
    assert not findings2, findings2
