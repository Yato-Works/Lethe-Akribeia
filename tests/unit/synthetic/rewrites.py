"""Counterfactual rewrites.

The R8.1 failure was an operator that returned ``total = 140`` — a constant
baked from one benchmark item.  A counterfactual rewrite changes the *inputs*
while preserving the *task*, and requires the answer to track the change.

If an operator survives every rewrite below, it is operating on structure,
not memory.
"""

from __future__ import annotations

import re
from typing import Sequence

from .corpus import (
    FIRST_NAMES,
    LAST_NAMES,
    CITIES,
    SyntheticItem,
    SyntheticTurn,
    make_rng,
)


# --------------------------------------------------------------------------- #
#  numbers — change every quantity, recompute nothing by hand
# --------------------------------------------------------------------------- #

def rewrite_numbers(item: SyntheticItem, seed: int) -> SyntheticItem:
    """Replace all quantity integers with fresh ones; recompute the expected answer.

    The whole point: the operator must re-extract from the rewritten text.
    A hardcoded ``total = 140`` cannot survive this.

    Only *quantity* integers are rewritten — those adjacent to a duration/count
    unit.  Date components (2021-01-25) are left intact so the temporal
    structure of the corpus is preserved and the answer stays well-defined.
    """
    rng = make_rng(seed)

    # A quantity is a number followed by a unit word (hours/days/points/...).
    _QTY = re.compile(r"\b(\d{1,3})\s*(hours?|hrs?|minutes?|mins?|days?|weeks?|"
                      r"months?|years?|points?|pts?|books?|coins?|records?|"
                      r"miles?|km|dollars?|\$|%)\b", re.I)

    def _sub_quantities(text: str) -> str:
        return _QTY.sub(lambda m: f"{rng.randint(2, 90)} {m.group(2)}", text)

    # Rewrite turn text (quantities only; dates preserved)
    new_turns = tuple(
        SyntheticTurn(speaker=t.speaker, text=_sub_quantities(t.text),
                      date=t.date, session_id=t.session_id)
        for t in item.turns
    )

    def _sum_of(turns) -> int:
        tot = 0
        for t in turns:
            nums = [int(n) for n, u in _QTY.findall(t.text)]
            tot += sum(nums)
        return tot

    expected = item.expected
    if item.operator == "SUM":
        total = _sum_of(new_turns)
        unit = item.expected.split()[-1]
        expected = f"{total} {unit}"
    elif item.operator == "GOAL_GAP":
        pass

    return SyntheticItem(
        item_id=f"{item.item_id}_num",
        question=item.question,
        turns=new_turns,
        expected=expected,
        operator=item.operator,
        family="numbers",
        provenance=f"{item.provenance}<-numbers:{seed}",
    )


# --------------------------------------------------------------------------- #
#  names — swap every proper noun; the answer must swap with them
# --------------------------------------------------------------------------- #

def rewrite_names(item: SyntheticItem, seed: int) -> SyntheticItem:
    """Replace every First/Last name with a fresh one, consistently.

    A consistent substitution map means 'Aveline Ashgrove' -> 'Bodhi Bramblewood'
    everywhere; the operator must resolve to the NEW name.
    """
    rng = make_rng(seed)
    old_firsts = sorted(set(re.findall(r"\b[A-Z][a-z]+\b", " ".join(t.text for t in item.turns))))

    fmap: dict[str, str] = {}
    for name in old_firsts:
        # avoid collisions
        pool = [n for n in FIRST_NAMES + LAST_NAMES if n not in fmap.values()]
        fmap[name] = rng.choice(pool) if pool else name

    def _sub(text: str) -> str:
        for old, new in fmap.items():
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        return text

    new_turns = tuple(
        SyntheticTurn(speaker=t.speaker, text=_sub(t.text),
                      date=t.date, session_id=t.session_id)
        for t in item.turns
    )
    new_expected = _sub(item.expected)

    return SyntheticItem(
        item_id=f"{item.item_id}_name",
        question=_sub(item.question),
        turns=new_turns,
        expected=new_expected,
        operator=item.operator,
        family="names",
        provenance=f"{item.provenance}<-names:{seed}",
    )


# --------------------------------------------------------------------------- #
#  order — shuffle session order; date-based reasoning must still win
# --------------------------------------------------------------------------- #

def rewrite_order(item: SyntheticItem, seed: int) -> SyntheticItem:
    """Shuffle the physical order of turns/sessions.

    Chronological answers must be driven by DATES, not list position.  An
    operator that takes 'the last session' or 'the first item' positionally
    will break here.
    """
    rng = make_rng(seed)
    turns = list(item.turns)
    rng.shuffle(turns)
    return SyntheticItem(
        item_id=f"{item.item_id}_ord",
        question=item.question,
        turns=tuple(turns),
        expected=item.expected,   # semantics unchanged; only ordering moves
        operator=item.operator,
        family="order",
        provenance=f"{item.provenance}<-order:{seed}",
    )


# --------------------------------------------------------------------------- #
#  rephrase — same task, different surface wording
# --------------------------------------------------------------------------- #

REPHRASE_MAP = [
    (r"\bhow many\b", "what is the total number of"),
    (r"\bin total\b", "altogether"),
    (r"\bwhat was the\b", "which was the"),
    (r"\bwhere do i currently keep\b", "where are my"),
    (r"\bwhich of these events happened first\b", "of these, which occurred earliest"),
    (r"\bwhat is\b", "tell me"),
]


def rewrite_rephrase(item: SyntheticItem, seed: int) -> SyntheticItem:
    """Surface rephrase of the question; expected answer unchanged."""
    rng = make_rng(seed)
    q = item.question
    # apply a deterministic subset based on seed
    applied = 0
    for pat, rep in REPHRASE_MAP:
        if rng.random() < 0.4:
            new_q = re.sub(pat, rep, q, flags=re.I)
            if new_q != q:
                q = new_q
                applied += 1
    return SyntheticItem(
        item_id=f"{item.item_id}_rep",
        question=q,
        turns=item.turns,
        expected=item.expected,
        operator=item.operator,
        family="rephrase",
        provenance=f"{item.provenance}<-rephrase:{seed}",
    )


# --------------------------------------------------------------------------- #
#  count — change the NUMBER of items/sessions; answer must scale
# --------------------------------------------------------------------------- #

def rewrite_count(item: SyntheticItem, seed: int) -> SyntheticItem:
    """Change the NUMBER of contributing items/sessions.

    For every operator family this appends a fresh contributing turn so the
    input genuinely differs from the base (defeating a turn-keyed memoriser),
    and recomputes the expected answer where it is count-sensitive.
    """
    rng = make_rng(seed)
    turns = list(item.turns)
    new_date = "2099-01-01"

    # State / lookup operators are not count-sensitive.  Appending a
    # topic-less quantity turn must be IGNORED by a correctly topic-scoped
    # operator, so the answer is unchanged — that is exactly what we assert.
    if item.operator in ("SUPERSEDES", "ORDER_FIRST", "SLOT_LOOKUP", "INDEX", "HOP_CHAIN"):
        extra_n = rng.randint(2, 30)
        extra = SyntheticTurn(
            speaker="user",
            text=f"On {new_date} I also spent {extra_n} hours on something else.",
            date=new_date,
            session_id=f"s{len({t.session_id for t in turns})}",
        )
        turns.append(extra)
        return SyntheticItem(
            item_id=f"{item.item_id}_cnt",
            question=item.question,
            turns=tuple(turns),
            expected=item.expected,   # unchanged — distractor must be ignored
            operator=item.operator,
            family="count",
            provenance=f"{item.provenance}<-count:{seed}",
        )

    extra_n = rng.randint(2, 30)
    # Match the unit of the base item so the appended turn is a genuine
    # additional data point for the SAME quantity (not a unit-mismatched one).
    base_unit = item.expected.split()[-1] if item.expected else "hours"
    # Include a topic word so a topic-scoped operator trusts this turn as a
    # genuine same-topic data point (a topic-less "doing it" would be — and
    # should be — rejected by the confidence gate).
    topic_word = "the same activity"
    if item.operator == "SUM":
        # reuse the QUESTION's topic entity so the added turn is on-topic
        import re as _re
        qwords = [w for w in _re.findall(r"[a-z]+", item.question.lower())
                  if len(w) >= 5 and w not in ("total", "spend", "spent", "since")]
        if qwords:
            topic_word = qwords[-1]
    extra = SyntheticTurn(
        speaker="user",
        text=f"On {new_date} I also spent {extra_n} {base_unit} on {topic_word}.",
        date=new_date,
        session_id=f"s{len({t.session_id for t in turns})}",
    )
    turns.append(extra)

    # A quantity is a number adjacent to a unit; dates are not quantities.
    _QTY = re.compile(r"\b(\d{1,3})\s*(hours?|hrs?|minutes?|mins?|days?|weeks?|"
                      r"months?|years?|points?|pts?|books?|coins?|records?|"
                      r"miles?|km|dollars?|\$|%)\b", re.I)

    if item.operator in ("SUM",):
        # recompute the summed total over ALL turns (quantities only)
        quantities = []
        for t in turns:
            for n, _u in _QTY.findall(t.text):
                quantities.append(int(n))
        unit = item.expected.split()[-1]
        new_expected = f"{sum(quantities)} {unit}"
    elif item.operator in ("ENUMERATE", "COUNT", "FILTER"):
        # count-sensitive: bump the expected integer if it is one
        m = re.search(r"\d+", item.expected)
        if m:
            new_expected = item.expected.replace(m.group(0), str(int(m.group(0)) + 1), 1)
        else:
            new_expected = item.expected
    else:
        # ORDER / INDEX / HOP / SUPERSEDES / SLOT answers are stable under an
        # appended later turn ONLY if that turn does not participate; we add a
        # turn dated 2099 so date-based 'latest' logic WOULD change, which is
        # exactly what we want to test.  For supersedes the latest value wins,
        # so the expected becomes the new turn's content.
        if item.operator == "SUPERSEDES":
            new_expected = extra.text
        else:
            new_expected = item.expected

    return SyntheticItem(
        item_id=f"{item.item_id}_cnt",
        question=item.question,
        turns=tuple(turns),
        expected=new_expected,
        operator=item.operator,
        family="count",
        provenance=f"{item.provenance}<-count:{seed}",
    )


# --------------------------------------------------------------------------- #
#  registry
# --------------------------------------------------------------------------- #

REWRITERS = {
    "numbers": rewrite_numbers,
    "names": rewrite_names,
    "order": rewrite_order,
    "rephrase": rewrite_rephrase,
    "count": rewrite_count,
}
