"""Synthetic corpus generators.

Each generator emits ``SyntheticItem`` objects whose answers are COMPUTED from
the generated turns, never hardcoded.  Every generator is seeded, so a given
seed always yields an identical corpus — the suite is fully reproducible.

LoCoMo coverage (1540 Qs, 5 categories):
  cat 1 multi-hop     -> hop_chain generator
  cat 2 temporal      -> temporal_interval / temporal_order generators
  cat 3 open-domain   -> open_domain generator (abstention-leaning)
  cat 4 single-hop    -> slot_lookup / ordinal generators
  cat 5 adversarial   -> adversarial generator (wrong-entity trap)
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import date, timedelta

from .corpus import (
    CITIES,
    COLLECTIBLE_NOUNS,
    DAYS_OF_WEEK,
    DURATION_NOUNS,
    EVENT_NOUNS,
    FIRST_NAMES,
    LAST_NAMES,
    MONTHS,
    SyntheticItem,
    SyntheticTurn,
    make_rng,
)


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _full_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _session_turns(rng, session_id: str, date_str: str, pairs: Sequence[tuple[str, str]]):
    return tuple(
        SyntheticTurn(speaker=spk, text=txt, date=date_str, session_id=session_id)
        for spk, txt in pairs
    )


def _distinct_dates(rng, n: int, start_year: int = 2021) -> list[str]:
    """n distinct ISO dates, chronologically ordered."""
    base = date(start_year, 1, 1)
    offsets = sorted(rng.sample(range(0, 900), n))
    return [(base + timedelta(days=o)).isoformat() for o in offsets]


# --------------------------------------------------------------------------- #
#  cat 2 — temporal interval sum (SUM across sessions)
# --------------------------------------------------------------------------- #

def gen_temporal_sum(seed: int) -> SyntheticItem:
    """Distributed numeric quantities summed across sessions.

    e.g. 'How many hours in total?' where hours are spread over sessions.
    The answer is COMPUTED as sum(extracted), never written as a constant.
    """
    rng = make_rng(seed)
    unit = rng.choice(["hours", "days", "weeks"])
    n = rng.randint(3, 6)
    values = [rng.randint(2, 40) for _ in range(n)]
    activity = rng.choice(["practising the cello", "cycling", "woodworking",
                           "studying Japanese", "baking sourdough"])
    dates = _distinct_dates(rng, n)

    turns = []
    for i, (v, d) in enumerate(zip(values, dates)):
        turns.extend(_session_turns(rng, f"s{i}", d, [
            ("user", f"On {d} I spent {v} {unit} {activity}."),
        ]))
    total = sum(values)
    q = f"How many {unit} in total did I spend {activity}?"
    return SyntheticItem(
        item_id=f"tsum_{seed}",
        question=q,
        turns=tuple(turns),
        expected=f"{total} {unit}",
        operator="SUM",
    )


# --------------------------------------------------------------------------- #
#  cat 2 — temporal ordering / "which happened first"
# --------------------------------------------------------------------------- #

def gen_temporal_order(seed: int) -> SyntheticItem:
    """Ordering question across dated sessions.

    The critical failure mode from the CLEAN-R7 holdout was order inversion
    and off-by-one date arithmetic; this generator stresses exactly that.
    """
    rng = make_rng(seed)
    n = rng.randint(3, 5)
    names = rng.sample(EVENT_NOUNS, n)
    dates = _distinct_dates(rng, n)
    turns = []
    for i, (ev, d) in enumerate(zip(names, dates)):
        turns.extend(_session_turns(rng, f"s{i}", d, [
            ("user", f"On {d} I went to the {ev}."),
        ]))
    # chronological first
    first_idx = min(range(n), key=lambda i: dates[i])
    q = "Which of these events happened first: " + ", ".join(names) + "?"
    return SyntheticItem(
        item_id=f"tord_{seed}",
        question=q,
        turns=tuple(turns),
        expected=f"the {names[first_idx]}",
        operator="ORDER_FIRST",
    )


# --------------------------------------------------------------------------- #
#  cat 4 — slot lookup (single-hop attribute retrieval)
# --------------------------------------------------------------------------- #

def gen_slot_lookup(seed: int) -> SyntheticItem:
    """Key-value attribute lookup within a session."""
    rng = make_rng(seed)
    entity = _full_name(rng)
    slots = {
        "phone number": f"555-{rng.randint(1000,9999)}",
        "email": f"{rng.choice(FIRST_NAMES).lower()}@example.com",
        "home city": rng.choice(CITIES),
        "favourite book": f"The {rng.choice(LAST_NAMES)} Codex",
    }
    ask_slot = rng.choice(list(slots.keys()))
    date_str = _distinct_dates(rng, 1)[0]
    other = ", ".join(f"{k} is {v}" for k, v in slots.items() if k != ask_slot)
    turns = _session_turns(rng, "s0", date_str, [
        ("user", f"My friend {entity} — {other}."),
        ("assistant", f"Noted. {entity}'s {ask_slot} is {slots[ask_slot]}."),
    ])
    q = f"What is {entity}'s {ask_slot}?"
    return SyntheticItem(
        item_id=f"slot_{seed}",
        question=q,
        turns=tuple(turns),
        expected=slots[ask_slot],
        operator="SLOT_LOOKUP",
    )


# --------------------------------------------------------------------------- #
#  cat 4 — ordinal INDEX (Nth item in an enumerated list)
# --------------------------------------------------------------------------- #

def gen_ordinal_index(seed: int) -> SyntheticItem:
    """Nth / last item from a numbered list embedded in a turn."""
    rng = make_rng(seed)
    n = rng.randint(5, 30)
    noun = rng.choice(["parameter", "step", "ingredient", "rule"])
    items = [f"{rng.choice(LAST_NAMES)} {rng.choice(['alpha','beta','gamma','delta','omega'])}"
             for _ in range(n)]
    date_str = _distinct_dates(rng, 1)[0]
    listing = "\n".join(f"{i+1}. {it}" for i, it in enumerate(items))
    turns = _session_turns(rng, "s0", date_str, [
        ("user", f"Here is my list of {n} {noun}s:\n{listing}"),
    ])
    if rng.random() < 0.5:
        target = rng.randint(1, n)
        q = f"What was the {target}th {noun} on that list?"
        exp = items[target - 1]
    else:
        q = f"What was the last {noun} on that list?"
        exp = items[-1]
    return SyntheticItem(
        item_id=f"ord_{seed}",
        question=q,
        turns=tuple(turns),
        expected=exp,
        operator="INDEX",
    )


# --------------------------------------------------------------------------- #
#  cat 1 — multi-hop chain (entity A -> relation -> entity B -> attribute)
# --------------------------------------------------------------------------- #

def gen_hop_chain(seed: int) -> SyntheticItem:
    """Two-hop reasoning: person -> their item -> that item's attribute."""
    rng = make_rng(seed)
    p1, p2 = _full_name(rng), _full_name(rng)
    item = rng.choice(["bicycle", "guitar", "kayak", "telescope"])
    attr = rng.choice(["color", "brand", "year"])
    attr_val = rng.choice(["crimson", "Harrowden", "2019", "teakwood"])
    dates = _distinct_dates(rng, 3)
    turns = []
    turns.extend(_session_turns(rng, "s0", dates[0], [
        ("user", f"{p1} owns a {item}."),
    ]))
    turns.extend(_session_turns(rng, "s1", dates[1], [
        ("user", f"That {item}'s {attr} is {attr_val}."),
    ]))
    turns.extend(_session_turns(rng, "s2", dates[2], [
        ("user", f"{p2} has a completely different {item}."),
    ]))
    q = f"What is the {attr} of the {item} that {p1} owns?"
    return SyntheticItem(
        item_id=f"hop_{seed}",
        question=q,
        turns=tuple(turns),
        expected=attr_val,
        operator="HOP_CHAIN",
    )


# --------------------------------------------------------------------------- #
#  cat 3 — open-domain / abstention (answer may be a judgement call)
# --------------------------------------------------------------------------- #

def gen_open_domain(seed: int) -> SyntheticItem:
    """Open-domain preference question with a grounded, computable rubric."""
    rng = make_rng(seed)
    topic = rng.choice(["weekend hikes", "board games", "espresso", "houseplants"])
    constraint = rng.choice(["on a budget", "indoors", "for beginners"])
    dates = _distinct_dates(rng, 1)
    turns = _session_turns(rng, "s0", dates[0], [
        ("user", f"I really enjoy {topic}, especially {constraint}."),
    ])
    q = f"Any ideas for {topic} {constraint}?"
    return SyntheticItem(
        item_id=f"open_{seed}",
        question=q,
        turns=tuple(turns),
        expected=topic,   # rubric: answer must reference the topic
        operator="OPEN_DOMAIN",
    )


# --------------------------------------------------------------------------- #
#  cat 5 — adversarial (distractor with the same shape as the target)
# --------------------------------------------------------------------------- #

def gen_adversarial(seed: int) -> SyntheticItem:
    """A near-miss distractor must NOT be selected.

    The CLEAN-R7 holdout showed 'last venue' resolving to an unrelated venue;
    this generator reproduces that trap structurally.
    """
    rng = make_rng(seed)
    target_city = rng.choice(CITIES)
    other_city = rng.choice([c for c in CITIES if c != target_city])
    n_target = rng.randint(4, 8)
    target_items = [rng.choice(LAST_NAMES) + " Hall" for _ in range(n_target)]
    dates = _distinct_dates(rng, 2)
    listing = "\n".join(f"{i+1}. {it}" for i, it in enumerate(target_items))
    turns = []
    turns.extend(_session_turns(rng, "s0", dates[0], [
        ("user", f"Venues in {other_city}:\n1. {rng.choice(LAST_NAMES)} Amphitheatre\n2. {rng.choice(LAST_NAMES)} Arena"),
    ]))
    turns.extend(_session_turns(rng, "s1", dates[1], [
        ("user", f"Here are popular venues in {target_city} for shows:\n{listing}"),
    ]))
    q = f"What was the last venue recommended in the list of venues in {target_city}?"
    return SyntheticItem(
        item_id=f"adv_{seed}",
        question=q,
        turns=tuple(turns),
        expected=target_items[-1],
        operator="INDEX_SCOPED",
    )


# --------------------------------------------------------------------------- #
#  knowledge update — supersedes (R7.3 trap, rebuilt generically)
# --------------------------------------------------------------------------- #

def gen_supersedes(seed: int) -> SyntheticItem:
    """A fact is stated, then updated later; the LATEST value wins."""
    rng = make_rng(seed)
    entity = rng.choice(COLLECTIBLE_NOUNS)
    locations = [f"in the {rng.choice(LAST_NAMES)} drawer",
                 f"on the {rng.choice(LAST_NAMES)} shelf",
                 f"inside the {rng.choice(LAST_NAMES)} chest"]
    dates = _distinct_dates(rng, 3)
    turns = []
    for i, (loc, d) in enumerate(zip(locations, dates)):
        turns.extend(_session_turns(rng, f"s{i}", d, [
            ("user", f"I now keep my {entity} {loc}."),
        ]))
    q = f"Where do I currently keep my {entity}?"
    return SyntheticItem(
        item_id=f"super_{seed}",
        question=q,
        turns=tuple(turns),
        expected=locations[-1],
        operator="SUPERSEDES",
    )


# --------------------------------------------------------------------------- #
#  registry
# --------------------------------------------------------------------------- #

GENERATORS = {
    "temporal_sum": gen_temporal_sum,
    "temporal_order": gen_temporal_order,
    "slot_lookup": gen_slot_lookup,
    "ordinal_index": gen_ordinal_index,
    "hop_chain": gen_hop_chain,
    "open_domain": gen_open_domain,
    "adversarial": gen_adversarial,
    "supersedes": gen_supersedes,
}
