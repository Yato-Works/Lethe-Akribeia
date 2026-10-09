"""Core primitives for the Synthetic Counterfactual Test Suite.

Design rule (R8.1 lesson): an operator that cannot survive a *counterfactual*
rewrite of its input has memorised the benchmark, not learned the task.
Every operator must be validated here before it is allowed into production.

This module is deliberately free of any benchmark import: the synthetic corpus
is generated from scratch, so no gold answer is ever embedded anywhere.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Sequence


# --------------------------------------------------------------------------- #
#  Synthetic corpus primitives
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SyntheticTurn:
    """A single conversational turn.  Immutable so rewrites are explicit."""
    speaker: str
    text: str
    date: str = ""
    session_id: str = ""


@dataclass(frozen=True)
class SyntheticItem:
    """A synthetic question plus its haystack.

    ``expected`` is computed by the generator, never copied from a real
    dataset.  The operator under test never sees it until grading.
    """
    item_id: str
    question: str
    turns: tuple[SyntheticTurn, ...]
    expected: str
    operator: str            # which operator this item is meant to exercise
    family: str = "base"     # base | numbers | names | order | rephrase | count
    provenance: str = ""     # lineage string, e.g. "base:42<-numbers:42<-order:7"


# --------------------------------------------------------------------------- #
#  Value pools — deliberately disjoint from any benchmark vocabulary
# --------------------------------------------------------------------------- #

FIRST_NAMES = [
    "Aveline", "Bodhi", "Cyrus", "Delphine", "Emeric", "Fennel", "Gwen",
    "Hadrian", "Isolde", "Jorunn", "Kestrel", "Lirien", "Marisol", "Nyx",
    "Odalys", "Peregrine", "Quillon", "Rowan", "Sable", "Theron", "Ursa",
    "Vesper", "Wren", "Xanthe", "Yarrow", "Zephyr",
]

LAST_NAMES = [
    "Ashgrove", "Bramblewood", "Cindermere", "Duskwater", "Emberfall",
    "Fairholt", "Glimmerstone", "Hollowmere", "Ironvale", "Jadefen",
    "Kestrelby", "Larkmount", "Moonwhisper", "Nightbreeze", "Oakhart",
    "Pellhaven", "Quicksilver", "Ravenhold", "Starhaven", "Thornwick",
    "Umberglen", "Vexmoor", "Windrune", "Yarrowfield", "Zephyrdale",
]

CITIES = [
    "Aurelia", "Bellharbor", "Cragmoor", "Dawnspire", "Elderbrook",
    "Foxglen", "Greycliff", "Havenport", "Ironforge", "Jasperbay",
    "Kilnhurst", "Larchmont", "Mistvale", "Northgate", "Ottercombe",
    "Pinehollow", "Quarrytown", "Rosewick", "Stonebridge", "Thistledown",
]

# Nouns for countable entities — each is a distinct, countable thing.
COLLECTIBLE_NOUNS = [
    "brass compasses", "amber rings", "clay figurines", "ivory combs",
    "silver thimbles", "copper lanterns", "jade pendants", "wool tapestries",
    "bronze bells", "glass marbles",
]

DURATION_NOUNS = [
    "hours", "days", "weeks", "months", "minutes",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

EVENT_NOUNS = [
    "lantern festival", "harvest fair", "solstice market", "founders day",
    "river regatta", "orchard walk", "cider tasting", "book bazaar",
]

DAYS_OF_WEEK = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]


# --------------------------------------------------------------------------- #
#  RNG — every generator is seeded for reproducibility
# --------------------------------------------------------------------------- #

class _SeededRandom(random.Random):
    """random.Random with an explicit, logged seed."""
    pass


def make_rng(seed: int) -> _SeededRandom:
    return _SeededRandom(seed)


def item_hash(item: SyntheticItem) -> str:
    """Stable content hash so regenerated corpora can be diffed."""
    payload = f"{item.question}|{item.expected}|{item.operator}|{item.family}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
#  Operators under test (the AST surface the synthetic suite validates)
# --------------------------------------------------------------------------- #

# The synthetic suite exercises an *interface*, not an implementation.  A
# production operator is passed as a callable ``fn(question, turns) -> str``.
# This keeps the suite implementation-agnostic: it can validate the real
# operator engine, a prototype, or a stub.

OperatorFn = "callable"  # documented alias: fn(question, turns) -> str


@dataclass(frozen=True)
class Rewrite:
    """One counterfactual rewrite of a base item."""
    kind: str
    rng_seed: int


REWRITE_KINDS = ("numbers", "names", "order", "rephrase", "count")
