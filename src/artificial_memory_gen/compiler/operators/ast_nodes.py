"""Operator AST — the algebra at the core of the Query Operator Compiler.

DESIGN CONTRACT (learned from the R8.1 overfitting failure):

  1. An operator NEVER knows a benchmark topic.  It knows only *concepts*
     ("festival"), *units* ("hours"), and *structure* (sum, union, count).
  2. An operator NEVER contains an answer constant.  Every value is extracted
     from records at execution time.
  3. Parser and Executor are separate: NL -> AST -> result.  The Executor
     receives an AST and records; it never sees the raw question string.

Because of rule 3, no benchmark-topic ``if`` can creep into execution logic.
An operator that survives the synthetic counterfactual suite operates on
structure, not memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- #
#  AST nodes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Node:
    """Base AST node."""


@dataclass(frozen=True)
class Extract(Node):
    """Pull values out of records matching a concept / unit / relation.

    concept : a generic category (never a proper noun), e.g. "event", "person"
    unit    : a measurement unit, e.g. "hours", "points"; None for text
    action  : an optional verb sense to scope extraction, e.g. "attend"
    """
    concept: str
    unit: str | None = None
    action: str | None = None


@dataclass(frozen=True)
class Sum(Node):
    operand: Node
    topic_terms: tuple = ()  # scope words; quantities must come from records
                             # mentioning the topic, else defer to the LLM
    time_scope: str | None = None  # relative window ("last_month", "last_year");
                                   # quantities outside the window are dropped


@dataclass(frozen=True)
class Union(Node):
    operand: Node
    dedup_key: str = "value"


@dataclass(frozen=True)
class Count(Node):
    operand: Node


@dataclass(frozen=True)
class Filter(Node):
    operand: Node
    predicate: Predicate


@dataclass(frozen=True)
class Index(Node):
    operand: Node
    ordinal: int          # 1-based; -1 means "last"
    selector: str = "nth"  # "nth" | "last" | "first"
    scope: str | None = None  # restrict to the record containing this token


@dataclass(frozen=True)
class Ratio(Node):
    numerator: Node
    denominator: Node
    as_percent: bool = True
    topic_terms: tuple = ()  # numerator & denominator must be topic-scoped
    # 'percentage of <DENOM> do <NUM> hold' — the noun phrase after 'of' is the
    # denominator (English general rule), the subject doing the holding is the
    # numerator.  Used to orient the two values instead of guessing by order.
    denominator_terms: tuple = ()
    numerator_terms: tuple = ()


@dataclass(frozen=True)
class Gap(Node):
    """target - current, both extracted."""
    target: Node
    current: Node
    topic_terms: tuple = ()  # the two point values must be topic-scoped


@dataclass(frozen=True)
class Supersedes(Node):
    """Latest-dated observation of a topic supersedes earlier ones."""
    topic: Node
    want: str = "latest"   # "latest" | "earliest"
    topic_terms: tuple = ()  # scope words (e.g. "sneakers", "compasses")
                              # used to pick the value phrase, not the whole turn


@dataclass(frozen=True)
class OrderFirst(Node):
    """Which of a set of dated events occurred earliest (or latest)."""
    operand: Node
    want: str = "earliest"   # "earliest" | "latest"


@dataclass(frozen=True)
class DateDiff(Node):
    """Elapsed time between two dated events.

    Deterministic calendar arithmetic: find the two dates in the records
    (scoped by topic_terms) and return their difference in the asked unit
    (days/weeks/months).  This is the R7 interval-algebra capability the LLM
    reader keeps getting wrong (off-by-one, week/month confusion) on holdout.
    """
    unit: str                       # "days" | "weeks" | "months"
    topic_terms: tuple = ()


@dataclass(frozen=True)
class SlotLookup(Node):
    """Retrieve a single attribute value for an entity."""
    operand: Node
    attribute: str | None = None


# --------------------------------------------------------------------------- #
#  Predicates (generic filters — never topic-specific)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Predicate:
    kind: str                       # "unit_equals" | "relation_equals" | "recency"
    value: Any = None


# --------------------------------------------------------------------------- #
#  Pretty-printer (for proof / audit)
# --------------------------------------------------------------------------- #

def render(node: Node) -> str:
    """Render an AST to a canonical string for auditability."""
    if isinstance(node, Extract):
        parts = [f'EXTRACT(concept="{node.concept}"']
        if node.unit:
            parts.append(f', unit="{node.unit}"')
        if node.action:
            parts.append(f', action="{node.action}"')
        return "".join(parts) + ")"
    if isinstance(node, Sum):
        terms = f", terms={list(node.topic_terms)}" if node.topic_terms else ""
        scope = f', scope="{node.time_scope}"' if node.time_scope else ""
        return f"SUM({render(node.operand)}{terms}{scope})"
    if isinstance(node, Union):
        return f'UNION({render(node.operand)}, dedup="{node.dedup_key}")'
    if isinstance(node, Count):
        return f"COUNT({render(node.operand)})"
    if isinstance(node, Filter):
        return f"FILTER({render(node.operand)}, {node.predicate.kind}={node.predicate.value!r})"
    if isinstance(node, Index):
        scope = f', scope="{node.scope}"' if node.scope else ""
        if node.selector == "last":
            return f"INDEX({render(node.operand)}, LAST{scope})"
        return f"INDEX({render(node.operand)}, {node.ordinal}{scope})"
    if isinstance(node, Ratio):
        return f"RATIO({render(node.numerator)} / {render(node.denominator)})"
    if isinstance(node, Gap):
        return f"GAP({render(node.target)} - {render(node.current)})"
    if isinstance(node, Supersedes):
        terms = f", terms={list(node.topic_terms)}" if node.topic_terms else ""
        return f'SUPERSEDES({render(node.topic)}, want="{node.want}"{terms})'
    if isinstance(node, DateDiff):
        terms = f", terms={list(node.topic_terms)}" if node.topic_terms else ""
        return f'DATE_DIFF(unit="{node.unit}"{terms})'
    if isinstance(node, OrderFirst):
        return f'ORDER_FIRST({render(node.operand)}, want="{node.want}")'
    if isinstance(node, SlotLookup):
        attr = f', attribute="{node.attribute}"' if node.attribute else ""
        return f"SLOT_LOOKUP({render(node.operand)}{attr})"
    return repr(node)


def node_types(node: Node) -> set[str]:
    """All node type names in the tree (for guard assertions)."""
    out: set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        out.add(type(n).__name__)
        for v in vars(n).values():
            if isinstance(v, Node):
                stack.append(v)
            elif isinstance(v, Predicate):
                out.add("Predicate")
    return out
