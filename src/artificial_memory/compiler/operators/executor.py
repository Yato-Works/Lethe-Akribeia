"""Deterministic Executor — runs an Operator AST against records.

The Executor NEVER sees the raw question string.  It receives an AST and a
record set, and returns an ``ExecutionResult`` carrying both the answer and an
auditable proof.  This separation is what makes benchmark-topic dispatch
structurally impossible: execution has no topic to dispatch on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from artificial_memory.compiler.operators import ast_nodes as A
from artificial_memory.compiler.operators.extractors import (
    Observation,
    _date_of,
    _text_of,
    extract_events,
    extract_list,
    extract_numeric,
    extract_persons,
    extract_slots,
    parse_date,
)


@dataclass(frozen=True)
class ExecutionResult:
    has_result: bool
    answer: str | None
    proof: str
    ast: str


def _num(obs: Observation) -> float:
    try:
        return float(obs.value.split()[0].replace(",", ""))
    except (ValueError, IndexError):
        return 0.0


def _unit(obs: Observation) -> str:
    parts = obs.value.split()
    return parts[1] if len(parts) > 1 else ""


class Executor:
    """Executes an Operator AST deterministically.  No question string in."""

    def execute(self, ast: A.Node, records: Sequence) -> ExecutionResult:
        rendered = A.render(ast)
        try:
            answer, proof = self._eval(ast, records)
        except Exception as exc:  # a crash must never fabricate an answer
            return ExecutionResult(
                has_result=False, answer=None,
                proof=f"[EXECUTION_ERROR] {type(exc).__name__}: {exc}",
                ast=rendered,
            )
        if answer is None:
            return ExecutionResult(has_result=False, answer=None, proof=proof, ast=rendered)
        return ExecutionResult(has_result=True, answer=answer, proof=proof, ast=rendered)

    # ------------------------------------------------------------------ #

    def _eval(self, node: A.Node, records: Sequence) -> tuple[str | None, str]:
        if isinstance(node, A.Extract):
            return self._eval_extract(node, records)

        if isinstance(node, A.Sum):
            inner, proof = self._eval(node.operand, records)
            if inner is None:
                return None, proof
            obs = extract_numeric(records, unit=self._unit_of(node.operand))
            if not obs:
                return None, proof
            total = sum(_num(o) for o in obs)
            unit = _unit(obs[0])
            terms = " + ".join(o.value for o in obs)
            return (
                f"{_fmt_num(total)} {unit}".strip(),
                f"{proof}\nSUM: {terms} = {_fmt_num(total)} {unit}",
            )

        if isinstance(node, A.Union):
            inner, proof = self._eval(node.operand, records)
            obs = self._collect(node.operand, records)
            seen: dict[str, Observation] = {}
            for o in obs:
                key = o.value.strip().lower()
                if key and key not in seen:
                    seen[key] = o
            if not seen:
                return None, proof
            vals = sorted(o.value for o in seen.values())
            return (
                ", ".join(vals),
                f"{proof}\nUNION ({len(vals)} unique): {', '.join(vals)}",
            )

        if isinstance(node, A.Count):
            inner, proof = self._eval(node.operand, records)
            obs = self._collect(node.operand, records)
            seen = {o.value.strip().lower() for o in obs if o.value.strip()}
            if not seen:
                return None, proof
            return str(len(seen)), f"{proof}\nCOUNT: {len(seen)} unique"

        if isinstance(node, A.Filter):
            inner, proof = self._eval(node.operand, records)
            obs = self._collect(node.operand, records)
            pred = node.predicate
            kept: list[Observation] = []
            if pred.kind == "unit_equals":
                for o in obs:
                    if _unit(o) == str(pred.value).lower():
                        kept.append(o)
            else:
                kept = list(obs)
            if not kept:
                return None, proof
            seen = {o.value.strip().lower() for o in kept}
            return str(len(seen)), f"{proof}\nFILTER({pred.kind}={pred.value}): kept {len(kept)}"

        if isinstance(node, A.Index):
            items = extract_list(records, scope=node.scope)
            if not items:
                return None, "[INDEX] no list found"
            if node.selector == "last" or node.ordinal == -1:
                chosen = items[-1]
                return chosen, f"[INDEX LAST] of {len(items)}: {chosen}"
            if 1 <= node.ordinal <= len(items):
                chosen = items[node.ordinal - 1]
                return chosen, f"[INDEX {node.ordinal}] of {len(items)}: {chosen}"
            return None, f"[INDEX] ordinal {node.ordinal} out of range (len {len(items)})"

        if isinstance(node, A.Ratio):
            num_obs = extract_numeric(records)
            # numerator = first matching, denominator = second distinct value
            vals = [(_num(o), o) for o in num_obs]
            if len(vals) < 2:
                return None, "[RATIO] need two numeric values"
            (n_num, n_obs), (d_num, d_obs) = vals[0], vals[1]
            if d_num == 0:
                return None, "[RATIO] denominator is zero"
            pct = (n_num / d_num) * 100
            proof = f"[RATIO] {n_obs.value} / {d_obs.value} * 100 = {pct:.2f}%"
            if node.as_percent:
                return f"{_pct(pct)}%", proof
            return f"{pct:.4f}", proof

        if isinstance(node, A.Gap):
            num_obs = extract_numeric(records, unit="points")
            nums = [_num(o) for o in num_obs]
            if len(nums) < 2:
                return None, "[GAP] need target and current points"
            target, current = max(nums), min(nums)
            gap = target - current
            return str(_fmt_num(gap)), f"[GAP] {_fmt_num(target)} - {_fmt_num(current)} = {_fmt_num(gap)}"

        if isinstance(node, A.Supersedes):
            obs = self._collect_state(node.topic, records, None)
            dated = [(parse_date(o.date) or parse_date(o.source), o) for o in obs]
            dated = [(d, o) for d, o in dated if d is not None]
            if len(dated) < 2:
                return None, "[SUPERSEDES] need >=2 dated observations"
            dated.sort(key=lambda x: x[0])
            chosen = dated[-1][1] if node.want == "latest" else dated[0][1]
            return chosen.value, f"[SUPERSEDES {node.want}] {chosen.value}"

        if isinstance(node, A.OrderFirst):
            obs = extract_events(records)
            dated = [(parse_date(o.date) or parse_date(o.source), o) for o in obs]
            dated = [(d, o) for d, o in dated if d is not None]
            if len(dated) < 2:
                return None, "[ORDER_FIRST] need >=2 dated events"
            dated.sort(key=lambda x: x[0])
            chosen = dated[0][1] if node.want == "earliest" else dated[-1][1]
            return chosen.value, f"[ORDER_FIRST {node.want}] {chosen.value}"

        if isinstance(node, A.SlotLookup):
            obs = extract_slots(records, slot_hint=node.attribute)
            if not obs:
                return None, "[SLOT_LOOKUP] no slot found"
            # first slot's value (the part after '=')
            first = obs[0].value
            val = first.split("=", 1)[1] if "=" in first else first
            return val, f"[SLOT_LOOKUP] {first}"

        return None, f"[UNSUPPORTED_NODE] {type(node).__name__}"

    # ------------------------------------------------------------------ #

    def _eval_extract(self, node: A.Extract, records: Sequence) -> tuple[str | None, str]:
        if node.concept == "event":
            obs = extract_events(records)
        elif node.concept == "person":
            obs = extract_persons(records)
        elif node.concept == "slot":
            obs = extract_slots(records)
        elif node.concept == "numeric":
            obs = extract_numeric(records, unit=node.unit)
        else:
            obs = extract_numeric(records, unit=node.unit)
        if not obs:
            return None, f"[EXTRACT {node.concept}] none found"
        # return first value; composition (SUM/COUNT) re-collects
        return obs[0].value, f"[EXTRACT {node.concept}] {obs[0].value}"

    def _collect(self, node: A.Node, records: Sequence) -> list[Observation]:
        """Collect the Observation list that a node operates over."""
        if isinstance(node, A.Extract):
            if node.concept == "event":
                return extract_events(records)
            if node.concept == "person":
                return extract_persons(records)
            if node.concept == "slot":
                return extract_slots(records)
            return extract_numeric(records, unit=node.unit)
        # unwrap single-child nodes to their operand
        for attr in ("operand",):
            child = getattr(node, attr, None)
            if isinstance(child, A.Node):
                return self._collect(child, records)
        return extract_numeric(records)

    def _collect_state(self, node: A.Node, records: Sequence, _unused) -> list[Observation]:
        """Collect dated text observations for state evolution (SUPERSEDES).

        Unlike numeric extraction, a 'state' is any dated utterance; we keep
        them all and let date-ordering pick the latest/earliest.
        """
        out: list[Observation] = []
        for r in records:
            text = _text_of(r)
            date = _date_of(r)
            if not text or not date:
                continue
            out.append(Observation(value=text, source=text, date=date))
        return out

    def _unit_of(self, node: A.Node) -> str | None:
        if isinstance(node, A.Extract):
            return node.unit
        child = getattr(node, "operand", None)
        if isinstance(child, A.Node):
            return self._unit_of(child)
        return None


# --------------------------------------------------------------------------- #
#  formatting helpers
# --------------------------------------------------------------------------- #

def _fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _pct(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}".rstrip("0").rstrip(".")
