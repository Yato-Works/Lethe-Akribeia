"""Deterministic Executor — runs an Operator AST against records.

The Executor NEVER sees the raw question string.  It receives an AST and a
record set, and returns an ``ExecutionResult`` carrying both the answer and an
auditable proof.  This separation is what makes benchmark-topic dispatch
structurally impossible: execution has no topic to dispatch on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
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

# Measurement units whose semantics are incompatible with a unit-agnostic
# fallback: summing "hours" against "books" would be meaningless, so when a
# question asks for one of these and none are found, we abstain rather than
# relax the filter.  Countable objects (stories, pieces, books) may relax.
_MEASUREMENT_UNITS = {
    "hours", "hrs", "hr", "minutes", "mins", "min", "days", "weeks",
    "months", "years", "points", "pts", "miles", "km", "dollars", "%",
}


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

    def execute(self, ast: A.Node, records: Sequence,
                reference_date: str | None = None) -> ExecutionResult:
        rendered = A.render(ast)
        try:
            answer, proof = self._eval(ast, records, reference_date)
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

    def _eval(self, node: A.Node, records: Sequence,
              reference_date: str | None = None) -> tuple[str | None, str]:
        if isinstance(node, A.Extract):
            return self._eval_extract(node, records)

        if isinstance(node, A.Sum):
            # NOTE: we deliberately do NOT gate on the inner Extract result.
            # A strict unit filter may return nothing at the leaf even when a
            # countable quantity exists under a hypernym noun; the SUM body
            # handles that with its own fallback, so an early None here would
            # wrongly suppress it.
            _inner, proof = self._eval(node.operand, records, reference_date)
            unit = self._unit_of(node.operand)
            obs = extract_numeric(records, unit=unit)
            # Unit-agnostic fallback: when the question's counted unit (e.g.
            # "pieces of writing") is a hypernym the records phrase with more
            # specific nouns ("short stories", "poems"), a strict unit filter
            # matches nothing.  In that case — and ONLY when the unit is a
            # countable object rather than a measurement — sum every countable
            # quantity.  This is safe: "pieces of writing" genuinely covers
            # stories + poems + challenge pieces.  Measurement units (hours,
            # dollars) are never relaxed, so we never sum unrelated measures.
            if not obs and unit and unit.lower() not in _MEASUREMENT_UNITS:
                # Relax to countable quantities, dropping measurement units
                # (weeks/hours/dollars — a "17 poems in two weeks" time scope is
                # not part of the count).  Prefer COMPLETION-scoped quantities:
                # "have written five stories" asserts a completed count, whereas
                # "a link between the two stories" merely mentions the object.
                # Fall back to all countable quantities if none are completion-
                # scoped, so we never return empty on an unusual phrasing.
                all_obs = extract_numeric(records, unit=None)
                countable = [o for o in all_obs
                             if _unit(o).lower() not in _MEASUREMENT_UNITS]
                completed = extract_numeric(records, unit=None, completion_only=True)
                completed = [o for o in completed
                             if _unit(o).lower() not in _MEASUREMENT_UNITS]
                obs = self._dedup_same_source(completed or countable)
            if not obs:
                return None, proof
            # CONFIDENCE GATE: every summed quantity must come from a record
            # that mentions the question's topic.  A distractor-heavy corpus
            # carries unrelated same-unit quantities ("6 years", "1000
            # classes") that would pollute the total; if ANY source lacks the
            # topic, we defer to the LLM rather than override with noise.
            if node.topic_terms and not self._all_topic_scoped(obs, node.topic_terms):
                return None, "[SUM] untrusted sources; defer to LLM"
            # TIME-SCOPE FILTER: a relative window ("last month", "past few
            # months") must bound the sum.  Without a reference date we cannot
            # resolve the window, so we defer to the LLM rather than sum the
            # entire history.  With one, drop quantities dated outside the
            # window; if any lack a resolvable date, defer (never guess).
            if node.time_scope:
                if not reference_date:
                    return None, "[SUM] relative window without reference date; defer to LLM"
                windowed = self._filter_by_window(obs, node.time_scope, reference_date)
                if windowed is None:
                    return None, "[SUM] undated quantity in window; defer to LLM"
                obs = windowed
                if not obs:
                    return None, "[SUM] no quantities in window; defer to LLM"
            # CONFIDENCE GATE: a legit distributed quantity lives in a handful
            # of turns.  If the same-unit values come from many distinct
            # records, the corpus is distractor-heavy and the total is
            # untrustworthy (e.g. 'how many hours playing games' summed 1675h
            # from unrelated turns).  Defer to the LLM rather than emit noise.
            distinct_sources = {o.source for o in obs}
            if len(distinct_sources) > 8:
                return None, f"[SUM] {len(distinct_sources)} distinct sources; distractor-heavy; defer to LLM"
            # CONFIDENCE GATE: a single source carrying many same-unit values
            # (e.g. a turn listing '$5,000, $15,000, $20,000 ...' as illustrative
            # figures) is reference material, not the user's own quantities.
            # A real contribution is one value per source.  Collapse each source
            # to its single value and drop sources that carry a suspicious pile.
            _per_source: dict[str, list] = {}
            for o in obs:
                _per_source.setdefault(o.source, []).append(o)
            collapsed = []
            for _src, _lst in _per_source.items():
                if len(_lst) > 3:
                    # >3 same-unit values in one turn is a distractor list;
                    # take only the first (the salient one) if the turn asserts
                    # a single topic value, else defer.
                    collapsed.extend(_lst[:1])
                else:
                    collapsed.extend(_lst)
            obs = collapsed
            total = sum(_num(o) for o in obs)
            out_unit = _unit(obs[0])
            terms = " + ".join(o.value for o in obs)
            return (
                f"{_fmt_num(total)} {out_unit}".strip(),
                f"{proof}\nSUM: {terms} = {_fmt_num(total)} {out_unit}",
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
            # CONFIDENCE GATE: a genuine object/people count is small.  A huge
            # unique-value count (e.g. 440 'persons' from every proper noun in
            # a distractor-heavy corpus) means the extractor grabbed noise, so
            # defer to the LLM rather than emit a wild number.
            if len(seen) > 15:
                return None, f"[COUNT] {len(seen)} unique values; distractor-heavy; defer to LLM"
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
            # CONFIDENCE GATE: numerator and denominator must be topic-scoped
            # and there must be EXACTLY two candidate values.  A distractor-
            # heavy corpus carries unrelated numbers that would otherwise be
            # picked by position and produce a nonsense ratio.
            if node.topic_terms:
                num_obs = [o for o in num_obs if self._all_topic_scoped([o], node.topic_terms)]
            if len(num_obs) != 2:
                return None, f"[RATIO] need exactly 2 topic values, got {len(num_obs)}; defer to LLM"
            vals = [(_num(o), o) for o in num_obs]
            # ORIENT numerator/denominator by the 'percentage of X do Y hold'
            # structure (English general rule): the value whose source mentions
            # the 'of' phrase is the denominator; the subject is the numerator.
            # Falling back to order-of-appearance is what produced 80% for a
            # 20% question.  Only orient when BOTH sides are unambiguously
            # matched; otherwise defer to the LLM rather than guess.
            if node.denominator_terms and node.numerator_terms:
                def _matches(obs, terms):
                    low = _text_of(obs).lower()
                    return any(t in low for t in terms)
                den = [v for v in vals if _matches(v[1], node.denominator_terms)]
                num = [v for v in vals if _matches(v[1], node.numerator_terms)]
                if len(den) == 1 and len(num) == 1 and den[0] is not num[0]:
                    (d_num, d_obs) = den[0]
                    (n_num, n_obs) = num[0]
                else:
                    return None, "[RATIO] cannot orient numerator/denominator; defer to LLM"
            else:
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
            # CONFIDENCE GATE: exactly two topic-scoped point values, so
            # target/current are unambiguous.  Otherwise defer to the LLM.
            if node.topic_terms:
                num_obs = [o for o in num_obs if self._all_topic_scoped([o], node.topic_terms)]
            if len(num_obs) != 2:
                return None, f"[GAP] need exactly 2 topic points, got {len(num_obs)}; defer to LLM"
            nums = [_num(o) for o in num_obs]
            target, current = max(nums), min(nums)
            gap = target - current
            return str(_fmt_num(gap)), f"[GAP] {_fmt_num(target)} - {_fmt_num(current)} = {_fmt_num(gap)}"

        if isinstance(node, A.Supersedes):
            obs = self._collect_state(node.topic, records, None,
                                      topic_terms=node.topic_terms)
            dated = [(parse_date(o.date) or parse_date(o.source), o) for o in obs]
            dated = [(d, o) for d, o in dated if d is not None]
            if len(dated) < 2:
                return None, "[SUPERSEDES] need >=2 dated observations"
            dated.sort(key=lambda x: x[0])
            chosen = dated[-1][1] if node.want == "latest" else dated[0][1]
            # Extract the topic-scoped value PHRASE, never the raw turn text.
            # A raw header like "[session on date] User: ..." is never a valid
            # answer, so we refuse (return None -> LLM handles it) rather than
            # override with noise.
            phrase = self._topic_phrase(chosen.value, node.topic_terms)
            if not phrase:
                return None, "[SUPERSEDES] no clean value phrase; defer to LLM"
            return phrase, f"[SUPERSEDES {node.want}] {phrase}"

        if isinstance(node, A.DateDiff):
            # Collect distinct dates across the (topic-scoped) records, take the
            # earliest and latest, and return their difference in the asked
            # unit.  Defer if fewer than 2 distinct dates are resolvable.
            dates = self._collect_distinct_dates(records, node.topic_terms)
            if len(dates) < 2:
                return None, f"[DATE_DIFF] need >=2 distinct dates, got {len(dates)}; defer to LLM"
            dates.sort()
            delta = (dates[-1] - dates[0]).days
            if node.unit == "weeks":
                val = round(delta / 7)
            elif node.unit == "months":
                val = round(delta / 30.44)
            else:
                val = delta
            proof = (f"[DATE_DIFF] {dates[0].date()} -> {dates[-1].date()} "
                     f"= {delta} days ~ {val} {node.unit}")
            return f"{val} {node.unit}", proof

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
            # CONFIDENCE GATE: a slot value is a short, self-contained value
            # (a name, a handle, a city) — never a clause.  A capture like
            # 'tant: Great! You can name it ...' (a ':' misparse) is a long
            # sentence fragment and must defer to the LLM, not override.
            # Also reject values that start with a function word (article,
            # conjunction, quote) — those are clause fragments, not values.
            _clause_starters = ("the ", "a ", "an ", "your ", "my ", "and ",
                                "or ", "but ", "with ", "\"", "'", "along ",
                                "for ", "to ", "of ", "'s ", "s name", "s the",
                                "it ", "its ", "their ", "his ", "her ",
                                "dear ", "hi ", "hello ", "hey ", "[", "{",
                                "<", "re:", "fwd:", "subject:", "note:")
            # reject values carrying stray quote/punctuation artifacts — a
            # clean slot value has none.  ':' / '[' / '{' mark a label or a
            # header, never a value.
            _bad_chars = ('"', "!", "...", "—", ":", "[", "{", "<")
            if (not val or len(val.split()) > 6
                    or val.lower().startswith(_clause_starters)
                    or any(c in val for c in _bad_chars)):
                return None, "[SLOT_LOOKUP] value is a sentence fragment; defer to LLM"
            return val.strip(), f"[SLOT_LOOKUP] {first}"

        return None, f"[UNSUPPORTED_NODE] {type(node).__name__}"

    @staticmethod
    def _filter_by_window(obs, time_scope: str, reference_date: str):
        """Keep observations whose date falls in the relative window.

        Returns the filtered list, or None if any observation lacks a
        resolvable date (so the caller defers to the LLM rather than summing a
        partially-dated set).  Window is measured back from ``reference_date``.
        """
        from datetime import datetime, timedelta
        ref = parse_date(reference_date)
        if ref is None:
            return None
        windows = {
            "last_week": timedelta(days=7),
            "last_month": timedelta(days=31),
            "last_year": timedelta(days=366),
            "past_weeks": timedelta(days=28),
            "past_months": timedelta(days=120),
        }
        delta = windows.get(time_scope)
        if delta is None:
            return None
        start = ref - delta
        kept = []
        for o in obs:
            d = parse_date(o.date) or parse_date(o.source)
            if d is None:
                return None  # undated -> cannot confirm it is in-window
            if start <= d <= ref:
                kept.append(o)
        return kept

    @staticmethod
    def _all_topic_scoped(obs, topic_terms: tuple) -> bool:
        """True iff every observation's source mentions a topic term.

        Used as a confidence gate: if any contributing quantity comes from a
        record unrelated to the question's topic, the aggregate is untrustworthy
        and we defer to the LLM instead of overriding.
        """
        terms = [t.lower() for t in topic_terms]
        if not terms:
            return True
        for o in obs:
            src = (o.source or "").lower()
            hit = False
            for t in terms:
                b = t[:-1] if t.endswith("s") else t
                if t in src or b in src:
                    hit = True
                    break
            if not hit:
                return False
        return True

    @staticmethod
    def _collect_distinct_dates(records, topic_terms: tuple):
        """Collect distinct resolvable dates from records (topic-scoped if terms given).

        Pulls dates from the structured date field AND from inline date
        mentions inside the raw text ("recovered on 5/10", "on 2023/05/01"),
        because the session-level time_scope is identical across a session's
        records — the *event* dates live in the text.
        """
        terms = [t.lower() for t in topic_terms]
        seen: set = set()
        out = []
        for r in records:
            text = _text_of(r)
            date_str = _date_of(r)
            if terms:
                low = text.lower()
                if not any((t in low) or ((t[:-1] if t.endswith("s") else t) in low)
                           for t in terms):
                    continue
            candidates = []
            d0 = parse_date(date_str) or parse_date(text)
            if d0 is not None:
                candidates.append(d0)
            # inline date mentions: YYYY/MM/DD, MM/DD/YYYY, MM/DD (year inferred)
            for m in re.finditer(r"\b(\d{4})[/](\d{1,2})[/](\d{1,2})\b", text):
                y, mo, dy = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try:
                    candidates.append(datetime(y, mo, dy))
                except ValueError:
                    pass
            for m in re.finditer(r"\b(\d{1,2})[/](\d{1,2})[/](\d{4})\b", text):
                mo, dy, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try:
                    candidates.append(datetime(y, mo, dy))
                except ValueError:
                    pass
            # bare MM/DD (year inherited from the record's own date, else the
            # session time_scope year) — event dates are usually written short
            ref_year = None
            if date_str:
                ym = re.search(r"\b(\d{4})\b", date_str)
                if ym:
                    ref_year = int(ym.group(1))
            for m in re.finditer(r"\b(?<!\d)(\d{1,2})[/](\d{1,2})(?!\d)(?!\s*/)\b", text):
                mo, dy = int(m.group(1)), int(m.group(2))
                if mo < 1 or mo > 12 or dy < 1 or dy > 31:
                    continue
                y = ref_year or datetime.now().year
                try:
                    candidates.append(datetime(y, mo, dy))
                except ValueError:
                    pass
            for d in candidates:
                key = d.date()
                if key in seen:
                    continue
                seen.add(key)
                out.append(d)
        return out

    @staticmethod
    def _topic_phrase(text: str, topic_terms: tuple) -> str | None:
        """Extract the topic-scoped value phrase from a raw turn.

        Strips the conversation header (``[session on date] Role:``) and
        returns the sentence that mentions the topic entity.  Never returns the
        raw turn text: if no topic-bearing sentence is found, returns None so
        the caller defers to the LLM instead of overriding with noise.
        """
        # strip header prefix like "[sess on 2023/...] User:"
        body = re.sub(r"^\s*\[[^\]]*\]\s*\w*:?\s*", "", text)
        sentences = re.split(r"(?<=[.!?])\s+", body)
        terms = [t.lower() for t in topic_terms]
        for s in sentences:
            sl = s.lower()
            for t in terms:
                b = t[:-1] if t.endswith("s") else t
                if t in sl or b in sl:
                    phrase = s.strip().rstrip(".,;:")
                    # CONFIDENCE GATE: a state value is a short phrase (a
                    # location, a brand, a name) — never a full sentence.  A
                    # 40-word 'value' is descriptive prose and must defer to
                    # the LLM.  Cap at 15 words.
                    if 2 <= len(phrase.split()) <= 15:
                        return phrase
        return None

    @staticmethod
    def _dedup_same_source(obs):
        """Collapse repeated identical values within the same session/day.

        Conversations restate one fact across adjacent user and assistant
        turns in the SAME session ("17 poems" said, then echoed back).  Those
        are one real quantity, not two.  We key on (value, date) so echoes
        inside a session collapse, while the same quantity genuinely stated on
        a different day (a real second occurrence) is still summed.
        """
        seen: set[tuple[str, str]] = set()
        out = []
        for o in obs:
            key = (o.value, o.date)
            if key in seen:
                continue
            seen.add(key)
            out.append(o)
        return out

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

    def _collect_state(self, node: A.Node, records: Sequence, _unused,
                       topic_terms: tuple = ()) -> list[Observation]:
        """Collect dated text observations for state evolution (SUPERSEDES).

        Unlike numeric extraction, a 'state' is any dated utterance; we keep
        them all and let date-ordering pick the latest/earliest.  When
        ``topic_terms`` is given, only records mentioning the topic entity are
        kept, so an unrelated later turn cannot hijack the 'latest' pick.
        """
        out: list[Observation] = []
        terms = [t.lower() for t in topic_terms]
        for r in records:
            text = _text_of(r)
            date = _date_of(r)
            if not text or not date:
                continue
            if terms:
                low = text.lower()
                base_hit = False
                for t in terms:
                    b = t[:-1] if t.endswith("s") else t
                    if t in low or b in low:
                        base_hit = True
                        break
                if not base_hit:
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
