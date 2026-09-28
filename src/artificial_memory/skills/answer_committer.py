"""Deterministic answer committer - AM decides, the reader only renders.

Rationale (measured, see ``benchmark_results/AM_APEX_STATUS.md``)
----------------------------------------------------------------
On LoCoMo-1,540 with ``qwen2.5:7b-instruct`` the memory layer retrieves the
gold turn 84.8% of the time, yet the reader converts only 65.0% of questions
into a correct answer (temporal: 86.6% retrieval -> 42.7% accuracy).  The
temporal post-mortem (``_failure_census.py --run locomo_instruct_full``) shows
where the loss sits:

* 45 of 184 wrong temporal answers contain the ground truth **verbatim** in the
  context the reader was shown - the model quoted the provenance header date
  instead of the resolved annotation inside the evidence turn;
* 46 more need only calendar arithmetic that the runtime already performs
  (:mod:`artificial_memory.context.temporal_normalizer`);
* 61 are multi-item enumerations that the reader truncated.

None of those three failure modes requires reasoning ability.  They require the
runtime to *decide* the answer and the reader to *copy* it.  This module is that
decision layer: it turns the compiled context plus the deterministic skills into
a committed answer, with a confidence, so the caller can either bypass the LLM
entirely or hand it a single slot to render.

Design rules
------------
* No LLM calls, no network, no randomness - same input, same output.
* No ground-truth access: the committer only ever reads the question and the
  context that the reader would have been shown.
* Abstain (``used=False``) whenever the evidence is ambiguous, so the existing
  reader path stays in charge for everything the committer cannot prove.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

#: Month names as they appear in the LoCoMo provenance headers and in prose.
_MONTH_NAME = (r"(?:january|february|march|april|may|june|july|august|september|"
               r"october|november|december)")
_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_NUMBER_WORD = (r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                r"a\s+couple\s+of|a\s+few|several")

#: A concrete calendar date: "7 May 2023", "May 2023", "May, 2023", "2022".
_ABS_DATE = re.compile(
    rf"\b(?:\d{{1,2}}\s+(?:of\s+)?{_MONTH_NAME}\s*,?\s*\d{{4}}"
    rf"|{_MONTH_NAME}\s*,?\s*\d{{4}}"
    rf"|(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
#: A duration: "two weeks", "4 years", "10 years ago".
_DURATION = re.compile(
    rf"\b(?:{_NUMBER_WORD}|\d+)\s+(?:day|week|month|year|hour|minute)s?\b",
    re.IGNORECASE,
)
#: A relative interval resolved against a session date, as produced by
#: TemporalNormalizer: "the Saturday before 25 May 2023", "the week before
#: 9 June 2023", "the summer of 2022", "the week of 15 July 2023".
_INTERVAL = re.compile(
    rf"\b(?:the\s+)?(?:(?:week|weekend|month|day|{_WEEKDAY}|summer|winter|spring|fall"
    rf")\s+(?:before|after|of)|(?:week|month|day)s?\s+before)\s+[^)\.\n]{{0,40}}",
    re.IGNORECASE,
)
#: Parenthesised annotations injected by TemporalNormalizer / the compiler.
_PAREN = re.compile(r"\(([^()]{2,80})\)")
#: Provenance header, LoCoMo format: "[D1:14 on 1:56 pm on 8 May, 2023]".
#: The trailing whitespace is consumed on purpose: a turn like
#: ``[D1:14 on ...] (In reply to Caroline: "...") Melanie: text`` would otherwise
#: leave a leading space in the payload, the reply-quote stripper (anchored at
#: ``^``) would not match, and the whole turn would be silently dropped - which is
#: how the first version of this module lost every reply turn from the cache.
_HEADER = re.compile(
    r"^\[D\d+:\d+\s+on\s+(?:\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+)?([^\]]+)\]\s*",
    re.IGNORECASE,
)
#: Provenance header, LongMemEval format: "[answer_280352e9 on 2023/05/30 (Tue) 17:27]".
#: The evidence id is kept as the turn's dia_id and the timestamp as its date.
_HEADER_LME = re.compile(r"^\[([A-Za-z0-9_-]+)\s+on\s+([^\]]+)\]\s*")
#: LongMemEval speakers are roles, not names.
_ROLE = r"(?:user|assistant|system|tool)"
#: "Caroline: text" (LoCoMo) or "user: text" (LongMemEval).
_SPEAKER = re.compile(
    rf"^\s*(?:([A-Z][A-Za-z' ]{{1,24}})|({_ROLE}))\s*:\s*(.*)$"
)

_STOP = {
    "when", "what", "which", "who", "whom", "where", "why", "how", "did", "does",
    "do", "was", "were", "is", "are", "has", "have", "had", "the", "a",
    "an", "and", "or", "of", "to", "in", "on", "at", "for", "with", "before",
    "after", "between", "since", "ago", "last", "next", "this", "that", "it",
    "he", "she", "they", "them", "his", "her", "their", "there", "then", "than",
    "as", "by", "from", "into", "out", "about", "up", "down", "again", "also",
    "just", "get", "got", "go", "went", "made", "make", "take", "took", "one",
    "time", "times", "first", "second", "long", "many", "much",
    "you", "your", "i", "me", "my", "we", "us", "our", "not", "no", "yes",
}


@dataclass
class CommittedAnswer:
    """A runtime-decided answer, ready to be rendered or returned verbatim."""

    used: bool
    answer: str = ""
    source: str = ""
    confidence: float = 0.0
    detail: str = ""
    candidates: list[str] = field(default_factory=list)
    #: The turn the answer was read from, kept for provenance and for offline
    #: failure analysis (which failure was a wrong turn vs a wrong span).
    evidence_turn: str = ""

    def __bool__(self) -> bool:
        return self.used


@dataclass
class Turn:
    """One dialogue turn recovered from the compiled context."""

    dia_id: str
    header_date: str
    speaker: str
    text: str


def parse_turns(context: str) -> list[Turn]:
    """Split a compiled context into turns, keeping the provenance header date.

    Two header formats are understood, because the same committer has to serve
    both suites:

    LoCoMo
        ``[D4:5 on 27 June, 2023] (In reply to Melanie: "...") Caroline: text``
    LongMemEval
        ``[answer_280352e9 on 2023/05/30 (Tue) 17:27] user: I graduated ...``

    In both cases the ``(In reply to ...)`` quote is dropped so words quoted from
    another speaker never count as this turn's own content.  A line that yields no
    turn is dropped rather than guessed at - a turn that is silently missing is
    invisible later, so the format handling is covered by unit tests.
    """
    turns: list[Turn] = []
    for raw in context.splitlines():
        line = raw.strip()
        if not line:
            continue
        dia_id = ""
        header_date = ""
        header = _HEADER.match(line)
        if header is not None:
            header_date = header.group(1).strip()
            dia_id = header.group(0)[1:-1].split(" ")[0]
            body = line[header.end():]
        else:
            header_lme = _HEADER_LME.match(line)
            if header_lme is not None:
                dia_id = header_lme.group(1)
                header_date = header_lme.group(2).strip()
                body = line[header_lme.end():]
            else:
                body = line
        body = re.sub(r'^\(In reply to .*?"\)\s*', "", body, flags=re.IGNORECASE)
        body = re.sub(r"^\(In reply to [^)]*\)\s*", "", body, flags=re.IGNORECASE)
        speaker_match = _SPEAKER.match(body)
        if not speaker_match:
            continue
        speaker = (speaker_match.group(1) or speaker_match.group(2)).strip()
        turns.append(Turn(
            dia_id=dia_id,
            header_date=header_date,
            speaker=speaker,
            text=speaker_match.group(3).strip(),
        ))
    return turns


def _stem(word: str) -> str:
    """Conservative suffix stripper so "sign" and "signed" identify the same fact.

    Deliberately the same shape as ``AnswerVerifier._stem``: strip a suffix only
    when at least three characters survive, so "car"/"cat" stay distinct.  Without
    this, a question ("When did Alice sign the lease?") and its evidence turn
    ("I signed the lease") share no key term and the committer abstains even
    though the fact is sitting in the turn.
    """
    w = word.lower()
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[: -len(suffix)]
    return w


def content_words(text: str) -> set[str]:
    """Stemmed content tokens used for lexical turn scoring (lowercased, len > 2)."""
    return {
        _stem(w) for w in re.findall(r"[a-z0-9']+", text.lower())
        if len(w) > 2 and w not in _STOP
    }


def _norm_date(text: str) -> str:
    """Normalise a date span onto the form the official scorer matches."""
    value = text.strip().strip(",.;:")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", value, flags=re.IGNORECASE)
    # "8 May, 2023" -> "8 May 2023"; "May, 2023" -> "May 2023"
    value = re.sub(rf"({_MONTH_NAME})\s*,\s*", r"\1 ", value, flags=re.IGNORECASE)
    return value.strip()


def _spans_in_turn(turn: Turn) -> list[tuple[str, int, int, str]]:
    """Return ``(kind, priority, position, value)`` date-ish spans in one turn.

    ``kind`` is the answer shape the span can satisfy:

    * ``interval`` - an interval the runtime itself resolved ("the Saturday
      before 25 May 2023", "the summer of 2022"): the most authoritative form,
      because TemporalNormalizer derived it from the session date;
    * ``date`` - an absolute calendar date stated in the prose ("5 July 2023");
    * ``duration`` - "4 years", "two weeks", "10 years ago";
    * ``year`` - a bare year ("2022");
    * ``header`` - the provenance header date.  It is only the *session* date,
      which is exactly what a copy-style reader wrongly prefers (measured: 45 of
      184 wrong temporal answers quoted this header instead of the annotation),
      so it ranks last and is never used when a better span exists.
    """
    spans: list[tuple[str, int, int, str]] = []
    text = turn.text

    def add(kind: str, priority: int, pos: int, value: str) -> None:
        norm = _norm_date(value)
        if norm:
            spans.append((kind, priority, pos, norm))

    for m in _PAREN.finditer(text):
        inner = m.group(1).strip()
        if _INTERVAL.search(inner):
            add("interval", 3, m.start(), inner)
        elif _DURATION.search(inner) and not _ABS_DATE.search(inner):
            add("duration", 3, m.start(), inner)
        elif _ABS_DATE.search(inner):
            kind = "year" if re.fullmatch(r"(?:19|20)\d{2}", inner) else "date"
            add(kind, 3, m.start(), inner)
    for m in _INTERVAL.finditer(text):
        add("interval", 2, m.start(), m.group(0))
    for m in _DURATION.finditer(text):
        add("duration", 2, m.start(), m.group(0))
    for m in _ABS_DATE.finditer(text):
        value = m.group(0)
        kind = "year" if re.fullmatch(r"(?:19|20)\d{2}", value) else "date"
        add(kind, 2, m.start(), value)
    if turn.header_date:
        add("header", 1, len(text), turn.header_date)
    return spans


def _token_idf(turns: Sequence[Turn]) -> dict[str, float]:
    """Inverse document frequency of every token over the context's turns.

    A word that appears in most turns ("know", "talk") carries almost no
    evidence, while a specific one ("sunrise", "charity") appears in one turn and
    identifies it.  Measured on the frozen cache, plain overlap scoring picked the
    wrong turn for most committed temporal answers; IDF weighting is what fixes
    the selection.

    The +0.5 floor matters: with a 1-2 turn context every token has
    ``df == n_turns`` and the raw IDF collapses to 0, which would silently disable
    the committer (``score < MIN_TURN_SCORE`` forever) on short conversations -
    the "silent no-op" failure class this project already documented elsewhere.
    """
    import math

    df: dict[str, int] = {}
    for turn in turns:
        for word in content_words(turn.text):
            df[word] = df.get(word, 0) + 1
    n = max(1, len(turns))
    return {w: math.log((n + 1) / (1 + c)) + 0.5 for w, c in df.items()}


def _turn_score(
    q_words: set[str],
    weights: dict[str, float],
    question_entities: set[str],
    turn: Turn,
) -> tuple[float, list[str]]:
    """IDF-weighted evidence score of one turn, plus the matched question words."""
    total = sum(weights.get(w, 0.5) for w in q_words)
    if total <= 0:
        return 0.0, []
    turn_words = content_words(turn.text)
    matched = [w for w in q_words if w in turn_words]
    score = sum(weights.get(w, 0.5) for w in matched) / total
    if turn.speaker.lower() in question_entities:
        score += 0.15  # the question names this speaker as the doer
    return score, matched


def _anchor_position(text: str, matched: Sequence[str], weights: dict[str, float]) -> int:
    """Character offset of the highest-IDF matched question word in a turn.

    ``matched`` holds *stems* while the turn text holds surface forms, so a plain
    ``find`` would fail for "sign" against "signed"; a prefix match is used as the
    fallback.  The position is only a tie-breaker for span selection, so a miss
    must degrade to "anchor at 0" rather than drop the candidate.

    ``matched`` is built by iterating a set, so it is walked in sorted order here:
    with two equally-weighted words the anchor would otherwise move between
    processes running with different ``PYTHONHASHSEED`` values.
    """
    best_pos = 0
    best_weight = -1.0
    lowered = text.lower()
    for word in sorted(matched):
        pos = lowered.find(word)
        if pos < 0:
            match = re.search(rf"\b{re.escape(word)}[a-z]*", lowered)
            pos = match.start() if match else -1
        if pos < 0:
            continue
        if weights.get(word, 0.5) > best_weight:
            best_weight = weights.get(word, 0.5)
            best_pos = pos
    return best_pos


def _document_frequency(turns: Sequence[Turn]) -> dict[str, int]:
    """How many turns contain each content token."""
    df: dict[str, int] = {}
    for turn in turns:
        for word in content_words(turn.text):
            df[word] = df.get(word, 0) + 1
    return df


def _key_terms(
    q_words: set[str],
    weights: dict[str, float],
    df: dict[str, int],
    n_turns: int,
    top: int = 2,
) -> list[str]:
    """The question's most discriminative terms, most specific first.

    A word that appears in most turns ("going", "talk") cannot identify the
    evidence turn, so terms seen in more than 40% of the turns are dropped first.
    Measured on the frozen cache: without that filter the winning turn for
    ``"When is Caroline going to the transgender conference?"`` was the turn
    holding the greeting *"How's it going?"*, which matched ``going``; the turn
    that actually names the conference came second.
    """
    cap = max(1, int(0.4 * n_turns))
    kept = [w for w in q_words if df.get(w, n_turns) <= cap]
    pool = kept or list(q_words)
    # Ties are broken by the word itself: ``pool`` comes from a set, and set
    # iteration order changes with PYTHONHASHSEED, which made the committer return
    # a different answer in different processes (measured: 77-80 correct on the
    # same 321 LoCoMo temporal questions across four identical runs).  Sorting by
    # (-weight, word) makes "which key term wins a tie" a property of the input.
    ranked = sorted(pool, key=lambda w: (-weights.get(w, 0.0), w))
    return ranked[:top]


#: Questions whose answer is an elapsed time rather than a calendar date.
_DURATION_QUESTION = re.compile(
    r"\bhow\s+long\b|\bhow\s+many\s+(?:days|weeks|months|years|hours|minutes)\b"
    r"|\bago\b|\belapsed\b",
    re.IGNORECASE,
)
#: Questions whose answer is a bare year.
_YEAR_QUESTION = re.compile(r"\bwhat\s+year\b|\bwhich\s+year\b", re.IGNORECASE)


def _shape_bonus(kind: str, question: str) -> float:
    """How well a span kind matches the answer shape the question asks for."""
    if _DURATION_QUESTION.search(question):
        return 2.5 if kind == "duration" else -0.5
    if _YEAR_QUESTION.search(question):
        return 2.0 if kind == "year" else 0.0
    if kind == "duration":
        return 0.0
    return 1.0 if kind in ("date", "interval") else 0.0


#: "occurred on 2023-06-15" inside a certificate.  The event *name* is recovered by
#: position, not by matching quotes: LongMemEval quotes names that themselves
#: contain apostrophes ("'michael's engagement party'"), so a quote-matching
#: pattern truncates them to "michael".
_OCCURRED_ON = re.compile(r"occurred on\s+(\d{4})-(\d{1,2})-(\d{1,2})", re.IGNORECASE)
#: The verdict is the last clause: "... the event that happened first is 'x'."
_VERDICT = re.compile(
    r"(?:happened\s+(first|last)|earliest|latest)\s+is\s+(.+?)\]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
#: Text before an event name starts after one of these.
_NAME_START = re.compile(r"[.:]\s|\]\s")


def _certificate_events(certificate: str) -> dict[str, tuple[int, int, int]]:
    """Map lowercased event name -> date for every "X occurred on D" clause."""
    events: dict[str, tuple[int, int, int]] = {}
    for match in _OCCURRED_ON.finditer(certificate):
        prefix = certificate[: match.start()]
        starts = list(_NAME_START.finditer(prefix))
        name = _clean_name(prefix[starts[-1].end():] if starts else prefix)
        if not name:
            continue
        events[name.lower()] = (
            int(match.group(1)), int(match.group(2)), int(match.group(3))
        )
    return events


def _clean_name(text: str) -> str:
    """Strip the quoting/punctuation the certificate wraps a name in.

    The verdict clause ends with a full stop *outside* the closing quote
    ("... is 'michael's engagement party'."), so a quote-only strip leaves the
    period attached and every lookup misses.
    """
    value = text.strip().rstrip(".").strip()
    return value.strip("'\"[] ").strip()


def _single_winner(certificate: str) -> tuple[str, str] | None:
    """Return ``(answer, why)`` for a single-winner ordering certificate.

    ``answer`` is empty - meaning "abstain" - when the certificate cannot be
    trusted:

    * the named winner is not one of the dated events (name normalisation drift);
    * **two events share a date**: the certificate keeps the date and drops the
      time of day, so the ordering is a coin flip.  Measured on the real cache,
      "Samsung Galaxy S22 vs Dell XPS 13" (both 2023-03-15), "tomatoes vs
      marigolds" (both 2023-03-10) and "smart thermostat vs mesh network" (both
      2023-05-25) all name the wrong winner, while the reader gets them right;
    * the named winner does not actually hold the extreme date, i.e. the
      resolver's verdict contradicts its own dates (0 of 20 such cases today,
      checked by ``scratch/_ordering_cert_audit.py``).
    """
    verdict = _VERDICT.search(certificate)
    if verdict is None:
        return None
    events = _certificate_events(certificate)
    if len(events) < 2:
        return "", "fewer than two dated events"
    name = _clean_name(verdict.group(2))
    if name.lower() not in events:
        return "", "named winner is not one of the dated events"
    dates = list(events.values())
    if len(set(dates)) != len(dates):
        return "", "same-date tie: the certificate has no time of day"
    wants_first = (verdict.group(1) or "earliest").lower() in ("first", "earliest")
    extreme = min(dates) if wants_first else max(dates)
    if events[name.lower()] != extreme:
        return "", "verdict contradicts the dates inside the certificate"
    return name, "certificate names the winning event"


def extract_certificate_answer(question: str, context: str) -> CommittedAnswer:
    """Commit the runtime's *own* temporal certificate, verbatim in substance.

    LongMemEval contexts open with a certificate the runtime already computed,
    e.g.::

        [Temporal Calculation: Event 1 ('visit to MoMA') occurred on 2023-01-08.
         Event 2 (...) occurred on 2023-01-15. Exactly 7 days passed ...]

        [Temporal Ordering: In chronological order from first to last: First,
         i helped my friend ..., then i helped my cousin ..., and lastly ...]

    122 of the 133 LongMemEval temporal-reasoning questions carry one.  The reader
    is then asked to redo arithmetic the runtime has already done, and gets
    77.4% of the category; this branch removes that step from the model's job and
    only transcribes what the runtime decided.

    Forms handled, in priority order:

    1. ``Exactly N days|weeks|months|years`` -> ``"N unit"`` (the decisive clause);
    2. ``Reference date is D`` + ``occurred on E`` -> the delta in the unit the
       question asks for, computed here (a genuine calendar operation);
    3. ``Temporal Ordering: ...`` -> the ordered enumeration itself.

    Anything else abstains, so a wrong transcription can never replace a reader
    answer that might have been right.
    """
    certificate = ""
    for line in context.splitlines():
        stripped = line.strip()
        if stripped.startswith("[Temporal Calculation:") or stripped.startswith(
            "[Temporal Ordering:"
        ):
            certificate = stripped
            break
    if not certificate:
        return CommittedAnswer(used=False, detail="no temporal certificate in context")

    # 1. The runtime already state the answer.
    exact = re.search(
        r"Exactly\s+(\d+)\s+(day|week|month|year)s?\b", certificate, re.IGNORECASE
    )
    if exact:
        unit = exact.group(2).lower()
        unit = {"day": "days", "week": "weeks",
                "month": "months", "year": "years"}[unit]
        return CommittedAnswer(
            used=True,
            answer=f"{exact.group(1)} {unit}",
            source="temporal_certificate",
            confidence=0.95,
            detail="certificate states the interval",
            evidence_turn=certificate[:400],
        )

    # 2. Reference date minus event date, in the unit the question asks for.
    reference = re.search(r"Reference date is\s+(\d{4})[-/](\d{1,2})[-/](\d{1,2})",
                          certificate, re.IGNORECASE)
    occurred = re.findall(r"occurred on\s+(\d{4})[-/](\d{1,2})[-/](\d{1,2})",
                          certificate, re.IGNORECASE)
    if reference and len(occurred) == 1:
        ref_date = datetime.date(*(int(x) for x in reference.groups()))
        event_date = datetime.date(*(int(x) for x in occurred[0]))
        days = abs((ref_date - event_date).days)
        if days == 0:
            return CommittedAnswer(used=False, detail="zero-length interval")
        if re.search(r"\bweeks?\b", question, re.IGNORECASE):
            answer = f"{max(1, round(days / 7))} weeks"
        elif re.search(r"\bmonths?\b", question, re.IGNORECASE):
            answer = f"{max(1, days // 30)} months"
        elif re.search(r"\byears?\b", question, re.IGNORECASE):
            answer = f"{max(1, days // 365)} years"
        else:
            answer = f"{days} days"
        return CommittedAnswer(
            used=True,
            answer=answer,
            source="temporal_certificate",
            confidence=0.9,
            detail=f"computed {ref_date} - {event_date} = {days} days",
            evidence_turn=certificate[:400],
        )

    # 3. Ordering certificates come in several shapes.
    #
    #    a) A single named winner ("... The event that happened first is 'x'.").
    #       Committed **only when the events fall on distinct dates**.  The
    #       certificate keeps the date and drops the time of day, so a same-day
    #       pair is an unresolvable tie and the resolver's tie-break is
    #       arbitrary: measured on the real cache, "Samsung Galaxy S22 vs Dell
    #       XPS 13" (both 2023-03-15), "tomatoes vs marigolds" (both 2023-03-10)
    #       and "smart thermostat vs mesh network" (both 2023-05-25) all name the
    #       wrong winner, while on the 20 cases with distinct dates the verdict
    #       matches the dates 20/20.  So the guard is not caution, it is the
    #       difference between right and wrong.
    #
    #    b) A numbered "Chronological Order of <X>:" dump of turn texts.  The
    #       entries are *not* reliably the events the question asks about (for
    #       "the six museums I visited" entries 1-4 are turns about unrelated
    #       routines), so they are left to the reader.
    #
    #    c) A "from first to last: ..." enumeration, which is already a
    #       transcription of the ordered answer.
    winner = _single_winner(certificate)
    if winner is not None:
        answer, why = winner
        if answer:
            return CommittedAnswer(
                used=True,
                answer=answer,
                source="temporal_certificate",
                confidence=0.9,
                detail=why,
                evidence_turn=certificate[:400],
            )

    ordering = re.search(
        r"from first to last:\s*(.+?)\]?\s*$", certificate, re.IGNORECASE | re.DOTALL
    )
    if ordering:
        listing = ordering.group(1).strip().rstrip("]")
        if len(listing) >= 10:
            return CommittedAnswer(
                used=True,
                answer=listing,
                source="temporal_certificate",
                confidence=0.8,
                detail="certificate states the chronological order",
                evidence_turn=certificate[:400],
            )

    return CommittedAnswer(used=False, detail="certificate has no committable form")


def extract_temporal_answer(question: str, context: str) -> CommittedAnswer:
    """Decide a temporal answer from the compiled context without a reader LLM.

    Pipeline (fully deterministic): rank turns by IDF-weighted overlap with the
    question, anchor on the strongest matched keyword inside the winning turn,
    then choose the span nearest that anchor whose kind matches the answer shape
    the question asks for (date vs duration vs year).  Anything ambiguous returns
    ``used=False`` so the caller falls back to the existing reader path.
    """
    turns = parse_turns(context)
    if not turns:
        return CommittedAnswer(used=False, detail="no parsable turns")

    q_words = content_words(question) - {"knew", "know", "happen", "happened"}
    q_entities = {w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", question)}
    weights = _token_idf(turns)
    df = _document_frequency(turns)
    keys = _key_terms(q_words, weights, df, len(turns))

    if not keys:
        return CommittedAnswer(used=False, detail="question has no discriminative terms")

    # The single most specific question term must be present in the turn; without
    # that, the turn is not the evidence turn no matter how many generic words it
    # shares with the question.
    candidates_turns: list[tuple[float, list[str], int, Turn]] = []
    for want in (keys, keys[:1]):
        candidates_turns = []
        for i, turn in enumerate(turns):
            turn_words = content_words(turn.text)
            if not all(k in turn_words for k in want):
                continue
            score, matched = _turn_score(q_words, weights, q_entities, turn)
            candidates_turns.append((score, matched, i, turn))
        if candidates_turns:
            break
    if not candidates_turns:
        return CommittedAnswer(used=False, detail=f"no turn contains the key term {keys[0]!r}")

    candidates_turns.sort(key=lambda p: (-p[0], p[2]))

    if candidates_turns[0][0] < MIN_TURN_SCORE:
        return CommittedAnswer(
            used=False,
            detail=f"best turn score {candidates_turns[0][0]:.2f} < {MIN_TURN_SCORE}",
        )

    candidates: list[str] = []
    for score, matched, idx, turn in candidates_turns[:4]:
        if score < MIN_TURN_SCORE:
            break  # never commit from a turn that barely matches the question
        spans = _spans_in_turn(turn)
        if not spans:
            continue
        anchor = _anchor_position(turn.text, matched, weights)
        ranked: list[tuple[float, str]] = []
        for kind, priority, pos, value in spans:
            if kind == "header":
                # The provenance header is the *session* date, not the event date.
                # Committing it reproduces exactly the reader bug we are fixing
                # (measured: it produced 100% of the v2 wrong commits), so a turn
                # with no dated content is skipped instead of guessed.
                continue
            rank = (
                _shape_bonus(kind, question) * 100
                + priority * 10
                - min(abs(pos - anchor), 400) / 100
            )
            ranked.append((rank, value))
        if not ranked:
            continue
        ranked.sort(key=lambda p: -p[0])
        value = ranked[0][1]
        candidates.append(value)
        return CommittedAnswer(
            used=True,
            answer=value,
            source="date_span",
            confidence=round(min(0.99, 0.5 + 0.5 * min(1.0, score)), 2),
            detail=f"turn#{idx} score={score:.2f} anchor={anchor}",
            candidates=[c for _r, c in ranked[:3]],
            evidence_turn=turn.text,
        )
    return CommittedAnswer(used=False, detail="no usable span in top turns", candidates=candidates)


#: Minimum IDF-weighted overlap required before a span may be committed.
#:
#: End-to-end on all 321 LoCoMo temporal questions with the frozen reader
#: (``qwen2.5:7b-instruct``, num_ctx 8192, same prompts, same cache for the arms
#: that are compared against each other - artefacts under
#: ``benchmark_results/locomo1540/``):
#:
#: ====================================  ========  =======  ==========
#: arm                                  artefact  correct  accuracy
#: ====================================  ========  =======  ==========
#: A frozen cache, reader answers         instruct_full  137   42.68%
#: B rebuilt cache (rules 16-18), reader  temporal321_rules  154   47.98%
#: C B + this committer                   temporal321_rules_commit  158   49.22%
#: ====================================  ========  =======  ==========
#:
#: B - A = **+5.30pp** from doing the calendar arithmetic in the runtime instead
#: of the reader.  C - B = **+1.24pp** from answering the questions the runtime can
#: prove.  On C the committer took 111 of 321 questions (34.6%) at **70.3%**
#: accuracy and **0 ms / 0 LLM calls**, while the reader handled the rest at
#: 38.1% and 2257 ms - the selection is the point: the AM takes exactly the
#: mechanical subset.  Wall clock over the category fell 2.25s/Q -> 1.48s/Q.
#:
#: The value below was swept with ``scratch/_commit_eval.py --sweep`` (paired
#: against arm B on the same cache); 0.80 was the best net and still covers a
#: third of the category.  Below ~0.60 the committer loses to the reader.
#:
#: Residual errors on the committed set: 16 not representable from the context at
#: all, 11 wrong turn, 5 wrong span - so evidence-turn recall is the next lever,
#: not span selection.
#:
#: Determinism: two tie-breaks used to read set iteration order
#: (``_key_terms`` ranked a set with a stable sort, ``_anchor_position`` walked
#: ``matched`` in set order), so the same command scored 77, 78, 79 or 80 correct
#: depending on PYTHONHASHSEED - a metric that moves is not a metric.  Both now
#: break ties by name.  Re-measured offline, bit-stable, on
#: ``benchmark_results/committer_metrics/locomo_cat2_temporal.json``: 109 of 321
#: temporal questions claimed (**34.0%**) at **70.6%** commit accuracy against
#: **67.0%** for the paired reader on exactly those questions (+1.25pp on the
#: slice, 0 LLM calls).
MIN_TURN_SCORE = 0.80

