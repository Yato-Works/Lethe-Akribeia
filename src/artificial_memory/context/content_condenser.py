"""Deterministic conversational-fat stripper for compiled context units.

Eli's "10-passenger principle": a selected turn is kept for its facts, not for
its dialogue envelope.  This module purges greetings, sign-offs, discourse
fillers and repetitive agreement from a unit's body, and optionally removes the
duplicated ``(In reply to ...)`` quote that makes up ~36% of a compiled context.

Everything here is pure string surgery -- no LLM, no network, fully
reproducible.  The leading ``[D7:22 on ...]`` provenance header and the speaker
name are never touched: the Layer-1 oracle predicate is a substring test for
``D7:22`` inside ``context_text``, so dropping the label would silently destroy
oracle recall measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- patterns -----------------------------------------------------------------

# Opening chit-chat.  Applied repeatedly so "Hey! Hi there. How are you?" dies.
# Each alternative is \b-terminated so "hi" cannot eat "History works.".
_GREETING = re.compile(
    r"^(?:hey\b|hi\b|hello\b|howdy\b|good (?:morning|afternoon|evening)\b"
    r"|glad to hear\b|thanks for asking\b|how are you\b|hope you'?re doing well\b)"
    r"[^.!?\n]*[.!?][\s\"']*",
    re.IGNORECASE,
)

# Speaker prefix ("Caroline: ") -- stripped so the body-level patterns below can
# match at ^, then re-attached verbatim.
_SPEAKER = re.compile(r"^([A-Za-z][A-Za-z .'-]{0,30}):\s*")

# Polite sign-offs at the very end of a unit.
_SIGNOFF = re.compile(
    r"[\s,]*(?:have a great day|have a good (?:one|day|week)|let me know if you need anything"
    r"|let me know if there'?s anything|take care|talk soon|catch you later"
    r"|i'?m always here|talk to you (?:soon|later)|thanks again|thanks so much)"
    r"[^.!?\n]*[.!?]?[\s\"']*$",
    re.IGNORECASE,
)

# Discourse fillers, sentence-initial only (so "do you know," survives).
_FILLER = re.compile(
    r"^(?:well|you know|to be honest|honestly|as a matter of fact|anyway|so yeah"
    r"|i mean|basically|actually)[,:\s]+",
    re.IGNORECASE,
)

# Repetitive agreement: the utterance is praise/consent carrying no fact.
_AGREEMENT = re.compile(
    r"^(?:(?:yeah|yep|yes|yup|absolutely|totally|definitely|for sure|i agree"
    r"|that'?s right|exactly)[,!.\s]*)+"
    r"(?:that sounds (?:really |so |pretty )?(?:awesome|great|amazing|wonderful|fantastic|cool)"
    r"[^.!?\n]*[.!?][\s\"']*)?",
    re.IGNORECASE,
)

# ``(In reply to Melanie: "....")`` -- the quoted run may itself contain ')',
# so the block terminates at the closing quote+paren.
_QUOTE_OPEN = "(In reply to "
_QUOTE_CLOSE = '")'


@dataclass(frozen=True)
class CondenseOptions:
    """Which deterministic strippers are active."""

    strip_greetings: bool = True
    strip_signoffs: bool = True
    strip_fillers: bool = True
    condense_agreements: bool = True
    # keep      -> leave the quote in place (frozen behaviour)
    # drop      -> remove it outright (maximum compression)
    # provenance-> keep "reply to <speaker>" but discard the quoted run
    # redundant -> drop ONLY when the same quoted text already appears in
    #              another selected unit's body (zero-loss by construction)
    quote_mode: str = "keep"
    # Rewrite ``[D7:22 on 8:56 pm on 20 July, 2023]`` to
    # ``[D7:22 20 July, 2023 20:56]``.  The label, the calendar date and the
    # instant are all preserved exactly (24-hour form); only the filler words
    # ``on``/``on``/``pm`` are dropped.  An ISO date is deliberately NOT used:
    # measured on LoCoMo it made the reader copy ``2023-02-08`` where the
    # ground truth reads ``February, 2023``.  The ``D7:22`` provenance label
    # the Layer-1 oracle tests for is preserved verbatim.
    compact_header: bool = False


DEFAULT = CondenseOptions()
OFF = CondenseOptions(
    strip_greetings=False,
    strip_signoffs=False,
    strip_fillers=False,
    condense_agreements=False,
    quote_mode="keep",
    compact_header=False,
)


def _split_header(text: str) -> tuple[str, str]:
    m = re.match(r"^(\[[^\]]*\]\s*)", text)
    if m:
        return m.group(1), text[m.end():]
    return "", text


def _quote_span(text: str) -> tuple[int, int] | None:
    start = text.find(_QUOTE_OPEN)
    if start < 0:
        return None
    end = text.find(_QUOTE_CLOSE, start)
    if end == -1:
        return None
    return start, end + len(_QUOTE_CLOSE)


def _handle_quote(body: str, mode: str) -> str:
    if mode == "keep":
        return body
    span = _quote_span(body)
    if span is None:
        return body
    start, end = span
    quote = body[start:end]
    if mode == "drop":
        return (body[:start] + body[end:]).lstrip()
    m = re.match(r"\(In reply to ([^:\"]+)", quote)
    who = m.group(1).strip() if m else ""
    keep = f"(reply to {who})" if who else ""
    return (body[:start] + keep + body[end:]).lstrip()


# ``[D7:22 on 8:56 pm on 20 July, 2023]`` -> ``[D7:22 2023-07-20 20:56]``.
_HEADER_RE = re.compile(
    r"^\[(?P<label>D\d+:\d+) on (?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<apm>am|pm)"
    r" on (?P<d>\d{1,2}) (?P<mon>[A-Za-z]{3,9}), (?P<y>(?:19|20)\d{2})\]"
)
_MONTH_NO = {
    name: i
    for i, name in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"], start=1
    )
}


def compact_header(text: str) -> str:
    """Rewrite ``[D7:22 on 8:56 pm on 20 July, 2023]`` to ``[D7:22 20 July, 2023 20:56]``.

    Nothing is lost: the provenance label, the calendar date and the instant
    all survive (the time is merely rendered in 24-hour form).  The Layer-1
    oracle predicate is a substring test for the label, so it is copied through
    untouched.
    """
    m = _HEADER_RE.match(text)
    if not m:
        return text
    mon = _MONTH_NO.get(m.group("mon").capitalize())
    if mon is None:
        return text
    hour = int(m.group("h")) % 12
    if m.group("apm") == "pm":
        hour += 12
    month_name = m.group("mon")
    return (
        f"[{m.group('label')} {int(m.group('d'))} {month_name}, "
        f"{m.group('y')} {hour:02d}:{m.group('m')}]"
    ) + text[m.end():]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _quote_text(quote: str) -> str:
    s, e = quote.find('"'), quote.rfind('"')
    return quote[s + 1:e] if s >= 0 and e > s else quote


def drop_redundant_quotes(lines: list[str], *, min_words: int = 4) -> list[str]:
    """Remove ``(In reply to ...)`` quotes whose text appears elsewhere.

    "Zero-loss by construction": a quote is dropped only when its quoted
    payload is already present verbatim in another selected unit's body, so no
    fact leaves the context.  Short quotes are kept because a substring test on
    a handful of words would match by coincidence.
    """
    corpus: list[str] = []
    for line in lines:
        span = _quote_span(line)
        if span is None:
            corpus.append(line)
        else:
            corpus.append(line[:span[0]] + line[span[1]:])
    haystack = _norm(" ".join(corpus))

    out: list[str] = []
    for line in lines:
        span = _quote_span(line)
        if span is None:
            out.append(line)
            continue
        payload = _norm(_quote_text(line[span[0]:span[1]]))
        if len(payload.split()) >= min_words and payload in haystack:
            out.append((line[:span[0]] + line[span[1]:]).strip())
        else:
            out.append(line)
    return out


def _condense_body(body: str, opts: CondenseOptions) -> str:
    if opts.strip_greetings:
        for _ in range(4):
            stripped = _GREETING.sub("", body, count=1)
            if stripped == body:
                break
            body = stripped

    if opts.condense_agreements:
        # Only collapse an agreement that leads into generic praise; if the
        # utterance carries its own content after the agreement, keep it all.
        new = _AGREEMENT.sub("", body, count=1)
        if new != body and len(new.split()) >= 3:
            body = new
        elif new != body and not new.strip():
            body = "Yeah."

    if opts.strip_fillers:
        for _ in range(4):
            stripped = _FILLER.sub("", body, count=1)
            if stripped == body:
                break
            body = stripped
        # sentence-initial fillers further down the body
        body = re.sub(r"(?<=[.!?])\s+(?:well|you know|to be honest|honestly"
                      r"|as a matter of fact|basically|actually|anyway)[,:\s]+",
                      " ", body, flags=re.IGNORECASE)

    if opts.strip_signoffs:
        body = _SIGNOFF.sub("", body)

    return body


def condense_turn(text: str, opts: CondenseOptions = DEFAULT) -> str:
    """Strip conversational fat from one rendered context unit.

    Preserves the ``[D7:22 on ...]`` provenance header verbatim (Layer-1 oracle
    recall depends on it) and preserves the speaker name (the reader needs it
    for "who said what" questions).
    """
    header, rest = _split_header(text)
    if opts.compact_header:
        header = compact_header(header)
    # "redundant" cannot be judged from one unit alone -- it is resolved by
    # drop_redundant_quotes() once every selected unit is known.
    rest = _handle_quote(rest, "keep" if opts.quote_mode == "redundant" else opts.quote_mode)
    m = _SPEAKER.match(rest)
    speaker = f"{m.group(1)}: " if m else ""
    body = rest[m.end():] if m else rest
    body = _condense_body(body, opts)
    body = re.sub(r"\s{2,}", " ", body).strip()
    if not body:
        # Never delete the unit outright: the provenance label is exactly what
        # the Layer-1 oracle predicate tests for, so an emptied body keeps its
        # header (and msc_compiler would otherwise drop the line entirely).
        return header.rstrip() if header else ""
    return header + speaker + body
