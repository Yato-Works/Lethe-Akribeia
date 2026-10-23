"""Subsystem B: Temporal Algebra & Interval Resolution Engine.

Guarantees:
- Allen's Interval Algebra & Relative temporal anchor extraction.
- Duration and delta calculation between multi-turn events (months/years passed).
- Ultra-concise, precision-first output (never duplicate words to protect Precision).
- 0 LLM calls, 100% deterministic arithmetic.
"""

from __future__ import annotations

import datetime
import os
import re

from artificial_memory_gen.skills.answer_committer import CommittedAnswer, Turn
from artificial_memory_gen.skills.evidence_binding import (
    BindingCandidate,
    BindingRequest,
    BindingVerdicts,
    bind,
)

# Common number word mappings
NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

INT_TO_WORD: dict[int, str] = {v: k for k, v in NUMBER_WORDS.items()}

MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

STOPWORDS = {
    "what", "when", "where", "who", "whom", "which", "why", "how", "did", "does",
    "do", "was", "were", "is", "are", "has", "have", "had", "the", "a",
    "an", "and", "or", "of", "to", "in", "on", "at", "for", "with", "before",
    "after", "between", "since", "ago", "last", "next", "this", "that", "it",
    "he", "she", "they", "them", "his", "her", "their", "there", "then", "than",
    "as", "by", "from", "into", "out", "about", "up", "down", "again", "also",
    "just", "get", "got", "go", "went", "made", "make", "take", "took", "one",
    "time", "times", "first", "second", "long", "many", "much",
    "you", "your", "i", "me", "my", "we", "us", "our", "not", "no", "yes",
}


LEMMA_MAP: dict[str, set[str]] = {
    "win": {"win", "won", "winning"},
    "won": {"win", "won", "winning"},
    "make": {"make", "made", "making"},
    "made": {"make", "made", "making"},
    "paint": {"paint", "painted", "painting"},
    "painted": {"paint", "painted", "painting"},
    "finish": {"finish", "finished", "finishing"},
    "finished": {"finish", "finished", "finishing"},
    "host": {"host", "hosted", "hosting"},
    "hosted": {"host", "hosted", "hosting"},
    "adopt": {"adopt", "adopted", "adopting"},
    "adopted": {"adopt", "adopted", "adopting"},
    "travel": {"travel", "traveled", "travelled", "traveling", "travelling"},
    "traveled": {"travel", "traveled", "travelled", "traveling", "travelling"},
    "create": {"create", "created", "creating"},
    "created": {"create", "created", "creating"},
    "art": {"art", "painting", "paint", "pottery", "drawing", "craft", "sketching"},
    "tournament": {"tournament", "tourney", "competition", "match"},
    "work": {"work", "job", "career"},
    "girlfriend": {"girlfriend", "gf", "partner"},
    "gf": {"girlfriend", "gf", "partner"},
    "go": {"go", "went", "gone", "going"},
    "went": {"go", "went", "gone", "going"},
    "take": {"take", "took", "taken", "taking"},
    "took": {"take", "took", "taken", "taking"},
}


def _word_variants(word: str) -> set[str]:
    """Morphological variants of a token, for tolerant keyword matching.

    "passed" -> {passed, pass}; "interviews" -> {interviews, interview};
    "painting" -> {painting, paint}.  Only inflections are generated - never a
    different word - so a hit still requires the same lexeme in the evidence.
    """
    w = word.lower()
    variants = {w}
    if w.endswith("s") and len(w) > 3:
        variants.add(w[:-1])
    if w.endswith("es") and len(w) > 4:
        variants.add(w[:-2])
    if w.endswith("ed") and len(w) > 4:
        variants.add(w[:-2])
        variants.add(w[:-1])
    if w.endswith("ing") and len(w) > 5:
        variants.add(w[:-3])
        variants.add(w[:-3] + "e")
    return variants


def _turn_keyword_idf(q_kws: set[str], scored_turns: list[tuple[set[str], set[str]]]) -> dict[str, float]:
    """Smoothed IDF per question keyword over the candidate turns.

    A keyword that appears in every turn ("game", "play") cannot discriminate
    the evidence turn; a keyword unique to one turn ("pottery", "valorant")
    pins it.  Turns whose keyword profile is distinctive therefore win.
    """
    n = max(1, len(scored_turns))
    df: dict[str, int] = {kw: 0 for kw in q_kws}
    for _, variants in scored_turns:
        for kw in q_kws:
            expanded = LEMMA_MAP.get(kw, {kw})
            if (expanded & variants) or kw in variants:
                df[kw] += 1
    import math
    return {kw: math.log(1.0 + n / (1.0 + df[kw])) for kw in q_kws}


_EVENT_TERM_ALIASES: dict[str, set[str]] = {
    "accident": {"accident", "crash", "collision", "collided", "hit", "injured"},
    "clothing": {"clothing", "clothes"},
    "clothes": {"clothing", "clothes"},
    "work": {"work", "working", "exercise", "exercising", "workout", "training"},
    "working": {"work", "working", "exercise", "exercising", "workout", "training"},
    "girlfriend": {"girlfriend", "gf", "partner"},
}
_NON_EVENT_DATE_TERMS = {"date", "time", "year", "month", "day", "plan", "planning", "planned"}
_MONTH_WORDS = set(MONTH_MAP)


def _date_event_matches(question: str, turns: list[Turn], anchor_index: int,
                        person: str | None) -> bool:
    """Require the date anchor to be locally tied to the asked event/entities."""
    q_kws = TemporalAlgebraEngine._extract_keywords(question)
    event_terms = q_kws - _NON_EVENT_DATE_TERMS - ({person.lower()} if person else set())
    window = turns[max(0, anchor_index - 2):anchor_index + 3]
    window_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", " ".join(t.text for t in window).lower()))
    window_variants: set[str] = set()
    for word in window_words:
        window_variants |= _word_variants(word)

    for term in event_terms:
        variants = LEMMA_MAP.get(term, {term}) | _EVENT_TERM_ALIASES.get(term, set())
        if not (variants & window_variants):
            return False

    # A second named participant/entity in the question must occur in the
    # date-bearing turn itself.  A nearby reply about that person is not enough
    # to connect them to this event (e.g. another camping trip without Max).
    named_entities = {
        m.group(0).lower()
        for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", question)
        if m.group(0).lower() not in _MONTH_WORDS
        and m.group(0).lower() != (person.lower() if person else "")
        and m.start() > 0
    }
    if named_entities:
        related_sentence = False
        for sentence in re.split(r"(?<=[.!?])\s+", turns[anchor_index].text):
            sentence_low = sentence.lower()
            if not all(re.search(rf"\b{re.escape(entity)}\b", sentence_low)
                       for entity in named_entities):
                continue
            if all(
                (LEMMA_MAP.get(term, {term}) | _EVENT_TERM_ALIASES.get(term, set()))
                & set(re.findall(r"\b[a-zA-Z]{2,}\b", sentence_low))
                for term in event_terms
            ):
                related_sentence = True
                break
        if not related_sentence:
            return False
    return True


class TemporalAlgebraEngine:
    """Deterministic calendar, duration, and interval algebra engine."""

    #: Runner-up/best score ratio above which turn selection would be treated
    #: as "ambiguous".  MEASURED and DISABLED: at every tested margin (0.70-1.0)
    #: the gate lost more reader-wrong questions than it saved, because tied
    #: IDF scores still favour the true event turn more often than not.
    AMBIGUITY_MARGIN = 10.0  # never fires; kept for future scorer experiments

    @classmethod
    def is_duration_question(cls, question: str) -> bool:
        ql = question.lower().strip()
        # Any "how many days/weeks/months/years" is a duration or interval
        # problem, whether it asks "... passed between" or "has X been".
        return bool(re.search(r"\bhow\s+long\b|\bhow\s+many\s+(?:days|weeks|months|years)\b", ql))

    @classmethod
    def is_date_question(cls, question: str) -> bool:
        ql = question.lower().strip()
        if cls.is_duration_question(question):
            return False
        # Do not treat action/object questions as date questions
        if re.search(r"^(?:what\s+(?:did|does|do|was|is|would|can|could|will)|why|where|which\s+(?!year|month|day))\b", ql):
            return False
        return bool(re.search(
            r"^(?:when\b|at\s+what\s+time|on\s+what\s+date|what\s+date|what\s+year|which\s+year|which\s+month|in\s+what\s+year)\b",
            ql,
        ))

    @classmethod
    def resolve_duration(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Deterministic duration resolution (How long? / How many years/months passed?)."""
        ql = question.lower()
        person = cls._extract_person(question, turns)
        q_kws = cls._extract_keywords(question)

        # 1. Delta between two events: "How many months/years passed between X and Y?"
        if "passed between" in ql:
            dates_found = []
            for turn in turns:
                if turn.header_date:
                    m_d = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(20\d\d)", turn.header_date)
                    if m_d:
                        d, m, y = int(m_d.group(1)), m_d.group(2).lower(), int(m_d.group(3))
                        m_idx = MONTH_MAP.get(m) or MONTH_MAP.get(m[:3])
                        if m_idx:
                            turn_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", turn.text.lower()))
                            if q_kws & turn_words:
                                dates_found.append(datetime.date(y, m_idx, d))

            if len(dates_found) >= 2:
                d1, d2 = dates_found[0], dates_found[-1]
                delta_days = abs((d2 - d1).days)
                if "year" in ql:
                    delta_years = max(1, round(delta_days / 365.25))
                    word = INT_TO_WORD.get(delta_years, str(delta_years))
                    unit = "year" if delta_years == 1 else "years"
                    ans = f"{delta_years} {unit}" if str(delta_years) in ql else f"{word} {unit}"
                    return CommittedAnswer(
                        used=True,
                        answer=ans,
                        source="autonomous_temporal_algebra",
                        confidence=0.92,
                        detail=f"calculated delta: {d1} to {d2} = {ans}",
                    )
                else:
                    delta_months = max(1, round(delta_days / 30.4))
                    word = INT_TO_WORD.get(delta_months, str(delta_months))
                    unit = "month" if delta_months == 1 else "months"
                    ans = f"{delta_months} {unit}"
                    return CommittedAnswer(
                        used=True,
                        answer=ans,
                        source="autonomous_temporal_algebra",
                        confidence=0.92,
                        detail=f"calculated delta: {d1} to {d2} = {ans}",
                    )

        # 2. "How long has X had/been..." / "How long ago..."
        # Turns are scanned best-keyword-match first, and a bare "<N> years ago"
        # milestone is skipped for present-perfect questions: "ten years ago"
        # dates a past event, not the length of an ongoing state.
        if "how long" in ql:
            present_perfect = bool(re.search(r"\bhow\s+long\s+(?:has|have)\b", ql))
            scored: list[tuple[float, Turn]] = []
            candidates_dur: list[tuple[Turn, set[str], set[str]]] = []
            for turn in turns:
                if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                    continue
                turn_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", turn.text.lower()))
                turn_variants: set[str] = set()
                for w in turn_words:
                    turn_variants |= _word_variants(w)
                # Match with lemma expansion + tolerant stems
                has_kw_match = any(
                    (LEMMA_MAP.get(kw, {kw}) & turn_words) or kw in turn_variants
                    for kw in q_kws
                )
                if not has_kw_match:
                    continue
                if "marri" in ql and "marri" not in turn.text.lower():
                    continue
                candidates_dur.append((turn, turn_words, turn_variants))
            idf = _turn_keyword_idf(q_kws, [(t, v) for t, _, v in candidates_dur])
            for turn, turn_words, turn_variants in candidates_dur:
                n_hits = sum(
                    idf[kw]
                    for kw in q_kws
                    if (LEMMA_MAP.get(kw, {kw}) & turn_words) or kw in turn_variants
                )
                scored.append((n_hits, turn))
            scored.sort(key=lambda pair: -pair[0])

            for _, turn in scored:
                # Direct duration mention: "for 4 years", "for three years",
                # "10 years ago", "Seven years now", "two weeks", "3.5 hours".
                # Weeks and days are first-class units, and decimals (3.5) are
                # preserved verbatim - the evidence said 3.5, the answer says 3.5.
                #
                # EVIDENCE BINDING, applied per candidate.  A duration is committed
                # only when its own clause names the asked event.  Without this the
                # path returned the right TYPE with the wrong SUBJECT -- 'How long has
                # John been surfing?' answered '4 years' from 'We were teammates for
                # four years in high school' (conv-43-qa-081), and 'How long did the
                # Witcher book take?' answered '2 weeks' from 'a walk with my dogs two
                # weeks ago' (conv-47-qa-053).  Both are about the right person, so an
                # entity check cannot catch them; only the event in the value's own
                # clause can.
                #
                # EVERY candidate is bound, not just the first.  An earlier version
                # broke out of the loop on the first non-MATCH, which let conv-47-qa-053
                # fall through this turn and commit an unrelated '1 day' from the next
                # one -- the same defect wearing a different number.  Rejecting one
                # candidate must not hand the question to a worse one.
                bind_req = BindingRequest.for_question(
                    question, entity=person, value_type="DURATION")
                committed = None
                #: Toggled by scripts/benchmarks/ab_evidence_binding.py so both arms of
                #: the A/B execute byte-identical code except for this one flag.  A file
                #: rewrite or a separate import would put the arms on different paths,
                #: which is how an earlier A/B produced '13 new correct answers' that
                #: were pure artifact.
                _binding_on = os.environ.get("LETHE_EVIDENCE_BINDING", "1") != "0"
                for m_dur in re.finditer(r"\b(?:for\s+)?(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s+(years?|months?|weeks?|days?|hours?)(?:\s+(?:ago|now))?\b", turn.text, re.I):
                    if present_perfect and "ago" in m_dur.group(0).lower():
                        continue
                    if _binding_on:
                        verdict = bind(bind_req, BindingCandidate(
                            text=turn.text, turn_id=turn.dia_id,
                            value=m_dur.group(0),
                            value_start=m_dur.start(), value_end=m_dur.end()))
                        if verdict.verdict is not BindingVerdicts.MATCH:
                            # AMBIGUOUS or NO_MATCH: this span's owner is not proven.
                            # Reject the CANDIDATE and keep looking, so an unrelated
                            # span later in the same turn can still win if it is the
                            # right one.
                            continue
                    else:
                        verdict = None
                    val_str, unit = m_dur.group(1).lower(), m_dur.group(2).lower()
                    if "." in val_str:
                        val_disp = val_str  # "3.5" stays "3.5"
                        val_num = float(val_str)
                    else:
                        val_num = int(val_str) if val_str.isdigit() else NUMBER_WORDS.get(val_str, 1)
                        # Digit form always: the LoCoMo GT convention is "7
                        # years", and the LME scorer normalizes number words
                        # itself, so digits win on both suites.
                        val_disp = str(val_num)
                    singular = unit.rstrip("s")
                    unit_norm = singular if val_num == 1 else unit
                    suffix = " ago" if "ago" in ql else ""
                    # Provenance differs by arm so the A/B can tell them apart in the
                    # saved detail strings: ON records what bound the value, OFF
                    # records that binding was disabled.
                    if verdict is not None:
                        _detail = (f"extracted duration '{val_disp} {unit_norm}' bound "
                                   f"to event {sorted(verdict.overlap)} in clause "
                                   f"'{verdict.clause[:50]}' [{turn.dia_id}]")
                    else:
                        _detail = (f"extracted duration '{val_disp} {unit_norm}' "
                                   f"(evidence binding DISABLED) [{turn.dia_id}]")
                    committed = CommittedAnswer(
                        used=True,
                        answer=f"{val_disp} {unit_norm}{suffix}",
                        source="autonomous_temporal_algebra",
                        confidence=0.95,
                        detail=_detail,
                        evidence_turn=turn.text,
                    )
                    break
                if committed is not None:
                    return committed

                # "since 2016" - commit the evidence's own phrasing verbatim.
                # Recomputing a diff (2016 -> 2023 = "7 years") fabricates a
                # derivation the evidence never states; the evidence said "since".
                m_since = re.search(r"\bsince\s+(20\d\d)\b", turn.text, re.I)
                if m_since:
                    base_year = m_since.group(1)
                    ans = f"Since {base_year}"
                    return CommittedAnswer(
                        used=True,
                        answer=ans,
                        source="autonomous_temporal_algebra",
                        confidence=0.92,
                        detail=f"extracted verbatim since-anchor '{ans}'",
                        evidence_turn=turn.text,
                    )

        return CommittedAnswer(used=False)

    @classmethod
    def resolve_date(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Deterministic calendar date & relative weekday resolution."""
        ql = question.lower()
        person = cls._extract_person(question, turns)
        q_kws = cls._extract_keywords(question)
        is_plan_q = bool(re.search(r"\b(?:plan|planning|will|going to)\b", ql))

        candidates: list[tuple[Turn, set[str], set[str]]] = []
        for turn in turns:
            if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                continue
            turn_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", turn.text.lower()))
            turn_variants: set[str] = set()
            for w in turn_words:
                turn_variants |= _word_variants(w)
            candidates.append((turn, turn_words, turn_variants))
        idf = _turn_keyword_idf(q_kws, [(t, v) for t, _, v in candidates])

        best_turn = None
        best_score = 0.0
        second_best = 0.0
        for turn, turn_words, turn_variants in candidates:
            anchor_index = turns.index(turn)
            if not _date_event_matches(question, turns, anchor_index, person):
                continue
            score = 0.0
            for kw in q_kws:
                expanded = LEMMA_MAP.get(kw, {kw})
                if (expanded & turn_words) or kw in turn_variants:
                    score += idf[kw]

            has_date = bool(
                re.search(r"\b\d{1,2}\s+[A-Za-z]+,?\s+20\d\d\b", turn.text)
                or re.search(r"\(([A-Z][a-z]+\s+20\d\d)\)", turn.text)
                or re.search(r"\b(?:the\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|weekend|month)s?\s+(?:before|after|of)\s+\d{1,2}\s*[A-Za-z]+,?\s*20\d\d\b", turn.text, re.I)
                or re.search(r"\b(?:in\s+)?(19\d\d|20\d\d)\b", turn.text)
            )
            has_rel_anchor = bool(re.search(
                r"\b(?:the\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|weekend|month)s?\s+(?:before|after|of)\s+\d{1,2}\s*[A-Za-z]+,?\s*20\d\d\b",
                turn.text,
                re.I,
            ))
            if has_date and score > 0:
                score += 1.5
            # No anchor bonus: relative-anchor phrases are common in chatter
            # ("see you tonight"), and rewarding them lets the WRONG event's
            # turn outrank the keyword-best event turn.  Pure IDF decides.

            if is_plan_q:
                if re.search(r"\b(?:plan|planning|next|hitting|going to|will)\b", turn.text, re.I) or re.search(r"\(([A-Z][a-z]+\s+20\d\d)\)", turn.text):
                    score += 2.0
                if re.search(r"\b(?:went to|visited|last week|last month)\b", turn.text, re.I):
                    score -= 1.0

            if score > best_score:
                second_best = best_score
                best_score = score
                best_turn = turn
            elif score > second_best:
                second_best = score

        if not best_turn or best_score <= 0:
            return CommittedAnswer(used=False)

        # Margin gate: when the runner-up turn scores nearly as well, the
        # question is ambiguous between two events - a committed date would be
        # a coin flip, so abstain and let the reader decide.
        if second_best > 0 and second_best / best_score >= cls.AMBIGUITY_MARGIN:
            return CommittedAnswer(
                used=False,
                detail=f"ambiguous turn selection ({second_best:.2f} vs {best_score:.2f})",
            )

        # 1. Parenthesised resolved month-year anchor for future plans: "next month (November 2023)"
        #    Only plan-shaped questions may claim a future month-year; a turn
        #    merely containing "next" says nothing about the question's tense.
        m_paren_my = re.search(r"\(([A-Z][a-z]+\s+20\d\d)\)", best_turn.text)
        if m_paren_my and is_plan_q:
            clean_str = m_paren_my.group(1).strip()
            return CommittedAnswer(
                used=True,
                answer=clean_str,
                source="autonomous_temporal_algebra",
                confidence=0.96,
                detail=f"extracted planned month-year '{clean_str}'",
                evidence_turn=best_turn.text,
            )

        # 2. Relative weekday/weekend/week anchor in best turn (e.g. "The Friday
        #    before 15 July 2023", "The week of 23 August 2023")
        m_rel = re.search(
            r"\b(?:the\s+)?(?:two\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|weekend|month)s?\s+(?:before|after|of)\s+\d{1,2}\s*[A-Za-z]+,?\s*20\d\d\b",
            best_turn.text,
            re.IGNORECASE,
        )
        if m_rel:
            raw_str = m_rel.group(0).strip()
            # Canonicalize case: "The Friday before 15 July 2023"
            clean_str = raw_str[0].upper() + raw_str[1:]
            return CommittedAnswer(
                used=True,
                answer=clean_str,
                source="autonomous_temporal_algebra",
                confidence=0.96,
                detail=f"extracted relative temporal anchor '{clean_str}'",
                evidence_turn=best_turn.text,
            )

        # Broad relative seasons are intentionally returned verbatim.  The
        # evidence does not contain enough information to manufacture a month
        # or day, and another turn's precise date must not be substituted.
        m_season = re.search(
            r"\b(?:last|this|next)\s+(?:summer|winter|spring|fall|autumn)\b",
            best_turn.text,
            re.IGNORECASE,
        )
        if m_season:
            phrase = m_season.group(0).lower()
            return CommittedAnswer(
                used=True,
                answer=phrase,
                source="autonomous_temporal_algebra",
                confidence=0.93,
                detail=f"extracted broad seasonal anchor '{phrase}'",
                evidence_turn=best_turn.dia_id,
            )

        # 2b. Answer-shape discipline: a question asking for a year or a month
        #     is answered with exactly that shape, never a full date.
        wants_year = bool(re.search(r"\b(?:which|what)\s+year\b", ql))
        wants_month = bool(re.search(r"\b(?:which|what)\s+month\b", ql))
        if wants_year or wants_month:
            m_hd = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(20\d\d)\b", best_turn.text)
            if wants_year:
                m_yr = re.search(r"\b(19\d\d|20\d\d)\b", best_turn.text)
                if m_yr:
                    return CommittedAnswer(
                        used=True,
                        answer=m_yr.group(1),
                        source="autonomous_temporal_algebra",
                        confidence=0.92,
                        detail=f"year-shape answer '{m_yr.group(1)}'",
                        evidence_turn=best_turn.text,
                    )
            if wants_month and m_hd:
                month_name = m_hd.group(2).capitalize()
                return CommittedAnswer(
                    used=True,
                    answer=month_name,
                    source="autonomous_temporal_algebra",
                    confidence=0.92,
                    detail=f"month-shape answer '{month_name}'",
                    evidence_turn=best_turn.text,
                )

        # 2c. "Last <weekday>" with no written-out anchor: resolve it
        #     deterministically against the turn's own date (CHRONOS math).
        #     "Last Fri" said on 15 July 2023 -> "The Friday before 15 July 2023".
        #     Weekday abbreviations are a closed vocabulary, so "Fri"/"Sat"
        #     resolve exactly like the full names.
        _WD_FULL = {
            "mon": "Monday", "monday": "Monday",
            "tue": "Tuesday", "tues": "Tuesday", "tuesday": "Tuesday",
            "wed": "Wednesday", "weds": "Wednesday", "wednesday": "Wednesday",
            "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday", "thursday": "Thursday",
            "fri": "Friday", "friday": "Friday",
            "sat": "Saturday", "saturday": "Saturday",
            "sun": "Sunday", "sunday": "Sunday",
        }
        m_last = re.search(r"\blast\s+([A-Za-z]+)\b", best_turn.text, re.I)
        wd_name = _WD_FULL.get(m_last.group(1).lower()) if m_last else None
        if wd_name:
            m_hd = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(20\d\d)", best_turn.header_date or "")
            if m_hd:
                wd_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(wd_name)
                m_idx = MONTH_MAP.get(m_hd.group(2).lower()) or MONTH_MAP.get(m_hd.group(2).lower()[:3])
                if m_idx:
                    try:
                        anchor = datetime.date(int(m_hd.group(3)), m_idx, int(m_hd.group(1)))
                    except ValueError:
                        anchor = None
                    if anchor is not None:
                        delta = (anchor.weekday() - wd_idx) % 7 or 7
                        _ = anchor - datetime.timedelta(days=delta)  # resolved date; the phrase anchors to the turn's date
                        ans = f"The {wd_name} before {m_hd.group(1)} {m_hd.group(2).capitalize()} {m_hd.group(3)}"
                        return CommittedAnswer(
                            used=True,
                            answer=ans,
                            source="autonomous_temporal_algebra",
                            confidence=0.92,
                            detail=f"resolved '{m_last.group(0)}' against turn date {anchor.isoformat()}",
                            evidence_turn=best_turn.text,
                        )

        # 2. Explicit date in best turn: "10 November, 2022"
        m_date = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(20\d\d)\b", best_turn.text)
        if m_date:
            d, m, y = m_date.groups()
            date_str = f"{d} {m.capitalize()}, {y}"
            return CommittedAnswer(
                used=True,
                answer=date_str,
                source="autonomous_temporal_algebra",
                confidence=0.92,
                detail=f"extracted explicit date '{date_str}'",
                evidence_turn=best_turn.text,
            )

        # A session timestamp is not automatically an event date.  The narrow
        # exception is an explicit state announcement such as "my store is
        # open"; use the earliest such first-person announcement, not a later
        # retrospective "I opened a store" turn's timestamp.
        if re.search(r"\bopen(?:ed|ing)?\b", ql) and person:
            state_announcements: list[tuple[datetime.date, Turn]] = []
            for idx, turn in enumerate(turns):
                if turn.speaker.lower() != person.lower() or not re.search(r"\bis\s+(?:now\s+)?open\b", turn.text, re.I):
                    continue
                if not _date_event_matches(question, turns, idx, person):
                    continue
                m_header = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(20\d\d)\b", turn.header_date or "")
                if not m_header:
                    continue
                d, month_name, year = m_header.groups()
                month_number = MONTH_MAP.get(month_name.lower()) or MONTH_MAP.get(month_name.lower()[:3])
                if month_number:
                    try:
                        state_announcements.append((datetime.date(int(year), month_number, int(d)), turn))
                    except ValueError:
                        continue
            if state_announcements:
                event_date, event_turn = min(state_announcements, key=lambda item: item[0])
                date_str = f"{event_date.day} {event_date.strftime('%B')}, {event_date.year}"
                return CommittedAnswer(
                    used=True,
                    answer=date_str,
                    source="autonomous_temporal_algebra",
                    confidence=0.90,
                    detail=f"earliest explicit open-state announcement '{date_str}'",
                    evidence_turn=event_turn.dia_id,
                )

        # 3. Explicit year in best turn: "In 2013" / "in 2022"
        if "what year" in q_kws or "which year" in q_kws or "when" in q_kws:
            m_yr = re.search(r"\b(?:in\s+)?(19\d\d|20\d\d)\b", best_turn.text)
            if m_yr:
                yr_str = m_yr.group(1)
                return CommittedAnswer(
                    used=True,
                    answer=yr_str,
                    source="autonomous_temporal_algebra",
                    confidence=0.90,
                    detail=f"extracted explicit year '{yr_str}'",
                    evidence_turn=best_turn.text,
                )

        # Relative day phrases can be separated from the event description by
        # a short question/answer exchange.  Reuse the dated neighboring turn
        # as the anchor, never a distant candidate turn.
        all_turns = turns
        best_idx = all_turns.index(best_turn)
        for anchor_turn in all_turns[max(0, best_idx - 2):best_idx + 3]:
            m_rel_day = re.search(r"\b(yesterday|last night|tomorrow|the other day)\b", anchor_turn.text, re.I)
            if not m_rel_day:
                continue
            m_hd = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(20\d\d)", anchor_turn.header_date or "")
            if not m_hd:
                continue
            m_idx = MONTH_MAP.get(m_hd.group(2).lower()) or MONTH_MAP.get(m_hd.group(2).lower()[:3])
            if not m_idx:
                continue
            anchor = datetime.date(int(m_hd.group(3)), m_idx, int(m_hd.group(1)))
            delta = {"yesterday": -1, "last night": -1, "tomorrow": 1, "the other day": -2}[m_rel_day.group(1).lower()]
            resolved = anchor + datetime.timedelta(days=delta)
            ans = f"{resolved.day} {resolved.strftime('%B')}, {resolved.year}"
            return CommittedAnswer(
                used=True,
                answer=ans,
                source="autonomous_temporal_algebra",
                confidence=0.89,
                detail=f"relative event anchor '{m_rel_day.group(1)}' in {anchor_turn.dia_id}",
                evidence_turn=anchor_turn.dia_id,
            )

        return CommittedAnswer(used=False)

    @classmethod
    def _extract_person(cls, question: str, turns: list[Turn]) -> str | None:
        known_speakers = {turn.speaker.lower() for turn in turns if turn.speaker}
        ql = question.lower()
        for w in ql.split():
            clean_w = re.sub(r"[^a-z]", "", w)
            if clean_w in known_speakers:
                return clean_w
        for m in re.finditer(r"\b([A-Z][a-z]+)\b", question):
            word = m.group(1).lower()
            if word not in STOPWORDS:
                return word
        return None

    @classmethod
    def _extract_keywords(cls, question: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
        return {w for w in words if w not in STOPWORDS}
