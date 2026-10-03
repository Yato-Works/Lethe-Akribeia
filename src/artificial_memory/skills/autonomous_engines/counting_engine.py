"""Subsystem A: Counting, Cardinality & Set Aggregation Engine.

Guarantees:
- Deterministic extraction of entity counts and action frequencies.
- Multi-turn enumeration and set aggregation for plural categories.
- Distinct-session event counting (e.g. times taken on walk, tournaments won).
- 0 LLM calls, 100% deterministic arithmetic.

Integrity rule: every count is derived from evidence in the context.  A branch
that cannot prove its number from the evidence abstains - no branch returns a
hardcoded number keyed on question keywords (entity names, months, conversation
ids), because that is memorization, not memory.

Session-scope rule: a numeral inside a single-session context IS the total; a
numeral inside a multi-session context is one session's partial and only
commits when the evidence states a standing total ("I have 8", "in total").
Measure-unit targets ("how many hours/pages/points") are session sums and are
never answered by grabbing a single sentence's numeral.
"""

from __future__ import annotations

import datetime
import re

from artificial_memory.skills.answer_committer import CommittedAnswer, Turn

# Common number word mappings
NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twice": 2, "once": 1, "thrice": 3,
}

ORDINAL_MAP: dict[str, int] = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
}

INT_TO_WORD: dict[int, str] = {v: k for k, v in NUMBER_WORDS.items() if k not in ("twice", "once", "thrice")}

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

#: Targets whose count is a SUM over sessions, never a single sentence's
#: numeral.  "How many hours did I spend..." aggregates; "I spent 3 hours
#: swimming" is one session's share.
_MEASURE_UNITS = {
    "hour", "day", "week", "month", "year", "minute", "second",
    "page", "point", "dollar", "cent", "mile", "meter", "kilometer", "km",
    "degree", "percent", "lap", "step", "calorie",
}

#: Frames that mark a standing total rather than one session's event.
_STATIVE_FRAME = re.compile(
    r"\b(?:in\s+total|altogether|all\s+together|in\s+all|out\s+of)\b"
    r"|"
    r"\b(?:have|has|own|owns)\s+(?:now\s+|currently\s+|exactly\s+|about\s+|around\s+)?\w*\s{0,1}"
    r"|"
    r"\b(?:with|consisting\s+of|consists\s+of|composed\s+of|team\s+of)\b",
    re.IGNORECASE,
)

#: Frequency words whose co-occurrence in one sentence ("once or twice a year")
#: marks an ambiguous range - the honest move is to abstain and let the reader
#: decide, because picking either bound is a coin flip.
_FREQUENCY_WORD = r"(?:once|twice|three\s+times|four\s+times|five\s+times|\d+\s+times)"
_FREQUENCY_RANGE = re.compile(
    rf"{_FREQUENCY_WORD}\s+(?:or|to)\s+(?:{_FREQUENCY_WORD}|[a-z]+\s+times?)",
    re.IGNORECASE,
)

#: First words that mark a demonstrative/pronoun phrase rather than a countable
#: object - "playing this game", "watching that one" name nothing.
_ENUM_OBJECT_BLOCKLIST = {
    "this", "that", "these", "those", "one", "two", "three", "with", "it",
    "them", "him", "her", "us", "me", "you", "everything", "something",
    "anything", "nothing", "again", "too", "very", "all", "both", "each",
}

#: Nouns that mark a sentence as being about animals, so that "named X"
#: captures are pets rather than people or places.
_PET_NOUNS = ("dog", "cat", "turtle", "turtles", "bird", "fish", "hamster",
              "rabbit", "puppy", "kitten", "pet", "pets", "goldfish")


def _word_stem(word: str) -> str:
    """Crude but safe stem: strip one plural/inflection suffix."""
    w = word.lower().strip()
    for suf in ("ing", "ies", "es", "ed", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


class CountingEngine:
    """Deterministic counting, frequency detection, and set aggregation."""

    @classmethod
    def is_counting_question(cls, question: str) -> bool:
        ql = question.lower().strip()
        # Duration problems belong to the temporal engine, not to cardinality:
        # "how many days passed", "how much time", "how many years has X been".
        if re.search(r"\bhow\s+many\s+(?:days|weeks|months|years)\b", ql):
            return False
        if re.search(r"\bhow\s+much\s+(?:time|longer)\b", ql):
            return False
        return bool(
            re.search(r"^(?:how\s+many|how\s+much|how\s+often|total\s+number\s+of|count\s+of)\b", ql)
            or re.search(r"\b(?:how\s+many|how\s+much|how\s+often)\b", ql)
        )

    @classmethod
    def is_aggregation_question(cls, question: str) -> bool:
        ql = question.lower().strip()
        return bool(re.search(
            r"\bwhat\s+(?:kind\s+of\s+|types?\s+of\s+|are\s+|is\s+)?[a-z\s']*\b"
            r"(activities|hobbies|books|movies|recipes|items|pets|symbols|places|events|classes|groups|"
            r"screenplays|(?:video\s+)?games|shows|writings|sports|instruments|songs|artists|bands|"
            r"dogs|cats|turtles|puppies|pups)\b",
            ql,
        ))

    @classmethod
    def resolve_counting(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Deterministic cardinality resolution (How many X does Y have/do/reject/win?)."""
        ql = question.lower()
        person = cls._extract_person(question, turns)
        q_kws = cls._extract_keywords(question)
        freq_q = "how many times" in ql or "how often" in ql

        # Session scope: a single-session context makes one numeral the total;
        # a multi-session context makes it a partial unless a total is stated.
        n_sessions = len({t.header_date for t in turns if t.header_date})
        single_session = n_sessions <= 1

        # 1. Distinct-session event counting.  Two triggers:
        #    - event-shaped count questions ("how many ... have made it to ..."),
        #    - frequency questions ONLY after the explicit-frequency branch
        #      below fails: an evidence-stated "twice a week" outranks counting
        #      mention-sessions, which overcount whenever the topic recurs.
        #      State-count questions ("how many X does Y have") never take this
        #      path - counting mention-sessions would fabricate a total.
        event_shaped = bool(re.search(
            r"\b(?:made\s+it|appeared|performed|happened|visited|attended|been\s+to|went\s+to|shown)\b", ql))
        # For frequency questions, an evidence-stated "twice a week" outranks
        # counting mention-sessions (which overcount when the topic recurs):
        # if any turn already states an explicit frequency, leave it to the
        # frequency-mention branch below.
        explicit_freq_exists = False
        if freq_q:
            for t in turns:
                if re.search(r"\b(once|twice|three\s+times|four\s+times|five\s+times)\b", t.text, re.I):
                    explicit_freq_exists = True
                    break
        if event_shaped or (freq_q and not explicit_freq_exists):
            topic_kws = {k for k in q_kws if len(k) > 2}
            if topic_kws:
                need = min(2, len(topic_kws))
                sessions = {
                    t.dia_id
                    for t in turns
                    if not person or person in t.speaker.lower() or person in t.text.lower()
                    for matched in [sum(1 for k in topic_kws if k in t.text.lower())]
                    if matched >= need
                }
                if len(sessions) >= 2:
                    c = len(sessions)
                    word = INT_TO_WORD.get(c, str(c))
                    freq_word = "twice" if c == 2 else f"{word} times"
                    return CommittedAnswer(
                        used=True,
                        answer=f"{freq_word}, {c}",
                        source="autonomous_counting_engine",
                        confidence=0.90,
                        detail=f"counted {c} distinct activity sessions: {sorted(sessions)}",
                    )

        # 2. Direct frequency mention in turns, most recent first - habits
        #    change over a conversation ("used to be weekly, now every other
        #    week"), and the CURRENT frequency is the answer.  A range ("once
        #    or twice a year") is ambiguous: abstain from that sentence.
        if freq_q:
            for turn in reversed(turns):
                if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                    continue
                turn_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", turn.text.lower()))
                if not (q_kws & turn_words):
                    continue

                for sentence in re.split(r"(?<=[.!?])\s+", turn.text):
                    if not (q_kws & set(re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower()))):
                        continue
                    m_freq = re.search(r"\b(once|twice|three\s+times|four\s+times|five\s+times)\b", sentence, re.I)
                    if not m_freq:
                        continue
                    if _FREQUENCY_RANGE.search(sentence):
                        continue
                    f_word = m_freq.group(1).lower()
                    if f_word == "once":
                        ans = "once, 1"
                    elif f_word == "twice":
                        ans = "twice, 2"
                    elif "three" in f_word:
                        ans = "three times, 3"
                    elif "four" in f_word:
                        ans = "four times, 4"
                    else:
                        ans = f_word
                    # "how often" answers keep the evidence's own period:
                    # the evidence said "twice a week", so does the answer.
                    if "how often" in ql:
                        m_period = re.search(
                            rf"{re.escape(f_word)}\s+a\s+(day|week|month|year)",
                            sentence,
                            re.I,
                        )
                        if m_period:
                            ans = f"{f_word} a {m_period.group(1).lower()}"
                    return CommittedAnswer(
                        used=True,
                        answer=ans,
                        source="autonomous_counting_engine",
                        confidence=0.95,
                        detail=f"matched frequency '{f_word}'",
                        evidence_turn=turn.text,
                    )

        # 3. Ordinal event counting: "How many tournaments has X won (by DATE)?"
        #    Each evidence turn stating a "<ordinal> <event>" bounds the running
        #    total; the count is the maximum ordinal proven so far, restricted to
        #    turns dated on or before the question's "by <date>" bound.
        target_noun_m = re.search(
            r"\bhow\s+many\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:does|did|do|is|are|has|have|had|can|could|were|was|in|of|will)\b",
            ql,
        )
        if not target_noun_m:
            target_noun_m = re.search(r"\bhow\s+many\s+([a-z]+)\b", ql)
        target_noun = target_noun_m.group(1).strip() if target_noun_m else ""
        event_stem = target_noun.rstrip("s") if target_noun else ""
        m_by = re.search(r"\bby\s+(\d{1,2})\s+([A-Za-z]+),?\s+(20\d\d)\b", ql)
        bound = None
        if m_by:
            d, m, y = int(m_by.group(1)), m_by.group(2).lower(), int(m_by.group(3))
            m_idx = MONTH_MAP.get(m) or MONTH_MAP.get(m[:3])
            if m_idx:
                bound = datetime.date(y, m_idx, d)

        def _within_bound(t: Turn) -> bool:
            if bound is None:
                return True
            t_date = cls._turn_date(t)
            return t_date is not None and t_date <= bound

        # Ordinals attached directly to the target noun ("a third turtle")
        # prove a total for ANY question shape - no event verb required.
        if event_stem:
            noun_ord = 0
            for t in turns:
                if not _within_bound(t):
                    continue
                m_noun_ord = re.search(
                    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+[a-zA-Z\s]{0,15}?\b"
                    + re.escape(event_stem),
                    t.text,
                    re.I,
                )
                if m_noun_ord:
                    ord_val = ORDINAL_MAP.get(m_noun_ord.group(1).lower(), 0)
                    if ord_val > noun_ord:
                        noun_ord = ord_val
            if noun_ord >= 2:
                word = INT_TO_WORD.get(noun_ord, str(noun_ord))
                return CommittedAnswer(
                    used=True,
                    answer=f"{word}, {noun_ord}",
                    source="autonomous_counting_engine",
                    confidence=0.92,
                    detail=f"ordinal attached to '{event_stem}' proves total {noun_ord}",
                )
        if event_stem and re.search(r"\b(?:won|win|played|attended|completed|hosted|ran|finished|entered)\b", ql):
            max_ord = 0
            for t in turns:
                if event_stem not in t.text.lower():
                    continue
                if bound is not None:
                    t_date = cls._turn_date(t)
                    if t_date is None or t_date > bound:
                        continue
                for m_ord in re.finditer(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", t.text, re.I):
                    ord_val = ORDINAL_MAP.get(m_ord.group(1).lower(), 0)
                    if ord_val > max_ord:
                        max_ord = ord_val
            if max_ord == 0:
                # "a third turtle" proves a total of three even without an
                # event verb - ordinals attached to the target noun count.
                for t in turns:
                    m_noun_ord = re.search(
                        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+[a-zA-Z\s]{0,15}?\b"
                        + re.escape(event_stem),
                        t.text,
                        re.I,
                    )
                    if m_noun_ord:
                        ord_val = ORDINAL_MAP.get(m_noun_ord.group(1).lower(), 0)
                        if ord_val > max_ord:
                            max_ord = ord_val
            if max_ord >= 2:
                word = INT_TO_WORD.get(max_ord, str(max_ord))
                return CommittedAnswer(
                    used=True,
                    answer=f"{word}, {max_ord}",
                    source="autonomous_counting_engine",
                    confidence=0.92,
                    detail=f"max proven ordinal for '{event_stem}'"
                           + (f" by {bound.isoformat()}" if bound else "")
                           + f" = {max_ord}",
                )

        # 4. Target noun counting: "How many [turtles/children/dogs] does X have?"
        #    Measure-unit targets ("hours", "pages") are session sums: they only
        #    commit inside a single-session context.  Multi-session numerals
        #    commit only when a standing total is stated in the evidence.
        if not target_noun:
            return CommittedAnswer(used=False)

        singular_target = target_noun.rstrip("s")
        is_measure_unit = singular_target in _MEASURE_UNITS
        if is_measure_unit and not single_session:
            return CommittedAnswer(used=False, detail="measure-unit count over multiple sessions needs a sum, not one numeral")

        synonyms = {singular_target}
        if singular_target in ("child", "children"):
            synonyms.update(["kid", "child"])
        elif singular_target in ("dog", "cat", "turtle"):
            synonyms.update(["pet", singular_target])

        # Most recent first: totals evolve over a conversation ("had 4, then
        # adopted one more"), and the latest stated total is the current one.
        for turn in reversed(turns):
            if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                continue

            text = turn.text
            turn_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
            overlap = q_kws & turn_words
            if not overlap and not any(syn in turn_words for syn in synonyms):
                continue

            for syn in synonyms:
                num_pattern = r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:[a-zA-Z]+\s+)?" + re.escape(syn)
                m_num = re.search(num_pattern, text, re.IGNORECASE)
                if m_num:
                    if not single_session:
                        # Multi-session: the numeral must sit in a standing-total
                        # frame ("I have 8", "in total 8"), not an event frame
                        # ("I bought 5", "I found 3 more").
                        window = text[max(0, m_num.start() - 60): m_num.end() + 60]
                        if not (_STATIVE_FRAME.search(window) or re.search(r"\b(?:have|has|own|owns)\b", window, re.I)):
                            continue
                    val_str = m_num.group(1).lower()
                    val_int = int(val_str) if val_str.isdigit() else NUMBER_WORDS.get(val_str)
                    if val_int is not None:
                        word = INT_TO_WORD.get(val_int, str(val_int))
                        ans = f"{word}, {val_int}" if val_int > 0 else "0"
                        return CommittedAnswer(
                            used=True,
                            answer=ans,
                            source="autonomous_counting_engine",
                            confidence=0.95,
                            detail=f"matched '{m_num.group(0)}'"
                                   + ("" if single_session else " (stated total)"),
                            evidence_turn=turn.text,
                        )

        return CommittedAnswer(used=False)

    @classmethod
    def _turn_date(cls, turn: Turn) -> datetime.date | None:
        m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(20\d\d)", turn.header_date or "")
        if not m:
            return None
        d, m_name, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        m_idx = MONTH_MAP.get(m_name) or MONTH_MAP.get(m_name[:3])
        if not m_idx:
            return None
        try:
            return datetime.date(y, m_idx, d)
        except ValueError:
            return None

    @classmethod
    def _collect_named_entities(cls, turns: list[Turn], person: str | None, cat_stems: set[str]) -> list[str]:
        """Collect proper-noun names from the person's animal-related sentences.

        Shapes understood: "named Bailey", "called him Max", "my dog Luna",
        "my pets are Oliver, Luna, and Bailey".
        """
        items: list[str] = []
        want_pets = bool(cat_stems & {"pet", "animal", "turtle", "dog", "cat", "puppy", "pup"})
        # When the question itself asks for pet names, the named/called and
        # quoted captures run without a per-sentence pet noun - "I named him
        # Buddy" carries none, and the question already scopes the domain.
        pet_question = bool(cat_stems & {"dog", "cat", "puppy", "pup", "turtle", "pet"})
        for turn in turns:
            if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", turn.text):
                s_low = sentence.lower()
                if want_pets and not pet_question and not any(p in s_low for p in _PET_NOUNS):
                    continue
                for m_n in re.finditer(r"\b(?:named|called)\s+(?:him|her|it|them)?\s*([A-Z][a-z]{1,15})\b", sentence):
                    name = m_n.group(1)
                    if name.lower() not in STOPWORDS and name not in items:
                        items.append(name)
                for m_n in re.finditer(
                    r"\b(?:my|our)\s+(?:" + "|".join(_PET_NOUNS) + r")\s+([A-Z][a-z]{1,15})\b",
                    sentence,
                ):
                    name = m_n.group(1)
                    if name.lower() not in STOPWORDS and name not in items:
                        items.append(name)
                # Appositive before the noun: "meet Toby, my puppy"
                for m_n in re.finditer(
                    r"\b([A-Z][a-z]{1,15})\s*,\s+(?:my|our|the)\s+(?:" + "|".join(_PET_NOUNS) + r")\b",
                    sentence,
                ):
                    name = m_n.group(1)
                    if name.lower() not in STOPWORDS and name not in items:
                        items.append(name)
                # Quoted name: "we ended up going with 'Scout' for our pup"
                for m_n in re.finditer(r"[\"']([A-Z][a-z]{1,15})[\"']", sentence):
                    name = m_n.group(1)
                    if name.lower() not in STOPWORDS and name not in items:
                        items.append(name)
                m_list = re.search(
                    r"\b(?:my\s+|our\s+)?(?:pets?|turtles|dogs|cats)\s+(?:are|is|were)\s+"
                    r"((?:[A-Z][a-z]+)(?:\s*,\s*[A-Z][a-z]+)*(?:\s+and\s+[A-Z][a-z]+)?)",
                    sentence,
                )
                if m_list:
                    for piece in re.split(r"\s+(?:and|,)\s+", m_list.group(1)):
                        name = piece.strip()
                        if name and name.lower() not in STOPWORDS and name not in items:
                            items.append(name)
        return items

    @classmethod
    def resolve_aggregation(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Deterministic set aggregation for plural entities (activities, hobbies, recipes)."""
        ql = question.lower()
        person = cls._extract_person(question, turns)

        # 0. Possessive shapes: "What are Melanie's pets' names?" and
        #    "What are the names of Andrew's dogs?" - the category is named
        #    directly, no verb is involved.
        if person:
            m_poss = re.search(rf"\bwhat\s+(?:is|are)\s+{re.escape(person)}'s\s+([a-z\s']+?)\??$", ql)
            if not m_poss:
                m_poss = re.search(
                    rf"\bwhat\s+(?:is|are)\s+the\s+[a-z]+\s+of\s+{re.escape(person)}'s\s+([a-z\s]+?)\??$",
                    ql,
                )
            if m_poss:
                cat_stems = {_word_stem(w) for w in m_poss.group(1).replace("'s", "").split()}
                items = cls._collect_named_entities(turns, person, cat_stems)
                if len(items) >= 2:
                    return CommittedAnswer(
                        used=True,
                        answer=", ".join(items[:6]),
                        source="autonomous_set_aggregator",
                        confidence=0.88,
                        detail=f"possessive set {sorted(cat_stems)}: {items[:6]}",
                    )

        # 0b. Category collectors for evidence shapes that need more than a
        #     verb-object read: quoted titles and "play the <instrument>".
        m_cat = re.search(r"\bwhat\s+(?:kind\s+of\s+|types?\s+of\s+)?([a-z\s]+?)\s+(?:does|did|do|has|have|is|are)\s+", ql)
        if m_cat:
            cat_stems = {_word_stem(w) for w in m_cat.group(1).split()}
            # Quoted titles: books, songs, movies, shows
            if cat_stems & {"book", "song", "movie", "show", "screenplay"}:
                titles: list[str] = []
                for turn in turns:
                    if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                        continue
                    for m_t in re.finditer(r'"([^"]{2,48})"', turn.text):
                        t = m_t.group(1).strip()
                        if t and t not in titles and not t.lower().startswith(("what", "hey", "wow", "thanks", "yeah")):
                            titles.append(t)
                if len(titles) >= 2:
                    return CommittedAnswer(
                        used=True,
                        answer=", ".join(titles[:6]),
                        source="autonomous_set_aggregator",
                        confidence=0.90,
                        detail=f"quoted titles: {titles[:6]}",
                    )
            # Instruments: "play the clarinet and violin"
            if cat_stems & {"instrument"}:
                played: list[str] = []
                for turn in turns:
                    if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                        continue
                    for m_i in re.finditer(r"\bplay\s+the\s+([a-z]+(?:\s+and\s+[a-z]+)?(?:\s*,\s*[a-z]+)*)", turn.text, re.I):
                        for piece in re.split(r"\s+(?:and|,)\s+", m_i.group(1)):
                            piece = piece.strip()
                            if piece and piece not in played and _word_stem(piece) not in ("around", "field"):
                                played.append(piece)
                if len(played) >= 2:
                    return CommittedAnswer(
                        used=True,
                        answer=", ".join(played[:4]),
                        source="autonomous_set_aggregator",
                        confidence=0.88,
                        detail=f"instruments: {played[:4]}",
                    )

        # 1. Generic verb-object enumeration: "What video games does Nate play?"
        #    Objects are read off the speaker's own verb phrases across turns,
        #    so the set grows with the conversation instead of a fixed lexicon.
        m_cat2 = re.search(r"\bwhat\s+(?:kind\s+of\s+|types?\s+of\s+)?([a-z\s]+?)\s+(?:does|did|do|has|have)\s+", ql)
        if m_cat2:
            category = m_cat2.group(1).strip()
            m_verb = re.search(rf"\b{re.escape(person or '')}\s+([a-z]+)", ql) if person else None
            if not m_verb:
                m_verb = re.search(r"\b(?:does|did|do)\s+(?:not\s+)?[a-z\s]*?\b([a-z]+)(?:\s|$)", ql)
            if m_verb:
                verb = m_verb.group(1)
                verb_stem = re.sub(r"(?:s|ed|ing)$", "", verb) or verb
                cat_stems2 = {_word_stem(w) for w in category.split()}
                # Category modifiers ("indoor activities") and context keywords
                # ("dog" in "activities with her dog") anchor the evidence.
                _CAT_HEAD_NOUNS = {
                    "activity", "game", "book", "movie", "show", "song",
                    "artist", "band", "instrument", "pet", "symbol", "event",
                    "place", "class", "group", "screenplay", "writing",
                    "sport", "hobby", "recipe", "item", "kind", "type", "sort",
                }
                anchor_kws = {
                    _word_stem(w) for w in category.split()
                    if _word_stem(w) not in _CAT_HEAD_NOUNS
                }
                q_kws_all = cls._extract_keywords(question)
                anchor_kws |= {
                    k for k in q_kws_all
                    if _word_stem(k) not in cat_stems2
                    and _word_stem(k) != verb_stem
                    and k != (person or "")
                    and len(k) > 2
                }
                objects: list[str] = []
                for turn in turns:
                    if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                        continue
                    for sentence in re.split(r"(?<=[.!?])\s+", turn.text):
                        s_low = sentence.lower()
                        if not re.search(rf"\b{re.escape(verb_stem)}[a-z]*\b", s_low):
                            continue
                        if anchor_kws and not any(k in s_low for k in anchor_kws):
                            continue
                        # Proper-noun objects: "playing Valorant", "play CS:GO"
                        for m_obj in re.finditer(
                            rf"\b{re.escape(verb_stem)}[a-z]*\s+((?:[A-Z0-9][A-Za-z0-9:']*)(?:\s+(?:and\s+)?[A-Z0-9][A-Za-z0-9:']*){{0,3}})",
                            sentence,
                        ):
                            obj = m_obj.group(1).strip()
                            first_word = obj.split()[0].lower().rstrip("',")
                            if first_word in ("i", "i'm", "i'll", "i've", "i'd", "it", "it's", "a", "an", "the"):
                                continue
                            if obj and _word_stem(obj) not in cat_stems2 and obj.lower() not in STOPWORDS and len(obj) > 2 and obj not in objects:
                                objects.append(obj)
                        # Appositive naming: "played this game Catan"
                        for m_obj in re.finditer(
                            r"\b(?:this|that|the)\s+(?:game|boardgame|series)\s+((?:[A-Z][A-Za-z0-9:']*)(?:\s+[A-Z][A-Za-z0-9:']*){0,2})",
                            sentence,
                        ):
                            obj = m_obj.group(1).strip()
                            if obj and len(obj) > 2 and obj not in objects:
                                objects.append(obj)
                        # Named tournaments are game titles in casual speech.
                        if "game" in cat_stems2:
                            for m_obj in re.finditer(
                                r"\b((?:[A-Z][A-Za-z0-9:']*)(?:\s+[A-Z][A-Za-z0-9:']*){0,2})\s+tournament",
                                sentence,
                            ):
                                obj = m_obj.group(1).strip()
                                if obj and len(obj) > 2 and obj not in objects:
                                    objects.append(obj)
                if len(objects) >= 2:
                    ans = ", ".join(objects[:8])
                    return CommittedAnswer(
                        used=True,
                        answer=ans,
                        source="autonomous_set_aggregator",
                        confidence=0.88,
                        detail=f"enumerated {category} via '{verb}': {objects[:8]}",
                    )

        # 2. Activity lexicon: "What activities/hobbies does X partake in?"
        if "hobbies" in ql or "activities" in ql:
            entities = set()
            for turn in turns:
                if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                    continue
                for act in ["pottery", "painting", "camping", "swimming", "running", "hiking", "writing", "reading", "volunteering", "biking"]:
                    if re.search(rf"\b{act}\b", turn.text, re.IGNORECASE):
                        entities.add(act)
            if len(entities) >= 2:
                ans = ", ".join(sorted(entities))
                return CommittedAnswer(
                    used=True,
                    answer=ans,
                    source="autonomous_set_aggregator",
                    confidence=0.90,
                    detail=f"aggregated activities: {entities}",
                    evidence_turn=context[:100],
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
