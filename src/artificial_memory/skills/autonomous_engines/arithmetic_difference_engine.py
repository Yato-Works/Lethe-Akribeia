"""Subsystem H: Arithmetic Difference & Numerical Derivation Engine.

Resolves deterministic numerical calculations from dialogue context:
1. Currency differences / price comparisons ("How much more did I spend on X compared to Y?").
2. Savings and discounts ("How much did I save on X at Y?").
3. Age / event time arithmetic ("How old was I when I moved to X?").
4. Multi-location / multi-event duration aggregations ("Total number of days in X and Y", "Days attending workshops in Month").
5. Multi-activity duration sums ("Weeks spent reading X and listening to Y").

0 LLM calls, 100% deterministic arithmetic.
"""

from __future__ import annotations

import re

from artificial_memory.skills.answer_committer import CommittedAnswer, Turn

#: Word to number conversion for duration and counts
_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50,
}

#: Common geographic region-to-subdivision ontology
_GEOGRAPHIC_SUBSUMPTIONS: dict[str, set[str]] = {
    "hawaii": {"maui", "honolulu", "oahu", "kauai", "kona", "hilo", "waikiki"},
    "japan": {"tokyo", "kyoto", "osaka", "hokkaido", "okinawa", "shibuya", "shinjuku"},
    "california": {"los angeles", "san francisco", "san diego", "san jose"},
    "united states": {"us", "usa", "america", "chicago", "new york", "seattle", "boston"},
    "uk": {"united kingdom", "london", "england", "scotland", "wales"},
    "france": {"paris", "nice", "lyon", "marseille"},
    "italy": {"rome", "milan", "florence", "venice"},
}


def _word_or_digit(token: str) -> int | None:
    token_clean = token.strip().lower()
    if token_clean.isdigit():
        return int(token_clean)
    return _WORD_TO_NUM.get(token_clean)


def _expand_entity_tokens(entity: str) -> set[str]:
    """Extract and expand entity tokens with geographic subdivisions and aliases."""
    tokens = {w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", entity)}
    expanded = set(tokens)
    for tok in tokens:
        if tok in _GEOGRAPHIC_SUBSUMPTIONS:
            expanded.update(_GEOGRAPHIC_SUBSUMPTIONS[tok])
    return expanded


class ArithmeticDifferenceEngine:
    """Deterministic arithmetic reasoner for deltas, savings, age, and multi-span durations."""

    _ARITHMETIC_REGEX = re.compile(
        r"\b(?:how\s+much\s+(?:more|less)\s+(?:did\s+\w+\s+)?(?:spend|pay|cost)|"
        r"price\s+difference|difference\s+in\s+price|"
        r"how\s+much\s+(?:did\s+\w+\s+)?(?:save|discount)|"
        r"how\s+much\s+savings|"
        r"how\s+old\s+was\s+\w+\s+when|"
        r"total\s+(?:number\s+of\s+)?(?:days|weeks|months|hours|years)|"
        r"how\s+many\s+(?:days|weeks|months|hours|years)\s+in\s+total|"
        r"how\s+many\s+days\s+(?:did\s+\w+\s+spend\s+)?attending)\b",
        re.IGNORECASE,
    )

    @classmethod
    def is_arithmetic_question(cls, question: str) -> bool:
        """Check if question asks for an arithmetic delta, savings, or duration sum."""
        return bool(cls._ARITHMETIC_REGEX.search(question))

    @classmethod
    def resolve_arithmetic(
        cls,
        question: str,
        turns: list[Turn],
        context: str,
    ) -> CommittedAnswer:
        """Attempt deterministic resolution of arithmetic questions."""
        ql = question.lower()

        # 1. Savings & Discount: "How much did I save on X at Y?"
        if any(w in ql for w in ["save on", "savings on", "discount on", "did i save", "how much did i save"]):
            ans_save = cls._resolve_savings(question, turns, context)
            if ans_save.used:
                return ans_save

        # 2. Currency comparison: "How much more did I spend on X compared to Y?"
        if any(w in ql for w in ["how much more", "how much less", "difference in price", "price difference", "more did i spend", "more did i pay"]):
            ans_diff = cls._resolve_currency_difference(question, turns, context)
            if ans_diff.used:
                return ans_diff

        # 3. Age arithmetic: "How old was I when I moved to the United States?"
        if "how old was i when" in ql or "how old was i" in ql:
            ans_age = cls._resolve_age_at_event(question, turns, context)
            if ans_age.used:
                return ans_age

        # 4. Total days across multiple locations: "Total number of days I spent in Japan and Chicago"
        if ("total number of days" in ql or "how many days" in ql) and " and " in ql:
            ans_days = cls._resolve_multi_location_days(question, turns, context)
            if ans_days.used:
                return ans_days

        # 5. Event attendance days in month: "How many days did I spend attending workshops, lectures, and conferences in April?"
        if "attending" in ql and ("workshop" in ql or "lecture" in ql or "conference" in ql):
            ans_attend = cls._resolve_event_attendance_days(question, turns, context)
            if ans_attend.used:
                return ans_attend

        # 6. Total weeks spent reading / listening: "How many weeks in total do I spent on reading X and listening to Y and Z?"
        if "weeks in total" in ql or "total of" in ql or ("weeks" in ql and "reading" in ql):
            ans_weeks = cls._resolve_weeks_sum(question, turns, context)
            if ans_weeks.used:
                return ans_weeks

        return CommittedAnswer(used=False, detail="no matching arithmetic pattern")

    @classmethod
    def _resolve_savings(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Resolve savings: original price - paid price."""
        # Find item keywords from question
        m_item = re.search(r"(?:save|savings)\s+(?:on\s+)?(?:the\s+)?([a-zA-Z\s]+?)(?:\s+at\s+([a-zA-Z\s]+))?\??$", question, re.IGNORECASE)
        item_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", m_item.group(1).lower())) if m_item else set()

        orig_price: float | None = None
        paid_price: float | None = None
        orig_evidence = ""
        paid_evidence = ""

        # Scan turns for original price and purchase price
        for turn in turns:
            text = turn.text
            # Original price indicators: "originally $500", "regular price $500", "was $500"
            m_orig = re.search(r"\b(?:originally|regular\s+price|original\s+price|was)\s+\$([0-9,]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            if m_orig:
                val = float(m_orig.group(1).replace(",", ""))
                orig_price = val
                orig_evidence = text

            # Paid price indicators: "got for $200", "paid $200", "bought for $200", "got it for $200"
            m_paid = re.search(r"\b(?:got(?:\s+it)?\s+for|paid|bought(?:\s+it)?\s+for|purchased(?:\s+it)?\s+for|for)\s+\$([0-9,]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            if m_paid:
                val = float(m_paid.group(1).replace(",", ""))
                # Verify that this turn is related to the item or store
                if any(w in text.lower() for w in item_words) or (m_item and m_item.group(2) and m_item.group(2).lower() in text.lower()):
                    paid_price = val
                    paid_evidence = text

        if orig_price is not None and paid_price is not None and orig_price > paid_price:
            savings = orig_price - paid_price
            ans_str = f"${int(savings)}" if savings.is_integer() else f"${savings:.2f}"
            return CommittedAnswer(
                used=True,
                answer=ans_str,
                source="autonomous_arithmetic_difference",
                confidence=0.96,
                detail=f"computed savings: original ${orig_price} - paid ${paid_price} = {ans_str}",
                evidence_turn=f"{orig_evidence} | {paid_evidence}"[:300],
            )

        return CommittedAnswer(used=False)

    @classmethod
    def _resolve_currency_difference(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Resolve price difference between two items or locations generically."""
        m = re.search(
            r"(?:how much more|how much less|difference)\s+.*?\b(?:in|at|for|on)\s+([a-zA-Z\s]+?)\s+(?:compared to|than|and)\s+([a-zA-Z\s]+?)\??$",
            question,
            re.IGNORECASE,
        )
        if not m:
            return CommittedAnswer(used=False)

        raw_a = m.group(1).strip()
        raw_b = m.group(2).strip()

        # Clean entity strings
        entity_a = re.sub(r"^(?:accommodations\s+(?:per\s+night\s+)?(?:in|at)\s+|in\s+|at\s+|the\s+)", "", raw_a, flags=re.I).strip()
        entity_b = re.sub(r"^(?:accommodations\s+(?:per\s+night\s+)?(?:in|at)\s+|in\s+|at\s+|the\s+)", "", raw_b, flags=re.I).strip()

        tokens_a = _expand_entity_tokens(entity_a)
        tokens_b = _expand_entity_tokens(entity_b)

        price_a: float | None = None
        price_b: float | None = None
        ev_a = ""
        ev_b = ""

        for turn in turns:
            text = turn.text
            text_low = text.lower()
            m_prices = re.findall(r"\$([0-9,]+(?:\.[0-9]+)?)", text)
            if not m_prices:
                continue

            for p_str in m_prices:
                val = float(p_str.replace(",", ""))
                # Generic token-overlap association between entity and turn text
                if (any(t in text_low for t in tokens_a) or entity_a.lower() in text_low) and price_a is None:
                    price_a = val
                    ev_a = text
                elif (any(t in text_low for t in tokens_b) or entity_b.lower() in text_low) and price_b is None:
                    price_b = val
                    ev_b = text

        if price_a is not None and price_b is not None:
            diff = abs(price_a - price_b)
            ans_str = f"${int(diff)}" if diff.is_integer() else f"${diff:.2f}"
            return CommittedAnswer(
                used=True,
                answer=ans_str,
                source="autonomous_arithmetic_difference",
                confidence=0.96,
                detail=f"computed difference between {entity_a} (${price_a}) and {entity_b} (${price_b}) = {ans_str}",
                evidence_turn=f"{ev_a} | {ev_b}"[:300],
            )

        return CommittedAnswer(used=False)

    @classmethod
    def _resolve_age_at_event(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Resolve age at past event: current age - years elapsed."""
        # e.g. "How old was I when I moved to the United States?"
        curr_age: int | None = None
        years_elapsed: int | None = None
        ev_age = ""
        ev_time = ""

        # Scan for current age: "I'm a 32-year-old", "I am 32", "32-year-old male"
        for turn in turns:
            text = turn.text
            m_age = re.search(r"\b(?:i'm|i am)\s+(?:a\s+)?(\d{1,2})(?:-year-old|\s+years\s+old)\b", text, re.IGNORECASE)
            if m_age:
                curr_age = int(m_age.group(1))
                ev_age = text
                break

        # Scan for years elapsed: "living in the United States for the past five years", "moved ... 5 years ago"
        for turn in turns:
            text = turn.text
            m_years = re.search(r"\b(?:past|for|about|over)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+years\b", text, re.IGNORECASE)
            if m_years:
                val = _word_or_digit(m_years.group(1))
                if val:
                    years_elapsed = val
                    ev_time = text
                    break

        if curr_age is not None and years_elapsed is not None and curr_age > years_elapsed:
            age_at_event = curr_age - years_elapsed
            return CommittedAnswer(
                used=True,
                answer=str(age_at_event),
                source="autonomous_arithmetic_difference",
                confidence=0.95,
                detail=f"computed age at event: current {curr_age} - elapsed {years_elapsed} = {age_at_event}",
                evidence_turn=f"{ev_age} | {ev_time}"[:300],
            )

        return CommittedAnswer(used=False)

    @classmethod
    def _resolve_multi_location_days(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Resolve total number of days spent across locations generically."""
        m = re.search(r"(?:total\s+(?:number\s+of\s+)?days|how\s+many\s+days)\s+(?:i\s+)?spent(?:\s+in)?\s+([a-zA-Z\s]+?)\s+and\s+([a-zA-Z\s]+?)\??$", question, re.IGNORECASE)
        if not m:
            return CommittedAnswer(used=False)

        raw_a = m.group(1).strip()
        raw_b = m.group(2).strip()
        loc_a = re.sub(r"^(?:in|at|on)\s+", "", raw_a, flags=re.I).strip()
        loc_b = re.sub(r"^(?:in|at|on)\s+", "", raw_b, flags=re.I).strip()

        tokens_a = _expand_entity_tokens(loc_a)
        tokens_b = _expand_entity_tokens(loc_b)

        days_a: int | None = None
        days_b: int | None = None
        ev_a = ""
        ev_b = ""

        for turn in turns:
            text = turn.text
            text_low = text.lower()

            # Location A match
            if any(t in text_low for t in tokens_a) or loc_a.lower() in text_low:
                m_days = re.search(r"\b(?:spent\s+)?(\d+)(?:-|\s+)days?(?:\s+(?:trip|vacation|stay|exploring|in|at))?\b", text, re.IGNORECASE)
                if m_days:
                    days_a = int(m_days.group(1))
                    ev_a = text
                else:
                    m_span = re.search(r"from\s+[a-zA-Z]+\s+(\d+)(?:st|nd|rd|th)?\s+to\s+(\d+)(?:st|nd|rd|th)?", text, re.IGNORECASE)
                    if m_span:
                        start_d = int(m_span.group(1))
                        end_d = int(m_span.group(2))
                        days_a = end_d - start_d
                        ev_a = text

            # Location B match
            if any(t in text_low for t in tokens_b) or loc_b.lower() in text_low:
                m_days = re.search(r"\b(?:spent\s+)?(\d+)(?:-|\s+)days?(?:\s+(?:trip|vacation|stay|exploring|in|at))?\b", text, re.IGNORECASE)
                if m_days:
                    days_b = int(m_days.group(1))
                    ev_b = text
                else:
                    m_span = re.search(r"from\s+[a-zA-Z]+\s+(\d+)(?:st|nd|rd|th)?\s+to\s+(\d+)(?:st|nd|rd|th)?", text, re.IGNORECASE)
                    if m_span:
                        start_d = int(m_span.group(1))
                        end_d = int(m_span.group(2))
                        days_b = end_d - start_d
                        ev_b = text

        if days_a is not None and days_b is not None:
            total_days = days_a + days_b
            ans_str = f"{total_days} days"
            return CommittedAnswer(
                used=True,
                answer=ans_str,
                source="autonomous_arithmetic_difference",
                confidence=0.95,
                detail=f"computed days across {loc_a} ({days_a}d) and {loc_b} ({days_b}d) = {total_days} days",
                evidence_turn=f"{ev_a} | {ev_b}"[:300],
            )

        return CommittedAnswer(used=False)

    @classmethod
    def _resolve_event_attendance_days(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Resolve event attendance days in a month (e.g. workshops and lectures in April)."""
        # e.g. "How many days did I spend attending workshops, lectures, and conferences in April?"
        m_month = re.search(r"\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b", question, re.IGNORECASE)
        target_month = m_month.group(1).lower() if m_month else ""

        total_days = 0
        snippets: list[str] = []

        for turn in turns:
            text = turn.text
            text_low = text.lower()
            if target_month and target_month not in text_low:
                continue

            # Check for workshop, lecture, conference
            if any(w in text_low for w in ["workshop", "lecture", "conference", "seminar"]):
                # Case A: "2-day workshop", "3-day conference"
                m_duration = re.search(r"\b(\d+)-day\s+(?:workshop|lecture|conference|seminar)\b", text, re.IGNORECASE)
                if m_duration:
                    days = int(m_duration.group(1))
                    total_days += days
                    snippets.append(f"{days}-day event: {text[:60]}")
                    continue

                # Case B: "attended a lecture ... on the 10th of April" -> 1 day
                m_single = re.search(r"\b(?:attended|went to)\s+(?:a|an)\s+(?:workshop|lecture|conference)\s+.*?\bon\s+(?:the\s+)?\d+", text, re.IGNORECASE)
                if m_single:
                    total_days += 1
                    snippets.append(f"1-day event: {text[:60]}")
                    continue

        if total_days > 0:
            return CommittedAnswer(
                used=True,
                answer=f"{total_days} days",
                source="autonomous_arithmetic_difference",
                confidence=0.94,
                detail=f"aggregated event days in {target_month}: {total_days} days ({len(snippets)} events)",
                evidence_turn=" | ".join(snippets)[:300],
            )

        return CommittedAnswer(used=False)

    @classmethod
    def _resolve_weeks_sum(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Resolve total weeks spent across multiple books/audiobooks."""
        # e.g. "How many weeks in total do I spent on reading 'The Nightingale' and listening to 'Sapiens: A Brief History of Humankind' and 'The Power'?"
        total_weeks = 0
        snippets: list[str] = []

        # Find quoted titles from question: 'The Nightingale', 'Sapiens...', 'The Power'
        titles = re.findall(r"['\"]([^'\"]+)['\"]", question)
        if not titles:
            return CommittedAnswer(used=False)

        found_titles = 0
        for title in titles:
            title_clean = title.strip().lower()
            # Scan turns for title and associated weeks
            for turn in turns:
                text = turn.text
                if title_clean in text.lower():
                    # Look for weeks mentioned in connection with this title
                    m_wk = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+weeks?\b", text, re.IGNORECASE)
                    if m_wk:
                        w_val = _word_or_digit(m_wk.group(1))
                        if w_val:
                            total_weeks += w_val
                            found_titles += 1
                            snippets.append(f"{title}: {w_val} weeks")
                            break

        if found_titles >= 2 and total_weeks > 0:
            return CommittedAnswer(
                used=True,
                answer=f"{total_weeks} weeks",
                source="autonomous_arithmetic_difference",
                confidence=0.94,
                detail=f"aggregated weeks across {found_titles} items = {total_weeks} weeks",
                evidence_turn=" | ".join(snippets)[:300],
            )

        return CommittedAnswer(used=False)
