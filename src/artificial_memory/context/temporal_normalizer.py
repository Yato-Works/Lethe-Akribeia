"""Deterministic Temporal Normalizer for AM Apex.

Resolves relative temporal expressions ("yesterday", "last year", "2 days ago",
"last month", etc.) into absolute calendar dates based on session/reference timestamps.
Offloads temporal arithmetic from the downstream LLM into the memory runtime.
"""

from __future__ import annotations

import calendar
import datetime
import re


class TemporalNormalizer:
    """Normalizes relative time expressions in text against a reference date."""

    MONTH_MAP = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    def parse_reference_date(self, date_str: str) -> datetime.date | None:
        """Parse a variety of session date strings into a datetime.date object.

        Examples:
            - '1:56 pm on 8 May, 2023'
            - '8 May, 2023'
            - '2023-05-08'
            - 'May 8, 2023'
        """
        if not date_str:
            return None

        # 1. ISO format: YYYY-MM-DD
        m_iso = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", date_str)
        if m_iso:
            return datetime.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))

        # 2. 'DD Month, YYYY' or 'DD Month YYYY'
        m_dmy = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b", date_str)
        if m_dmy:
            day = int(m_dmy.group(1))
            month_str = m_dmy.group(2).lower()
            year = int(m_dmy.group(3))
            month = self.MONTH_MAP.get(month_str[:3])
            if month:
                return datetime.date(year, month, day)

        # 3. 'Month DD, YYYY' or 'Month DD YYYY'
        m_mdy = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", date_str)
        if m_mdy:
            month_str = m_mdy.group(1).lower()
            day = int(m_mdy.group(2))
            year = int(m_mdy.group(3))
            month = self.MONTH_MAP.get(month_str[:3])
            if month:
                return datetime.date(year, month, day)

        return None

    def format_date(self, d: datetime.date) -> str:
        """Format a date into clean readable format (e.g. '7 May 2023')."""
        month_name = calendar.month_name[d.month]
        return f"{d.day} {month_name} {d.year}"

    def normalize(
        self,
        text: str,
        reference_date_str: str,
        enabled_rules: set[str] | None = None,
    ) -> str:
        """Resolve relative dates in text using the reference date string.

        Args:
            text: Input turn text.
            reference_date_str: Reference timestamp of the session.
            enabled_rules: If provided, only execute rules present in this set:
                - 'yesterday': yesterday, tomorrow, the day before yesterday
                - 'last_week': last week, last weekend, this week, next week
                - 'month': this month, next month, last month
                - 'days_ago': X days ago, X weeks ago, two days ago
                - 'weekdays': last [day of week]
                - 'last_year': last year, last <season>
                - 'years_ago': X years ago (resolved to the year)
                - 'months_ago': X months ago (resolved to Month YYYY)
        """
        ref_date = self.parse_reference_date(reference_date_str)
        if not ref_date:
            return text

        result = text
        run_all = enabled_rules is None

        # 1. 'the day before yesterday'
        if (run_all or "yesterday" in enabled_rules) and re.search(r"\bthe day before yesterday\b", result, flags=re.IGNORECASE):
            d = ref_date - datetime.timedelta(days=2)
            d_str = self.format_date(d)
            result = re.sub(
                r"\bthe day before yesterday\b",
                f"the day before yesterday ({d_str})",
                result,
                flags=re.IGNORECASE,
            )

        # 2. 'yesterday'
        if (run_all or "yesterday" in enabled_rules) and re.search(r"\byesterday\b", result, flags=re.IGNORECASE):
            d = ref_date - datetime.timedelta(days=1)
            d_str = self.format_date(d)
            result = re.sub(
                r"\byesterday\b",
                f"yesterday ({d_str})",
                result,
                flags=re.IGNORECASE,
            )

        # 3. 'tomorrow'
        if (run_all or "yesterday" in enabled_rules) and re.search(r"\btomorrow\b", result, flags=re.IGNORECASE):
            d = ref_date + datetime.timedelta(days=1)
            d_str = self.format_date(d)
            result = re.sub(
                r"\btomorrow\b",
                f"tomorrow ({d_str})",
                result,
                flags=re.IGNORECASE,
            )

        # 4. 'last year'
        if (run_all or "last_year" in enabled_rules) and re.search(r"\blast year\b", result, flags=re.IGNORECASE):
            d_str = str(ref_date.year - 1)
            result = re.sub(
                r"\blast year\b",
                f"last year ({d_str})",
                result,
                flags=re.IGNORECASE,
            )

        # 5. 'last month'
        if (run_all or "month" in enabled_rules) and re.search(r"\blast month\b", result, flags=re.IGNORECASE):
            prev_m = ref_date.month - 1 or 12
            prev_y = ref_date.year if ref_date.month > 1 else ref_date.year - 1
            m_name = calendar.month_name[prev_m]
            result = re.sub(
                r"\blast month\b",
                f"last month ({m_name} {prev_y})",
                result,
                flags=re.IGNORECASE,
            )

        # 6. 'X days ago'
        if run_all or "days_ago" in enabled_rules:
            def replace_days_ago(match: re.Match) -> str:
                count = int(match.group(1))
                d = ref_date - datetime.timedelta(days=count)
                return f"{count} days ago ({self.format_date(d)})"

            result = re.sub(r"\b(\d+)\s+days?\s+ago\b", replace_days_ago, result, flags=re.IGNORECASE)

        # 7. 'X weeks ago'
        if run_all or "days_ago" in enabled_rules:
            def replace_weeks_ago(match: re.Match) -> str:
                count = int(match.group(1))
                d = ref_date - datetime.timedelta(weeks=count)
                return f"{count} weeks ago ({self.format_date(d)})"

            result = re.sub(r"\b(\d+)\s+weeks?\s+ago\b", replace_weeks_ago, result, flags=re.IGNORECASE)

        # 8. 'last week' -> 'last week (the week before DD Month YYYY)'
        if (run_all or "last_week" in enabled_rules) and re.search(r"\blast week\b", result, flags=re.IGNORECASE):
            d_str = self.format_date(ref_date)
            result = re.sub(r"\blast week\b", f"last week (the week before {d_str})", result, flags=re.IGNORECASE)

        # 9. 'last weekend' -> 'last weekend (the weekend before DD Month YYYY)'
        if (run_all or "last_week" in enabled_rules) and re.search(r"\blast weekend\b", result, flags=re.IGNORECASE):
            d_str = self.format_date(ref_date)
            result = re.sub(r"\blast weekend\b", f"last weekend (the weekend before {d_str})", result, flags=re.IGNORECASE)

        # 10. 'last [Day of Week]' -> 'last [Day] (the [Day] before DD Month YYYY)'
        #     Also matches common abbreviations ("Last Fri", "last Tues") that
        #     appear frequently in chat transcripts: the LLM sees the resolved
        #     calendar date without having to do any arithmetic itself.
        if run_all or "weekdays" in enabled_rules:
            DAY_ABBREV_MAP = {
                "mon": "Monday", "monday": "Monday",
                "tue": "Tuesday", "tues": "Tuesday", "tuesday": "Tuesday",
                "wed": "Wednesday", "weds": "Wednesday", "wednesday": "Wednesday",
                "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday", "thursday": "Thursday",
                "fri": "Friday", "friday": "Friday",
                "sat": "Saturday", "saturday": "Saturday",
                "sun": "Sunday", "sunday": "Sunday",
            }

            def replace_last_day(match: re.Match) -> str:
                day_name = DAY_ABBREV_MAP[match.group(1).lower()]
                d_str = self.format_date(ref_date)
                return f"last {day_name} (the {day_name} before {d_str})"

            result = re.sub(
                r"\blast\s+(monday|mon|tuesday|tues|tue|wednesday|weds|wed|thursday|thurs|thur|thu|friday|fri|saturday|sat|sunday|sun)\b",
                replace_last_day,
                result,
                flags=re.IGNORECASE,
            )

        # 11. 'this month' -> 'this month (Month YYYY)'
        if (run_all or "month" in enabled_rules) and re.search(r"\bthis month\b", result, flags=re.IGNORECASE):
            m_name = calendar.month_name[ref_date.month]
            result = re.sub(r"\bthis month\b", f"this month ({m_name} {ref_date.year})", result, flags=re.IGNORECASE)

        # 12. 'next month' -> 'next month (NextMonth YYYY)'
        if (run_all or "month" in enabled_rules) and re.search(r"\bnext month\b", result, flags=re.IGNORECASE):
            next_m = (ref_date.month % 12) + 1
            next_y = ref_date.year if ref_date.month < 12 else ref_date.year + 1
            m_name = calendar.month_name[next_m]
            result = re.sub(r"\bnext month\b", f"next month ({m_name} {next_y})", result, flags=re.IGNORECASE)

        # 13. 'two days ago' -> 'two days ago (DD Month YYYY)'
        if (run_all or "days_ago" in enabled_rules) and re.search(r"\btwo days ago\b", result, flags=re.IGNORECASE):
            d = ref_date - datetime.timedelta(days=2)
            d_str = self.format_date(d)
            result = re.sub(r"\btwo days ago\b", f"two days ago ({d_str})", result, flags=re.IGNORECASE)
        # 14. 'this week' -> 'this week (the week of DD Month YYYY)'
        if (run_all or "last_week" in enabled_rules) and re.search(r"\bthis week\b", result, flags=re.IGNORECASE):
            d_str = self.format_date(ref_date)
            result = re.sub(r"\bthis week\b", f"this week (the week of {d_str})", result, flags=re.IGNORECASE)

        # 15. 'next week' -> 'next week (the week after DD Month YYYY)'
        if (run_all or "last_week" in enabled_rules) and re.search(r"\bnext week\b", result, flags=re.IGNORECASE):
            d_str = self.format_date(ref_date)
            result = re.sub(r"\bnext week\b", f"next week (the week after {d_str})", result, flags=re.IGNORECASE)

        # 16. 'last <season>' -> 'last <season> (the <season> of YYYY)'
        #     "Last <season>" means *the most recent one that has already ended*,
        #     so the year depends on where the reference date sits relative to the
        #     season: from 15 July 2023 "last summer" is 2022 (2023's summer is
        #     still running) but "last winter" is also 2022 (the winter that ended
        #     in February 2023).  Wrapping into the previous year unconditionally
        #     (the v1 rule) is wrong for every reference date that sits inside or
        #     after the named season.
        if run_all or "last_year" in enabled_rules:
            season_end = {"spring": 5, "summer": 8, "fall": 11, "autumn": 11}

            def replace_last_season(match: re.Match) -> str:
                season = match.group(1).lower()
                if season == "winter":
                    # The most recent winter that ended before the reference date
                    # started in ref.year-1, unless we are still inside it.
                    year = ref_date.year - 1 if ref_date.month >= 3 else ref_date.year - 2
                else:
                    end = season_end.get(season, 11)
                    year = ref_date.year - 1 if ref_date.month <= end else ref_date.year
                return f"last {match.group(1)} (the {match.group(1)} of {year})"

            result = re.sub(
                r"\blast\s+(summer|winter|spring|fall|autumn)\b",
                replace_last_season,
                result,
                flags=re.IGNORECASE,
            )

        # 17. 'N years ago' -> 'N years ago (YYYY)'.  This is calendar arithmetic
        #     the reader cannot be trusted with: LoCoMo answers these questions
        #     with the year ("In 2013") while the turn only says "10 years ago",
        #     so without the resolved year the answer is not present anywhere in
        #     the context and the question is unanswerable by construction.
        if run_all or "years_ago" in enabled_rules:
            spelled = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            }

            def replace_years_ago(match: re.Match) -> str:
                raw = match.group(1).lower()
                count = spelled.get(raw) if raw in spelled else int(raw)
                return f"{match.group(1)} years ago ({ref_date.year - count})"

            result = re.sub(
                r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
                r"\s+years?\s+ago\b",
                replace_years_ago,
                result,
                flags=re.IGNORECASE,
            )

        # 18. 'N months ago' -> 'N months ago (Month YYYY)'
        if run_all or "months_ago" in enabled_rules:
            def replace_months_ago(match: re.Match) -> str:
                count = int(match.group(1))
                total = ref_date.year * 12 + (ref_date.month - 1) - count
                year, month_index = divmod(total, 12)
                return f"{count} months ago ({calendar.month_name[month_index + 1]} {year})"

            result = re.sub(
                r"\b(\d{1,2})\s+months?\s+ago\b",
                replace_months_ago,
                result,
                flags=re.IGNORECASE,
            )
        return result
