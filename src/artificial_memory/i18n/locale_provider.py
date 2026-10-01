"""Locale provider for deterministic multilingual normalization and reasoning (P4).

Decouples hardcoded English token regexes and calendar patterns into
pluggable locale providers, laying the foundation for zero-leak multilingual
long-term AI memory.
"""

from __future__ import annotations

import calendar
import datetime
from abc import ABC, abstractmethod


class LocaleProvider(ABC):
    """Abstract base class for locale-specific calendar and language tokens."""

    @property
    @abstractmethod
    def locale_code(self) -> str:
        """IETF language tag (e.g. 'en', 'ja')."""

    @property
    @abstractmethod
    def month_map(self) -> dict[str, int]:
        """Mapping from lowercase month names/abbreviations to 1..12."""

    @property
    @abstractmethod
    def weekday_map(self) -> dict[str, int]:
        """Mapping from lowercase weekday names to 0..6 (0=Monday)."""

    @abstractmethod
    def format_date(self, d: datetime.date) -> str:
        """Format a calendar date according to locale conventions."""


class EnLocaleProvider(LocaleProvider):
    """Standard English locale provider."""

    @property
    def locale_code(self) -> str:
        return "en"

    @property
    def month_map(self) -> dict[str, int]:
        return {
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

    @property
    def weekday_map(self) -> dict[str, int]:
        return {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }

    def format_date(self, d: datetime.date) -> str:
        month_name = calendar.month_name[d.month]
        return f"{d.day} {month_name} {d.year}"


class JaLocaleProvider(LocaleProvider):
    """Japanese locale provider for CJK temporal normalization."""

    @property
    def locale_code(self) -> str:
        return "ja"

    @property
    def month_map(self) -> dict[str, int]:
        return {
            "1月": 1, "2月": 2, "3月": 3, "4月": 4, "5月": 5, "6月": 6,
            "7月": 7, "8月": 8, "9月": 9, "10月": 10, "11月": 11, "12月": 12,
            "一月": 1, "二月": 2, "三月": 3, "四月": 4, "五月": 5, "六月": 6,
            "七月": 7, "八月": 8, "九月": 9, "十月": 10, "十一月": 11, "十二月": 12,
        }

    @property
    def weekday_map(self) -> dict[str, int]:
        return {
            "月曜日": 0, "火曜日": 1, "水曜日": 2, "木曜日": 3,
            "金曜日": 4, "土曜日": 5, "日曜日": 6,
            "月曜": 0, "火曜": 1, "水曜": 2, "木曜": 3,
            "金曜": 4, "土曜": 5, "日曜": 6,
        }

    def format_date(self, d: datetime.date) -> str:
        return f"{d.year}年{d.month}月{d.day}日"


_PROVIDERS: dict[str, LocaleProvider] = {
    "en": EnLocaleProvider(),
    "ja": JaLocaleProvider(),
}


def get_locale_provider(locale: str = "en") -> LocaleProvider:
    """Get the appropriate LocaleProvider by locale code (defaulting to 'en')."""
    code = locale.lower().split("-")[0].split("_")[0]
    return _PROVIDERS.get(code, _PROVIDERS["en"])
