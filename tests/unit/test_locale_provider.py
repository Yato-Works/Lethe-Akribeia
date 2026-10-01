"""Unit tests for LocaleProvider and multilingual foundations."""

from __future__ import annotations

import datetime
from artificial_memory.i18n.locale_provider import (
    EnLocaleProvider,
    JaLocaleProvider,
    get_locale_provider,
)


def test_en_locale_provider() -> None:
    en = get_locale_provider("en-US")
    assert isinstance(en, EnLocaleProvider)
    assert en.locale_code == "en"
    assert en.month_map["oct"] == 10
    assert en.weekday_map["wednesday"] == 2

    d = datetime.date(2023, 5, 8)
    assert en.format_date(d) == "8 May 2023"


def test_ja_locale_provider() -> None:
    ja = get_locale_provider("ja_JP")
    assert isinstance(ja, JaLocaleProvider)
    assert ja.locale_code == "ja"
    assert ja.month_map["10月"] == 10
    assert ja.month_map["十月"] == 10
    assert ja.weekday_map["水曜日"] == 2
    assert ja.weekday_map["水曜"] == 2

    d = datetime.date(2023, 5, 8)
    assert ja.format_date(d) == "2023年5月8日"


def test_fallback_to_en() -> None:
    fallback = get_locale_provider("fr_FR")
    assert isinstance(fallback, EnLocaleProvider)
