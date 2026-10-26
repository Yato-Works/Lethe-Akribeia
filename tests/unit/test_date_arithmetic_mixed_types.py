"""Calendar arithmetic must accept date and datetime mixed.

Measured defect: building the LongMemEval context cache aborted the whole
250-question run on one question, because `calculate_recency` subtracted a
`datetime` from a `date` and Python raises TypeError.  The resolver is fed both
shapes (parsed calendar text yields `date`; upstream callers carrying a time
yield `datetime`), so normalisation belongs at the arithmetic boundary.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.recall.temporal_resolver import DateArithmeticEngine  # noqa: E402

D = datetime.date(2023, 5, 1)
DT = datetime.datetime(2023, 6, 1, 10, 30)


def test_recency_accepts_mixed_shapes():
    # recency is signed: reference minus event.
    assert DateArithmeticEngine.calculate_recency(D, DT) == -31
    assert DateArithmeticEngine.calculate_recency(DT, D) == 31


def test_difference_accepts_mixed_shapes():
    assert DateArithmeticEngine.calculate_difference(D, DT) == 31
    assert DateArithmeticEngine.calculate_difference(DT, D) == 31


def test_unit_conversion_still_works():
    assert DateArithmeticEngine.calculate_recency(D, datetime.datetime(2023, 5, 8)) == -7
    assert DateArithmeticEngine.calculate_recency(D, datetime.datetime(2023, 5, 29), "week") == -4
