"""Guard: an A/B report must never label noise as an improvement.

WHY THIS EXISTS
---------------
An experiment was reported as "conditional adoption" on a table like:

    baseline   ece    delta
        76%    78%    +2%  OK
        62%    62%     0%  OK
        58%    56%    -2%  WARN

McNemar on the real paired predictions gave b=0/c=0 for the first two slices
(the two implementations emitted IDENTICAL answers -- the gate never fired)
and b=1/c=0, p=1.0 for the third (a single question).  There was no effect at
all.  "OK" and "conditional adoption" dressed up nothing as something.

These tests lock in the honest behaviour:
  * a not-significant delta is labelled NOISE, never OK
  * identical predictions are called out as NO EFFECT
  * a one-question difference cannot be reported as a regression
  * pairing is by question_id, so reordering cannot invent a delta
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from artificial_memory.compiler.eval.honest_report import (
    compare_slices,
    format_report,
)


def _close(actual, expected, tol=1e-6):
    assert abs(float(actual) - expected) < tol, f"{actual} != {expected}"


BASE = [{"question_id": f"q{i}", "is_correct": True} for i in range(50)]


def _write(rows, path: Path) -> Path:
    path.write_text(json.dumps({"results": rows}), encoding="utf-8")
    return path


def test_identical_predictions_is_no_effect_not_ok():
    """b=0/c=0 must never be reported as an improvement."""
    with tempfile.TemporaryDirectory() as d:
        a = _write(BASE, Path(d) / "a.json")
        b = _write([dict(r) for r in BASE], Path(d) / "b.json")
        (v,) = compare_slices([("identical", a, b)])
        assert v.n_paired == 50
        assert v.delta_pt == 0.0
        assert v.b == 0 and v.c == 0
        assert v.p_value == 1.0
        assert not v.significant
        assert v.verdict == "NO EFFECT (identical predictions)"
        assert "OK" not in v.verdict


def test_single_question_difference_is_noise():
    """A one-question delta (2pt at n=50) must be NOISE, not a regression."""
    rows_b = [dict(r) for r in BASE]
    rows_b[0]["is_correct"] = False  # exactly one question flips
    with tempfile.TemporaryDirectory() as d:
        a = _write(BASE, Path(d) / "a.json")
        b = _write(rows_b, Path(d) / "b.json")
        (v,) = compare_slices([("one-q", a, b)])
        _close(v.delta_pt, -2.0)
        assert int(v.b) == 1 and int(v.c) == 0
        assert not bool(v.significant)
        assert v.verdict == "NOISE (not significant)"


def test_report_never_prints_ok_for_noise():
    """The rendered table must not contain an 'OK' verdict for noise."""
    rows_b = [dict(r) for r in BASE]
    rows_b[0]["is_correct"] = False
    with tempfile.TemporaryDirectory() as d:
        a = _write(BASE, Path(d) / "a.json")
        b = _write(rows_b, Path(d) / "b.json")
        vs = compare_slices([("slice", a, b)])
        report = format_report(vs, "TITLE")
        assert "NOISE" in report
        assert "conditional" not in report.lower()
        assert " OK" not in report


def test_pairing_is_by_question_id_not_position():
    """Reordering the treatment file must not change the verdict."""
    rows_b = [dict(r) for r in BASE]
    rows_b[3]["is_correct"] = False
    with tempfile.TemporaryDirectory() as d:
        a = _write(BASE, Path(d) / "a.json")
        ordered = _write(rows_b, Path(d) / "b_ordered.json")
        shuffled = _write(list(reversed(rows_b)), Path(d) / "b_shuffled.json")
        (v1,) = compare_slices([("x", a, ordered)])
        (v2,) = compare_slices([("x", a, shuffled)])
        assert int(v1.b) == int(v2.b) == 1
        assert int(v1.c) == int(v2.c) == 0
        _close(v1.delta_pt, -2.0)
        _close(v2.delta_pt, -2.0)


def test_real_improvement_is_labelled_significant():
    """A large real gain must be labelled SIGNIFICANT IMPROVEMENT."""
    rows_b = []
    for i, r in enumerate(BASE):
        row = dict(r)
        # flip 20 questions from wrong-ish state: make 20 extra correct
        row["is_correct"] = True if i % 5 else (i % 2 == 0)
        rows_b.append(row)
    # Construct a clear gain: baseline half correct, treatment all correct
    base = [{"question_id": f"q{i}", "is_correct": i < 25} for i in range(50)]
    treat = [{"question_id": f"q{i}", "is_correct": True} for i in range(50)]
    with tempfile.TemporaryDirectory() as d:
        a = _write(base, Path(d) / "a.json")
        b = _write(treat, Path(d) / "b.json")
        (v,) = compare_slices([("gain", a, b)])
        assert v.delta_pt == 50.0
        assert v.significant
        assert v.verdict == "SIGNIFICANT IMPROVEMENT"
