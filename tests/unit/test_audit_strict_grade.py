"""Guard: the strict-grading rubric is stable and does not credit subsets.

WHY THIS EXISTS
---------------
The strict rubric is what separates a real improvement from a tolerant partial
match.  If it drifts, audits stop being comparable and a noise-level change can
be reported as progress.  These tests pin the rubric's decisions.

The decisive cases are the enumeration ones: naming one item of a multi-item
gold must NOT be credited, because that is an incomplete answer.
"""
from __future__ import annotations

from artificial_memory.compiler.eval.audit_strict_grade import (  # noqa: F401
    NEAR, PARTIAL, STRICT, WRONG, audit, grade, is_list_gold,
)

# ---- tier assignment ----------------------------------------------------
CASES = [
    # (prediction, gold, expected tier)
    ("Sweden", "Sweden", STRICT),
    ("  SWEDEN ", "sweden", STRICT),
    ("magical", "it's magical", NEAR),
    ("Likely no", "Likely no, though she likes reading", NEAR),
    # tolerant partial -- what the production scorer already accepts
    ("lactose intolerance", "She is lactose intolerant", PARTIAL),
    # single-token typo: the fuzzy rule needs >3 tokens, so this stays WRONG
    ("Xeonoblade", "Xenoblade", WRONG),
    # enumeration: one item of a list is NOT a match
    ("Rome", "Paris, Rome", WRONG),
    ("Shadow", "Coco, Shadow", WRONG),
    ("Florida", "Oregon, Florida", WRONG),
    ("figurines", "Figurines, shoes", WRONG),
    ("banana split sundae", "Banana split sundae, Peach cobbler", WRONG),
    ("pottery, camping, painting, swimming", "pottery, camping, painting", WRONG),
    # abstention against a real gold is wrong
    ("I don't know.", "necklace", WRONG),
    ("", "necklace", WRONG),
    # genuinely different answers
    ("Yes, she would still want to", "no", WRONG),
]


def test_rubric_tiers_are_stable():
    for pred, gold, expected in CASES:
        got = grade(pred, gold)
        assert got == expected, f"grade({pred!r}, {gold!r}) = {got}, want {expected}"


def test_list_gold_gets_no_subset_credit():
    assert is_list_gold("Paris, Rome")
    assert is_list_gold("pottery, camping, painting, swimming")
    assert not is_list_gold("Sweden")
    assert not is_list_gold("a letter and a card")  # single comma-free item


def test_single_item_gold_still_allows_containment():
    """Non-list golds keep the containment credit the production scorer gives."""
    assert grade("her mental health", "mental health") == NEAR
    assert grade("to de-stress", "To de-stress and clear her mind") == NEAR
    # substring, but not a contiguous word-boundary containment -> tolerated
    assert grade("a handwritten letter", "a letter") == PARTIAL


def test_audit_reports_noise_as_noise():
    """A net change inside the tolerant band must not be called significant."""
    import json
    import tempfile
    from pathlib import Path

    rows_a, rows_b = [], []
    for i in range(120):
        rows_a.append({"question_id": f"q{i}", "predicted_answer": "I don't know.",
                       "ground_truth": "something else entirely", "is_correct": False})
        rows_b.append({"question_id": f"q{i}", "predicted_answer": "I don't know.",
                       "ground_truth": "something else entirely", "is_correct": False})
    # three questions move only inside the tolerant band (wrong -> partial)
    for i in range(3):
        rows_a[i] = {"question_id": f"q{i}", "predicted_answer": "wrong thing",
                     "ground_truth": "a partially overlapping phrase here",
                     "is_correct": False}
        rows_b[i] = {"question_id": f"q{i}", "predicted_answer": "a partially overlapping here",
                     "ground_truth": "a partially overlapping phrase here",
                     "is_correct": False}

    with tempfile.TemporaryDirectory() as d:
        pa = Path(d) / "a.json"
        pb = Path(d) / "b.json"
        pa.write_text(json.dumps({"results": rows_a}), encoding="utf-8")
        pb.write_text(json.dumps({"results": rows_b}), encoding="utf-8")
        report = audit(pa, pb, "A", "B")

    assert "REAL NET: +0 questions" in report
    assert "NOT SIGNIFICANT at any tier" in report
    assert "the headline is noise" in report


def test_audit_detects_a_real_improvement():
    """A large strict gain must still be called significant."""
    import json
    import tempfile
    from pathlib import Path

    rows_a, rows_b = [], []
    for i in range(200):
        rows_a.append({"question_id": f"q{i}", "predicted_answer": "I don't know.",
                       "ground_truth": f"answer number {i}", "is_correct": False})
        rows_b.append({"question_id": f"q{i}", "predicted_answer": f"answer number {i}",
                       "ground_truth": f"answer number {i}", "is_correct": True})

    with tempfile.TemporaryDirectory() as d:
        pa = Path(d) / "a.json"
        pb = Path(d) / "b.json"
        pa.write_text(json.dumps({"results": rows_a}), encoding="utf-8")
        pb.write_text(json.dumps({"results": rows_b}), encoding="utf-8")
        report = audit(pa, pb, "A", "B")

    assert "REAL NET: +200 questions" in report
    assert "at least one tier is significant" in report
