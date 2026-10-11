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
    assert is_list_gold("Figurines, shoes")
    # enumerated with 'and' rather than a comma
    assert is_list_gold("clarinet and violin")
    assert is_list_gold("a letter and a card")
    assert is_list_gold("Thoughtful, authentic, driven")
    assert not is_list_gold("Sweden")
    # prose that merely contains a comma is not an enumeration
    assert not is_list_gold("Likely no, though she likes reading")
    assert not is_list_gold("Yes, she is supportive")


def test_clause_matching_is_word_boundary_not_prefix():
    """'Thoughtful' must not be read as the clause starter 'though'.

    Prefix matching demoted a real 3-item enumeration answer, which let a
    subset prediction earn containment credit.
    """
    assert is_list_gold("Thoughtful, authentic, driven")
    assert grade("Thoughtful, authentic", "Thoughtful, authentic, driven") == WRONG


def test_enumeration_uses_and_as_well_as_comma():
    """'clarinet and violin' is an enumeration even without a comma."""
    assert grade("clarinet", "clarinet and violin") == WRONG
    assert grade("clarinet and violin", "clarinet and violin") == STRICT


def test_and_enumeration_vs_clause():
    """'and' joins an enumeration only between coordinate noun phrases.

    Rule derived from the 91 comma-free golds in the corpus, not invented:
      enumeration : "Bach and Mozart", "Roses and dahlias", "Free and light"
      clause      : "Changed his diet and started walking regularly",
                    "Read a book and paint", "to de-stress and clear her mind"
    The discriminator is a finite verb or an infinitive leading the part.
    """
    # enumerations
    for g in ("Bach and Mozart", "Fetch and Frisbee", "Roses and dahlias",
              "Free and light", "Two cats and a dog", "Music and art"):
        assert is_list_gold(g), g
        assert grade(g.split(" and ")[0], g) == WRONG, g
    # clauses -- containment credit must still work
    for g in ("Changed his diet and started walking regularly",
              "Read a book and paint",
              "to de-stress and clear her mind"):
        assert not is_list_gold(g), g


def test_clause_lead_regex_is_balanced():
    """The pattern must compile: an earlier version had an unterminated group."""
    from artificial_memory.compiler.eval.audit_strict_grade import _CLAUSE_LEAD
    assert _CLAUSE_LEAD.match("Changed his diet")
    assert _CLAUSE_LEAD.match("Read a book")
    assert _CLAUSE_LEAD.match("to de-stress")
    assert not _CLAUSE_LEAD.match("Bach")
    assert not _CLAUSE_LEAD.match("Roses")


def test_single_item_gold_still_allows_containment():
    """Non-list golds keep the containment credit the production scorer gives."""
    assert grade("her mental health", "mental health") == NEAR
    assert grade("to de-stress", "To de-stress and clear her mind") == NEAR
    # substring, but not a contiguous word-boundary containment -> tolerated
    assert grade("a handwritten letter", "a letter") == PARTIAL


def test_list_gold_is_graded_by_set_not_substring():
    """Any enumeration answer is settled by item-set relation, never by overlap.

    Two items of a three-item gold is an incomplete answer, and an extra item
    is a wrong one.  Token overlap would credit both.
    """
    assert grade("Thoughtful, authentic", "Thoughtful, authentic, driven") == WRONG
    assert grade("clarinet", "clarinet and violin") == WRONG
    assert grade("Rome", "Paris, Rome") == WRONG
    # order does not matter
    assert grade("Rome, Paris", "Paris, Rome") == STRICT
    assert grade("painting, pottery", "pottery, painting") == STRICT


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
