"""Unit tests for the safe answer contract.

The contract is a *measurement* of what a deterministic post-processor can do, so
the tests pin the two properties that decide whether it is shippable at all: it
never returns an empty answer, and it never removes a token that carries the
value.  The measured result on the deployed 7B is that it recovers ~0.00pp - the
padding is not boilerplate, it is the answer written out in a sentence - and
these tests are what make that a safe negative rather than a broken tool.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "benchmarks" / "answer_contract.py"


def load_script():
    spec = importlib.util.spec_from_file_location("_answer_contract_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ac = load_script()


def test_markdown_and_whitespace_are_normalised() -> None:
    assert ac._strip_markdown("**24 February 2023**") == "24 February 2023"
    assert ac._strip_markdown("see [the garden](https://x.y)") == "see the garden"


def test_boilerplate_is_dropped_only_when_something_remains() -> None:
    assert ac._drop_boilerplate("The answer is 24 February 2023.") == "24 February 2023."
    assert ac._drop_boilerplate("According to the context, 2019") == "2019"
    # A prefix that would empty the answer is not applied.
    assert ac._drop_boilerplate("The answer is") == "The answer is"


def test_repeated_information_is_collapsed_once() -> None:
    out = ac._dedupe("The garden is lovely. The garden is lovely. It rained.")
    assert out == "The garden is lovely. It rained."


def test_a_trailing_attribution_is_dropped_only_if_something_remains() -> None:
    out = ac._drop_trailing_meta("24 February 2023. According to the context, that is the date.")
    assert out.startswith("24 February 2023")
    assert ac._drop_trailing_meta("According to the context, that is the date.") != ""


def test_polarity_trim_applies_to_category_three_only() -> None:
    assert ac._polarity("Yes, it is classical music.", 3) == "Yes."
    # Category 4 keeps its sentence: the gold there is the whole phrase.
    assert ac._polarity("Yes, it is classical music.", 4) == "Yes, it is classical music."


def test_dates_are_canonicalised_to_one_order() -> None:
    assert ac._canonical_dates("February 24, 2023") == "24 February 2023"
    assert ac._canonical_dates("24 February 2023") == "24 February 2023"
    assert ac._canonical_dates("in 2019") == "in 2019"  # no reorderable date


def test_the_contract_never_empties_an_answer() -> None:
    for text in ("**", "```", "  ", "According to the context,", "-", "Yes"):
        assert ac.contract(text, 3).strip() != "" or text.strip() == ""
