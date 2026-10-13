"""Both-direction tests for the NUMERAL CONTRADICTION guard.

A guard that only closes the hole is untested -- it may equally be eating the
legitimate credits it was never supposed to touch.  Every case here is asserted
in BOTH directions: the wrong-numeral cases must go WRONG, and the
correct-numeral cases must stay CORRECT.
"""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = mod
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
score = mod.score


# ---------------------------------------------------------------- must be WRONG
#: Conflicting numerals that were previously credited correct (the defect).
MUST_BE_WRONG = [
    ("two dogs", "three dogs"),
    ("five dogs", "two dogs"),
    ("four times", "two times"),
    ("three times", "two times"),
    ("ten times", "two times"),
    ("four times, 4", "three times"),      # the lc_conv-44_17 case
    ("six kids", "two kids"),
    ("nine cats", "two cats"),
    ("2 dogs", "3 dogs"),
    ("5 dogs", "2 dogs"),
    ("2 weeks", "three weeks"),            # lc_conv-43_16
    ("5 months", "4 months"),              # lc_conv-44_59
    ("5 months", "six months"),            # lc_conv-30_31
    ("0 years", "three years"),            # lc_conv-44_7
    ("2 years ago", "10 years ago"),       # lc_conv-26_12
    ("22 August, 2023", "August 2022"),    # lc_conv-50_51, conflicting YEAR
    ("8 October 2023", "October 2022"),    # lc_conv-49_58, conflicting YEAR
]


# --------------------------------------------------------------- must be CORRECT
#: Legitimate credits the guard must NOT eat.  These are the false-positive
#: risk: if any of these starts failing, the guard is too wide.
MUST_BE_CORRECT = [
    # numerals agree
    ("two dogs", "two dogs"),
    ("three times", "three times"),
    ("2 dogs", "two dogs"),
    ("4 months", "four months"),
    ("10 years ago", "10 years ago"),
    ("August 2022", "August 2022"),
    # gold names no quantity at all -> guard must stay silent
    ("two dogs", "she has dogs"),
    ("5 cats", "cats"),
    # prediction names several quantities, one of which matches gold -> not a
    # contradiction (the gold value is present)
    ("two dogs and three cats", "three cats"),
    # digit vs word for the same value
    ("3 dogs", "three dogs"),
]


#: Guard must stay SILENT on these -- the disagreement is not a numeral one, so
#: whatever the scorer did before this guard must be unchanged.  These pin the
#: guard's blast radius: it may not alter any verdict it was not meant to.
GUARD_STAYS_SILENT = [
    ("twice", "two times"),        # quantity {2} == {2}, not a contradiction
    ("three times", "often"),      # gold names no quantity at all
]


def main() -> int:
    failures: list[str] = []

    for pred, gold in MUST_BE_WRONG:
        if score(pred, gold):
            failures.append(f"WRONG-expected: pred={pred!r} gold={gold!r} scored CORRECT")

    for pred, gold in MUST_BE_CORRECT:
        if not score(pred, gold):
            failures.append(f"CORRECT-expected: pred={pred!r} gold={gold!r} scored WRONG")

    # The guard must not claim a contradiction it has no business claiming.
    for pred, gold in GUARD_STAYS_SILENT:
        if mod._numeral_contradiction(pred, gold):
            failures.append(f"guard fired where it must not: pred={pred!r} gold={gold!r}")

    checked = len(MUST_BE_WRONG) + len(MUST_BE_CORRECT) + len(GUARD_STAYS_SILENT)
    print(f"numeral guard cases checked : {checked}")
    print(f"  must-be-wrong       : {len(MUST_BE_WRONG)}")
    print(f"  must-be-correct     : {len(MUST_BE_CORRECT)}")
    print(f"  guard-must-be-silent: {len(GUARD_STAYS_SILENT)}")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for line in failures:
            print(f"  {line}")
        return 1
    print("\nall cases pass in BOTH directions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
