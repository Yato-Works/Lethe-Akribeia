"""Diagnose the two false positives the numeral guard produced.

Reported by check_numeral_guard.py:
  pred 'twice'        gold 'two times'  scored WRONG (should be CORRECT)
  pred 'three times'  gold 'often'      scored WRONG (should be CORRECT)
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


def show(pred: str, gold: str) -> None:
    print(f"=== pred={pred!r}  gold={gold!r}")
    print(f"  _quantity_values(pred) = {mod._quantity_values(pred)}")
    print(f"  _quantity_values(gold) = {mod._quantity_values(gold)}")
    print(f"  _year_values(pred)     = {mod._year_values(pred)}")
    print(f"  _year_values(gold)     = {mod._year_values(gold)}")
    print(f"  _numeral_contradiction = {mod._numeral_contradiction(pred, gold)}")
    print(f"  _date_contradiction    = {mod._date_contradiction(pred, gold)}")
    print(f"  score                  = {mod.score(pred, gold)}")
    for m in mod._QUANTITY_RE.finditer(pred.lower()):
        tok = m.group(0).split()[0]
        print(f"  pred match {m.group(0)!r} -> token {tok!r} -> "
              f"{'digit' if tok.isdigit() else mod._NUMERAL_WORDS.get(tok, '??')}")
    print()


show("twice", "two times")
show("three times", "often")
# a control that still behaves
show("four times", "two times")
show("two dogs", "three dogs")
