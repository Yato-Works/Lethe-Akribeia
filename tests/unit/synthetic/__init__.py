"""Synthetic Counterfactual Test Suite for Lethe-Akribeia.

Prevents recurrence of the R8.1 failure mode (benchmark answers embedded as
constants).  Any operator promoted to production must survive every
counterfactual rewrite in this suite.
"""

from .corpus import SyntheticItem, SyntheticTurn
from .generators import GENERATORS
from .guards import scan_source_for_overfitting
from .rewrites import REWRITERS

__all__ = [
    "SyntheticItem",
    "SyntheticTurn",
    "GENERATORS",
    "REWRITERS",
    "scan_source_for_overfitting",
]
