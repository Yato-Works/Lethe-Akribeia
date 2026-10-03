"""Unit test verifying that every benchmark metric in RESULTS_REGISTRY.json has an auditable artifact and matches measured numbers.

Guarantees:
- Zero silent disappearance of headline benchmark artifacts (anti-purge invariant).
- Exact numerical concordance between published documentation and raw JSON artifacts.
- Enforced across local test runs and CI pipelines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from scripts.benchmarks.verify_registry import verify_entry, REGISTRY_PATH


def test_registry_file_exists() -> None:
    assert REGISTRY_PATH.exists(), f"Results registry not found at {REGISTRY_PATH}"


def test_all_registry_entries_pass() -> None:
    assert REGISTRY_PATH.exists()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    entries = registry.get("entries", [])
    assert len(entries) >= 10, "Registry must contain at least 10 canonical headline metrics"

    failures = []
    for entry in entries:
        ok, msg = verify_entry(entry)
        if not ok:
            failures.append(msg)

    assert not failures, f"Registry verification failed on {len(failures)} entries:\n" + "\n".join(failures)
