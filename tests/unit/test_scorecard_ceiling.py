"""Unit tests for §8 of the Apex scorecard: the failure ceiling.

The section publishes two claims that can only be trusted if they are read off
stored counts: that the buckets **partition** the failures, and that the numbers
carry the oracle revision they were measured under.  A section that quietly showed
the legacy label-based figures would be worse than no section, so both are pinned.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCORECARD = REPO / "scripts" / "benchmarks" / "apex_scorecard.py"


def load_scorecard():
    spec = importlib.util.spec_from_file_location("_apex_scorecard_ceiling_under_test",
                                                  SCORECARD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = load_scorecard()


@contextmanager
def temp_artefact_dir():
    path = Path(tempfile.mkdtemp(prefix="ceiling-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def census(buckets: dict, **extra) -> dict:
    payload = {
        "status": "MEASURED",
        "n": 1000, "correct": 640, "wrong": 360, "committed_in_run": 100,
        "a": {"run": "benchmark_results/locomo1540/run.oraclefix.json", "model": "7b"},
        "buckets": buckets, "signals": {"reader_determined": 171, "neither_reader": 275,
                                        "unresolved_n": 255},
    }
    payload.update(extra)
    return payload


def test_without_an_artefact_the_section_says_not_measured() -> None:
    with temp_artefact_dir() as tmp:
        payload = sc.collect_failure_ceiling(tmp)
    assert payload["status"] == "NOT MEASURED"
    section = "\n".join(sc.render_ceiling_section(payload))
    assert "## 8. Failure ceiling" in section
    assert "NOT MEASURED" in section
    assert "recount_oracle.py" in section


def test_the_buckets_are_printed_with_both_denominators() -> None:
    payload = census({"retrieval": 87, "commit": 21,
                      "unresolved_reader_sensitive": 67, "unresolved_reader_insensitive": 185})
    section = "\n".join(sc.render_ceiling_section(payload))
    assert "| retrieval | 87 | 8.7% | 24.2% |" in section
    assert "Only 171 of 1000 questions (17.1%) change outcome" in section


def test_the_gold_context_ceiling_and_the_oracle_provenance_are_published() -> None:
    payload = census({"retrieval": 87, "commit": 21,
                      "unresolved_reader_sensitive": 67, "unresolved_reader_insensitive": 185},
                     gold_context=[{"cell": "context_limited"}, {"cell": "semantic",
                                                                  "answer_in_gold_context": "absent"},
                                   {"cell": "semantic", "answer_in_gold_context": "verbatim"}])
    section = "\n".join(sc.render_ceiling_section(payload))
    assert "### 8.1 Gold-context ceiling" in section
    assert "recovered by **both** readers" in section
    assert "not in the labelled evidence at all" in section
    assert "### 8.2 Provenance of the retrieval oracle" in section
    # Both revisions are on the page: the legacy figure is kept, not deleted.
    assert "| Legacy label oracle | 211 / 1,540 | 301 / 1,540 |" in section
    assert "| Corrected content+date oracle | 87 / 1,000 | 255 / 1,000 |" in section


def test_a_corrupt_artefact_degrades_to_not_measured() -> None:
    with temp_artefact_dir() as tmp:
        (tmp / sc.CEILING_ARTEFACT).write_text("{not json", encoding="utf-8")
        payload = sc.collect_failure_ceiling(tmp)
    assert payload["status"] == "NOT MEASURED"
