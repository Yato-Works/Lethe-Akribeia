"""Unit tests for the deterministic-engine KPIs on the Apex scorecard (§6).

These pin the contract of the *reporting* layer, not the skills themselves: the
artefacts written by ``commit_sweep.py --report-out`` store raw counts, and the
scorecard must derive every ratio from them, add slices up without averaging
percentages, and say "NOT MEASURED" instead of silently rendering nothing when a
measurement is missing.

The scorecard is a script, not a library module, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCORECARD = REPO / "scripts" / "benchmarks" / "apex_scorecard.py"
COMMITTER_DIR = REPO / "benchmark_results" / "committer_metrics"


def load_scorecard():
    """Import ``scripts/benchmarks/apex_scorecard.py`` as a module."""
    spec = importlib.util.spec_from_file_location("_apex_scorecard_under_test", SCORECARD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = load_scorecard()


@contextmanager
def temp_artefact_dir():
    """A scratch directory for synthetic artefacts.

    The pytest ``tmp_path`` fixture is not used on purpose: its shared temp root
    (``...\\Temp\\pytest-of-<user>``) is not writable in every environment this
    suite runs in, while ``tempfile.mkdtemp`` always is.
    """
    path = Path(tempfile.mkdtemp(prefix="committer-metrics-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def artefact(counts: dict, **extra) -> dict:
    base = {
        "label": "synthetic",
        "suite": "locomo",
        "cache": "benchmark_results/synthetic.jsonl",
        "reader_run": "benchmark_results/locomo1540/synthetic_reader.json",
        "min_turn_score": 0.80,
        "counts": counts,
        "groups": [],
    }
    base.update(extra)
    return base


def test_the_attribution_dump_is_not_read_as_a_second_measurement() -> None:
    """`commit_sweep.py --dump-committed` writes `*_claims.json` into the same
    directory, with a `counts` summary of its own.  Counting it would double the
    slice and invent a 0% reader baseline (it has no paired reader run)."""
    with temp_artefact_dir() as tmp:
        counts = {"n": 100, "claimed": 20, "commit_correct": 18, "reader_correct": 15}
        (tmp / "slice.json").write_text(json.dumps(artefact(counts)), encoding="utf-8")
        (tmp / "slice_claims.json").write_text(json.dumps(artefact(
            {"n": 100, "claimed": 20, "commit_correct": 18},
            records=[{"question_id": "q1", "claimed": True, "correct": True}],
        )), encoding="utf-8")
        payload = sc.collect_committer_metrics(tmp)

    assert [entry["label"] for entry in payload["slices"]] == ["synthetic"]
    assert payload["total"]["n"] == 100
    assert payload["total"]["claimed"] == 20
    assert payload["total"]["reader_correct"] == 15


def test_metrics_are_derived_from_raw_counts() -> None:
    m = sc.committer_metrics({"n": 321, "claimed": 112, "commit_correct": 80,
                              "reader_correct": 75})
    assert m["deterministic_coverage"] == 112 / 321
    assert m["fallback_rate"] == 1 - 112 / 321
    assert m["commit_accuracy"] == 80 / 112
    assert m["reader_same"] == 75 / 112
    # The delta is over the whole slice, not over the claimed subset.
    assert m["commit_delta_pp"] == (80 - 75) / 321 * 100
    assert m["llm_calls_saved"] == 112


def test_metrics_survive_a_committer_that_never_claims() -> None:
    """No division by zero, and a real 0% coverage instead of a missing row."""
    m = sc.committer_metrics({"n": 133, "claimed": 0, "commit_correct": 0, "reader_correct": 0})
    assert m["deterministic_coverage"] == 0.0
    assert m["fallback_rate"] == 1.0
    assert m["commit_accuracy"] is None
    assert m["reader_same"] is None
    assert m["llm_calls_saved"] == 0
    assert sc.pct_or(m["commit_accuracy"]) == "n/a"


def test_totals_add_counts_instead_of_averaging_percentages() -> None:
    """A 90% slice and a 10% slice must not average to 50%."""
    small = sc.committer_metrics({"n": 10, "claimed": 10, "commit_correct": 9,
                                  "reader_correct": 1})
    large = sc.committer_metrics({"n": 90, "claimed": 10, "commit_correct": 1,
                                  "reader_correct": 9})
    assert small["commit_accuracy"] == 0.9 and large["commit_accuracy"] == 0.1

    counts = {"n": 0, "claimed": 0, "commit_correct": 0, "reader_correct": 0}
    for entry in (small, large):
        for key in counts:
            counts[key] += entry[key]
    total = sc.committer_metrics(counts)
    assert total["n"] == 100
    assert total["deterministic_coverage"] == 0.2
    assert total["commit_accuracy"] == 10 / 20  # 50%, not an average of 90% and 10%
    assert total["commit_delta_pp"] == 0.0


def test_collect_reads_artefacts_and_skips_malformed_files() -> None:
    with temp_artefact_dir() as tmp_path:
        (tmp_path / "b.json").write_text(json.dumps(artefact(
            {"n": 100, "claimed": 40, "commit_correct": 30, "reader_correct": 20},
            label="B")), encoding="utf-8")
        (tmp_path / "a.json").write_text(json.dumps(artefact(
            {"n": 10, "claimed": 5, "commit_correct": 5, "reader_correct": 0},
            label="A")), encoding="utf-8")
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "notes.json").write_text(json.dumps({"label": "no counts here"}),
                                             encoding="utf-8")
        (tmp_path / "readme.txt").write_text("ignored", encoding="utf-8")

        payload = sc.collect_committer_metrics(tmp_path)

    assert payload["status"] == "MEASURED"
    assert [s["label"] for s in payload["slices"]] == ["A", "B"]  # sorted, deterministic
    total = payload["total"]
    assert total["n"] == 110 and total["claimed"] == 45
    assert total["commit_accuracy"] == 35 / 45
    assert payload["slices"][1]["metrics"]["commit_delta_pp"] == (30 - 20) / 100 * 100


def test_missing_artefact_directory_reports_not_measured() -> None:
    with temp_artefact_dir() as tmp_path:
        payload = sc.collect_committer_metrics(tmp_path / "does_not_exist")
    assert payload["status"] == "NOT MEASURED"
    assert payload["slices"] == [] and payload["total"] is None
    section = "\n".join(sc.render_committer_section(payload))
    assert "## 6. Deterministic memory engine (answer committer)" in section
    assert "NOT MEASURED" in section


def test_section_renders_the_kpis_and_the_zero_coverage_rows() -> None:
    """The renderer must show every KPI, the total, and the honest 0% rows."""
    with temp_artefact_dir() as tmp_path:
        (tmp_path / "synthetic.json").write_text(json.dumps(artefact(
            {"n": 500, "claimed": 59, "commit_correct": 55, "reader_correct": 56},
            label="LongMemEval - all types",
            suite="lme",
            groups=[
                {"name": "temporal-reasoning",
                 "counts": {"n": 133, "claimed": 59, "commit_correct": 55,
                            "reader_correct": 56}},
                {"name": "multi-session",
                 "counts": {"n": 133, "claimed": 0, "commit_correct": 0,
                            "reader_correct": 0}},
            ])), encoding="utf-8")

        section = "\n".join(sc.render_committer_section(sc.collect_committer_metrics(tmp_path)))

    assert "## 6. Deterministic memory engine (answer committer)" in section
    assert "Commit accuracy" in section and "LLM calls saved" in section
    assert "Total (all measurements)" in section
    assert "### 6.1 Per-capability breakdown" in section
    assert "multi-session" in section
    assert "n/a" in section  # the never-claiming slice has no accuracy, not a fake 0%
    assert "-0.20pp" in section  # (55 - 56) / 500 questions, over the whole slice


def test_render_markdown_always_carries_the_engine_section() -> None:
    """Even without a measurement the scorecard must name the gap."""
    md = sc.render_markdown([], [], {"status": "NOT MEASURED", "slices": [], "total": None,
                                     "artefact_dir": "benchmark_results/committer_metrics"})
    assert "## 6. Deterministic memory engine" in md
    assert "commit_sweep.py --report-out" in md
    # The newest section is the tail now, and it names its own gap too.
    assert "## 8. Failure ceiling" in md
    assert md.rstrip().endswith("`scripts/benchmarks/recount_oracle.py`.")
