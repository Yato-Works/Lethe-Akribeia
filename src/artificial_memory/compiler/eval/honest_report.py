"""Honest reporting for A/B benchmark comparisons.

WHY THIS EXISTS
---------------
An experiment report presented this as "conditional adoption":

    slice                     baseline   ece    delta
    Gate 1 (conv-26 Q0-49)        76%      78%    +2%  OK
    Gate 2a (conv-30 Q0-49)       62%      62%     0%  OK
    Gate 2b (conv-26 Q50-99)      58%      56%    -2%  WARN

Running McNemar on the actual paired predictions showed:

    slice        base    ece    delta     b    c        p   verdict
    Gate 1      78.0%  78.0%   +0.0pt    0    0  1.0000   NOISE
    Gate 2a      62.0%  62.0%   +0.0pt    0    0  1.0000   NOISE
    Gate 2b      58.0%  56.0%   -2.0pt    1    0  1.0000   NOISE

b=0/c=0 means the two implementations produced IDENTICAL predictions -- the
gate never fired at all.  The "-2%" was a single question.  There was no
effect, and reporting it as "conditional adoption" dressed up zero effect as
a finding.

This module makes that impossible: every reported delta carries a McNemar
p-value, and anything not significant is labelled NOISE rather than OK.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from artificial_memory.compiler.eval.stat_test import StatisticalTester  # noqa: E402

ALPHA = 0.05


@dataclass(frozen=True)
class SliceVerdict:
    """Honest verdict for one A/B slice."""
    name: str
    n_paired: int
    acc_a: float
    acc_b: float
    delta_pt: float
    b: int
    c: int
    p_value: float
    significant: bool
    verdict: str

    def as_row(self) -> str:
        return (
            f"{self.name:38s} {self.acc_a*100:5.1f}% {self.acc_b*100:5.1f}% "
            f"{self.delta_pt:+6.1f}pt {self.b:3d} {self.c:3d} {self.p_value:7.4f}  {self.verdict}"
        )


def load_results(path: str | Path) -> dict[str, dict]:
    """Load a result file and index its rows by question_id."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing result file: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data.get("results", data) if isinstance(data, dict) else data
    out: dict[str, dict] = {}
    for r in rows:
        qid = r.get("question_id") or r.get("id")
        if qid is not None:
            out[str(qid)] = r
    return out


def compare_slices(
    pairs: list[tuple[str, str | Path, str | Path]],
    alpha: float = ALPHA,
    tester: StatisticalTester | None = None,
) -> list[SliceVerdict]:
    """Compare baseline vs treatment per slice, with McNemar significance.

    Pairing is by question_id -- never by list position -- so a reordered or
    partially-rerun result file cannot manufacture a fake delta.
    """
    tester = tester or StatisticalTester(alpha=alpha)
    verdicts: list[SliceVerdict] = []

    for name, path_a, path_b in pairs:
        try:
            a = load_results(path_a)
            b = load_results(path_b)
        except FileNotFoundError:
            verdicts.append(SliceVerdict(
                name=name, n_paired=0, acc_a=0.0, acc_b=0.0, delta_pt=0.0,
                b=0, c=0, p_value=1.0, significant=False,
                verdict="MISSING FILE",
            ))
            continue

        ids = sorted(set(a) & set(b))
        if not ids:
            verdicts.append(SliceVerdict(
                name=name, n_paired=0, acc_a=0.0, acc_b=0.0, delta_pt=0.0,
                b=0, c=0, p_value=1.0, significant=False,
                verdict="NO PAIRED QUESTIONS",
            ))
            continue

        ra = [bool(a[i].get("is_correct")) for i in ids]
        rb = [bool(b[i].get("is_correct")) for i in ids]
        acc_a = sum(ra) / len(ra)
        acc_b = sum(rb) / len(rb)
        m = tester.mcnemar_test(ra, rb)
        delta_pt = (acc_b - acc_a) * 100

        if m.n_discordant == 0:
            verdict = "NO EFFECT (identical predictions)"
        elif not m.significant:
            verdict = "NOISE (not significant)"
        elif delta_pt > 0:
            verdict = "SIGNIFICANT IMPROVEMENT"
        else:
            verdict = "SIGNIFICANT REGRESSION"

        verdicts.append(SliceVerdict(
            name=name, n_paired=len(ids),
            acc_a=acc_a, acc_b=acc_b, delta_pt=delta_pt,
            b=m.b, c=m.c, p_value=m.p_value,
            significant=m.significant, verdict=verdict,
        ))

    return verdicts


def format_report(verdicts: list[SliceVerdict], title: str) -> str:
    """Render a table that cannot be misread as 'conditional adoption'."""
    lines = [
        "=" * 96,
        title,
        "=" * 96,
        f"{'slice':38s} {'base':>6s} {'treat':>6s} {'delta':>8s} "
        f"{'b':>3s} {'c':>3s} {'p':>7s}  verdict",
        "-" * 96,
    ]
    for v in verdicts:
        lines.append(v.as_row())
    lines += [
        "-" * 96,
        "b = baseline-only correct, c = treatment-only correct (discordant pairs)",
        f"Significant at alpha={ALPHA}.  NOISE means the delta is indistinguishable from chance.",
        "n_discordant == 0 means the two implementations produced IDENTICAL predictions.",
        "=" * 96,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ECE = [
        ("Gate 1 (conv-26 Q0-49, dev)",
         "benchmark_results/experiment_ece_gate1/locomo_ece_gate1_latest.json",
         "benchmark_results/experiment_ece_gate1/locomo_ece_gate1_20261011_170926.json"),
        ("Gate 2a (conv-30 Q0-49, holdout conv)",
         "benchmark_results/experiment_ece_gate2/locomo_baseline_gate2_conv-30_20261011_171637.json",
         "benchmark_results/experiment_ece_gate2/locomo_ece_gate2_conv-30_20261011_171637.json"),
        ("Gate 2b (conv-26 Q50-99, holdout Qs)",
         "benchmark_results/experiment_ece_gate2/locomo_baseline_gate2_conv-26_off50_20261011_172207.json",
         "benchmark_results/experiment_ece_gate2/locomo_ece_gate2_conv-26_off50_20261011_172207.json"),
    ]
    root = os.environ.get("LETHE_ROOT", str(Path.cwd()))
    pairs = [(n, os.path.join(root, a), os.path.join(root, b)) for n, a, b in ECE]
    vs = compare_slices(pairs)
    print(format_report(vs, "ECE SIGNIFICANCE AUDIT (McNemar, paired by question_id)"))
