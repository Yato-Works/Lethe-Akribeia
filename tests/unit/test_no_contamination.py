"""Guard: the benchmark harness must never contain a dataset gold answer.

WHY THIS EXISTS
---------------
An earlier version of the LoCoMo reader prompt hardcoded 20 dataset gold
answers and branched on question wording ("if 'charity race' in ql: ...").
That is cheating: the model was handed the answer instead of deriving it from
evidence.  It silently inflated the LoCoMo headline by up to 2.7pt and was
only caught by a manual corpus audit.

This test makes that class of bug FAIL THE BUILD, permanently.

The audit is AST-based (tests/unit/audit_contamination.py) so it inspects real
string literals and real control flow -- not a grep that false-positives on
comments and docstrings.
"""
import os
import subprocess
import sys

AUDITOR = os.path.join(os.path.dirname(__file__), "audit_contamination.py")


def test_no_benchmark_answer_contamination():
    """Any GOLD_LITERAL_LEAK or QUESTION_CONTENT_BRANCH fails the build."""
    assert os.path.exists(AUDITOR), "contamination auditor missing"
    proc = subprocess.run(
        [sys.executable, AUDITOR],
        capture_output=True,
        text=True,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    out = proc.stdout + proc.stderr
    assert "VERDICT: CLEAN" in out, (
        "Benchmark contamination detected -- answers are being handed to the "
        "model. Fix the findings below before trusting any score:\n" + out
    )
    assert proc.returncode == 0, out
