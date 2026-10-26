"""
Evaluate Multi-Session Coverage Engine (R6 Core Phase 1) on Dev Multi-Session 67.
Runs end-to-end with OllamaAnswerer (Qwen2.5 7B).
"""

import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
qids_path = REPO / "benchmark_results" / "splits" / "lme_dev_ms_qids.json"
qids = json.loads(qids_path.read_text(encoding="utf-8"))
qids_str = ",".join(qids)

out_file = REPO / "benchmark_results" / "pipeline_v4" / "lme_dev_ms67_R6_coverage.json"

cmd = [
    sys.executable,
    str(REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py"),
    "--cond", "R6",
    "--fresh",
    "--qids", qids_str,
    "--out", str(out_file),
]

print("Executing Multi-Session 67 evaluation under R6 Multi-Session Coverage Engine...")
res = subprocess.run(cmd, cwd=str(REPO))
sys.exit(res.returncode)
