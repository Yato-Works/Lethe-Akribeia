"""Verify that all benchmark metrics in RESULTS_REGISTRY.json have corresponding artifacts and match measured values.

Fails with non-zero exit code if any artifact is missing or if any metric diverges
beyond the specified tolerance.

Usage:
    python scripts/benchmarks/verify_registry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "benchmark_results" / "RESULTS_REGISTRY.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_nested_val(data: dict, key_path: str):
    parts = key_path.split(".")
    curr = data
    for p in parts:
        if isinstance(curr, dict) and p in curr:
            curr = curr[p]
        else:
            return None
    return curr


def verify_entry(entry: dict) -> tuple[bool, str]:
    eid = entry["id"]
    art_path = REPO_ROOT / entry["artifact_path"]

    # 1. Existence check
    if not art_path.exists():
        return False, f"[{eid}] Artifact NOT FOUND: {entry['artifact_path']}"

    # 2. JSON parse
    try:
        data = json.loads(art_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"[{eid}] Failed to parse JSON from {art_path}: {e}"

    target = float(entry["target_value"])
    tol = float(entry.get("tolerance", 0.1))
    measured = None

    # 3. Value extraction
    if "json_path" in entry:
        v = get_nested_val(data, entry["json_path"])
        if v is None:
            return False, f"[{eid}] json_path '{entry['json_path']}' not found in artifact"
        measured = float(v)

    elif "json_calc" in entry:
        calc = entry["json_calc"]
        if calc == "counts.commit_correct / counts.claimed * 100":
            counts = data.get("counts", {})
            claimed = counts.get("claimed", 0)
            corr = counts.get("commit_correct", 0)
            if claimed == 0:
                return False, f"[{eid}] claimed is 0"
            measured = (corr / claimed) * 100
        elif calc == "oracle_recall * 100":
            measured = float(data.get("oracle_recall", 0.0)) * 100
        elif calc == "overall_accuracy * 100":
            raw_acc = float(data.get("overall_accuracy", 0.0))
            measured = raw_acc * 100 if raw_acc <= 1.0 else raw_acc
        else:
            return False, f"[{eid}] Unknown json_calc expression: {calc}"

    elif entry.get("custom_eval") == "holdout_convs":
        results = data.get("results", [])
        holdout_corr = 0
        holdout_tot = 0
        for r in results:
            qid = r.get("question_id", "")
            if "conv-42" in qid or "conv-48" in qid:
                holdout_tot += 1
                if r.get("is_correct"):
                    holdout_corr += 1
        if holdout_tot == 0:
            return False, f"[{eid}] No holdout questions found"
        measured = (holdout_corr / holdout_tot) * 100

    elif entry.get("custom_eval") == "dev_convs":
        results = data.get("results", [])
        dev_corr = 0
        dev_tot = 0
        for r in results:
            qid = r.get("question_id", "")
            if "conv-42" not in qid and "conv-48" not in qid:
                dev_tot += 1
                if r.get("is_correct"):
                    dev_corr += 1
        if dev_tot == 0:
            return False, f"[{eid}] No dev questions found"
        measured = (dev_corr / dev_tot) * 100

    elif str(entry.get("custom_eval", "")).startswith("cat"):
        target_cat = int(entry["custom_eval"].replace("cat", ""))
        results = data.get("results", [])
        cat_corr = 0
        cat_tot = 0
        for r in results:
            if int(r.get("category", 0)) == target_cat:
                cat_tot += 1
                if r.get("is_correct"):
                    cat_corr += 1
        if cat_tot == 0:
            return False, f"[{eid}] No category {target_cat} questions found"
        measured = (cat_corr / cat_tot) * 100

    if measured is None:
        return False, f"[{eid}] Could not extract measured value"

    # 4. Numerical validation
    diff = abs(measured - target)
    if diff > tol:
        return False, (
            f"[{eid}] Value mismatch: target={target:.4f}, measured={measured:.4f}, "
            f"diff={diff:.4f} > tol={tol:.4f}"
        )

    # 5. Check counts if provided
    if "sample_count" in entry:
        exp_n = entry["sample_count"]
        act_n = None
        if "results" in data and isinstance(data["results"], list):
            if entry.get("custom_eval") == "holdout_convs":
                act_n = sum(1 for r in data["results"] if "conv-42" in r.get("question_id", "") or "conv-48" in r.get("question_id", ""))
            elif entry.get("custom_eval") == "dev_convs":
                act_n = sum(1 for r in data["results"] if "conv-42" not in r.get("question_id", "") and "conv-48" not in r.get("question_id", ""))
            elif str(entry.get("custom_eval", "")).startswith("cat"):
                target_cat = int(entry["custom_eval"].replace("cat", ""))
                act_n = sum(1 for r in data["results"] if int(r.get("category", 0)) == target_cat)
            else:
                act_n = len(data["results"])
        elif entry.get("json_path") == "holdout.official_f1":
            act_n = data.get("holdout", {}).get("n")
        elif entry.get("json_path") == "dev.official_f1":
            act_n = data.get("dev", {}).get("n")
        elif "total_questions" in data:
            act_n = data["total_questions"]
        elif "shared" in data:
            act_n = data["shared"]
        elif "n" in data:
            act_n = data["n"]
        elif "counts" in data and "claimed" in data["counts"]:
            act_n = data["counts"]["claimed"]
        elif "counts" in data and "n" in data["counts"]:
            act_n = data["counts"]["n"]

        if act_n is not None and act_n != exp_n:
            return False, f"[{eid}] Sample count mismatch: expected {exp_n}, got {act_n}"

    if entry.get("metric_type") == "p_value":
        return True, f"[{eid}] PASS (target={target:.4f}, measured={measured:.4f})"
    return True, f"[{eid}] PASS (target={target:.2f}%, measured={measured:.2f}%)"


def verify_all() -> int:
    if not REGISTRY_PATH.exists():
        print(f"ERROR: Registry file not found at {REGISTRY_PATH}")
        return 1

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    entries = registry.get("entries", [])
    print(f"Verifying {len(entries)} benchmark metrics from {REGISTRY_PATH.name}...\n")

    failed = 0
    passed = 0
    for e in entries:
        ok, msg = verify_entry(e)
        if ok:
            print(f"  ✓ {msg}")
            passed += 1
        else:
            print(f"  ✗ {msg}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Verification Summary: {passed} passed, {failed} failed.")
    print("=" * 60)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(verify_all())
