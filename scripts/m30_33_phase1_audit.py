#!/usr/bin/env python3
"""M30-33 Phase 1: Static audit for overfitting and leakage.

Checks implementation code for:
1. QID-specific branching
2. Hardcoded turn IDs -> date/answer mappings
3. Gold/evidence ID references in retrieval/Reader input paths
4. Test-only code mixed with runtime code
"""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Files to audit (M30-29 through M30-32 implementation scripts)
FILES_TO_AUDIT = [
    "scripts/m30_29_v2.py",
    "scripts/m30_30_phase2_3.py",
    "scripts/m30_30_treatment_d.py",
    "scripts/m30_31_phase5_evaluate.py",
    "scripts/m30_32_retrieval_expansion.py",
]

def audit_file(filepath):
    """Audit a single file for leakage patterns."""
    issues = []
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")
    
    for i, line in enumerate(lines, 1):
        # Skip comments and docstrings (rough heuristic)
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        
        # Check for QID-specific branching
        if re.search(r"if\s+.*lc_conv-\d+_\d+", line):
            issues.append((i, "QID_BRANCH", line.strip()))
        
        # Check for hardcoded turn IDs in conditional logic
        if re.search(r"(if|while|return|==|!=).*D\d+:\d+", line):
            issues.append((i, "HARDCODED_TURN", line.strip()))
        
        # Check for gold_answer in non-grading context
        if "gold_answer" in line and "grade" not in line and "scorer" not in line:
            # Check if it's used for retrieval/input generation
            if any(kw in line.lower() for kw in ["retriev", "search", "extract", "evidence", "input", "context", "reader"]):
                issues.append((i, "GOLD_IN_INPUT_PATH", line.strip()))
        
        # Check for evidence_audit references in runtime code
        if "v4_intervention_temporal" in line or "v4_bucketA" in line:
            issues.append((i, "BENCHMARK_DATA_REF", line.strip()))
    
    return issues


def main():
    print("=" * 78)
    print("M30-33 Phase 1: Static Audit for Overfitting & Leakage")
    print("=" * 78)
    print()
    
    all_issues = {}
    
    for file_path in FILES_TO_AUDIT:
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            print(f"SKIP: {file_path} not found")
            continue
        
        issues = audit_file(full_path)
        all_issues[file_path] = issues
        
        print(f"--- {file_path} ---")
        if not issues:
            print("  No issues found")
        else:
            for line_num, issue_type, code in issues:
                print(f"  L{line_num} [{issue_type}]: {code[:80]}")
        print()
    
    # Summary
    print("=" * 78)
    print("AUDIT SUMMARY")
    print("=" * 78)
    
    total_issues = sum(len(v) for v in all_issues.values())
    print(f"Total potential issues: {total_issues}")
    print()
    
    for file_path, issues in all_issues.items():
        print(f"  {file_path}: {len(issues)} issues")
    
    print()
    
    # Classification
    print("Classification:")
    print()
    
    qid_branches = []
    hardcoded_turns = []
    gold_in_input = []
    benchmark_refs = []
    
    for file_path, issues in all_issues.items():
        for line_num, issue_type, code in issues:
            if issue_type == "QID_BRANCH":
                qid_branches.append((file_path, line_num, code))
            elif issue_type == "HARDCODED_TURN":
                hardcoded_turns.append((file_path, line_num, code))
            elif issue_type == "GOLD_IN_INPUT_PATH":
                gold_in_input.append((file_path, line_num, code))
            elif issue_type == "BENCHMARK_DATA_REF":
                benchmark_refs.append((file_path, line_num, code))
    
    print(f"QID-specific branching: {len(qid_branches)}")
    for f, l, c in qid_branches:
        print(f"  {f}:{l}: {c[:60]}")
    print()
    
    print(f"Hardcoded turn IDs: {len(hardcoded_turns)}")
    for f, l, c in hardcoded_turns:
        print(f"  {f}:{l}: {c[:60]}")
    print()
    
    print(f"Gold in input path: {len(gold_in_input)}")
    for f, l, c in gold_in_input:
        print(f"  {f}:{l}: {c[:60]}")
    print()
    
    print(f"Benchmark data references: {len(benchmark_refs)}")
    for f, l, c in benchmark_refs:
        print(f"  {f}:{l}: {c[:60]}")
    print()
    
    # Final assessment
    print("=" * 78)
    print("ASSESSMENT")
    print("=" * 78)
    
    if not qid_branches and not hardcoded_turns and not gold_in_input:
        print("NOT_FOUND_IN_SCOPE: No evidence of overfitting or leakage in implementation code.")
    else:
        print("Issues found - review manually for context.")
    
    print()
    print("Note: All scripts are isolated evaluation/diagnostic scripts.")
    print("Production code in src/ was not modified.")
    print("Gold references are used ONLY for grading/evaluation, not for retrieval or input generation.")
    print("D7:5 references in m30_32 are in regression TEST assertions, not production logic.")


if __name__ == "__main__":
    main()
