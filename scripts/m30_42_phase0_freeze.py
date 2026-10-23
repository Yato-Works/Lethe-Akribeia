#!/usr/bin/env python3
"""M30-42 Phase 0: Freeze the current code state (hashes only, no edits).

Records git HEAD, branch, working-tree status, and sha256 of the files on the
production path that carry benchmark-specific tokens.
"""
import hashlib, json, subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "artificial_memory"

FILES = [
    "recall/evidence_scorer.py",
    "recall/state_reconstructor.py",
    "recall/query_planner.py",
    "recall/evidence_graph.py",
    "memory/persona_store.py",
    "context/msc_compiler.py",
    "recall/adaptive_search.py",
    "recall/proposition_integrity_gate.py",
    "recall/answer_verifier.py",
    "compiler/speaker_normalizer.py",
    "compiler/closure/planner.py",
    "research/benchmarks/external/locomo_adapter.py",
]


def sh(cmd):
    try:
        return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                              shell=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def main():
    state = {
        "git_head": sh("git rev-parse HEAD"),
        "git_branch": sh("git branch --show-current"),
        "git_status_porcelain": sh("git status --porcelain").splitlines(),
        "files": {},
    }
    for rel in FILES:
        p = SRC / rel
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            state["files"][rel] = {"sha256": h[:16], "lines": len(p.read_text(encoding='utf-8', errors='ignore').splitlines())}

    print("=" * 78)
    print("M30-42 Phase 0: Code freeze record")
    print("=" * 78)
    print(f"HEAD:   {state['git_head']}")
    print(f"Branch: {state['git_branch']}")
    print(f"Modified files: {len(state['git_status_porcelain'])}")
    print()
    print("Production-path file hashes:")
    for rel, info in state["files"].items():
        print(f"  {rel:55s} {info['sha256']}  ({info['lines']} lines)")
    print()

    out = PROJECT_ROOT / "results/m30_42"
    out.mkdir(parents=True, exist_ok=True)
    (out / "freeze_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'freeze_state.json'}")


if __name__ == "__main__":
    main()
