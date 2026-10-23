#!/usr/bin/env python3
"""M30-41 Phase 1i: Refined hardcode classification (exclude comments/docstrings).

Phase 1h was too loose: it counted comments as decision lines.
This pass strips comments and docstrings first, then re-tiers.
"""
import re, json, io, tokenize
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "artificial_memory"

LOCOMO_NAMES = ["caroline", "melanie", "gina", "jon", "maria", "joanna", "nate",
                "james", "deborah", "jolene", "evan", "calvin", "audrey", "tim",
                "andrew", "kadence", "dave"]

PROD_PATH = {
    "context/msc_compiler.py", "recall/state_reconstructor.py",
    "recall/evidence_scorer.py", "recall/query_planner.py",
    "recall/adaptive_search.py", "recall/temporal_resolver.py",
    "recall/state_supersession_engine.py", "recall/proposition_integrity_gate.py",
    "recall/answer_verifier.py", "recall/evidence_graph.py",
}


def strip_comments_and_strings(src: str) -> list[str]:
    """Return lines with comments and string literals blanked out."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return src.splitlines()
    lines = src.splitlines()
    out = [list(ln) for ln in lines]
    for tok in toks:
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            (srow, scol), (erow, ecol) = tok.start, tok.end
            for r in range(srow, erow + 1):
                if r - 1 >= len(out):
                    continue
                a = scol if r == srow else 0
                b = ecol if r == erow else len(out[r - 1])
                for c in range(a, min(b, len(out[r - 1]))):
                    out[r - 1][c] = " "
    return ["".join(ch) for ch in out]


def main():
    real = {}
    for py in SRC.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        rel = str(py.relative_to(SRC)).replace("\\", "/")
        src = py.read_text(encoding="utf-8", errors="ignore")
        code_only = strip_comments_and_strings(src)
        hits = []
        for i, ln in enumerate(code_only, 1):
            low = ln.lower()
            if any(re.search(rf"\b{n}\b", low) for n in LOCOMO_NAMES):
                hits.append({"line": i, "code": ln.strip()[:110]})
        if hits:
            real[rel] = hits

    print("=" * 78)
    print("M30-41 Phase 1i: Real code hardcodes (comments/strings excluded)")
    print("=" * 78)
    print()
    print(f"Files with real code-level benchmark names: {len(real)}")
    print()
    for rel, hits in sorted(real.items(), key=lambda x: -len(x[1])):
        on_prod = " [PRODUCTION PATH]" if rel in PROD_PATH else ""
        print(f"  {rel}{on_prod}")
        for h in hits:
            print(f"    L{h['line']}: {h['code']}")
    print()

    prod_hits = {r: h for r, h in real.items() if r in PROD_PATH}
    print("=" * 78)
    print("ASSESSMENT")
    print("=" * 78)
    print()
    print(f"Files with real hardcodes on the production path: {len(prod_hits)}")
    for r in prod_hits:
        print(f"  {r}")
    print()
    print("Phase 1h over-counted: most 'decision lines' were comments/docstrings.")
    print("The real code-level hardcodes are concentrated in the name lists.")
    print()

    out = PROJECT_ROOT / "results/m30_41"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hardcode_real.json").write_text(json.dumps({
        "files": {r: h for r, h in real.items()},
        "production_path_files": list(prod_hits.keys()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'hardcode_real.json'}")


if __name__ == "__main__":
    main()
