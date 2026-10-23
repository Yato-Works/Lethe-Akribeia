#!/usr/bin/env python3
"""M30-41 Phase 1i-v2: Real code hardcodes — strip only comments and DOCSTRINGS.

v1 blanked all string literals, which erased the very name lists we want to
find. Correct approach: blank COMMENT tokens and docstring STRING tokens only;
keep ordinary string literals (which is where KNOWN_ACTORS lives).
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


def blank_spans(src, spans):
    lines = [list(ln) for ln in src.splitlines()]
    for (srow, scol), (erow, ecol) in spans:
        for r in range(srow, erow + 1):
            if r - 1 >= len(lines):
                continue
            a = scol if r == srow else 0
            b = ecol if r == erow else len(lines[r - 1])
            for c in range(a, min(b, len(lines[r - 1]))):
                lines[r - 1][c] = " "
    return ["".join(ch) for ch in lines]


def strip_comments_and_docstrings(src):
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return src.splitlines(), []
    spans = []
    # docstring = STRING token that is the first statement of a module/def/class
    prev_significant = None
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            spans.append((tok.start, tok.end))
        elif tok.type == tokenize.STRING:
            if prev_significant in ("INDENT", "NEWLINE", None, "DEDENT", ":"):
                spans.append((tok.start, tok.end))
            prev_significant = "STRING"
            continue
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.COMMENT):
            prev_significant = tokenize.tok_name.get(tok.type, str(tok.type))
    return blank_spans(src, spans), spans


def main():
    real = {}
    for py in SRC.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        rel = str(py.relative_to(SRC)).replace("\\", "/")
        src = py.read_text(encoding="utf-8", errors="ignore")
        code_only, _ = strip_comments_and_docstrings(src)
        hits = []
        for i, ln in enumerate(code_only, 1):
            low = ln.lower()
            found = [n for n in LOCOMO_NAMES if re.search(rf"\b{n}\b", low)]
            if found:
                hits.append({"line": i, "names": found, "code": ln.strip()[:110]})
        if hits:
            real[rel] = hits

    print("=" * 78)
    print("M30-41 Phase 1i-v2: Real code hardcodes (comments+docstrings excluded)")
    print("=" * 78)
    print()
    print(f"Files with real code-level benchmark names: {len(real)}")
    print()
    for rel, hits in sorted(real.items(), key=lambda x: -len(x[1])):
        on_prod = " [PRODUCTION PATH]" if rel in PROD_PATH else ""
        print(f"  {rel}{on_prod}  ({len(hits)} lines)")
        for h in hits[:4]:
            print(f"    L{h['line']}: {h['code']}")
        if len(hits) > 4:
            print(f"    ... +{len(hits)-4} more")
    print()

    prod = {r: h for r, h in real.items() if r in PROD_PATH}
    print("=" * 78)
    print("ASSESSMENT")
    print("=" * 78)
    print()
    print(f"Production-path files with real code hardcodes: {len(prod)}")
    for r in prod:
        print(f"  {r}")
    print()

    out = PROJECT_ROOT / "results/m30_41"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hardcode_real_v2.json").write_text(json.dumps({
        "files": real,
        "production_path_files": list(prod.keys()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'hardcode_real_v2.json'}")


if __name__ == "__main__":
    main()
