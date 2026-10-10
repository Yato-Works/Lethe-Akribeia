"""Anti-cheat auditor: detect benchmark-answer contamination in any prompt builder.

THIS IS A GUARD, NOT A FIXER.  It reports; it never edits.  Run it before
trusting any benchmark number.

Checks performed:
  1. GOLD-LITERAL LEAK  -- a string literal in a prompt-building function that
     is an exact match for a gold answer in the dataset.
  2. QUESTION-CONTENT BRANCH -- control flow keyed on the question's own text
     (if "charity race" in ql: ...).  Generic TYPE checks (startswith("where"))
     are fine; content checks are not.
  3. ANSWER-SHAPED RETURNS -- early returns of literal strings inside a
     prompt/answer builder.
"""
import ast
import glob
import json
import os
import re
import sys

PROMPT_FUNC_HINTS = ("prompt", "build_prompt", "reader", "build_reader", "answer_contract", "_COT")
DATASETS = {
    "locomo": "datasets/external/locomo10.json",
    "longmemeval": "datasets/external/longmemeval_s_cleaned.json",
}

# Question-content branches are keyed on >=2 real words of question wording.
CONTENT_BRANCH_RE = re.compile(r'if\s+["\']([a-z0-9][a-z0-9 ]{5,40})["\']\s+in\s+(q\b|ql\b|q_lower\b|question_lower\b|question\.lower)')
GENERIC_PREFIX_OK = re.compile(r"^(where|when|how many|how much|who|what|which|why|how)\b")


def load_golds():
    """Collect gold answers per dataset."""
    golds = {}
    # LoCoMo
    p = DATASETS["locomo"]
    if os.path.exists(p):
        s = set()
        for conv in json.load(open(p, encoding="utf-8")):
            for q in conv.get("qa", []):
                for k in ("answer", "adversarial_answer"):
                    v = q.get(k)
                    if isinstance(v, str) and v.strip():
                        s.add(v.strip())
        golds["locomo"] = s
    # LongMemEval
    p = DATASETS["longmemeval"]
    if os.path.exists(p):
        try:
            data = json.load(open(p, encoding="utf-8"))
        except Exception:
            data = []
        if isinstance(data, dict):
            data = data.get("data", [])
        s = set()
        for item in data:
            a = item.get("answer")
            if isinstance(a, str) and a.strip():
                s.add(a.strip())
        golds["longmemeval"] = s
    return golds


def iter_string_literals(node):
    """Yield (lineno, value) for every string literal under node."""
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            yield n.lineno, n.value
        # concatenated f-strings / adjacent literals inside JoinedStr
        elif isinstance(n, ast.FormattedValue):
            pass


def audit_file(path, golds):
    """Return list of findings for one python file."""
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception as e:
        return [{"kind": "parse_error", "path": path, "detail": str(e)[:120]}]

    findings = []
    lines = src.split("\n")

    def docstring_of(fn):
        d = ast.get_docstring(fn)
        return d or ""

    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        name = fn.name.lower()
        if not any(h in name for h in PROMPT_FUNC_HINTS):
            continue
        # skip obvious non-prompt helpers
        if name.endswith("_test"):
            continue
        ds = docstring_of(fn)
        ds_lines = set()
        for i, l in enumerate(lines, 1):
            if l.strip().startswith(("#", '"""', "'''")) or '"""' in l or "'''" in l:
                ds_lines.add(i)

        # 1. gold-literal leak (exclude lines inside comments/docstrings)
        for lineno, lit in iter_string_literals(fn):
            if lineno in ds_lines:
                continue
            v = lit.strip()
            # Only WHOLE-literal matches count as a leak.  A prompt builder
            # embedding a gold answer writes it as its own string
            # ("'Sweden'" or "'The Friday before 15 July 2023'"), never as a
            # passing substring inside a multi-line instruction paragraph --
            # and substring matching false-positives on exactly those.
            if len(v) < 3:
                continue
            for ds_name, gset in golds.items():
                if v in gset:
                    # Too-generic tokens that are legitimately part of every
                    # yes/no prompt must not trip the guard.
                    if v.lower() in (
                        "yes", "no", "unknown", "likely no", "yes.", "no.",
                        "yes,", "no,",
                    ):
                        continue
                    findings.append({
                        "kind": "GOLD_LITERAL_LEAK",
                        "path": path,
                        "func": fn.name,
                        "line": lineno,
                        "dataset": ds_name,
                        "literal": v[:70],
                    })
                    break

        # 2. question-content branch
        body_src = "\n".join(lines[fn.lineno - 1: (fn.end_lineno or fn.lineno)])
        for m in CONTENT_BRANCH_RE.finditer(body_src):
            phrase = m.group(1).strip()
            if GENERIC_PREFIX_OK.match(phrase):
                continue
            if len(phrase.split()) < 2:
                continue
            findings.append({
                "kind": "QUESTION_CONTENT_BRANCH",
                "path": path,
                "func": fn.name,
                "phrase": phrase,
            })

        # 3. answer-shaped return inside a prompt builder
        for n in ast.walk(fn):
            if isinstance(n, ast.Return) and n.value is not None:
                if isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                    v = n.value.value.strip()
                    if 3 <= len(v) <= 40 and v.lower() not in ("yes", "no", "unknown"):
                        findings.append({
                            "kind": "LITERAL_RETURN",
                            "path": path,
                            "func": fn.name,
                            "line": n.lineno,
                            "literal": v[:60],
                        })
    return findings


def main():
    golds = load_golds()
    total = {}
    for k, v in golds.items():
        total[k] = len(v)
    print(f"[audit] gold pools: {total}")

    targets = []
    for pat in ("scripts/benchmarks/**/*.py", "src/**/*.py"):
        targets.extend(glob.glob(pat, recursive=True))
    targets = [t for t in targets if not t.endswith("_refactor_prompt.py")]

    all_findings = []
    for p in sorted(set(targets)):
        all_findings.extend(audit_file(p, golds))

    # Group and print
    from collections import Counter, defaultdict
    by_kind = defaultdict(list)
    for f in all_findings:
        by_kind[f["kind"]].append(f)

    for kind in ("GOLD_LITERAL_LEAK", "QUESTION_CONTENT_BRANCH", "LITERAL_RETURN", "parse_error"):
        fl = by_kind.get(kind, [])
        print(f"\n=== {kind}: {len(fl)} ===")
        for f in fl[:40]:
            if kind == "GOLD_LITERAL_LEAK":
                print(f"  {f['path']}:{f['line']} [{f['dataset']}] {f['func']}() -> {f['literal']!r}")
            elif kind == "QUESTION_CONTENT_BRANCH":
                print(f"  {f['path']} [{f['func']}()] if {f['phrase']!r} in <question>")
            elif kind == "LITERAL_RETURN":
                print(f"  {f['path']}:{f['line']} [{f['func']}()] returns {f['literal']!r}")
            else:
                print(f"  {f['path']}: {f['detail']}")

    # verdict
    critical = by_kind.get("GOLD_LITERAL_LEAK", []) + by_kind.get("QUESTION_CONTENT_BRANCH", [])
    print("\n" + "=" * 60)
    print(f"VERDICT: {'CLEAN' if not critical else f'{len(critical)} CRITICAL FINDING(S)'}")
    return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(main())
