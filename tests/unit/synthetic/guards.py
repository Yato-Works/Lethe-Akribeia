"""Static guards against benchmark overfitting.

Two complementary checks:

1. ``scan_source_for_overfitting`` — an AST scan of operator/source files that
   flags forbidden patterns:
     - question-id literals (qid == "gpt4_...")
     - embedded answer constants (return 140, return {"Max", ...})
     - benchmark proper-noun dispatchers (if "sneaker" ...)

2. ``check_corpus_is_synthetic`` — asserts a corpus contains NO real benchmark
   question ids or gold answers.

The existing test_no_label_leak.py only checks the PROMPT for gold-label
tokens; it cannot see an answer baked into source as a constant.  These guards
close that hole.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


# Proper nouns / phrases that indicate a benchmark-specific dispatcher.
# Concept words (festival, restaurant) are ALLOWED; specific entities are not.
FORBIDDEN_ENTITY_PATTERNS = [
    r"sneaker", r"hawaii", r"kauai", r"oahu", r"korean", r"starbucks",
    r"phonak", r"accu-chek", r"revolution hall", r"jessica", r"sephora",
    r"austin film festival", r"seattle international film",
    r"portland film festival", r"afi fest",
]

# Question-id shapes used by LongMemEval / LoCoMo.
QID_PATTERNS = [
    r"\bgpt4_[0-9a-f]{8}\b",
    r"\b[0-9a-f]{8}_abs\b",
    r"\b[0-9a-f]{8}\b(?=\s*[\"'])",
]

# Numeric literal that looks like an embedded answer (bare int in a return
# or assignment to an answer-ish name).  Conservative: only bare ints 2-9999
# assigned to suspiciously-named targets.
ANSWER_VAR_NAMES = ("total", "answer", "cnt", "count", "result", "value")


def scan_source_for_overfitting(source: str) -> list[str]:
    """Return a list of human-readable overfitting findings in ``source``."""
    findings: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"SYNTAX_ERROR: {exc}"]

    # 1. entity-token dispatch
    lowered = source.lower()
    for pat in FORBIDDEN_ENTITY_PATTERNS:
        if re.search(pat, lowered):
            # allow if it only appears in a comment or docstring
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if re.search(pat, node.value.lower()):
                        findings.append(f"ENTITY_DISPATCH: forbidden token '{pat}' in string literal")

    # 2. embedded answer constants: `return <int>` or `total = <int>` etc.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id.lower() in ANSWER_VAR_NAMES:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                        if 1 < node.value.value < 10000:
                            findings.append(
                                f"EMBEDDED_ANSWER: {tgt.id} = {node.value.value} (line {node.lineno})"
                            )
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, int) and 1 < node.value.value < 10000:
                findings.append(
                    f"EMBEDDED_ANSWER_RETURN: return {node.value.value} (line {node.lineno})"
                )
            if isinstance(node.value.value, (list, tuple, set)):
                # a literal collection of strings that looks like an answer set
                if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.value.value):
                    if len(node.value.value) >= 2:
                        findings.append(
                            f"EMBEDDED_ANSWER_SET: return {{...}} literal (line {node.lineno})"
                        )

    # 3. question-id literals
    for pat in QID_PATTERNS:
        if re.search(pat, source):
            findings.append(f"QID_LITERAL: matches {pat}")

    return findings


def check_corpus_is_synthetic(items) -> list[str]:
    """Assert a synthetic corpus carries no real benchmark identifiers."""
    findings: list[str] = []
    for it in items:
        blob = f"{it.question} {it.expected} {getattr(it, 'item_id', '')}"
        for pat in QID_PATTERNS:
            if re.search(pat, blob):
                findings.append(f"REAL_QID_IN_SYNTHETIC: {it.item_id} matches {pat}")
        for pat in FORBIDDEN_ENTITY_PATTERNS:
            if re.search(pat, blob.lower()):
                findings.append(f"REAL_ENTITY_IN_SYNTHETIC: {it.item_id} contains '{pat}'")
    return findings


def scan_file(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return [f"MISSING_FILE: {p}"]
    return scan_source_for_overfitting(p.read_text(encoding="utf-8"))
