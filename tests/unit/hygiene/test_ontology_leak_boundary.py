"""Ontology leak-boundary guard.

Watched corpus fingerprints (ontology/data/leak_watch.yaml) must not appear
in *runtime logic*. Excluded from the scan, by design:

  - research/** benchmark harness plumbing,
  - the ontology data plane itself (ontology/data/*.yaml),
  - pure comment lines and docstrings (documentation may name behavior
    examples; executable code may not),
  - tests, scripts and benchmark result artifacts.

Remaining references live in a justified, *shrinking* allowlist
(tests/unit/hygiene/ontology_leak_allowlist.yaml). Any UNLISTED occurrence
fails this test, so benchmark vocabulary cannot silently re-enter runtime
code during future development. The goal state is an empty allowlist.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "artificial_memory"
_WATCH_FILE = _SRC_ROOT / "ontology" / "data" / "leak_watch.yaml"
_ALLOWLIST_FILE = Path(__file__).resolve().parent / "ontology_leak_allowlist.yaml"

# Out-of-scope subtrees: benchmark plumbing, the data plane and caches.
_EXCLUDED_PARTS = (
    ("research",),
    ("ontology", "data"),
    ("__pycache__",),
)


@dataclass(frozen=True)
class Leak:
    file: str
    token: str
    line_no: int
    line: str

    def describe(self) -> str:
        return f"{self.file}:{self.line_no}: {self.token!r} in {self.line.strip()!r}"


def _load_watched_tokens() -> list[str]:
    doc = yaml.safe_load(_WATCH_FILE.read_text(encoding="utf-8"))
    tokens: list[str] = []
    for group in doc.values():
        if isinstance(group, dict):
            for values in group.values():
                if isinstance(values, list):
                    tokens.extend(values)
    # Deterministic order, longest-first so multi-word fingerprints win.
    return sorted(set(tokens), key=lambda t: (-len(t), t))


def _load_allowlist() -> dict[str, str]:
    doc = yaml.safe_load(_ALLOWLIST_FILE.read_text(encoding="utf-8"))
    allowed: dict[str, str] = {}
    for section, entries in doc.items():
        if isinstance(entries, dict):
            for rel_path, meta in entries.items():
                rel = str(rel_path).replace("\\", "/")
                if rel.startswith("src/"):
                    rel = rel[len("src/"):]
                allowed[rel] = str((meta or {}).get("reason", section))
    return allowed


def _is_excluded(path: Path) -> bool:
    rel = path.relative_to(_SRC_ROOT).parts
    for excl in _EXCLUDED_PARTS:
        if rel[: len(excl)] == excl:
            return True
    return False


def _docstring_line_ranges(tree: ast.AST) -> set[int]:
    """Line numbers covered by module / class / function docstrings."""
    ranges: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc and node.body and isinstance(node.body[0], ast.Expr):
                const = node.body[0]
                if isinstance(const.value, ast.Constant) and isinstance(const.value.value, str):
                    start = const.lineno
                    end = getattr(const, "end_lineno", start)
                    ranges.update(range(start, end + 1))
    return ranges


def _scan() -> list[Leak]:
    tokens = _load_watched_tokens()
    patterns = [
        (t, re.compile(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])", re.IGNORECASE))
        for t in tokens
    ]
    allowlist = _load_allowlist()
    leaks: list[Leak] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        if _is_excluded(path):
            continue
        rel_path = str(path.relative_to(_SRC_ROOT.parent)).replace("\\", "/")
        if rel_path in allowlist:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            doc_lines = _docstring_line_ranges(ast.parse(text))
        except SyntaxError:
            doc_lines = set()
        for line_no, line in enumerate(text.splitlines(), 1):
            if len(line) > 4000:
                continue
            # Comments and docstrings document behavior examples; executable
            # logic may not embed corpus vocabulary.
            if line.lstrip().startswith("#") or line_no in doc_lines:
                continue
            for token, pat in patterns:
                if pat.search(line):
                    leaks.append(Leak(rel_path, token, line_no, line))
    return leaks


def test_runtime_logic_is_free_of_unlisted_corpus_fingerprints() -> None:
    """Every watched fingerprint occurrence in runtime logic must be allowlisted."""
    leaks = _scan()
    assert not leaks, (
        "Unlisted corpus fingerprints leaked into runtime logic:\n"
        + "\n".join(f"  - {lk.describe()}" for lk in leaks)
        + "\n\nIf this is a sanctioned migration-in-progress, add the file to "
        "tests/unit/hygiene/ontology_leak_allowlist.yaml with a justification. "
        "Otherwise move the vocabulary to the ontology data plane "
        "(artificial_memory/ontology/data/)."
    )


def test_allowlist_entries_point_at_existing_files() -> None:
    """Allowlist must not reference deleted/moved files (shrinking invariant)."""
    allowlist = _load_allowlist()
    stale = [rel for rel in allowlist if not (_SRC_ROOT.parent / rel).is_file()]
    assert not stale, (
        "Stale allowlist entries (file no longer exists) - remove them to keep "
        f"the allowlist shrinking: {stale}"
    )


def test_watched_token_registry_loads() -> None:
    tokens = _load_watched_tokens()
    assert len(tokens) >= 16, "leak_watch.yaml unexpectedly shrank or emptied"
    assert "matt patterson" in tokens and "caroline" in tokens
