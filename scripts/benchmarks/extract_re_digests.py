"""Extract the decision-relevant sections from the Letta and Cognee digests.

The subagent digests are complete JSON on disk; the consolidated message showed
only head+tail.  This pulls the sections that decide what Lethe should build:
the [DETERMINISTIC-FEASIBLE] vs [LLM-DEPENDENT] algorithm tags, the integration
ideas, and the license/reuse boundaries.
"""
import json
import re
from pathlib import Path

FILES = {
    "LETTA": Path("C:/Users/smily/AppData/Local/hermes/cache/delegation/"
                  "subagent-summary-0-20261013_221306_043743.txt"),
    "COGNEE": Path("C:/Users/smily/AppData/Local/hermes/cache/delegation/"
                   "subagent-summary-1-20261013_221306_048465.txt"),
}

SECTIONS = (
    "3. Algorithms", "4. What its tests", "5. Benchmarks", "6. License",
    "7. Top 3 integration", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.",
)


def load(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip().rstrip("`").strip()
    data = json.loads(raw)
    return data["digest_markdown"]


def main() -> int:
    for name, path in FILES.items():
        md = load(path)
        print("=" * 78)
        print(f"### {name}   ({len(md)} chars)")
        print("=" * 78)
        # Everything from the first tagged/decision section onward.
        start = len(md)
        for key in SECTIONS:
            i = md.find(key)
            if i != -1:
                start = min(start, i)
        if start < len(md):
            print(md[start:])
        else:
            print("(no decision section found -- printing tail)")
            print(md[-3000:])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
