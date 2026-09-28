"""Scratch comparison: two LongMemEval reports, question by question (read-only).

Used to test the hypothesis "the previous run lost accuracy because Ollama's default
context window truncated the prompt": if every prediction is byte-identical between
the default-context run and the explicit ``num_ctx 8192`` run, no truncation can have
occurred and the accuracy gap has a different cause.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["question_id"]: r for r in data["results"]}, data


def main() -> None:
    old_path = sys.argv[1]
    new_path = sys.argv[2]
    old, old_meta = load(old_path)
    new, new_meta = load(new_path)

    print(f"A = {old_path}  num_ctx={old_meta.get('reader_num_ctx')}  acc={old_meta['overall_accuracy']*100:.2f}%")
    print(f"B = {new_path}  num_ctx={new_meta.get('reader_num_ctx')}  acc={new_meta['overall_accuracy']*100:.2f}%")

    shared = sorted(set(old) & set(new))
    identical = [q for q in shared if str(old[q]["predicted_answer"]) == str(new[q]["predicted_answer"])]
    flipped = [q for q in shared if q not in set(identical)]
    print(f"\nquestions compared: {len(shared)}")
    print(f"identical predictions: {len(identical)} ({len(identical)/len(shared)*100:.1f}%)")
    print(f"different predictions: {len(flipped)}")
    for q in flipped[:10]:
        print(f"  {q} [{new[q]['question_type']}]")
        print(f"    A: {str(old[q]['predicted_answer'])[:70]}")
        print(f"    B: {str(new[q]['predicted_answer'])[:70]}")

    big = [q for q in shared if (old[q]["tokens_used"] or 0) > 8192]
    print(f"\nquestions whose prompt exceeded 8192 tokens: {len(big)}")
    print("verify: all such prompts produced identical output in A and B:",
          all(q in set(identical) for q in big))


if __name__ == "__main__":
    main()
