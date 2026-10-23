"""Phase C attribute failures -- classify the 19 wrong answers by OBSERVED cause.

The 19 preference/attribute wrong answers look like one disease but the `why` strings
show several.  Rather than trust the eye, each is classified mechanically against the
evidence turn, and the classifications are counted.

Candidate causes, each with a falsifiable test against the frozen context:

  NO_OBJECT_NOUN     the question names an object noun (movie, dish, genre, recipe)
                     and the extracted value is NOT that noun -- 'action' answering
                     'what movie', 'cooking' answering 'what recipe'
  GENRE_NOT_INSTANCE the value is a CATEGORY word (action, fantasy, sci-fi) where an
                     INSTANCE was asked ('what is the favourite MOVIE' -> 'Xenoblade')
  DEICTIC           the value is a pronoun/determiner ('this series', 'the smell',
                     'how's', 'pizza too') -- no noun phrase at all
  PARTIAL_SET       a set was extracted but an item is missing or wrong ('Sapiens,
                     Detroit' vs 'Sapiens, Avalanche by Neal Stephenson')
  WRONG_ATTRIBUTE   the extractor read a different attribute entirely (a console
                     'switch' answering a game question)
  OTHER             none of the above

Controls: the correct answers in the same shape are run through the same classifier,
because a discriminator that also fires on the correct set is not a discriminator.

Production code is NOT modified here.  This only locates the defect.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory_gen.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory_gen.skills.autonomous_engines.entity_attribute_resolver import (  # noqa: E402
    EntityAttributeResolver,
)
from artificial_memory_gen.skills.deterministic_memory_engine import (  # noqa: E402
    DeterministicMemoryEngine,
)

spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = scorer
try:
    spec.loader.exec_module(scorer)
except SystemExit:
    pass

BASELINE = ROOT / "benchmark_results/evidence_audit/phase_c_baseline.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/attribute_failure_causes.json"

#: Object nouns a 'what X' question asks for.
OBJECT_NOUNS = {
    "movie", "movies", "film", "films", "book", "books", "series", "trilogy",
    "dish", "dishes", "recipe", "recipes", "food", "meal", "pizza", "dessert",
    "treat", "game", "games", "show", "shows", "song", "songs", "artist",
    "artists", "band", "bands", "genre", "hobby", "hobbies", "activity",
    "activities", "style", "dance", "sport", "sports", "place", "pet", "pets",
    "color", "colour", "car", "console", "consoles", "instrument",
}

#: Category words -- a GENRE answers 'what genre', not 'what movie'.
CATEGORY_WORDS = {
    "action", "fantasy", "sci-fi", "scifi", "comedy", "drama", "horror",
    "thriller", "romance", "documentary", "mystery", "adventure", "contemporary",
    "classical", "jazz", "rock", "pop", "hip-hop", "country",
}

#: Values that are not noun phrases at all.
_DEICTIC = re.compile(
    r"^(?:this|that|these|those|the|a|an|it|its|his|her|their|how|what|how's)\b"
    r"|^(?:the\s+)?(?:smell|feel|feeling|way|thing|things|stuff|too|also)\b", re.I)


def object_noun(question: str) -> str | None:
    """The noun the question asks for, if it names one."""
    q = question.lower()
    # Prefer the noun closest to the end: 'favorite movie trilogy' -> 'trilogy'
    # would be wrong; the ANSWER slot is the head noun after 'favorite'.
    m = re.search(r"\bfavou?rite\s+([a-z-]+)", q)
    if m and m.group(1) in OBJECT_NOUNS:
        return m.group(1)
    for noun in sorted(OBJECT_NOUNS, key=len, reverse=True):
        if re.search(rf"\b{noun}\b", q):
            return noun
    return None


def classify(question: str, answer: str, gold: str, detail: str) -> tuple[str, str]:
    qnoun = object_noun(question)
    a = str(answer).strip()
    al = a.lower()

    if not a:
        return "EMPTY", "no answer"

    # DEICTIC / non-noun-phrase -- check first, it explains the worst cases.
    if _DEICTIC.match(al) or len(al.split()) == 1 and al.rstrip("'s") in {
            "how", "what", "it", "cooking", "action", "fantasy", "switch",
            "the", "too", "also"}:
        return "DEICTIC", f"value {a[:28]!r} is not an object noun phrase"

    if qnoun:
        # GENRE_NOT_INSTANCE: a category word where an instance was asked.
        if al in CATEGORY_WORDS and qnoun not in ("genre", "style", "type"):
            return ("GENRE_NOT_INSTANCE",
                    f"asked for a {qnoun!r}, got category {a[:24]!r}")
        # NO_OBJECT_NOUN: the answer is a verb phrase / different domain.
        if not re.search(rf"\b{re.escape(qnoun.rstrip('s'))}", al) and \
                re.search(r"\b(?:running|cooking|creating|getting|immersed|"
                          r"non-fiction|also|smell|feel)", al):
            return ("NO_OBJECT_NOUN",
                    f"asked for a {qnoun!r}, got {a[:26]!r}")

    # PARTIAL_SET: a list answer whose items are not all in gold.
    if "," in a and "," in str(gold):
        gold_items = {x.strip().lower() for x in str(gold).split(",")}
        ans_items = {x.strip().lower() for x in a.split(",")}
        if not ans_items <= gold_items:
            return ("PARTIAL_SET",
                    f"answer items {sorted(ans_items)[:3]} not all in gold "
                    f"{sorted(gold_items)[:3]}")

    if "consoles mentioned" in detail:
        return "WRONG_ATTRIBUTE", f"read a console, asked for something else: {a[:22]!r}"

    return "OTHER", f"unclassified; detail={detail[:50]}"


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    # The phase-C rows do not carry cache_qid.  Derive it and ASSERT the join,
    # because a silent zero-match here would report 19 causes from zero evidence.
    def cache_qid_of(qid: str) -> str | None:
        m = re.fullmatch(r"lc_conv-(\d+)_(\d+)", qid)
        return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}" if m else None

    for r in baseline["rows"]:
        cq = cache_qid_of(r["question_id"])
        r["cache_qid"] = cq
        assert cq in cache, (
            f"{r['question_id']} -> {cq!r} has no frozen context; refusing to "
            f"classify from no evidence")

    def examine(rows, label):
        out = []
        counts = Counter()
        for r in rows:
            ctx = cache.get(r["cache_qid"]) or {}
            context = ctx.get("context") or ""
            gold = r["gold"]
            ans = DeterministicMemoryEngine.resolve(r["question"], context)
            detail = getattr(ans, "detail", "") or ""
            cause, why = classify(r["question"], ans.answer, gold, detail)
            counts[cause] += 1
            out.append({
                "question_id": r["question_id"],
                "question": r["question"],
                "gold": gold,
                "answer": ans.answer,
                "detail": detail[:130],
                "cause": cause,
                "why": why,
            })
        print(f"{label}  n={len(out)}")
        for k, n in counts.most_common():
            print(f"    {k:<20} {n}")
        return out, counts

    wrong = [r for r in baseline["rows"]
             if r["shape"] == "preference/attribute" and r.get("current") == "wrong"]
    correct = [r for r in baseline["rows"]
               if r["shape"] == "preference/attribute" and r.get("current") == "correct"]

    print("=" * 74)
    print("CLASSIFICATION OF THE 19 WRONG ANSWERS (mechanical, not by eye)")
    print("=" * 74)
    wrong_out, wrong_counts = examine(wrong, "WRONG set")

    print()
    print("=" * 74)
    print("CONTROL: the same classifier on the CORRECT answers")
    print("=" * 74)
    correct_out, correct_counts = examine(correct, "CORRECT set")

    print()
    print("=" * 74)
    print("DISCRIMINATOR CHECK -- does a cause fire on BOTH sets?")
    print("=" * 74)
    shared = set(wrong_counts) & set(correct_counts)
    if shared:
        for k in sorted(shared):
            print(f"  {k:<20} fires on {wrong_counts[k]} wrong AND "
                  f"{correct_counts[k]} correct -> NOT a discriminator")
    else:
        print("  every cause fires only on the wrong set")
    print()

    print("PER-CASE, wrong set:")
    for r in wrong_out:
        print(f"  {r['question_id']:<16} {r['cause']}")
        print(f"    Q    : {r['question'][:60]}")
        print(f"    gold : {str(r['gold'])[:46]!r}")
        print(f"    gave : {str(r['answer'])[:46]!r}")
        print(f"    why  : {r['why'][:70]}")
    print()
    print("PER-CASE, correct set (these must NOT trip a discriminator):")
    for r in correct_out:
        print(f"  {r['question_id']:<16} {r['cause']}")
        print(f"    Q    : {r['question'][:60]}")
        print(f"    gold : {str(r['gold'])[:40]!r}  gave {str(r['answer'])[:30]!r}")

    OUT.write_text(json.dumps({
        "label": "phase C attribute failure causes",
        "wrong": {"n": len(wrong_out), "counts": dict(wrong_counts), "rows": wrong_out},
        "correct_control": {"n": len(correct_out),
                            "counts": dict(correct_counts), "rows": correct_out},
        "shared_causes": sorted(shared),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
