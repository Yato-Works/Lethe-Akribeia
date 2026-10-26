"""Out-of-sample check: does the lexical recall channel help on a REAL corpus?

The lexical channel was diagnosed and validated on the synthetic controlled arena
(11.5% -> 53.0%).  That is in-sample for the mechanism: the failure was found
there.  Before claiming the fix generalises, it must be measured on a corpus it
was not designed against.

This probe runs the SHIPPED runtime (not the R7 research pipeline) over a LoCoMo
conversation and asks one question only, at 0 LLM calls:

    does the recalled context contain the gold answer?

Both configurations are run on identical ingested memory, so the difference is
the retrieval channel alone.

Usage
    python scripts/benchmarks/probe_runtime_recall_locomo.py --conv-idx 0 --limit 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def gold_terms(gold: str) -> list[str]:
    raw = [x.strip().strip("\"'") for x in re.split(r",|;|\band\b", str(gold)) if x.strip()]
    out: list[str] = []
    for item in raw:
        toks = [w for w in re.findall(r"[a-z0-9]{3,}", item.lower())]
        out.extend(toks or [item.lower()])
    return out


def reached(context: str, gold: str) -> bool:
    low = context.lower()
    terms = gold_terms(gold)
    return bool(terms) and all(t in low for t in terms)


async def ingest(rt, turns, topic: str) -> None:
    for speaker, text in turns:
        await rt.remember(f"{speaker}: {text}", topic=topic)


async def evaluate(rt, questions, topic: str, level: int, max_tokens: int):
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        res = await rt.recall(q["question"], topic=topic, level=level, max_tokens=max_tokens)
        latency = (time.perf_counter() - t0) * 1000
        context = "\n".join(m.semantic_content.content for m in res.memories)
        rows.append({
            "category": q["category"],
            "reached": reached(context, q["answer"]),
            "tokens": res.tokens,
            "latency_ms": latency,
            "memories": res.memories_retrieved,
        })
    return rows


def summarise(rows) -> dict:
    by_cat: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        by_cat[r["category"]][0] += int(r["reached"])
        by_cat[r["category"]][1] += 1
    total = sum(int(r["reached"]) for r in rows)
    return {
        "n": len(rows),
        "reached": total,
        "rate": total / max(1, len(rows)) * 100,
        "mean_tokens": sum(r["tokens"] for r in rows) / max(1, len(rows)),
        "mean_latency_ms": sum(r["latency_ms"] for r in rows) / max(1, len(rows)),
        "by_cat": {CAT.get(c, str(c)): {"hit": v[0], "n": v[1]}
                   for c, v in sorted(by_cat.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--conv-idx", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

    data = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    conv = data[args.conv_idx]
    sid = conv["sample_id"]

    turns: list[tuple[str, str]] = []
    for key, value in conv["conversation"].items():
        if key.startswith("session_") and isinstance(value, list):
            for t in value:
                turns.append((str(t.get("speaker", "")), str(t.get("text", ""))))

    questions = []
    for qa in conv["qa"]:
        answer = qa.get("answer", qa.get("adversarial_answer", ""))
        questions.append({"question": qa["question"], "category": int(qa.get("category", 0)),
                          "answer": str(answer)})
    if args.limit:
        questions = questions[: args.limit]

    print(f"conversation {sid}: {len(turns)} turns, {len(questions)} questions")

    results = {}
    for name, lexical in (("dense-only", False), ("lexical", True)):
        rt = ArtificialMemoryRuntime(RuntimeConfig(
            database_path=":memory:", memory_files_path=None, vector_index_path=None,
            lexical_recall=lexical))
        asyncio.run(ingest(rt, turns, topic="LoCoMo"))
        rows = asyncio.run(evaluate(rt, questions, "LoCoMo", args.level, args.max_tokens))
        results[name] = summarise(rows)
        s = results[name]
        print(f"  {name:<10} gold-reached {s['reached']}/{s['n']} = {s['rate']:.1f}%  "
              f"mean tokens {s['mean_tokens']:.0f}  latency {s['mean_latency_ms']:.0f}ms")
        for cat, v in s["by_cat"].items():
            print(f"      {cat:<12} {v['hit']}/{v['n']}")

    delta = results["lexical"]["rate"] - results["dense-only"]["rate"]
    print(f"\ndelta (lexical - dense-only): {delta:+.1f}pt on a corpus the channel "
          f"was not designed against")
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"conversation": sid, "results": results,
                                   "delta_pt": round(delta, 2)},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
