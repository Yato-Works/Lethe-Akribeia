"""Paired control for the PPR spike: PPR vs the retrieval that already exists.

spike_ppr_entity_edges.py reported 37/40 gold turns reached by PPR top-k and
STOPPED THERE.  That number is worthless on its own: reaching 37 of 40 is only
evidence if the CURRENT retriever reaches fewer.  A control that loses to nothing
proves nothing, and this session already retracted one gain that turned out to be
a scoring illusion -- so the paired comparison is mandatory, not optional.

Three conditions on the IDENTICAL cohort, questions, contexts, and seeds:

  A  RRF-equivalent  current MultiViewRetriever ranking, top-k
  B  PPR over the existing graph (dialogue-flow + speaker edges only)
  C  PPR over the graph PLUS the missing entity-sharing edge

B isolates "does PPR help at all"; C isolates "does the entity edge help".
Reported as (n, reached_A, reached_B, reached_C) -- never a bare count.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory_gen.compiler.closure.retriever import MultiViewRetriever  # noqa: E402
from artificial_memory.skills.answer_committer import parse_turns  # noqa: E402

ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"

PPR_DAMPING = 0.5
PPR_ITERS = 30

_STOP = {
    "the", "a", "an", "and", "or", "but", "i", "you", "he", "she", "it", "we",
    "they", "my", "your", "his", "her", "its", "our", "their", "me", "him",
    "them", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "have", "has", "had", "to", "of", "in", "on", "at", "for", "with",
    "that", "this", "these", "those", "so", "if", "then", "than", "as", "by",
    "from", "about", "just", "really", "very", "too", "also", "not", "no", "yes",
    "what", "when", "where", "who", "why", "how", "which", "there", "here",
}


def content_terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


def _session_of(dia_id: str) -> str:
    """'D7:18' -> 'D7'.  Session key for a turn id."""
    return dia_id.split(":", 1)[0] if ":" in dia_id else dia_id


def turn_ids_at(indices, turns) -> set[str]:
    """Atlas turn ids ('D7:18') for the given positional indices."""
    return {turns[i].dia_id for i in indices}


def capitalised_terms(text: str) -> set[str]:
    return {m.group(1).lower() for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", text)}


def build_edges(turns: list, with_entity: bool) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for i in range(len(turns)):
        adj.setdefault(i, set())

    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(turns):
        by_speaker[t.speaker.lower()].append(i)
    for idxs in by_speaker.values():
        for a, b in zip(idxs, idxs[1:]):
            adj[a].add(b)
            adj[b].add(a)

    if with_entity:
        ent: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(turns):
            for e in capitalised_terms(t.text):
                ent[e].append(i)
        for holders in ent.values():
            for a in holders:
                for b in holders:
                    if a != b:
                        adj[a].add(b)
                        adj[b].add(a)
    return adj


def ppr(seeds: list[int], adj: dict[int, set[int]], n: int) -> list[float]:
    mass = [0.0] * n
    if not seeds:
        return mass
    share = 1.0 / len(seeds)
    seed_set = set(seeds)
    teleport = [share if i in seed_set else 0.0 for i in range(n)]
    for _ in range(PPR_ITERS):
        nxt = [PPR_DAMPING * teleport[i] for i in range(n)]
        for i in range(n):
            if mass[i] == 0.0:
                continue
            outs = adj.get(i) or set()
            if not outs:
                part = (1.0 - PPR_DAMPING) * mass[i] / n
                for j in range(n):
                    nxt[j] += part
                continue
            part = (1.0 - PPR_DAMPING) * mass[i] / len(outs)
            for j in outs:
                nxt[j] += part
        mass = nxt
    return mass


def main() -> int:
    atlas = {}
    for line in ATLAS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            atlas[row["question_id"]] = row
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    cohort = [
        r for r in atlas.values()
        if r.get("shape") in ("what_thing", "what_person")
        and r.get("gold_in_context") == "full"
        and r.get("rescored_correct") is False
    ][:40]

    retriever = MultiViewRetriever()
    scored = 0
    reached = {"A_rrf": 0, "B_ppr_no_entity": 0, "C_ppr_with_entity": 0}
    wins_over_baseline: list[str] = []

    for row in cohort:
        ctx_row = cache.get(row.get("cache_qid") or "")
        if not ctx_row:
            continue
        context = ctx_row.get("context") or ""
        if not context:
            continue
        turns = parse_turns(context)
        if len(turns) < 3:
            continue
        gold_turns = set(row.get("evidence_turns") or [])
        if not gold_turns:
            continue
        scored += 1

        k = max(5, min(len(turns), 12))

        # A -- the retrieval that already exists.
        #
        # ID WARNING, recorded because it nearly fabricated a win.  retrieve()
        # REGENERATES turn ids as f"{sid}:t{index}" internally (retriever.py:164
        # raises KeyError on the ids it synthesises), so no input format makes it
        # echo our atlas ids back.  Comparing its candidate ids against gold
        # evidence turns ('D7:18') would silently zero-match and report the
        # baseline as reaching nothing -- a fake win for condition C.
        #
        # So the comparison is POSITIONAL: turns are passed in a known order, and
        # 'D7:t8' is decoded back to index 8 of that same order.  The assertion
        # guards the decode rather than trusting it.
        from artificial_memory_gen.compiler.closure.planner import DeterministicQueryPlanner

        qplan = DeterministicQueryPlanner().plan(row["question"])
        raw_turns = [
            (_session_of(t.dia_id), t.header_date, t.speaker, t.text) for t in turns
        ]
        cands = retriever.retrieve(qplan, raw_turns, top_k=k)

        def _decode(cand_id: str) -> int | None:
            """"D7:t8" -> 8, or None if the shape is unexpected."""
            _, _, tail = cand_id.rpartition(":t")
            return int(tail) if tail.isdigit() else None

        a_indices = [_decode(c.turn_id) for c in cands]
        if any(i is None for i in a_indices):
            bad = [c.turn_id for c, i in zip(cands, a_indices) if i is None]
            raise SystemExit(
                f"{row['question_id']}: candidate ids not in 'sid:tN' form: {bad[:3]} "
                f"-- the A baseline cannot be compared positionally")
        assert all(0 <= i < len(turns) for i in a_indices), (
            f"{row['question_id']}: decoded index outside the turn list")
        a_hit = {turns[i].dia_id for i in a_indices} & gold_turns
        if a_hit:
            reached["A_rrf"] += 1

        qterms = content_terms(row["question"]) | capitalised_terms(row["question"])
        seeds = [i for i, t in enumerate(turns) if qterms & content_terms(t.text)]
        if not seeds:
            continue

        adj_b = build_edges(turns, with_entity=False)
        mass_b = ppr(seeds, adj_b, len(turns))
        top_b = sorted(range(len(turns)), key=lambda i: mass_b[i], reverse=True)[:k]
        if {turns[i].dia_id for i in top_b} & gold_turns:
            reached["B_ppr_no_entity"] += 1

        adj_c = build_edges(turns, with_entity=True)
        mass_c = ppr(seeds, adj_c, len(turns))
        top_c = sorted(range(len(turns)), key=lambda i: mass_c[i], reverse=True)[:k]
        c_hit = {turns[i].dia_id for i in top_c} & gold_turns
        if c_hit:
            reached["C_ppr_with_entity"] += 1

        if c_hit and not a_hit:
            wins_over_baseline.append(row["question_id"])

    print(f"cohort scored (context parsed, gold turns present) : {scored}")
    print()
    print(f"{'condition':<24} {'gold turns reached':>18}")
    print("-" * 46)
    print(f"{'A current RRF retriever':<24} {reached['A_rrf']:>18}")
    print(f"{'B PPR, no entity edge':<24} {reached['B_ppr_no_entity']:>18}")
    print(f"{'C PPR + entity edge':<24} {reached['C_ppr_with_entity']:>18}")
    print()
    print(f"questions where C reached gold and A did not : {len(wins_over_baseline)}")
    if wins_over_baseline:
        print(f"  {wins_over_baseline[:12]}")

    print()
    if reached["C_ppr_with_entity"] > reached["A_rrf"]:
        print("VERDICT: the entity edge + PPR reaches MORE gold turns than the")
        print("current retriever.  The gap is what a real implementation could win.")
    elif reached["C_ppr_with_entity"] == reached["A_rrf"]:
        print("VERDICT: no gain over the current retriever.  The 37/40 from the")
        print("unpaired spike was NOT evidence of improvement -- the baseline")
        print("already reaches them.  Do not build this.")
    else:
        print("VERDICT: PPR is WORSE than the current retriever on this cohort.")
        print("Do not build this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
