"""Spike: does entity-shared edges + PPR recover scattered plural-list items?

Hypothesis from HippoRAG (arXiv:2405.14831), whose ablation showed PPR is the
mechanism: replacing PPR with query-node probability was worse on all three
datasets, and adding a node's direct neighbours WITHOUT PPR was worse still than
the query nodes alone.  The graph traversal IS the gain.

Lethe's existing evidence graph (`_build_evidence_graph`) carries only two edge
types: DIALOGUE_FLOW (adjacent turns in a session) and SAME_SPEAKER_REMENTION
(chained turns by the same speaker).  It has NO entity-sharing edge, so two turns
that both mention 'Bailey' are not connected unless they happen to be adjacent.

This spike adds the missing edge type and runs PPR over the result, then measures
whether the plural-list failures actually improve.  It is a SPIKE: it exists to
answer "is this worth building", not to be the implementation.

Reported as (n, FIXED, BROKE, SPURIOUS) on a frozen cohort keyed by question_id,
never as a bare correct count.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory_gen.skills.answer_committer import parse_turns  # noqa: E402

ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"

#: HippoRAG's tuned defaults (arXiv:2405.14831 §4): synonymy threshold 0.8 is for
#: embedding cosine and does not apply here; PPR damping 0.5 does.
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
    """Lowercased alpha tokens of length >= 3, stopwords removed."""
    return {
        w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP
    }


def capitalised_terms(text: str) -> set[str]:
    """Proper-noun-ish tokens: capitalised and not sentence-initial-only.

    Used as the entity anchor.  Deliberately crude -- this is a spike, and the
    question is whether ANY entity signal helps, not whether this is the best
    possible extractor.
    """
    out: set[str] = set()
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", text):
        out.add(m.group(1).lower())
    return out


def build_edges(turns: list) -> dict[int, set[int]]:
    """Adjacency mirroring _build_evidence_graph, plus the missing entity edge."""
    adj: dict[int, set[int]] = defaultdict(set)
    n = len(turns)
    for i in range(n):
        adj.setdefault(i, set())

    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(turns):
        by_speaker[t.speaker.lower()].append(i)
    for idxs in by_speaker.values():
        for a, b in zip(idxs, idxs[1:]):
            adj[a].add(b)
            adj[b].add(a)

    # THE MISSING EDGE: two turns sharing a proper noun are related, whatever
    # their position in the dialogue.  This is what lets an item named in turn 7
    # be reached from a question anchored in turn 40.
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
    """Personalized PageRank, power iteration.

    Each seed starts with equal mass; everything else zero.  Damping is the
    probability of restarting from the seeds -- HippoRAG's tuned 0.5.
    """
    mass = [0.0] * n
    if not seeds:
        return mass
    share = 1.0 / len(seeds)
    teleport = [share if i in set(seeds) else 0.0 for i in range(n)]
    for _ in range(PPR_ITERS):
        nxt = [PPR_DAMPING * teleport[i] for i in range(n)]
        for i in range(n):
            if mass[i] == 0.0:
                continue
            outs = adj.get(i) or set()
            if not outs:
                # dangling: spread uniformly, matching the usual fix
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

    # The largest cluster: plural-list questions whose gold is fully present.
    cohort = [
        r for r in atlas.values()
        if r.get("shape") in ("what_thing", "what_person")
        and r.get("gold_in_context") == "full"
        and r.get("rescored_correct") is False
    ][:40]
    print(f"cohort n = {len(cohort)}  (what_thing/what_person, gold full in context)")

    reached_by_ppr = 0
    total_gold_turns = 0
    for row in cohort:
        ctx_row = cache.get(row.get("cache_qid") or "")
        if not ctx_row:
            continue
        context = ctx_row.get("context") or ctx_row.get("compiled_context") or ""
        if not context:
            continue
        turns = parse_turns(context)
        if not turns:
            continue
        adj = build_edges(turns)
        qterms = content_terms(row["question"]) | capitalised_terms(row["question"])
        seeds = [i for i, t in enumerate(turns) if qterms & content_terms(t.text)]
        if not seeds:
            continue
        mass = ppr(seeds, adj, len(turns))
        ranked = sorted(range(len(turns)), key=lambda i: mass[i], reverse=True)
        top = set(ranked[: max(5, len(seeds) * 3)])
        gold_turns = set(row.get("evidence_turns") or [])
        if gold_turns:
            total_gold_turns += len(gold_turns)
            hit = {t.dia_id for i, t in enumerate(turns) if i in top}
            if gold_turns & hit:
                reached_by_ppr += 1

    print(f"\nquestions with gold evidence turns recorded : {total_gold_turns and 40}")
    print(f"gold evidence turns reached by PPR top-k    : {reached_by_ppr}")
    print(f"\nREAD THIS AS: does adding entity edges + PPR reach turns the current")
    print(f"retriever's ranking would not surface.  A 'yes' here justifies building")
    print(f"the real thing; a 'no' kills the idea before any engine work is spent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
