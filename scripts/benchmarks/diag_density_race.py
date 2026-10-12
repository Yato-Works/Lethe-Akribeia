"""What separates a winning bundle from a losing one?  87 real cases.

diag_gain_starvation.py disproved the gain=0 hypothesis (0 of 87).  All 87
losers score positive gain and lose the density comparison.  So the question is
what the winners have that the losers lack.  Compare, per losing question, the
winning bundle's gain/density against the losing bundle's, and see whether the
loser was close or hopelessly behind.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from eval_v4_pipeline_dev import V4CandidateRetriever  # noqa: E402
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner  # noqa: E402
from artificial_memory.compiler.graph.graph_builder import GraphBuilder  # noqa: E402
from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"
STOP = frozenset({
    "i", "im", "ive", "id", "ill", "my", "me", "mine", "we", "our", "us",
    "you", "your", "the", "a", "an", "and", "or", "but", "to", "of", "in",
    "on", "at", "for", "with", "was", "were", "is", "are", "be", "been",
    "it", "its", "this", "that", "these", "those", "so", "very", "just",
    "went", "go", "going", "had", "has", "have", "did", "do", "does",
})


def probe(text: str, n: int = 5) -> str:
    w = [x for x in re.findall(r"[a-z0-9]+", str(text).lower())
         if x not in STOP and len(x) > 2]
    return " ".join(w[:n])


def present(ctx: str, probes) -> bool:
    words = set(re.findall(r"[a-z0-9]+", ctx.lower()))
    return any(sum(1 for w in p.split() if w in words) >= max(2, len(p.split()) - 1)
               for p in probes)


def gain_density(bundle, q_words, covered, anc, in_sel):
    b_words = set()
    for n in bundle.member_nodes:
        b_words.update(re.findall(r"\b[a-zA-Z0-9_-]+\b", n.raw_text.lower()))
    new_words = (b_words & q_words) - covered
    gain = anc * 2.5 + len(new_words) * 8.0 + (6.0 if in_sel else 0.0)
    ents = set()
    for n in bundle.member_nodes:
        ents |= set(n.entities)
    gain += min(3.0, len(ents) * 0.1)
    density = gain / (max(80, bundle.total_tokens) ** 0.15)
    return gain, density


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    retriever = V4CandidateRetriever()
    planner = CognitiveExecutionPlanner()
    gb = GraphBuilder()
    comp = BudgetedCognitiveCompiler()

    loser_density = []
    winner_density = []
    loser_tokens = []
    winner_tokens = []
    loser_anchor0 = 0
    shown = 0

    for conv in data:
        turns, dia_map = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        for q in E.load_questions(conv):
            ev = [e for e in (q.get("evidence") or []) if dia_map.get(e)]
            probes = [probe(dia_map[e]) for e in ev
                      if len(dia_map[e]) >= 40 and probe(dia_map[e])]
            if not probes:
                continue
            top, _ = retriever.rank_turns(q["question"], turns, top_k=45)
            gold = [t for t in top if present(t[3], probes)]
            if not gold:
                continue
            plan = planner.plan(q["question"], [])
            t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(top)}
            anchor_ids = [t_map[(t[0], t[3])] for t in top[:12]
                          if (t[0], t[3]) in t_map]
            edg = gb.build(list(top), [])
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
            budget = E._scaled_budget(plan.recommended_budget,
                                      sum(len(t[3].split()) * 4 // 3 for t in top))
            try:
                out = comp.compile(question=q["question"], bundles=bundles,
                                   token_budget=budget,
                                   anchor_scores={a: 5.0 for a in anchor_ids},
                                   is_multi_session=False, adaptive_escalation=True)
            except Exception:
                continue

            q_words = set(re.findall(r"[a-z0-9]+", q["question"].lower()))
            covered = set(re.findall(r"[a-z0-9]+", out.context_text.lower()))

            # winners = bundles whose content IS in the context
            winners, losers = [], []
            for b in bundles:
                if b.total_tokens > budget:
                    continue
                anc = max((5.0 for n in b.member_nodes if n.node_id in anchor_ids),
                          default=0.0)
                text = " ".join(n.raw_text for n in b.member_nodes)
                g, d = gain_density(b, q_words, covered, anc, False)
                if present(out.context_text, [probe(text)]):
                    winners.append((g, d, b))
                elif any(n.raw_text == t[3] for t in gold for n in b.member_nodes):
                    losers.append((g, d, b))

            if not losers or not winners:
                continue
            best_winner_d = max(w[1] for w in winners)
            best_winner_t = max(w[0] for w in winners)
            for g, d, b in losers:
                loser_density.append(d)
                winner_density.append(best_winner_d)
                loser_tokens.append(b.total_tokens)
                if anc == 0.0:
                    loser_anchor0 += 1
                if shown < 3:
                    shown += 1
                    print(f"--- Q: {q['question'][:58]}")
                    print(f"    loser : density={d:6.2f} gain={g:5.1f} tokens={b.total_tokens}")
                    print(f"    winner: density={best_winner_d:6.2f} gain={best_winner_t:5.1f}")
                    print(f"    loser needed {best_winner_d - d:+.2f} more density to win")

    print()
    print("=" * 70)
    print("WINNERS vs LOSERS on the density race")
    print("=" * 70)
    print(f"loser bundles analysed          : {len(loser_density)}")
    print(f"losers with anchor score 0      : {loser_anchor0} "
          f"({loser_anchor0/len(loser_density)*100:.1f}%)")
    print(f"median loser density            : {statistics.median(loser_density):.2f}")
    print(f"median best-winner density      : {statistics.median(winner_density):.2f}")
    print(f"median loser tokens             : {statistics.median(loser_tokens):.0f}")
    print()
    gaps = [w - l for l, w in zip(loser_density, winner_density)]
    close = sum(1 for g in gaps if g < 1.0)
    print(f"losers within 1.0 density of the winner : {close} "
          f"({close/len(gaps)*100:.1f}%)")
    print(f"median density gap                     : {statistics.median(gaps):.2f}")
    print()
    if statistics.median(gaps) < 1.0:
        print("VERDICT: losers are CLOSE. A small tie-break or anchor-floor change")
        print("  would flip a meaningful share of them.  Worth trying.")
    else:
        print("VERDICT: losers are far behind on density. The ranking itself is")
        print("  choosing better turns; adjusting the formula marginally will not")
        print("  recover these, and the loss is mostly inherent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
