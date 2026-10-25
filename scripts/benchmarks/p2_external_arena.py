"""P2: External baseline comparison arena (Mem0 / Zep / HippoRAG styles).

Runs the Controlled Arena with the external-style players plus the AM v0.2.0
System Under Test over the frozen dataset, then score-replays every run and
prints the P2 comparison table:

    player | accuracy | mean_score | answer tok in/out | retrieval ms | answer ms | context tok

Every player shares the frozen answer LLM, the frozen extraction LLM, the
frozen embedding model and the 2,000 token budget, so measured deltas
belong to memory / retrieval strategy, never to LLM configuration.

Run:
    python scripts/benchmarks/p2_external_arena.py                  # full 200 questions
    python scripts/benchmarks/p2_external_arena.py --limit 10      # 1/category E2E subset
    python scripts/benchmarks/p2_external_arena.py --dry-run       # no LLM (abstain mode)
    python scripts/benchmarks/p2_external_arena.py --players "Mem0-style,Zep-style,AM v0.2.0"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from artificial_memory.research.benchmarks.arena import ARENA_VERSION, build_dataset
from artificial_memory.research.benchmarks.players import (
    AMv020Player,
    HippoRAGStylePlayer,
    Mem0StylePlayer,
    SimpleRAGPlayer,
    ZepStylePlayer,
)
from artificial_memory.research.benchmarks.runner import ArenaRunner, RunConfig
from artificial_memory.research.benchmarks.scorer import (
    build_aggregate,
    player_aggregates,
    score_run_dir,
)

sys.stdout.reconfigure(encoding="utf-8")

PLAYER_FACTORIES = {
    "Simple RAG": SimpleRAGPlayer,
    "Mem0-style": Mem0StylePlayer,
    "Zep-style": ZepStylePlayer,
    "HippoRAG-style": HippoRAGStylePlayer,
    "AM v0.2.0": AMv020Player,
}
DEFAULT_PLAYERS = "Simple RAG,Mem0-style,Zep-style,HippoRAG-style,AM v0.2.0"

# S6: the R7 research pipeline (retrieval + compile) behind the arena
# interface -- V4_PROGRAM.md 4.17's named next measurement.  Defined in
# scripts (measurement machinery, not shipped code) and registered here.
try:
    from am_current_player import (
        AMConflictPlayer,
        AMCurrentPlayer,
        AMHybridPlayer,
        AMRuntimeConflictPlayer,
    )
    PLAYER_FACTORIES["AM current"] = AMCurrentPlayer
    PLAYER_FACTORIES["AM conflict"] = AMConflictPlayer

    def _hybrid_runtime(config, answerer=None):
        config = dict(config or {})
        config["hybrid_fill"] = "runtime"
        return AMHybridPlayer(config, answerer=answerer)

    PLAYER_FACTORIES["AM hybrid v2"] = _hybrid_runtime

    def _runtime_base(config, answerer=None):
        config = dict(config or {})
        config["conflict_surfacing"] = False
        return AMRuntimeConflictPlayer(config, answerer=answerer)

    PLAYER_FACTORIES["AM runtime"] = _runtime_base
    PLAYER_FACTORIES["AM runtime conflict"] = AMRuntimeConflictPlayer

    def _runtime_conflict_dt(config, answerer=None):
        config = dict(config or {})
        config["provenance_days"] = True
        return AMRuntimeConflictPlayer(config, answerer=answerer)

    PLAYER_FACTORIES["AM runtime conflict dt"] = _runtime_conflict_dt

    def _runtime_conflict_bare(config, answerer=None):
        config = dict(config or {})
        config["conflict_fill_ablation"] = True
        return AMRuntimeConflictPlayer(config, answerer=answerer)

    PLAYER_FACTORIES["AM runtime conflict bare"] = _runtime_conflict_bare
except ImportError as _e:  # pragma: no cover - keeps the 5-player default intact
    if os.environ.get("AM_DEBUG"):
        print(f"AM current player unavailable: {_e}")



def main() -> None:
    parser = argparse.ArgumentParser(description="P2 external baseline arena")
    parser.add_argument("--players", default=DEFAULT_PLAYERS,
                        help="comma list of players (default: all five)")
    parser.add_argument("--limit", type=int, default=0,
                        help="ask only the first N (per-category E2E subset) questions; 0 = full 200")
    parser.add_argument("--repeats", type=int, default=1, help="answer repeats per question")
    parser.add_argument("--output-dir", default="benchmark/results/p2_external")
    parser.add_argument("--am-lexical", action="store_true",
                        help="AM player only: enable the lexical recall channel")
    parser.add_argument("--dry-run", action="store_true",
                        help="unit mode: no LLM (deterministic abstention answers)")
    args = parser.parse_args()

    names = [n.strip() for n in args.players.split(",") if n.strip()]
    unknown = [n for n in names if n not in PLAYER_FACTORIES]
    if unknown:
        raise SystemExit(f"unknown players: {unknown}; choose from {sorted(PLAYER_FACTORIES)}")

    answerer = None  # dry-run mode: frozen abstention answers, zero LLM cost
    if not args.dry_run:
        from artificial_memory.research.benchmarks.players import get_shared_answerer
        answerer = get_shared_answerer()

    dataset = build_dataset(version=ARENA_VERSION, created_at=datetime.now().isoformat())

    # Per-category E2E subset: first `limit` questions of every category.
    if args.limit > 0:
        seen: dict[str, int] = {}
        subset: list[str] = []
        for q in dataset.questions:
            cat = q.category.value
            if seen.get(cat, 0) < args.limit:
                subset.append(q.question_id)
                seen[cat] = seen.get(cat, 0) + 1
        question_ids = subset
    else:
        question_ids = None

    players: list = []
    for name in names:
        factory = PLAYER_FACTORIES[name]
        config = {"context_budget": 2000}
        # Measurement knob for the AM player only: enable the IDF-weighted
        # lexical recall channel (off by default so the baseline is unchanged).
        if args.am_lexical and name == "AM v0.2.0":
            config["lexical_recall"] = True
        players.append(factory(config, answerer=answerer))

    runner = ArenaRunner(RunConfig(
        arena_class="controlled",
        players=players,
        dataset=dataset,
        answer_repeats=args.repeats,
        question_ids=question_ids,
        output_dir=args.output_dir,
    ))
    result = runner.run()
    run_dir = Path(args.output_dir) / "controlled" / result.run_id

    # Frozen scorer replay (deterministic, LLM-free).
    scores = score_run_dir(run_dir, dataset, write=True)

    # Comparison table.
    aggregates = {agg.player_name: agg for agg in player_aggregates(scores)}
    failure_lookup = {
        f["player"] for f in result.failures if f.get("stage") == "ingest"
    }

    print(f"\n=== P2 External Baseline Comparison (run {result.run_id}) ===")
    print(f"questions: {result.question_count}  repeats: {args.repeats}  dry-run: {args.dry_run}")
    header = (
        f"{'player':<16} {'acc%':>6} {'mean':>6} {'wIn':>7} {'wOut':>7} "
        f"{'rIn':>7} {'rOut':>7} {'retr.ms':>8} {'ans.ms':>8} {'ctx.tok':>8}"
    )
    print(header)
    print("-" * len(header))
    summary_rows: list[dict] = []
    for player in players:
        costs = result.player_results[player.name].final_costs
        agg = aggregates.get(player.name)
        if costs is None:
            print(f"{player.name:<16}  (ingest failed)")
            continue
        answer_calls = max(1, costs.answer_llm_calls)
        row = {
            "player": player.name,
            "accuracy": round(100 * agg.answer_accuracy, 1) if agg else None,
            "mean_score": round(agg.mean_score, 3) if agg else None,
            "write_prompt_tokens": costs.write_llm_prompt_tokens,
            "write_completion_tokens": costs.write_llm_completion_tokens,
            "answer_prompt_tokens": costs.answer_llm_prompt_tokens,
            "answer_completion_tokens": costs.answer_llm_completion_tokens,
            "answer_prompt_tokens_per_q": round(costs.answer_llm_prompt_tokens / answer_calls),
            "retrieval_latency_ms": round(costs.retrieval_latency_ms / answer_calls, 1),
            "answer_latency_ms": round(costs.answer_latency_ms / answer_calls, 1),
            "write_latency_ms": round(costs.write_latency_ms, 1),
            "ingest_failed": player.name in failure_lookup,
        }
        summary_rows.append(row)
        print(
            f"{player.name:<16} "
            f"{row['accuracy'] if row['accuracy'] is not None else '--':>6} "
            f"{row['mean_score'] if row['mean_score'] is not None else '--':>6} "
            f"{row['write_prompt_tokens']:>7} {row['write_completion_tokens']:>7} "
            f"{row['answer_prompt_tokens']:>7} {row['answer_completion_tokens']:>7} "
            f"{row['retrieval_latency_ms']:>8} {row['answer_latency_ms']:>8} "
            f"{0:>8}"
        )

    out = run_dir / "p2_summary.json"
    out.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"\nscores:  {run_dir / 'aggregates.csv'}")
    print(f"summary: {out}")


if __name__ == "__main__":
    main()
