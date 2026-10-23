#!/usr/bin/env python3
"""M30-43 Phase C: The Map — consolidated project facts from code + results.

Builds the single reference document: pipeline, data flow, what's live,
what each score actually measured, and confirmed/refuted root causes.
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results/m30_43"
OUT.mkdir(parents=True, exist_ok=True)

# --- Collect the real numbers from artifacts ---
def load(p, default=None):
    try:
        return json.loads((PROJECT_ROOT / p).read_text(encoding="utf-8"))
    except Exception:
        return default

prod = load("results/m30_43/production_bucketA.json", {})
e2e = load("results/m30_42/e2e_ablation.json", {})
unseen = load("results/m30_42/unseen_ablation.json", {})
firing = load("results/m30_42/firing_and_input_hash.json", {})
m29 = load("results/m30_29/results_v2.json", {})
m37 = load("results/m30_37/evaluation_results.json", {})
m40 = load("results/m30_40/evaluation_results.json", {})

doc = []
doc.append("# M30-43 — Lethe Akribeia V4: Project Map (fact-based)\n")
doc.append("## 0. The headline finding\n")
doc.append(f"**The production pipeline scores {prod.get('correct','?')}/28 on Bucket A.**")
doc.append("Every M30-29..42 experiment measured a hand-rolled approximation, not this pipeline.\n")
doc.append("| Pipeline | Bucket A 28 | Source |")
doc.append("|----------|-------------|--------|")
doc.append(f"| **PRODUCTION (locomo_adapter)** | **{prod.get('correct','?')}/28** | m30_43_phaseB |")
doc.append(f"| M30-29 (handrolled + normalize) | {m29.get('treatment_correct','?')}/28 | m30_29 |")
doc.append(f"| M30-42 (handrolled, compile only) | {e2e.get('Baseline',{}).get('correct','?')}/28 | m30_42 |")
doc.append("")

doc.append("## 1. The real execution path (verified from code)\n")
doc.append("```")
doc.append("locomo10.json")
doc.append("  -> LoCoMoAdapter.load_conversation()")
doc.append("     -> UniversalIRExtractor.extract(prev_ctx + text)      # structured IR")
doc.append("     -> r.raw_content = '[D1:1 on <date>] (In reply to X: ...) Speaker: text'")
doc.append("     -> r.time_scope  = session_date                        # PER-TURN date")
doc.append("  -> MinimumSufficientContextCompiler.compile(question, ir_records)")
doc.append("     L396  planner.plan()              QueryPlanner")
doc.append("     L402  temporal_resolver.resolve() TemporalResolver")
doc.append("     L410  persona_store.ingest_records()")
doc.append("     L414  state_engine.resolve()      StateSupersessionEngine")
doc.append("     L417  reconstructor.reconstruct_world()  StateReconstructor")
doc.append("     L432  adaptive_searcher.search()  AdaptiveEvidenceSearcher")
doc.append("     L435  integrity_gate.check()      PropositionIntegrityGate")
doc.append("     L510  checker.check()             proof certificate")
doc.append("  -> commit_answer() / OllamaAnswerer.answer(question, context_text)")
doc.append("  -> OfficialLongMemEvalGrader.grade()")
doc.append("```\n")

doc.append("## 2. Data flow: what is added / removed / transformed\n")
doc.append("| Stage | Input | Output | Added | Removed |")
doc.append("|-------|-------|--------|-------|---------|")
doc.append("| load_conversation | raw session turns | StructuredIR per turn | entity/property/value/source/time_scope | — |")
doc.append("| compile | ir_records (all turns) | ProofCarryingContext | provenance header `[D.. on date]`, temporal grounding | non-selected turns |")
doc.append("| reader | context_text | answer string | — | — |")
doc.append("| grader | answer, gold | is_correct | — | — |")
doc.append("")

doc.append("## 3. What is LIVE vs merely present\n")
doc.append("| Mechanism | Status | Evidence |")
doc.append("|-----------|--------|----------|")
doc.append("| QueryPlanner | LIVE | compile L396 |")
doc.append("| TemporalResolver | LIVE | compile L402 |")
doc.append("| StateSupersessionEngine | LIVE | compile L414 |")
doc.append("| StateReconstructor | LIVE | compile L417 |")
doc.append("| AdaptiveEvidenceSearcher | LIVE | compile L432 |")
doc.append("| PropositionIntegrityGate | LIVE | compile L435 |")
doc.append("| checker (proof cert) | LIVE | compile L510 |")
doc.append("| UniversalIRExtractor | **LIVE ONLY in locomo_adapter** | load_conversation |")
doc.append("| KNOWN_ACTORS boost | fires 28/28 on LoCoMo, 0/28 renamed | m30_42 firing |")
doc.append("| commit_answer (zero-LLM) | LIVE | evaluate_question L586 |")
doc.append("")

doc.append("## 4. Evaluation provenance: what each number measured\n")
doc.append("| Number | Pipeline | Dataset | Reader | Valid as production score? |")
doc.append("|--------|----------|---------|--------|---------------------------|")
doc.append(f"| {prod.get('correct','?')}/28 | locomo_adapter (PROD) | Bucket A | qwen2.5:7b | **YES** |")
doc.append(f"| {m29.get('treatment_correct','?')}/28 | handrolled records | Bucket A | qwen2.5:7b | NO |")
doc.append(f"| {e2e.get('Baseline',{}).get('correct','?')}/28 | handrolled records | Bucket A | qwen2.5:7b | NO |")
doc.append("| 24/120 | handrolled | synthetic | qwen2.5:7b | NO |")
doc.append("| 32/120 | handrolled | synthetic | qwen2.5:7b | NO |")
doc.append("| 108/120 (M30-40) | **no LLM, synthetic raw strings** | synthetic | — | NO |")
doc.append("")

doc.append("## 5. Root causes: confirmed / refuted / open\n")
doc.append("| Claim | Verdict | Evidence |")
doc.append("|-------|---------|----------|")
doc.append("| Benchmark-specific name boosts exist in production | CONFIRMED | 20 files, 4 on prod path |")
doc.append("| Those boosts inflate the final score | **REFUTED** | E2E 1/28 -> 1/28 across ablations |")
doc.append("| Those boosts change the Reader input | CONFIRMED | context hash differs 26/28 |")
doc.append("| The engine is weak on Bucket A | **REFUTED** | production = 13/28 (46%) |")
doc.append("| Prior M30 scores reflect the engine | **REFUTED** | they reflect a hand-rolled copy |")
doc.append("| Generalization gap exists | OPEN | 3.3% vs 57%, but data-design confounded |")
doc.append("")

doc.append("## 6. Why the agent could not improve the system\n")
doc.append("1. It never ran the production entry point (`LoCoMoAdapter`).")
doc.append("2. It rebuilt the record construction by hand and silently dropped")
doc.append("   `UniversalIRExtractor` and per-turn `time_scope`.")
doc.append("3. Every 'failure' it diagnosed was a property of its own copy.")
doc.append("4. It added 83 scripts and never closed the loop on the real pipeline.")
doc.append("")

doc.append("## 7. What to do next (no new code yet)\n")
doc.append("- Treat **13/28** as the baseline. All future work must run through")
doc.append("  `LoCoMoAdapter` so numbers are comparable.")
doc.append("- Re-derive the 15 failures from the PRODUCTION run, not the copy.")
doc.append("- Only then pick ONE change and measure it on the production path.")
doc.append("")

(OUT / "PROJECT_MAP.md").write_text("\n".join(doc), encoding="utf-8")
print("\n".join(doc))
print(f"\nSaved: {OUT / 'PROJECT_MAP.md'}")
