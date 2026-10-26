"""30-minute stage-by-stage speaker attribution trace.

Eli's instruction: before touching the compiler, locate the FIRST stage where the
speaker attribution breaks.

Stages traced, for the same question:
  ORIGINAL   the raw conversation turn that holds the gold answer
  RETRIEVED  what V4CandidateRetriever.rank_turns selected
  BUNDLED    what GraphBuilder.form_bundles produced
  COMPILED   what BudgetedCognitiveCompiler emitted
  COMMITTER  what parse_turns sees (the engine's actual input)

The output is per-question: gold, and at each stage the speaker the stage
attributes that gold-bearing turn to.  The first stage whose speaker differs from
the ORIGINAL is the fault site.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from eval_v4_pipeline_dev import V4CandidateRetriever  # noqa: E402
from artificial_memory_gen.compiler.execution_planner import (  # noqa: E402
    CognitiveExecutionPlanner,
)
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder  # noqa: E402
from artificial_memory_gen.compiler.budgeted_compiler import (  # noqa: E402
    BudgetedCognitiveCompiler,
)
from artificial_memory.skills.answer_committer import parse_turns  # noqa: E402


def norm_words(s: str) -> set:
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if len(w) > 2}


def holds_gold(gold: str, text: str) -> bool:
    g = norm_words(gold)
    if not g:
        return False
    return len(g & norm_words(text)) >= max(1, len(g) // 2)


def main() -> None:
    # 30 fixed questions: 15 the engine got WRONG, 15 it got RIGHT (control).
    d = json.load(open(ROOT / "datasets/external/locomo10.json", encoding="utf-8"))
    r_ = V4CandidateRetriever()
    pl_ = CognitiveExecutionPlanner()
    gb_ = GraphBuilder()
    cp_ = BudgetedCognitiveCompiler()

    import eval_locomo_full as _E
    from artificial_memory.skills.autonomous_engines.entity_attribute_resolver import (
        EntityAttributeResolver,
    )
    PAT = re.compile(r"^(what|which|who)\s+(did|does|is|was|has|have)\b", re.I)

    cases = []
    for conv in d:
        turns, _ = E.conversation_turns(conv["conversation"])
        for q in E.load_questions(conv):
            qq = q["question"].strip()
            if not PAT.match(qq):
                continue
            gold = str(q["answer"])
            top, _ = r_.rank_turns(qq, turns, top_k=45)
            pl = pl_.plan(qq, [])
            tm = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(top)}
            an = [tm[(t[0], t[3])] for t in top[:12] if (t[0], t[3]) in tm]
            edg = gb_.build(list(top), [])
            bs = edg.form_bundles(an, max_bundle_tokens=150)
            bud = _E._scaled_budget(pl.recommended_budget,
                                    sum(len(t[3].split()) * 4 // 3 for t in top))
            try:
                out = cp_.compile(
                    question=qq, bundles=bs, token_budget=bud,
                    anchor_scores={a: 5.0 for a in an},
                    is_multi_session=False, adaptive_escalation=True)
            except Exception:
                continue
            ct = parse_turns(out.context_text)
            claim = EntityAttributeResolver._extract_generic_attribute(qq, ct)
            if claim is None:
                continue
            answer, turn = claim
            ok = _E.score(answer, gold)
            if len([c for c in cases if c["ok"] is (ok is True)]) >= 15:
                continue
            cases.append({"q": qq, "gold": gold, "ans": answer, "ok": bool(ok),
                          "turn": turn, "ctx_turns": ct, "top": top,
                          "orig": turns})
            if len([c for c in cases if c["ok"]]) >= 15 and len(cases) >= 30:
                break
        if len([c for c in cases if c["ok"]]) >= 15 and len(cases) >= 30:
            break

    print(f"traced {len(cases)} questions: "
          f"{sum(1 for c in cases if c['ok'])} correct, "
          f"{sum(1 for c in cases if not c['ok'])} wrong\n")

    fault_stage: dict[str, int] = {}
    for i, c in enumerate(cases, 1):
        orig_speaker = None
        orig_text = None
        for t in c["orig"]:
            if holds_gold(c["gold"], t[3]):
                orig_speaker, orig_text = t[0], t[3]
                break

        retr = None
        for t in c["top"]:
            if holds_gold(c["gold"], t[3]):
                retr = t[0]
                break

        comp = c["turn"].speaker
        gold_in_compiled = holds_gold(c["gold"], c["turn"].text)

        # fault site = first stage that diverges from the original speaker
        fault = "NONE"
        if orig_speaker is None:
            fault = "GOLD_NOT_IN_ORIGINAL"
        elif retr is None:
            fault = "RETRIEVER_DROPPED"
        elif retr != orig_speaker:
            fault = "RETRIEVER_WRONG_SPEAKER"
        elif not gold_in_compiled:
            fault = "COMPILER_LOST_TEXT"
        elif comp.lower() != orig_speaker.lower():
            fault = "COMPILER_WRONG_SPEAKER"
        fault_stage[fault] = fault_stage.get(fault, 0) + 1

        if i <= 6 or not c["ok"]:
            print(f"--- [{i}] {'OK ' if c['ok'] else 'WRONG'} {c['q'][:52]}")
            print(f"    gold   : {c['gold'][:40]!r}")
            print(f"    ORIGINAL speaker : {orig_speaker!r}")
            print(f"    RETRIEVED speaker: {retr!r}")
            print(f"    COMPILED speaker : {comp!r}   gold_in_turn={gold_in_compiled}")
            if orig_text:
                print(f"    orig text : {orig_text[:78]!r}")
            print(f"    FAULT SITE: {fault}")
            print()

    print("=== fault site histogram (30 questions) ===")
    for k, v in sorted(fault_stage.items(), key=lambda x: -x[1]):
        print(f"  {k:26s}: {v}")


if __name__ == "__main__":
    main()
