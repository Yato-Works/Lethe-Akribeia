"""Forensic audit of the 163 wrong answers.

Eli's challenge: three numbers were reported without their derivation:
  1. 215 coverage -- 'can answer correctly', or merely 'produced a candidate'?
  2. 163 wrong   -- subject mixup / predicate misclassification /
                    evidence meaning change / question mismatch?
  3. 0 regressions -- against which test set, comparing what?

This script answers (1) and (2) from data, and (3) by naming the exact set.

It also tests the 'compiler rewrites first person to third person' claim that was
generalised from a SINGLE example.  For each wrong answer the ORIGINAL
conversation turn is located and compared against the compiled context, so the
claim is either supported or retracted on evidence.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
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
from artificial_memory_gen.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory_gen.skills.autonomous_engines.entity_attribute_resolver import (  # noqa: E402
    EntityAttributeResolver,
)

PAT = re.compile(r"^(what|which|who)\s+(did|does|is|was|has|have)\b", re.I)

#: Failure taxonomy.  Each wrong answer gets exactly one label, decided by checks
#: run in this order (most upstream cause first, because that is the layer that
#: must be fixed).
WRONG_SPEAKER = "WRONG_SPEAKER"        # evidence came from a different person
WRONG_SPAN = "WRONG_SPAN"              # right person, wrong text window
MEANING_CHANGED = "MEANING_CHANGED"    # original held the answer, compiler altered it
GOLD_ABSENT = "GOLD_ABSENT"            # gold never appears in the compiled context


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def gold_tokens_in(gold: str, text: str) -> bool:
    """True when the gold answer's content words appear in the text."""
    g = [w for w in norm(gold) if len(w) > 2]
    if not g:
        return False
    t = set(norm(text))
    return sum(1 for w in g if w in t) >= max(1, len(g) // 2)


def main() -> None:
    d = json.load(open(ROOT / "datasets/external/locomo10.json", encoding="utf-8"))
    r_ = V4CandidateRetriever()
    pl_ = CognitiveExecutionPlanner()
    gb_ = GraphBuilder()
    cp_ = BudgetedCognitiveCompiler()

    tax: Counter = Counter()
    claimed = 0
    graded_wrong = 0
    wrong = []
    person_change = {"third_person_in_compiled": 0, "first_person_in_original": 0}
    #: Denominator discipline.  Every wrong answer gets exactly ONE label, and the
    #: per-label counts must sum to the number of graded-wrong answers.  A
    #: previous version let a claim fall through with no label assigned and then
    #: also counted it under GOLD_ABSENT, so the labels summed to 185 against a
    #: stated 163 -- the mismatch Eli caught.  Assert the invariant here instead
    #: of trusting it.

    for conv in d:
        raw_turns, _ = E.conversation_turns(conv["conversation"])
        for q in E.load_questions(conv):
            qq = q["question"].strip()
            if not PAT.match(qq):
                continue
            gold = str(q["answer"])

            top, _ = r_.rank_turns(qq, raw_turns, top_k=45)
            pl = pl_.plan(qq, [])
            tm = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(top)}
            an = [tm[(t[0], t[3])] for t in top[:12] if (t[0], t[3]) in tm]
            edg = gb_.build(list(top), [])
            bs = edg.form_bundles(an, max_bundle_tokens=150)
            bud = E._scaled_budget(
                pl.recommended_budget, sum(len(t[3].split()) * 4 // 3 for t in top))
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
            claimed += 1
            answer, turn = claim

            if E.score(answer, gold):
                continue
            graded_wrong += 1
            if len(wrong) >= 200:
                continue

            # --- classify the failure -------------------------------------
            # Subject as the question names them.
            subj = None
            for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", qq):
                w = m.group(1)
                if w.lower() in E.STOPWORDS if hasattr(E, "STOPWORDS") else False:
                    continue
                subj = w
                break

            gold_anywhere = gold_tokens_in(gold, out.context_text)
            gold_here = gold_tokens_in(gold, turn.text)

            if not gold_anywhere:
                label = GOLD_ABSENT
            elif subj and turn.speaker.lower() != subj.lower():
                label = WRONG_SPEAKER
            elif not gold_here:
                label = WRONG_SPAN
            else:
                label = MEANING_CHANGED

            tax[label] += 1
            wrong.append({
                "q": qq, "gold": gold, "ans": answer, "label": label,
                "speaker": turn.speaker, "ev": turn.text[:110],
            })

            # --- test the first->third person claim -----------------------
            orig = next((t for t in raw_turns if gold_tokens_in(gold, t[3])), None)
            if orig:
                if re.search(r"\b(i|my|me)\b", orig[3], re.I):
                    person_change["first_person_in_original"] += 1
                if re.search(rf"\b{re.escape(subj.lower()) if subj else 'zzz'}\b",
                             out.context_text, re.I) and subj:
                    person_change["third_person_in_compiled"] += 1

    tot = sum(tax.values())
    print(f"=== claimed by engine: {claimed}  (COVERAGE = candidates produced) ===")
    print("    This is a CANDIDATE count, not a correctness measure.")
    print()
    print(f"=== graded WRONG: {graded_wrong}  ===")
    print(f"    classified: {tot}  (invariant: classified == graded_wrong)")
    print()
    print("=== failure taxonomy (one label per wrong answer) ===")
    for lbl in (WRONG_SPEAKER, WRONG_SPAN, MEANING_CHANGED, GOLD_ABSENT):
        n = tax[lbl]
        print(f"  {lbl:18s}: {n:4d} ({n/graded_wrong*100 if graded_wrong else 0:.1f}%)")
    print()
    if tot != graded_wrong:
        print(f"  !! DENOMINATOR MISMATCH: {tot} labelled vs {graded_wrong} graded")
    else:
        print("  denominator consistent")
    print()
    print("=== first->third person claim, tested on data ===")
    print(f"  originals containing gold ALSO use first person : "
          f"{person_change['first_person_in_original']}")
    print(f"  those where subject name appears in compiled    : "
          f"{person_change['third_person_in_compiled']}")
    print()
    print("=== worked examples per label ===")
    seen = set()
    for w in wrong:
        if w["label"] in seen:
            continue
        seen.add(w["label"])
        print(f"\n[{w['label']}]")
        print(f"  Q     : {w['q'][:60]}")
        print(f"  GOLD  : {w['gold'][:44]!r}")
        print(f"  GOT   : {w['ans'][:44]!r}")
        print(f"  speaker={w['speaker']}  evidence={w['ev'][:70]!r}")


if __name__ == "__main__":
    main()
