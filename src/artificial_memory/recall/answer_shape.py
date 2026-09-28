"""Deterministic answer-shape classification - a measured NEGATIVE result.

What was hypothesised
---------------------
Open-domain is the weakest LoCoMo category, and the first post-mortem
(``_failure_census.py --run locomo_instruct_full``, 2026-09-25) attributed 20 of
its 63 wrong answers to the prompt demanding "Yes / Likely no / No" from a
reader that was being asked open questions: *"What personality traits might
Melanie say Caroline has?"* answered *"Likely no"*.  If the runtime decided the
answer *form* instead of asking the model to decide it, those should be free.

What the measurement said
-------------------------
Paired A/B over all 96 category-3 questions, same rebuilt context cache, same
reader (``qwen2.5:7b-instruct``, num_ctx 8192), only the shape instruction
differing:

============================  ====  ==========  ==========  ======
arm                           n    frozen      gate        delta
============================  ====  ==========  ==========  ======
hand-written shape block      96     38.5% (37)      -        -
runtime-classified gate       96       -        35.4% (34)   -3
  polarity                   23     70%          61%         -2
  choice                      10     60%          70%         +1
  attribute                   63     24%          21%         -2
============================  ====  ==========  ==========  ======

3 questions gained, 6 lost.  **The hypothesis is refuted**, and the per-shape
split says why: the attribute class - two thirds of the category - is pinned at
21-24% *whatever* the prompt says, so its failures are not a form problem at all
(the typical loss is a wrong value, e.g. "screenwriter" for "filmmaker"), i.e. a
reader-capability ceiling for a 7B model.  The 20 "boolean leak" failures had
already been fixed by the hand-written block in AM_APEX_STATUS.md 0f, which is
what this arm was measured against.

Therefore: default **off** everywhere.  The module is kept because
* the classification is the useful diagnostic - it says which 63 questions are
  attribute-shaped and therefore not addressable by prompt engineering;
* the harness that refuted it (``scripts/benchmarks/commit_sweep.py`` and the
  two-arm runner) is reusable for the next idea.
"""

from __future__ import annotations

import re
from enum import StrEnum


class AnswerShape(StrEnum):
    """The surface form the answer is expected to take."""

    #: "Would Caroline pursue writing as a career?" -> Yes / Likely no / No
    POLARITY = "polarity"
    #: "... a national park or a theme park?" -> one of the offered options
    CHOICE = "choice"
    #: "What personality traits ...?" -> the attribute value(s), never yes/no
    ATTRIBUTE = "attribute"


#: A question opening with an auxiliary verb asks for a yes/no judgement.
_POLARITY_OPENING = re.compile(
    r"^\s*(would|does|do|did|is|are|was|were|has|have|had|can|could|should|"
    r"shall|will|must|may|might)\b",
    re.IGNORECASE,
)
#: "or" as a standalone word: the question offers a choice.  Word boundaries
#: keep "for", "more" and "before" out, which a bare " or " test would not.
_CHOICE = re.compile(r"\bor\b", re.IGNORECASE)
#: Question forms that want a value even though they start with "what/which".
_ATTRIBUTE_WH = re.compile(r"^\s*(what|which|who|whom|whose|where|how)\b", re.IGNORECASE)


def classify_answer_shape(question: str) -> AnswerShape:
    """Classify a question by the answer form it demands.

    Order matters and is deliberate:

    1. an explicit ``or`` between candidates is a CHOICE even when the sentence
       is phrased as a yes/no question - "Would Melanie prefer a national park
       or a theme park?" wants the chosen option, not "Likely no";
    2. an auxiliary opening is a POLARITY question;
    3. everything else is an ATTRIBUTE question, which must never be answered
       with a bare yes/no.
    """
    text = str(question).strip()
    if _CHOICE.search(text):
        return AnswerShape.CHOICE
    if _POLARITY_OPENING.match(text):
        return AnswerShape.POLARITY
    return AnswerShape.ATTRIBUTE


#: Instruction block injected into the reader prompt, per shape.  Kept next to
#: the classifier so the two cannot drift apart.
DIRECTIVES: dict[AnswerShape, str] = {
    AnswerShape.POLARITY: (
        "- The question is a yes/no judgement: answer with exactly one of "
        "'Yes', 'Likely no' or 'No'.\n"
        "- Weigh both supporting and contradicting evidence before choosing.\n"
        "- Prefer 'Likely no' when the question asks whether someone IS something "
        "and the evidence only shows them supporting it.\n"
        "- Prefer 'Likely no' when the evidence describes the thing negatively "
        "(accident, scary, bad, went wrong).\n"
    ),
    AnswerShape.CHOICE: (
        "- The question offers alternatives: answer with the ONE option the "
        "evidence supports, named as it appears in the question.\n"
        "- A short reason is allowed after the option, separated by a semicolon.\n"
        "- Do NOT answer with 'Yes' or 'No': the reader wants the chosen option.\n"
    ),
    AnswerShape.ATTRIBUTE: (
        "- The question asks for a value, trait, field or object - NOT a yes/no "
        "judgement.\n"
        "- Answer with the value itself, taken from the evidence, e.g. a field, a "
        "trait list, a title, a place.\n"
        "- Answering 'Yes', 'No' or 'Likely no' is WRONG even when the evidence is "
        "clear; the question did not ask a yes/no question.\n"
        "- If several values are supported, list them all, comma-separated.\n"
    ),
}


def shape_directive(question: str) -> str:
    """The instruction block for this question's shape (empty-safe)."""
    return DIRECTIVES[classify_answer_shape(question)]
