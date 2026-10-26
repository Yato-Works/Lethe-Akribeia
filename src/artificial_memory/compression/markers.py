"""Language-neutral cue detection for deterministic compression and extraction.

Both the compressor and the incremental memory compiler need to answer "does this
sentence record a decision / a reason / a risk?" without an LLM.  Those tables used
to be Japanese-only, which silently produced an empty ``decisions.md`` and a
no-op semantic compression for English conversations.

The markers live here so the two call sites cannot drift apart.  English matches
are word-boundary regular expressions (``undecided`` must not match ``decided``);
Japanese matches stay substring-based because the language has no word boundaries.
"""

from __future__ import annotations

import re

DECISION_MARKERS_JA: tuple[str, ...] = (
    "決めた", "決定", "採用", "却下", "選択", "選んだ", "決断",
)

DECISION_MARKERS_EN: tuple[str, ...] = (
    "decide", "decides", "decided", "deciding", "decision", "decisions",
    "adopt", "adopts", "adopted", "adopting",
    "reject", "rejects", "rejected", "rejecting",
    "select", "selects", "selected", "selecting", "selection",
    "choose", "chooses", "chose", "chosen", "choosing",
    "approve", "approves", "approved", "approving",
    "agree", "agrees", "agreed", "agreeing",
    "settle", "settles", "settled", "settling",
    "sign off", "signed off", "go with", "going with", "went with",
    "we will", "we'll", "final answer",
)

REASON_MARKERS_JA: tuple[str, ...] = (
    "理由", "なぜなら", "なぜ", "ため", "から",
)

REASON_MARKERS_EN: tuple[str, ...] = (
    "because", "since", "reason", "reasons", "rationale", "due to",
    "so that", "in order to", "the point is", "as a result",
)

IMPORTANT_MARKERS_JA: tuple[str, ...] = (
    "重要", "大事", "クリティカル", "必須", "マスト",
    "問題", "課題", "懸念", "リスク", "不安",
    "メリット", "デメリット", "利点", "欠点", "長所", "短所",
    "比較", "検討", "トレードオフ", "アーキテクチャ", "設計", "方針",
)

IMPORTANT_MARKERS_EN: tuple[str, ...] = (
    "important", "critical", "must", "requirement", "risk", "concern",
    "blocker", "trade-off", "tradeoff", "pros", "cons", "advantage",
    "drawback", "compare", "comparison", "architecture", "design",
    "deadline", "budget",
)

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9d]")
_LATIN = re.compile(r"[A-Za-z]")


def _en_pattern(words: tuple[str, ...]) -> re.Pattern[str]:
    # Longest first so "decided on" wins over "decided" when both are present.
    ordered = sorted(words, key=len, reverse=True)
    return re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in ordered) + r")\b",
        re.IGNORECASE,
    )


_DECISION_EN_RE = _en_pattern(DECISION_MARKERS_EN)
_REASON_EN_RE = _en_pattern(REASON_MARKERS_EN)
_IMPORTANT_EN_RE = _en_pattern(IMPORTANT_MARKERS_EN)


def has_decision(text: str) -> bool:
    """True when the text records a decision, in Japanese or English."""
    lowered = text.lower()
    if any(m in lowered for m in DECISION_MARKERS_JA):
        return True
    return _DECISION_EN_RE.search(text) is not None


def has_reason(text: str) -> bool:
    """True when the text states a rationale, in Japanese or English."""
    lowered = text.lower()
    if any(m in lowered for m in REASON_MARKERS_JA):
        return True
    return _REASON_EN_RE.search(text) is not None


def has_important(text: str) -> bool:
    """True when the text carries a decision-grade signal (risk, requirement...)."""
    lowered = text.lower()
    if any(m in lowered for m in IMPORTANT_MARKERS_JA):
        return True
    return _IMPORTANT_EN_RE.search(text) is not None


def detect_language(text: str) -> str:
    """Return ``"ja"`` or ``"en"`` from the script mix of the text.

    Deliberately crude: it only chooses output labels, and mixed text falls back
    to whichever script carries at least 20% of the letters.
    """
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    if cjk == 0:
        return "en"
    if latin == 0:
        return "ja"
    return "ja" if cjk / (cjk + latin) >= 0.2 else "en"


LABELS: dict[str, dict[str, str]] = {
    "ja": {
        "decisions": "**決定事項:**",
        "reasons": "\n**理由:**",
        "points": "\n**重要ポイント:**",
        "decision_event": "決定",
        "important_event": "重要",
        "topic": "トピック",
    },
    "en": {
        "decisions": "**Decisions:**",
        "reasons": "\n**Reasons:**",
        "points": "\n**Key points:**",
        "decision_event": "Decision",
        "important_event": "Important",
        "topic": "Topic",
    },
}


def labels_for(text: str) -> dict[str, str]:
    """Labels matching the language of ``text``."""
    return LABELS[detect_language(text)]
