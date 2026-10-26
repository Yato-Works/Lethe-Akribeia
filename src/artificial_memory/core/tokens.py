"""Canonical token counting & estimation utilities.

Every token/cost estimate in the runtime path must be a named function from
this module. Inline re-implementations (`len(text) // 3`,
`len(text.split()) * 4 // 3`, ...) are banned — grep for them in the
hygiene tests.

The module preserves the exact formulas of each historically shipped
estimate class (byte-equivalent), grouped and annotated:

``bpe``
    Real BPE counting (tiktoken ``cl100k_base``). Used for persisted usage
    metrics and context budgets.

``cjk``
    Char-based estimate calibrated for Japanese text
    (``1 token ≈ 3 chars``). Used by recall-side budget cuts.

``en4``
    Char-based estimate for English text (``1 token ≈ 4 chars``). Used by
    provider fallbacks and packers.

``words``
    Whitespace-word based estimates historically tuned per compiler
    generation. Variants differ in multiplier / overhead / floor.

No function here changes observable values relative to the formulas each
call site previously used; the module exists so the semantics live in ONE
auditable place.
"""

from __future__ import annotations

import threading

# ---------------------------------------------------------------------------
# bpe — real BPE token counts (cl100k_base)
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_encoding = None


def _get_encoding():
    global _encoding
    enc = _encoding
    if enc is None:
        with _LOCK:
            if _encoding is None:
                import tiktoken

                _encoding = tiktoken.get_encoding("cl100k_base")
            enc = _encoding
    return enc


def bpe_tokens(text: str) -> int:
    """Exact BPE token count (cl100k_base)."""
    if not text:
        return 0
    return len(_get_encoding().encode(text))


# ---------------------------------------------------------------------------
# cjk — char/3 estimates used by recall-side budgets
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Japanese-calibrated char estimate: ``max(1, len(text) // 3)``."""
    return max(1, len(text) // 3)


def estimate_tokens_raw(text: str) -> int:
    """Same as :func:`estimate_tokens` but without the floor (may return 0)."""
    return len(text) // 3


# ---------------------------------------------------------------------------
# en4 — char/4 estimates used by provider fallbacks and packers
# ---------------------------------------------------------------------------

def estimate_tokens_en(text: str) -> int:
    """English-calibrated char estimate: ``len(text) // 4``."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# words — whitespace-word based estimates (per compiler-generation tuning)
# ---------------------------------------------------------------------------

def word_tokens_floored(text: str, floor: int = 10) -> int:
    """Word estimate ``max(floor, len(text.split()) * 4 // 3)``."""
    return max(floor, len(text.split()) * 4 // 3)


def word_tokens_43(text: str) -> int:
    """Word estimate ``len(text.split()) * 4 // 3`` (no floor)."""
    return len(text.split()) * 4 // 3


def word_tokens_13(text: str, overhead: int = 0) -> int:
    """Word estimate ``int(len(text.split()) * 1.3) + overhead``."""
    return int(len(text.split()) * 1.3) + overhead


def word_tokens(text: str) -> int:
    """Bare whitespace word count used as a token proxy."""
    return len(text.split())
