"""Deterministic Speaker Attribution & Pronoun Disambiguation Engine.

Solves the Actor / Premise Ownership Gap in multi-session conversations:
- Disambiguates first-person pronouns ("I", "my", "me", "myself") to canonical speaker identity.
- Explicitly binds relational entities ("my sister", "my dog", "my boss") to the speaker.
- Prevents adversary premise attribution errors where actions of speaker B or speaker's relatives
  are incorrectly attributed to speaker A.
- Write LLM Calls = 0: Pure deterministic rule-based canonicalisation.
"""

from __future__ import annotations

import re


class SpeakerAttributionNormalizer:
    """Canonicalises dialogue turns to explicitly bind asserted facts to the true owner."""

    RELATIONAL_NOUNS = frozenset({
        "sister", "brother", "mother", "mom", "father", "dad", "parent", "parents",
        "husband", "wife", "partner", "spouse", "son", "daughter", "child", "children",
        "friend", "colleague", "coworker", "boss", "manager", "doctor", "dentist",
        "dog", "cat", "pet", "roommate", "neighbor", "cousin", "aunt", "uncle",
        "grandma", "grandmother", "grandpa", "grandfather",
    })

    # Contractions for first-person pronouns
    CONTRACTION_MAP = [
        (re.compile(r"\bI'm\b", re.IGNORECASE), " is "),
        (re.compile(r"\bI've\b", re.IGNORECASE), " has "),
        (re.compile(r"\bI'll\b", re.IGNORECASE), " will "),
        (re.compile(r"\bI'd\b", re.IGNORECASE), " would "),
    ]

    def is_generic_speaker(self, speaker: str) -> bool:
        """Check if the speaker label is a generic role rather than a named entity."""
        s = speaker.lower().strip()
        return s in {"user", "assistant", "system", "teammate", "bot", "tool", ""}

    def normalize_turn(self, text: str, speaker: str) -> str:
        """Normalize first-person statements in text to make speaker attribution explicit.

        Example:
            text: "I love hiking and my dog loves running."
            speaker: "Melanie"
            returns: "Melanie loves hiking and Melanie's dog loves running."
        """
        clean_speaker = speaker.strip()
        if not clean_speaker or self.is_generic_speaker(clean_speaker):
            return text

        # Separate reply quote if present: e.g. (In reply to X: "...")
        quote_prefix = ""
        body = text
        m_quote = re.match(r"^(\(In reply to [^:]+:\s*\"[^\"]*\"\)\s*)(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
        if m_quote:
            quote_prefix = m_quote.group(1)
            body = m_quote.group(2)

        speaker_name = clean_speaker
        speaker_possessive = f"{speaker_name}'s"

        # 1. Disambiguate relational nouns: "my <relation>" -> "<Speaker>'s <relation>"
        def replace_my_relation(m: re.Match) -> str:
            rel = m.group(1)
            return f"{speaker_possessive} {rel}"

        rel_pattern = rf"\bmy\s+({'|'.join(self.RELATIONAL_NOUNS)})\b"
        body = re.sub(rel_pattern, replace_my_relation, body, flags=re.IGNORECASE)

        # 2. Expand first-person contractions: "I'm" -> "Melanie is", "I've" -> "Melanie has"
        for pattern, replacement in self.CONTRACTION_MAP:
            body = pattern.sub(f"{speaker_name}{replacement}", body)

        # 3. Disambiguate remaining possessives: "my" -> "Melanie's"
        body = re.sub(r"\bmy\b", speaker_possessive, body, flags=re.IGNORECASE)

        # 4. Disambiguate subject "I": "I" -> "Melanie"
        # Match standalone 'I' not part of other words
        body = re.sub(r"\bI\b", speaker_name, body)

        # 5. Disambiguate objective "me" / "myself": "me" -> "Melanie", "myself" -> "Melanie herself"
        body = re.sub(r"\bme\b", speaker_name, body, flags=re.IGNORECASE)
        body = re.sub(r"\bmyself\b", f"{speaker_name} oneself", body, flags=re.IGNORECASE)

        # Clean up any duplicated spacing
        body = re.sub(r"\s+", " ", body).strip()

        return f"{quote_prefix}{body}" if quote_prefix else body

    def extract_ownership_bindings(self, text: str, speaker: str) -> list[tuple[str, str, str]]:
        """Extract explicit ownership and relationship facts from a dialogue turn.

        Returns list of (Subject, Relation, Object) tuples.
        Example:
            "My sister plays piano" with speaker "Caroline"
            -> [("Caroline", "has_relation", "sister"), ("Caroline's sister", "plays", "piano")]
        """
        clean_speaker = speaker.strip()
        if not clean_speaker or self.is_generic_speaker(clean_speaker):
            return []

        bindings: list[tuple[str, str, str]] = []
        # Find all relational mentions
        for m in re.finditer(rf"\bmy\s+({'|'.join(self.RELATIONAL_NOUNS)})\b", text, flags=re.IGNORECASE):
            rel = m.group(1).lower()
            rel_entity = f"{clean_speaker}'s {rel}"
            bindings.append((clean_speaker, "has_relation", rel_entity))

        return bindings
