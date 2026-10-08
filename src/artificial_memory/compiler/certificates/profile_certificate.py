"""
Profile Certificate Engine for Lethe-Akribeia V4.

Compiles user preferences, equipment, background, and stated tastes into a
grounded User Profile State for recommendation queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.core.ir.structured import IRRelation, StructuredIR


@dataclass(frozen=True)
class ProfileCertificate:
    """User profile state distilled from past dialogue turns."""
    has_certificate: bool
    profile_statements: list[str]
    formatted_certificate: str


class ProfileCertificateEngine:
    """Engine that distills user preferences and tastes into grounded profile states."""

    _PREF_PATTERNS = [
        re.compile(r"\b(i like|i love|i prefer|my favorite|i usually use|i own a|i have a|my gear is|i'm interested in)\b", re.I),
        re.compile(r"\b(i don't like|i dislike|i avoid|allergic to|can't eat)\b", re.I),
        re.compile(r"\b(my cat|my dog|my pet|high school|reunion|mixology|photography|camera|lens|deep clean|dust|allergies)\b", re.I),
    ]

    def generate(self, question: str, records: Sequence[StructuredIR], preference_demand: float = 0.0) -> ProfileCertificate:
        """Extract user profile state relevant to recommendation queries."""
        q_lower = question.lower()
        is_retrospective_recall = bool(re.search(
            r"\byou\s+(recommended|suggested|said|told|mentioned|advised)\b", q_lower
        ))
        is_recommendation = (
            not is_retrospective_recall and (
                preference_demand >= 0.4
                or any(w in q_lower for w in [
                    "recommend", "suggest", "what should i", "ideas for", "tips for",
                    "any suggestions", "do you think", "would it be a good idea",
                ])
            )
        )

        profile_statements = []
        seen = set()

        for r in records:
            # Explicit PREFERS relation
            if r.relation == IRRelation.PREFERS:
                stmt = f"- {r.entity} prefers {r.value}"
                if stmt not in seen:
                    seen.add(stmt)
                    profile_statements.append(stmt)
                continue

            # Textual preferences and background in user speech
            content = r.raw_content
            if any(p.search(content) for p in self._PREF_PATTERNS):
                clean_stmt = re.sub(r"\s+", " ", content).strip()
                if clean_stmt not in seen:
                    seen.add(clean_stmt)
                    profile_statements.append(f"- {clean_stmt[:160]}")

        if is_recommendation:
            cert_lines = ["[USER PROFILE & PREFERENCE STATE]"]
            if profile_statements:
                cert_lines.extend(profile_statements[:8])
            else:
                cert_lines.append("- (User personal background from conversation history)")
            cert_lines.append(
                "\n[INSTRUCTION]\n"
                "The user is asking for advice, recommendations, or thoughts based on their personal situation.\n"
                "You MUST tailor your answer directly to their background, personal experiences, equipment, or tastes mentioned above.\n"
                "Incorporate their specific details (e.g. equipment brand, pet, activities, classes) into your reasoning and advice.\n"
                "Do NOT say 'I don't know' or 'The information is not provided'. Provide a helpful, personalized response."
            )
            cert_lines.append("[/USER PROFILE & PREFERENCE STATE]")

            return ProfileCertificate(
                has_certificate=True,
                profile_statements=profile_statements,
                formatted_certificate="\n".join(cert_lines),
            )

        return ProfileCertificate(
            has_certificate=False,
            profile_statements=[],
            formatted_certificate="",
        )
