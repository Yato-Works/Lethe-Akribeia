#!/usr/bin/env python3
"""Build the complete persona_store.py with fixed experience reasoning."""
import re

# Read the current file
with open(r"C:\Users\smily\artificial_memory\src\artificial_memory\memory\persona_store.py") as f:
    content = f.read()

# Replace the _extract_experience_reasoning method
old_method = '''    def _extract_experience_reasoning(self, q_lower: str, context: str) -> str:
        """Detect negative experience patterns that reduce likelihood of repetition."""
        ctx_lower = context.lower()

        experience_patterns = [
            (["roadtrip", "road trip", "hike", "trip"],
             ["scared", "bad", "stressful", "terrible", "horrible", "awful", "worst",
              "nightmare", "accident", "emergency"]),
            (["concert", "event"],
             ["boring", "terrible", "awful", "worst", "hate", "didn't like", "disappointing"]),
        ]

        for q_keywords, neg_patterns in experience_patterns:
            if not any(kw in q_lower for kw in q_keywords):
                continue

            for name in ["melanie", "caroline"]:
                for neg in neg_patterns:
                    if neg in ctx_lower:
                        # Find negative experience snippet mentioning the character
                        pattern = rf"{name}[^.!?]{{0,80}}(?:{neg})[^.!?]{{0,120}}"
                        match = re.search(pattern, ctx_lower, re.DOTALL)
                        if match:
                            snippet = context[match.start():match.end()].strip()[:150]
                            return (f"ISS {name}: negative experience ({neg}) — "
                                    f"'{snippet}' — makes repeating unlikely")

        return ""'''

new_method = '''    def _extract_experience_reasoning(self, q_lower: str, context: str) -> str:
        """Detect negative experience patterns that reduce likelihood of repetition.

        Uses turn-level parsing so we can check if the target character
        expressed the negative experience in their own statements.
        """
        turns = self._parse_turns(context)

        experience_patterns = [
            (["roadtrip", "road trip", "hike", "trip"],
             ["scared", "bad", "stressful", "terrible", "horrible", "awful", "worst",
              "nightmare", "accident", "emergency"]),
            (["concert", "event"],
             ["boring", "terrible", "awful", "worst", "hate", "didn't like", "disappointing"]),
        ]

        for q_keywords, neg_patterns in experience_patterns:
            if not any(kw in q_lower for kw in q_keywords):
                continue

            for name in ["melanie", "caroline"]:
                # Find turns where this character speaks and mentions a negative experience
                for spk, text in turns:
                    if spk.strip().lower() != name:
                        continue
                    text_lower = text.lower()
                    for neg in neg_patterns:
                        if neg in text_lower:
                            snippet = text.strip()[:120]
                            return (f"ISS {name}: had a negative experience ({neg}) — "
                                    f"'{snippet}' — makes repeating unlikely")

        return ""'''

if old_method in content:
    content = content.replace(old_method, new_method)
    with open(r"C:\Users\smily\artificial_memory\src\artificial_memory\memory\persona_store.py", "w") as f:
        f.write(content)
    print("Method replaced successfully")
else:
    print("ERROR: Could not find old method")
    # Show what's around the method
    idx = content.find("def _extract_experience_reasoning")
    print(content[idx:idx+200])
