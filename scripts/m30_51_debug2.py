#!/usr/bin/env python3
import sys, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
exec(open("scripts/m30_51_phase2_fallback.py").read().split("def main")[0])

q = "When did Deborah receive an appreciation letter from her community?"
qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", q.lower())
          if w not in STOP and w not in {"when", "what", "where", "which", "who", "whose", "why", "how"})
print("qkw:", sorted(qkw))

# D2:7 body: "It is love, and openness that have kept us close all these years. Being there fo..."
# This turn is about "love, openness" — NOT about "appreciation letter"
# So the noun overlap score will be LOW for the correct turn.

# Check what the body contains
ctx_line = "[SPEAKER:Deborah] [D2:7 on 9:49 am on 27 January, 2023] (In reply to Jolene: \"What do you value in your relationship?\") Deborah: It is love, and openness that have kept us close all these years."
m = TURN_RE.match(ctx_line.strip())
if m:
    body = m.group(3)
    bkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", body.lower()) if w not in STOP)
    print("body keywords:", sorted(bkw))
    print("overlap:", sorted(qkw & bkw))
    print("score:", len(qkw & bkw))

# The issue: D2:7 body mentions "love, openness" - the appreciation letter is
# mentioned IN THE QUESTION about the letter, but the evidence turn D2:7
# says "yesterday" + mentions the letter received yesterday.
# Wait - let me check the actual body
print("\n--- D14:5 (has 'the day before') ---")
print("body has 'yesterday' + 'the day before'")
# D14:5: "You are amazing as always! Here are new photos... yesterday"
# This also won't have 'appreciation' or 'letter'
