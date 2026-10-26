import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]

data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
it = next(item for item in data if item["question_id"] == "a3332713")

for s in it["haystack_sessions"]:
    for t in s:
        if t["role"] == "user":
            c = t["content"]
            if "brother" in c.lower() or "coworker" in c.lower():
                print("TURN:", c)
                # Split sentences
                for sent in re.split(r"(?<=[.!?])\s+", c):
                    if any(w in sent.lower() for w in ["brother", "coworker", "baby clothes", "gift card", "baby shower"]):
                        m = re.findall(r"\$(\d+)", sent)
                        print("  SENT:", sent, "-> M:", m)
