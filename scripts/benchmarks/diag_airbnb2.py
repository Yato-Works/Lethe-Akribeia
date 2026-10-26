import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
it = next(item for item in data if item["question_id"] == "982b5123")

for s_idx, session in enumerate(it["haystack_sessions"]):
    s_date = it["haystack_dates"][s_idx]
    for turn in session:
        c = turn["content"]
        if "airbnb" in c.lower():
            print(f"[{s_date}] {turn['role']}: {c}")
