import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
it = next(item for item in data if item["question_id"] == "a1cc6108")

re_user_age = re.compile(
    r"\b(?:as a\s+(\d{1,2})-year-old|i\s+just\s+turned\s+(\d{1,2})|current\s+age(?::|\s+is)?\s+(\d{1,2})|"
    r"do you think\s+(\d{1,2})\s+is\s+considered\s+(?:young|old)|"
    r"i\s+turned\s+(\d{1,2})\s+last|i\s+am\s+(\d{1,2})\s+years\s+old)\b",
    re.I
)

for s_idx, session in enumerate(it.get("haystack_sessions", [])):
    sid = it["haystack_session_ids"][s_idx]
    s_date = it["haystack_dates"][s_idx]
    for turn in session:
        c = turn.get("content", "")
        if "alex" in c.lower() or "age" in c.lower() or "born" in c.lower():
            m_u = re_user_age.search(c)
            if m_u:
                print(f"[{sid}] USER AGE MATCH: {m_u.groups()} in: {c[:120]}")
            if "alex" in c.lower():
                print(f"[{sid}] ALEX MATCH: {turn['role']}: {c[:120]}")
