"""Diagnose why the week-long gold user turn scores below threshold."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory_gen.compiler.samplers.per_session_sampler import PerSessionSampler

q = "How many days did I take social media breaks in total?"
sampler = PerSessionSampler(turns_per_session=2)
terms = PerSessionSampler.extract_query_terms(q)
print("QUERY TERMS:", sorted(terms))

turns = {
    "10-day user": "I'm trying to get back into journaling. By the way, I've been making an effort to cut down on social media lately - I actually just got back from a 10-day break in mid-February.",
    "week-long user": "I'm trying to track my screen time and social media usage. By the way, I've been making an effort to cut down on social media lately - I even took a week-long break from it in January.",
    "10-day asst": "Congratulations on taking a 10-day break from social media! Here are some book recommendations.",
    "week-long asst": "Congratulations on taking a week-long break from social media! Here are some apps and tools.",
    "noise asst": "I'm glad you're excited about the recommendations! Now, let's talk about making the most out of your music festival experience! Have you taken a break today?",
}
for name, content in turns.items():
    speaker = "user" if "user" in name else "assistant"
    sc, matched = sampler.score_turn(content, terms, speaker=speaker)
    print(f"{name:15s} speaker={speaker:9s} score={sc:6.1f} matched={matched} "
          f"qty={bool(PerSessionSampler._QUANTITY_RE.search(content))}")
