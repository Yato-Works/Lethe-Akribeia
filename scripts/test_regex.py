#!/usr/bin/env python3
"""Test the fixed turn parsing regex."""
import re

context = """[D18:7 on 6:55 pm on 20 October, 2023] (In reply to Caroline: "The kids look so cute, Mel! I bet they bring lots of joy. How did they handle the accident?") Melanie: Thanks! They were scared but we reassured them and explained their brother would be OK. They're tough kids.
[D7:2 on 4:33 pm on 12 July, 2023] (In reply to Caroline: "Hey Mel, great to chat with you again! So much has happened since we last spoke - I went to an LGBTQ conference two days") Melanie: Wow, Caroline, that sounds awesome! So glad you felt accepted and supported. Events like these are great for reminding us of how strong community can be!
[D14:34 on 1:33 pm on 25 August, 2023] Melanie: Wow, Caroline, that's awesome! Can't wait to see your show - the LGBTQ community needs more platforms like this!"""

# Fixed regex: skip "(In reply to ...)" part
turn_pattern = re.compile(
    r"\[.*?\]\s*(?:\([^)]*\)\s*)*(?:Speaker: )?([^:]+):\s*(.*?)(?=\[.*?\]\s*(?:\([^)]*\)\s*)*(?:Speaker: )?[^:]+:|\Z)",
    re.DOTALL
)
turns = turn_pattern.findall(context)
print(f"Parsed turns: {len(turns)}")
for spk, text in turns:
    print(f"  Speaker: '{spk.strip()}'")
    print(f"  Text: '{text.strip()[:80]}'")
