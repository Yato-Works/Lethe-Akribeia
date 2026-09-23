#!/usr/bin/env python3
"""Debug the identity reasoning extraction."""
import sys, json, re
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, questions, ir = adapter.load_conversation(0)

q30 = next(q for q in questions if q.question_id == 'conv-26-qa-030')
pcc = adapter.compiler.compile(q30.question, ir)
context = pcc.context_text

# Debug the turn parsing
turn_pattern = re.compile(r"\[.*?\]\s*([^:]+):\s*(.*?)(?=\[.*?\]\s*[^:]+:|\Z)", re.DOTALL)
parsed_turns = turn_pattern.findall(context)
print(f"Parsed turns: {len(parsed_turns)}")
for spk, text in parsed_turns[:5]:
    print(f"  Speaker: '{spk.strip()[:20]}', Text: '{text.strip()[:80]}'")

# Check Melanie's own text
melanie_texts = [text.strip() for spk, text in parsed_turns if spk.strip().lower() == 'melanie']
combined = " ".join(melanie_texts).lower()
print(f"\nMelanie's combined text length: {len(combined)}")
print(f"Contains 'supportive': {'supportive' in combined}")
print(f"Contains 'lgbtq': {'lgbtq' in combined}")
print(f"Contains 'ally': {'ally' in combined}")

# Check for identity claims
ic = re.search(r"i\s*(?:'m|am|identif(?:y|ies) as)\s+(\S+)", combined)
print(f"\nIdentity claim: {ic.group(1) if ic else 'None'}")

# Check ally
ally_kw = ["supportive of", "supports", "backing", "cheering for", "ally", "amazing", "inspiring", "so proud", "so glad", "love that", "great for"]
has_ally = any(kw in combined for kw in ally_kw) and ("lgbtq" in combined or "community" in combined)
print(f"Has ally: {has_ally}")
