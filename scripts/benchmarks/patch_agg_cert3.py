"""Patch v3: relax duration-question regex to verb-medial order."""
from pathlib import Path

p = Path("src/artificial_memory/compiler/certificates/aggregation_certificate.py")
t = p.read_text(encoding="utf-8")
old = 'r"\\\\bhow many\\\\b.*\\\\b(days?|weeks?|months?|years?|hours?|minutes?)\\\\b.*\\\\b(in total|total|combined|altogether)\\\\b",'
assert t.count(old) == 1, f"count={t.count(old)}"
new = 'r"\\\\bhow many\\\\b.*\\\\b(days?|weeks?|months?|years?|hours?|minutes?)\\\\b.*\\\\b(total|combined|altogether|in total)\\\\b|\\\\bhow many\\\\b.*\\\\b(total|combined|altogether|in total)\\\\b",'
t = t.replace(old, new, 1)
p.write_text(t, encoding="utf-8")
print("PATCHED OK")
