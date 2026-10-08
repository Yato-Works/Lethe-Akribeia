"""Patch v4: fix double-escaped regexes back to single backslash."""
from pathlib import Path

p = Path("src/artificial_memory/compiler/certificates/aggregation_certificate.py")
t = p.read_text(encoding="utf-8")
pairs = [
    ("\\\\\\\\bhow many\\\\\\\\b", "\\\\bhow many\\\\b"),
    ("\\\\\\\\b(days?", "\\\\b(days?"),
    ("minutes?)\\\\\\\\b", "minutes?)\\\\b"),
    ("\\\\\\\\b(total", "\\\\b(total"),
    ("total)\\\\\\\\b", "total)\\\\b"),
    ("(\\\\\\\\d+)[\\\\\\\\s-]*", "(\\\\d+)[\\\\s-]*"),
    ("minutes?)\\\\\\\\b", "minutes?)\\\\b"),
    ("\\\\\\\\b(week-long", "\\\\b(week-long"),
    ("off)\\\\\\\\b", "off)\\\\b"),
    ("a week\\\\\\\\s+off", "a week\\\\s+off"),
    ("a month\\\\\\\\s+off", "a month\\\\s+off"),
]
for old, new in pairs:
    if old in t:
        t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("PATCHED OK")
