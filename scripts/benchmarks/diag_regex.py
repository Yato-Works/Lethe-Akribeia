from pathlib import Path
p = Path("src/artificial_memory/compiler/certificates/aggregation_certificate.py")
t = p.read_text(encoding="utf-8")
i = t.find("_RE_DURATION_Q")
print(repr(t[i:i+400]))
