from pathlib import Path
p = Path("src/artificial_memory/compiler/certificates/aggregation_certificate.py")
t = p.read_text(encoding="utf-8")
i = t.find("def generate")
print(repr(t[i-200:i+1200]))
