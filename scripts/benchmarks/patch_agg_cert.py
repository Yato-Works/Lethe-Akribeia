"""One-shot patcher: adds duration-sum certificate to aggregation_certificate.py."""
import io
from pathlib import Path

p = Path("src/artificial_memory/compiler/certificates/aggregation_certificate.py")
t = p.read_text(encoding="utf-8")

old = '''    def __init__(self) -> None:
        self.session_fuser = SessionFuser()

    def generate(self, question: str, records: Sequence[StructuredIR]) -> AggregationCertificate:
        """Deterministically resolve multi-session counting and differences into a certificate.

        Note: Global regex summation across full haystacks is disabled to prevent
        hallucinatory arithmetic and budget starvation. Verified multi-session turns
        are preserved for the reader via the budgeted cognitive compiler.
        """
        return AggregationCertificate(
            has_certificate=False,
            query_type="none",
            matched_events=[],
            total_count=None,
            difference_value=None,
            formatted_certificate="",
            direct_answer_candidate=None,
        )
'''

new = '''    _RE_DURATION_Q = re.compile(
        r"\\\\bhow many\\\\b.*\\\\b(days?|weeks?|months?|years?|hours?|minutes?)\\\\b.*\\\\b(in total|total|combined|altogether)\\\\b",
        re.I,
    )
    _RE_DUR_NUM = re.compile(
        r"(\\\\d+)[\\\\s-]*(days?|weeks?|months?|years?|hours?|minutes?)\\\\b", re.I
    )
    _RE_DUR_WORD = re.compile(
        r"\\\\b(week-long|month-long|year-long|day-long|"
        r"a week\\\\s+off|a month\\\\s+off)\\\\b", re.I
    )
    _DUR_WORD_TO_DAYS = {
        "week-long": 7, "month-long": 30, "year-long": 365, "day-long": 1,
        "a week off": 7, "a month off": 30,
    }
    _DUR_UNIT_TO_DAYS = {
        "day": 1, "days": 1, "week": 7, "weeks": 7,
        "month": 30, "months": 30, "year": 365, "years": 365,
    }

    def __init__(self) -> None:
        self.session_fuser = SessionFuser()

    def generate(self, question: str, records: Sequence[StructuredIR]) -> AggregationCertificate:
        """Deterministically resolve duration sums into a reader certificate.

        Duration-sum path (safe, narrow):
        - Only fires on 'how many <unit> ... in total/combined' questions.
        - Sums day-normalized durations found in record text.
        - Fires only when >= 2 distinct duration mentions exist, and is
          passed as a verified certificate for the reader (no override).
        Other counting/difference paths remain disabled (reader-side).
        """
        duration_cert = self._resolve_duration_sum(question, records)
        if duration_cert.has_certificate:
            return duration_cert
        return AggregationCertificate(
            has_certificate=False,
            query_type="none",
            matched_events=[],
            total_count=None,
            difference_value=None,
            formatted_certificate="",
            direct_answer_candidate=None,
        )

    def _resolve_duration_sum(
        self, question: str, records: Sequence[StructuredIR]
    ) -> AggregationCertificate:
        """Sum day-normalized durations ('10-day', 'week-long') for total-days questions."""
        none_cert = AggregationCertificate(
            has_certificate=False, query_type="none", matched_events=[],
            total_count=None, difference_value=None,
            formatted_certificate="", direct_answer_candidate=None,
        )
        if not self._RE_DURATION_Q.search(question):
            return none_cert
        seen: set[str] = set()
        events: list[str] = []
        total_days = 0.0
        q_words = {w for w in re.findall(r"[a-z0-9_-]+", question.lower()) if len(w) > 3}
        topical = q_words - {"many", "total", "combined", "altogether"}
        for r in records:
            text = re.sub(r"\\\\s+", " ", (r.raw_content or "")).strip()
            if not text or text in seen:
                continue
            c_lower = text.lower()
            if not any(w in c_lower for w in topical):
                continue
            turn_total = 0.0
            turn_bits: list[str] = []
            for m in self._RE_DUR_NUM.finditer(text):
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                unit = m.group(2).lower()
                if unit.startswith("hour") or unit.startswith("min"):
                    continue
                turn_total += val * self._DUR_UNIT_TO_DAYS.get(unit, 0)
                turn_bits.append(m.group(0))
            for m in self._RE_DUR_WORD.finditer(text):
                turn_total += self._DUR_WORD_TO_DAYS.get(m.group(1).lower(), 0)
                turn_bits.append(m.group(0))
            if turn_total > 0:
                seen.add(text)
                total_days += turn_total
                events.append(f"{text[:120]} [{', '.join(turn_bits)}]")
        if len(events) < 2 or total_days <= 0:
            return none_cert
        total_int = int(total_days) if float(total_days).is_integer() else total_days
        lines = ["[DURATION SUM CERTIFICATE]",
                 f"- Matched Duration Mentions ({len(events)} instances):"]
        for i, ev in enumerate(events[:10], 1):
            lines.append(f"  {i}. {ev}")
        lines.append(f"- Verified Total: {total_int} days")
        lines.append("[/DURATION SUM CERTIFICATE]")
        return AggregationCertificate(
            has_certificate=True,
            query_type="duration_sum",
            matched_events=events,
            total_count=int(total_days),
            difference_value=None,
            formatted_certificate="\\\\n".join(lines),
            direct_answer_candidate=None,
        )
'''

assert old in t, "OLD BLOCK NOT FOUND"
t = t.replace(old, new, 1)
p.write_text(t, encoding="utf-8")
print("PATCHED OK")
