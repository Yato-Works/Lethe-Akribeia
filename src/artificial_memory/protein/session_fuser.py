"""Deterministic Cross-Session Aggregator & Evidence Distiller (Phase SESSION FUSION).

Aggregates evidence across multiple conversation sessions for multi-hop / multi-session queries:
1. Groups records by session ID.
2. Identifies matching candidate statements across sessions using linguistic & semantic topic matching.
3. Performs deterministic quantity extraction and mathematical summation (currency, durations, counts).
4. Generates a Proof-Carrying Aggregation Certificate ([GLOBAL STATE AGGREGATION]).
"""

from __future__ import annotations

import datetime
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class AggregationResult:
    """Result of cross-session aggregation."""
    is_aggregation_query: bool
    total_value: float | None = None
    unit: str | None = None
    found_snippets: list[tuple[str, str, float]] = field(default_factory=list)  # [(sid, snippet, val)]
    certificate: str | None = None


class SessionFuser:
    """Deterministic Cross-Session Aggregator & Evidence Distiller."""

    STOP_WORDS = {
        "how", "many", "much", "what", "which", "when", "where", "total",
        "did", "have", "been", "was", "were", "are", "is", "the", "for",
        "from", "with", "and", "in", "to", "of", "my", "i", "a", "an",
        "this", "that", "these", "those", "combined", "altogether", "all",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "first", "second", "third", "recently", "lately", "past", "last",
        "you", "your", "can", "tell", "could", "would", "about", "different",
    }

    WORD_TO_NUM: dict[str, float] = {
        "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
        "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
        "ten": 10.0, "eleven": 11.0, "twelve": 12.0, "fifteen": 15.0, "twenty": 20.0,
        "half": 0.5, "a week and a half": 1.5, "one and a half": 1.5,
        "two and a half": 2.5, "three and a half": 3.5,
    }

    MONTH_MAP: dict[str, int] = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    _FINANCIAL_KEYWORDS = {
        "dollar", "dollars", "money", "cost", "price", "expense", "expenses",
        "spent", "spend", "earned", "earn", "total amount", "amount spent",
        "budget", "$", "paid", "pay", "fee", "sales",
    }

    _DOMAIN_SYNONYMS: dict[str, set[str]] = {
        "doctor": {"doctor", "doctors", "dr", "physician", "physicians", "specialist", "specialists", "dermatologist", "ent"},
        "clothing": {"clothing", "clothes", "blazer", "boots", "jacket", "jeans", "shirt", "pants", "dress", "sweater"},
        "plant": {"plant", "plants", "lily", "succulent", "fern", "basil", "snake"},
        "furniture": {"furniture", "bookshelf", "table", "chair", "desk", "couch", "sofa", "bed", "mattress", "cabinet", "dresser"},
        "run": {"jog", "jogging", "jogged", "run", "running", "ran"},
        "exercise": {"exercise", "workout", "fitness", "gym", "yoga", "pilates"},
    }

    def _split_sentences(self, content: str) -> list[str]:
        """Split content into sentences while protecting abbreviations like Dr., Mr., etc."""
        clean_content = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof)\.\s+", r"\1_DOT_ ", content)
        return [s.replace("_DOT_", ". ").strip() for s in re.split(r"[.!?]\s+", clean_content) if s.strip()]

    def is_aggregation_query(self, query: str) -> bool:
        """Check if query requires counting, summing, or aggregating across sessions."""
        ql = query.lower()
        # Exclude questions looking for single dates, times, or specific facts
        if any(w in ql for w in [
            "what time did i", "when did i", "at which university",
            "where did i", "who was", "why did i",
        ]):
            return False

        # Exclude comparisons and single relative queries unless asking for total/sum
        if any(w in ql for w in ["difference in", "difference between", "compared to"]) and not any(
            t in ql for t in ["total", "combined", "altogether", "sum"]
        ):
            return False

        return any(
            w in ql
            for w in [
                "how many", "how much", "total", "combined", "in total",
                "altogether", "sum of", "average", "how long", "page count",
            ]
        )

    def is_multi_session_reasoning_query(self, query: str) -> bool:
        """Check if query requires multi-session cross-comparison or quantitative synthesis."""
        ql = query.lower()
        return bool(re.search(
            r"\b(difference\s+in|difference\s+between|percentage\s+of|percentage\s+discount|"
            r"how\s+much\s+(?:more|faster|earlier|older)|how\s+many\s+years\s+older|"
            r"compared\s+to|across\s+(?:all\s+)?sessions)\b",
            ql,
        ))

    def determine_unit(self, query: str) -> str:
        """Determine target aggregation unit from query semantics."""
        ql = query.lower()

        # 1. Delivery Duration
        if "did it take" in ql and any(w in ql for w in ["arrive", "receive", "deliver"]):
            return "delivery days"

        # 2. Direct Time Span Counting
        if re.search(r"how many\s+(?:more\s+)?days\b", ql) or "number of days" in ql or "days did i" in ql or "days in total" in ql:
            return "days"
        if re.search(r"how many\s+(?:more\s+)?hours\b", ql) or "number of hours" in ql or "hours did i" in ql or "hours do i" in ql:
            return "hours"
        if re.search(r"how many\s+(?:more\s+)?weeks\b", ql) or "number of weeks" in ql or "weeks did i" in ql:
            return "weeks"
        if re.search(r"how many\s+(?:more\s+)?months\b", ql) or "number of months" in ql or "months did i" in ql:
            return "months"
        if re.search(r"how many\s+(?:more\s+)?years\b", ql) or "number of years" in ql or "years in total" in ql:
            return "years"
        if re.search(r"how many\s+(?:more\s+)?minutes\b", ql) or "number of minutes" in ql or "minutes did i" in ql:
            return "minutes"

        # 3. Financial Amount
        if any(w in ql for w in self._FINANCIAL_KEYWORDS):
            return "$"

        # 4. Noun Phrase Entity Extraction
        m_entity = re.search(
            r"how many\s+(?:different\s+)?([a-zA-Z\s_-]+?)(?:\s+did|\s+have|\s+do|\s+were|\s+are|\s+in\s+total|\s+altogether|\s+can|\s+could|\s+would|\s+will|\?|$)",
            ql,
        )
        if m_entity:
            ent = m_entity.group(1).strip()
            ent_tokens = [w for w in ent.split() if w not in self.STOP_WORDS]
            if ent_tokens:
                return " ".join(ent_tokens)

        if "day" in ql and not any(w in ql for w in ["in the past", "over the", "in the last"]):
            return "days"
        if "hour" in ql:
            return "hours"
        if "week" in ql and not any(w in ql for w in ["in the past", "over the", "in the last"]):
            return "weeks"

        return "items"

    def _word_root(self, w: str) -> str:
        """Extract simple stem/root for robust cross-session lemma matching."""
        w = w.lower()
        for suffix in ["ing", "ed", "es", "s"]:
            if len(w) > len(suffix) + 2 and w.endswith(suffix):
                return w[:-len(suffix)]
        return w

    def fuse(
        self,
        query: str,
        records: Sequence[StructuredIR] | str,
    ) -> AggregationResult:
        """Analyze query, extract cross-session evidence, and compute deterministic aggregation."""
        if not self.is_aggregation_query(query) and not self.is_multi_session_reasoning_query(query):
            return AggregationResult(is_aggregation_query=False)

        ql = query.lower()
        unit = self.determine_unit(query)

        # Extract salient query keywords (excluding stop words)
        q_words = set(
            w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", ql)
            if len(w) > 2 and w not in self.STOP_WORDS
        )
        for w in list(q_words):
            for syn_group in self._DOMAIN_SYNONYMS.values():
                if w in syn_group:
                    q_words.update(syn_group)
        q_roots = {self._word_root(w) for w in q_words}

        # Parse session lines
        session_lines: dict[str, list[str]] = defaultdict(list)
        curr_sid = "unknown"
        if isinstance(records, str):
            for line in records.split("\n"):
                if "Temporal Calculation:" in line and "passed between" in line:
                    continue
                m = re.search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]\s*(?:user|assistant)?:\s*(.*)", line)
                if m:
                    curr_sid = m.group(1)
                    content = m.group(2)
                    session_lines[curr_sid].append(content)
                elif line.strip() and not line.strip().startswith("[") and not line.strip().startswith("=="):
                    session_lines[curr_sid].append(line.strip())
        else:
            for r in records:
                content = r.raw_content or ""
                m = re.search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]\s*(?:user|assistant)?:\s*(.*)", content)
                sid = m.group(1) if m else (r.source or "unknown")
                text = m.group(2) if m else content
                session_lines[sid].append(text)

        found_snippets: list[tuple[str, str, float]] = []
        total_sum = 0.0

        # Case A: Financial Currency Aggregation ($)
        if unit == "$":
            seen_snippets = set()
            for sid, contents in session_lines.items():
                for content in contents:
                    for s in self._split_sentences(content):
                        s_lower = s.lower()
                        # Require financial notation or explicit currency token
                        m_curr = re.search(r"\$([0-9,]+(?:\.[0-9]{1,2})?)", s_lower) or re.search(r"\b([0-9,]+(?:\.[0-9]{1,2})?)\s+dollars\b", s_lower)
                        if not m_curr:
                            continue

                        val = float(m_curr.group(1).replace(",", ""))
                        if val <= 0:
                            continue

                        # Check relevance to query keywords or financial verbs
                        s_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", s_lower))
                        s_roots = {self._word_root(w) for w in s_words}
                        overlap = len((q_words & s_words) | (q_roots & s_roots))
                        has_finance_verb = any(v in s_lower for v in ["spent", "cost", "paid", "bought", "purchased", "earned", "price", "fee", "ordered", "picked up", "got"])

                        if overlap >= 1 or has_finance_verb:
                            snip_key = f"{sid}_{val}_{s.strip()[:40]}"
                            if snip_key not in seen_snippets:
                                seen_snippets.add(snip_key)
                                found_snippets.append((sid, s.strip(), val))
                                total_sum += val

        # Case B: Delivery Days
        elif unit == "delivery days":
            ord_date = None
            arr_date = None
            ord_sid, arr_sid = None, None
            ord_snip, arr_snip = "", ""

            for sid, contents in session_lines.items():
                for content in contents:
                    for s in self._split_sentences(content):
                        s_lower = s.lower()
                        m_d = re.search(r"\b(\d{1,2})/(\d{1,2})\b", s_lower)
                        m_w = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b", s_lower)
                        dt = None
                        if m_d:
                            dt = datetime.date(2024, int(m_d.group(1)), int(m_d.group(2)))
                        elif m_w:
                            m_name = m_w.group(1)[:3]
                            if m_name in self.MONTH_MAP:
                                dt = datetime.date(2024, self.MONTH_MAP[m_name], int(m_w.group(2)))

                        if dt and any(w in s_lower for w in ["bought", "ordered", "purchased"]):
                            ord_date, ord_sid, ord_snip = dt, sid, s.strip()
                        elif dt and any(w in s_lower for w in ["arrived", "received", "delivered"]):
                            arr_date, arr_sid, arr_snip = dt, sid, s.strip()

            if ord_date and arr_date:
                diff = float(abs((arr_date - ord_date).days))
                found_snippets.append((ord_sid or "s1", f"Ordered on {ord_date}: {ord_snip}", diff))
                found_snippets.append((arr_sid or "s2", f"Delivered on {arr_date}: {arr_snip}", diff))
                total_sum = diff
                unit = "days"

        # Case C: Temporal Duration Units (days, hours, weeks, months, years, minutes)
        elif unit in ["days", "hours", "weeks", "months", "years", "minutes"]:
            for sid, contents in session_lines.items():
                best_sentence = ""
                best_score = 0.0
                best_val = None

                for content in contents:
                    for s in self._split_sentences(content):
                        s_clean = s.strip()
                        if not s_clean:
                            continue
                        s_lower = s_clean.lower()
                        s_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", s_lower))
                        s_roots = {self._word_root(w) for w in s_words}
                        overlap = len((q_words & s_words) | (q_roots & s_roots))

                        val = None
                        if unit == "days":
                            m_day = re.search(r"\b(\d+(?:\.\d+)?)(?:-|\s+)days?\b", s_lower)
                            if m_day:
                                val = float(m_day.group(1))
                            elif any(w in s_lower for w in ["week-long", "a week", "one week", "1-week", "1 week"]):
                                val = 7.0
                            elif any(w in s_lower for w in ["two-week", "two weeks", "2-week", "2 weeks"]):
                                val = 14.0

                        elif unit == "hours":
                            m_min = re.search(r"\b(\d+)\s*-\s*minutes?\b|\b(\d+)\s+minutes?\b", s_lower)
                            if m_min:
                                val = float(m_min.group(1) or m_min.group(2)) / 60.0
                            else:
                                m_hr = re.search(r"\b(\d+(?:\.\d+)?)(?:-|\s+)hours?\b", s_lower)
                                if m_hr:
                                    val = float(m_hr.group(1))
                                else:
                                    for w, n in self.WORD_TO_NUM.items():
                                        if f"{w} hours" in s_lower:
                                            val = float(n)
                                            break

                        elif unit == "weeks":
                            if "week and a half" in s_lower:
                                val = 1.5
                            else:
                                m_wk = re.search(r"\b(\d+(?:\.\d+)?)(?:-|\s+)weeks?\b", s_lower)
                                if m_wk:
                                    val = float(m_wk.group(1))
                                else:
                                    for w, n in self.WORD_TO_NUM.items():
                                        if f"{w} weeks" in s_lower:
                                            val = float(n)
                                            break

                        elif unit == "minutes":
                            m_min = re.search(r"\b(\d+(?:\.\d+)?)(?:-|\s+)minutes?\b", s_lower)
                            if m_min:
                                val = float(m_min.group(1))

                        score = overlap + (2.0 if val is not None else 0.0)
                        if score > best_score and (overlap >= 1 or val is not None) and val is not None:
                            best_score = score
                            best_sentence = s_clean
                            best_val = val

                if best_sentence and best_val is not None:
                    found_snippets.append((sid, best_sentence, best_val))
                    total_sum += best_val

        # Case D: Quantified Entities / Counts
        else:
            seen_items = set()
            unit_root = self._word_root(unit)
            for sid, contents in session_lines.items():
                for content in contents:
                    for s in self._split_sentences(content):
                        s_lower = s.lower()
                        s_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", s_lower))
                        s_roots = {self._word_root(w) for w in s_words}
                        overlap = len((q_words & s_words) | (q_roots & s_roots))

                        if overlap >= 1 or unit_root in s_lower:
                            # Look for explicit numeric count
                            m_cnt = re.search(r"\b(\d+)\s+" + re.escape(unit_root) + r"[a-z]*\b", s_lower)
                            if m_cnt:
                                val = float(m_cnt.group(1))
                                found_snippets.append((sid, s.strip(), val))
                                total_sum += val
                                continue

                            # Look for distinct named entities or action occurrences
                            has_action = any(
                                act in s_lower
                                for act in [
                                    "bought", "ordered", "visited", "attended",
                                    "completed", "finished", "read", "tried", "watched",
                                ]
                            )
                            if (has_action or unit_root in s_lower) and s.strip() not in seen_items:
                                seen_items.add(s.strip())
                                found_snippets.append((sid, s.strip(), 1.0))
                                total_sum += 1.0

        if not found_snippets:
            return AggregationResult(is_aggregation_query=True, unit=unit, total_value=None, found_snippets=[])

        # Generate transparent, proof-carrying certificate from actual evidence
        lines = [
            f"[GLOBAL STATE AGGREGATION: Found {len(found_snippets)} distinct instance(s) across sessions:"
        ]
        for sid, snip, val in found_snippets:
            val_str = f"${val:g}" if unit == "$" else f"{val:g} {unit}" if unit else f"{val:g}"
            lines.append(f"- Session {sid}: \"{snip}\" ({val_str})")

        tot_str = f"${total_sum:g}" if unit == "$" else f"{total_sum:g} {unit}" if unit else f"{total_sum:g}"
        lines.append(f"Computed Total: {tot_str}.]")
        cert = "\n".join(lines)

        return AggregationResult(
            is_aggregation_query=True,
            total_value=total_sum,
            unit=unit,
            found_snippets=found_snippets,
            certificate=cert,
        )
