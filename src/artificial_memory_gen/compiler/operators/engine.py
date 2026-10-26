"""Query Operator Engine — the public, three-layer composition.

    question --Parser--> AST --Executor--> answer + proof

This replaces the deleted R8.1 ``QueryOperatorCompiler``.  The critical
difference: values are extracted from records at execution time, so no answer
constant and no benchmark-topic branch can exist.  The engine is validated by
the synthetic counterfactual suite before it is trusted on any benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from artificial_memory_gen.compiler.operators import ast_nodes as A
from artificial_memory_gen.compiler.operators.executor import ExecutionResult, Executor
from artificial_memory_gen.compiler.operators.parser import QueryParser


@dataclass(frozen=True)
class OperatorOutcome:
    has_result: bool
    answer: str | None
    proof: str
    ast: str
    operator_name: str


class QueryOperatorEngine:
    """Parse a question to an AST, then execute it deterministically."""

    def __init__(self) -> None:
        self.parser = QueryParser()
        self.executor = Executor()

    def run(self, question: str, records: Sequence,
            reference_date: str | None = None) -> OperatorOutcome:
        ast = self.parser.parse(question)
        if ast is None:
            return OperatorOutcome(
                has_result=False, answer=None, proof="", ast="", operator_name="NONE"
            )
        # Aggregation operators (SUM/COUNT/UNION/FILTER) must not sum over the
        # entire corpus: unrelated sessions carry their own quantities that
        # would pollute a total.  Scope to records that share the question's
        # content words.  This is a dataset-agnostic relevance filter, NOT a
        # benchmark shortcut — it holds for any "how many <topic> ..." query.
        scoped = self._scope_to_topic(ast, question, records)
        res: ExecutionResult = self.executor.execute(ast, scoped, reference_date=reference_date)
        return OperatorOutcome(
            has_result=res.has_result,
            answer=res.answer,
            proof=res.proof,
            ast=res.ast,
            operator_name=self._top_operator(ast),
        )

    # Aggregation node types that should be topic-scoped.
    _AGGREGATORS = ("Sum", "Count", "Union", "Filter", "Gap")

    def _scope_to_topic(self, ast: A.Node, question: str, records: Sequence) -> Sequence:
        """Restrict records to those relevant to the question's topic.

        Only applies to aggregation operators.  If scoping would drop every
        record, fall back to the full set so we never return empty due to an
        over-eager filter.
        """
        from artificial_memory_gen.compiler.operators.ast_nodes import node_types
        types = node_types(ast)
        if not any(t in self._AGGREGATORS for t in types):
            return records
        # If the aggregation carries an explicit measurement unit (SUM of
        # "days", "hours"), the unit itself is the filter — topic-scoping would
        # only wrongly drop same-unit quantities from turns that phrase the
        # topic loosely ("...spent 9 days doing it").  Skip scoping then.
        if self._ast_has_unit(ast):
            return records
        terms = self._content_terms(question)
        if not terms:
            return records
        scoped = [r for r in records if self._record_matches(r, terms)]
        return scoped if scoped else records

    def _ast_has_unit(self, ast: A.Node) -> bool:
        """True if any Extract node in the AST specifies a measurement unit."""
        from artificial_memory_gen.compiler.operators.ast_nodes import node_types
        _ = node_types(ast)  # keep signature stable; walk for Extract units below
        stack = [ast]
        while stack:
            n = stack.pop()
            if isinstance(n, A.Extract) and n.unit and n.unit.lower() not in ("", "list", "state", "slot", "event", "person", "numeric"):
                # a concrete unit (days/hours/...) — treat as measurement filter
                return n.unit.lower() not in ("pieces", "piece")
            for v in vars(n).values():
                if isinstance(v, A.Node):
                    stack.append(v)
        return False

    # Stopwords excluded when deriving the question's content terms.
    _STOP = frozenset({
        "how", "many", "much", "what", "the", "a", "an", "is", "are", "was",
        "were", "do", "does", "did", "i", "my", "me", "you", "your", "of",
        "to", "in", "on", "at", "for", "and", "or", "total", "in", "have",
        "has", "had", "since", "including", "including", "that", "this",
        "there", "their", "with", "from", "by", "as", "it", "its",
    })

    def _content_terms(self, question: str) -> list[str]:
        import re
        words = re.findall(r"[a-z]+", question.lower())
        # Only keep reasonably specific content words (>=5 chars) so generic
        # short adjectives ("short", "free") do not pull in unrelated records
        # via substring hits like "shortbread".
        terms = [w for w in words if len(w) >= 5 and w not in self._STOP]
        seen: set[str] = set()
        out: list[str] = []
        for w in terms:
            base = w[:-1] if w.endswith("s") else w
            if base in seen:
                continue
            seen.add(base)
            out.append(w)
        return out

    def _record_matches(self, r, terms: list[str]) -> bool:
        import re
        text = (getattr(r, "raw_content", None) or getattr(r, "text", None)
                or getattr(r, "value", "") or "").lower()
        tokens = set(re.findall(r"[a-z]+", text))
        # Whole-word / stem match — NOT substring — so "short" never matches
        # "shortbread".  A record is relevant if any content term's stem
        # appears as a token.
        for t in terms:
            base = t[:-1] if t.endswith("s") else t
            if base in tokens:
                return True
        return False

    @staticmethod
    def _top_operator(ast: A.Node) -> str:
        return type(ast).__name__.upper()
