"""Query Operator Engine — the public, three-layer composition.

    question --Parser--> AST --Executor--> answer + proof

This replaces the deleted R8.1 ``QueryOperatorCompiler``.  The critical
difference: values are extracted from records at execution time, so no answer
constant and no benchmark-topic branch can exist.  The engine is validated by
the synthetic counterfactual suite before it is trusted on any benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from artificial_memory.compiler.operators import ast_nodes as A
from artificial_memory.compiler.operators.executor import ExecutionResult, Executor
from artificial_memory.compiler.operators.parser import QueryParser


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

    def run(self, question: str, records: Sequence) -> OperatorOutcome:
        ast = self.parser.parse(question)
        if ast is None:
            return OperatorOutcome(
                has_result=False, answer=None, proof="", ast="", operator_name="NONE"
            )
        res: ExecutionResult = self.executor.execute(ast, records)
        return OperatorOutcome(
            has_result=res.has_result,
            answer=res.answer,
            proof=res.proof,
            ast=res.ast,
            operator_name=self._top_operator(ast),
        )

    @staticmethod
    def _top_operator(ast: A.Node) -> str:
        return type(ast).__name__.upper()
