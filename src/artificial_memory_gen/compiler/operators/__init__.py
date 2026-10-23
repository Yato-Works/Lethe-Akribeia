"""Query Operator Compiler (Lethe-Akribeia, CLEAN-R7 rebuild).

Three-layer design that structurally prevents benchmark overfitting:

    question ──Parser──▶ Operator AST ──Executor──▶ answer + proof

The Executor never sees the question; the Parser never sees the records.
Answers are extracted from records at run time — no constants, no topic
dispatch.  Validated by tests/unit/synthetic before use on any benchmark.
"""

from artificial_memory_gen.compiler.operators import ast_nodes as A
from artificial_memory_gen.compiler.operators.engine import OperatorOutcome, QueryOperatorEngine
from artificial_memory_gen.compiler.operators.executor import ExecutionResult, Executor
from artificial_memory_gen.compiler.operators.parser import QueryParser

__all__ = [
    "A",
    "QueryOperatorEngine",
    "QueryParser",
    "Executor",
    "OperatorOutcome",
    "ExecutionResult",
]
