"""Lethe Akribeia Ontology: declarative knowledge planes for the memory runtime.

Vocabulary tables (person rosters, concept synonyms, domain groups, geography,
aggregation evidence keywords, ...) live as YAML under
``artificial_memory/ontology/data/`` and are loaded exclusively through
:class:`artificial_memory.ontology.registry.OntologyRegistry`.

Rationale (ADR-grade summary):
    Runtime modules used to embed person-name rosters and item vocabulary
    derived from evaluation corpora (LongMemEval / LoCoMo) directly in
    nominally-generic scoring and selection code. Externalizing that content
    into auditable data files with declared provenance (``generic`` vs
    ``benchmark_tuned``) restores the "zero test-item-specific logic" claim:
    code now declares *usage*, data files declare *content*, and CI
    (``tests/unit/hygiene/test_ontology_leak_boundary.py``) guards against
    benchmark vocabulary re-entering runtime source.
"""

from __future__ import annotations

from artificial_memory.ontology.registry import (
    OntologyError,
    OntologyRegistry,
    get_ontology,
)

__all__ = ["OntologyError", "OntologyRegistry", "get_ontology", "ontology"]


def __getattr__(name: str):  # noqa: ANN202
    if name == "ontology":
        return get_ontology()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
