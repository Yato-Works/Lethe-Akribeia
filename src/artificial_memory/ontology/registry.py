"""Ontology Registry: the single data plane for vocabulary tables.

This module is the *only* sanctioned place from which knowledge tables (person
rosters, concept synonym expansions, domain groups, geographic subsumptions,
aggregation evidence keywords, ...) may be loaded by runtime code.

Design contract:
    - Code declares *usage*; data files declare *content*.
    - All tables live in ``artificial_memory/ontology/data/*.yaml``.
    - Loading is deterministic: YAML mappings are materialized with sorted
      keys and lists preserve their declared order.
    - Loading is strict: a missing, malformed, or non-canonical data file
      raises :class:`OntologyError` instead of silently returning empty
      tables (no "trust-me" fallbacks).
    - Each data file carries a ``_meta`` block declaring its ``origin``
      (``generic`` vs ``benchmark_tuned``) so downstream audit tooling can
      distinguish neutral world-knowledge from corpus-derived tuning.

Typical usage::

    from artificial_memory.ontology import ontology

    table = ontology.mapping("concept_synonyms")
    actors = ontology.list("actor_roster")
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DATA_DIR = Path(__file__).resolve().parent / "data"

_VALID_ORIGINS = frozenset({"generic", "benchmark_tuned"})

_TABLE_NAMES = (
    "actor_roster",
    "aggregation_evidence",
    "concept_synonyms",
    "content_seeds",
    "coreference_seeds",
    "domain_groups",
    "domain_synonyms",
    "event_span_anchors",
    "evidence_idioms",
    "geography",
    "persona_gear_patterns",
    "project_codenames",
    "query_patterns",
    "referent_resolutions",
    "speaker_aliases",
    "synonym_expansion_targets",
    "tech_properties",
    "value_canonicalizations",
)


class OntologyError(RuntimeError):
    """Raised when an ontology data file is missing, malformed, or invalid."""


def _validate_origin(name: str, meta: Any) -> str:
    if not isinstance(meta, Mapping) or "origin" not in meta:
        raise OntologyError(
            f"ontology/{name}.yaml must declare a '_meta' block with an 'origin' key"
        )
    origin = meta["origin"]
    if origin not in _VALID_ORIGINS:
        raise OntologyError(
            f"ontology/{name}.yaml has invalid _meta.origin={origin!r}; "
            f"expected one of {sorted(_VALID_ORIGINS)}"
        )
    return origin


def _canonicalize(value: Any) -> Any:
    """Recursively canonicalize tables so ordering is deterministic."""
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    return value


class OntologyRegistry:
    """Deterministic, strict loader for ontology data tables."""

    def __init__(self, tables: Mapping[str, Mapping[str, Any]], origins: Mapping[str, str]) -> None:
        self._tables = dict(tables)
        self._origins = dict(origins)

    # ------------------------------------------------------------------
    # Access API
    # ------------------------------------------------------------------
    def table(self, name: str) -> Mapping[str, Any]:
        """Return a full YAML table (the raw document minus ``_meta``)."""
        try:
            return self._tables[name]
        except KeyError as exc:
            raise OntologyError(f"unknown ontology table {name!r}") from exc

    def mapping(self, name: str) -> Mapping[str, Any]:
        """Return a table whose top level is a mapping (concept -> values)."""
        table = self.table(name)
        if not isinstance(table, Mapping):
            raise OntologyError(f"ontology table {name!r} is not a mapping")
        return table

    def list(self, name: str, key: str = "names") -> list[str]:
        """Return a list-valued table (or the ``key`` entry of a mapping)."""
        table = self.table(name)
        if isinstance(table, list):
            return list(table)
        if isinstance(table, Mapping) and key in table and isinstance(table[key], list):
            return list(table[key])
        raise OntologyError(
            f"ontology table {name!r} does not contain a list under key {key!r}"
        )

    def section(self, name: str, section: str) -> Mapping[str, Any]:
        """Return the ``section`` sub-mapping of a table."""
        table = self.table(name)
        if not isinstance(table, Mapping) or not isinstance(table.get(section), Mapping):
            raise OntologyError(
                f"ontology table {name!r} has no mapping section {section!r}"
            )
        return table[section]

    def origin(self, name: str) -> str:
        """Declared provenance of a table: 'generic' or 'benchmark_tuned'."""
        try:
            return self._origins[name]
        except KeyError as exc:
            raise OntologyError(f"unknown ontology table {name!r}") from exc

    def table_names(self) -> list[str]:
        """All loaded table names, sorted."""
        return sorted(self._tables)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, data_dir: Path | None = None) -> OntologyRegistry:
        """Load and validate every declared ontology table."""
        directory = data_dir if data_dir is not None else _DATA_DIR
        tables: dict[str, dict[str, Any]] = {}
        origins: dict[str, str] = {}
        for name in _TABLE_NAMES:
            path = directory / f"{name}.yaml"
            if not path.is_file():
                raise OntologyError(f"missing ontology data file: {path}")
            try:
                doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise OntologyError(f"malformed ontology data file {path}: {exc}") from exc
            if not isinstance(doc, Mapping):
                raise OntologyError(f"ontology data file {path} must be a YAML mapping")
            meta = doc.pop("_meta", None)
            origins[name] = _validate_origin(name, meta)
            tables[name] = _canonicalize(doc)
        return cls(tables, origins)


_LOCK = threading.Lock()
_registry: OntologyRegistry | None = None


@lru_cache(maxsize=1)
def _get_singleton() -> OntologyRegistry:
    return OntologyRegistry.load()


def get_ontology() -> OntologyRegistry:
    """Process-wide ontology singleton (deterministic, loaded once)."""
    global _registry
    if _registry is None:
        with _LOCK:
            if _registry is None:
                _registry = _get_singleton()
    return _registry
