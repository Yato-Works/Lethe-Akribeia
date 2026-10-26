"""Product defaults that the arena measurement changed.

The shipped runtime used to default to the configuration that scored 11.5% on the
controlled arena while the measured-better one stayed opt-in.  These tests pin the
new default, the environment override, and the lazy embedding load (the model used
to be loaded for commands that never encode anything).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from artificial_memory.memory.vector_search import FAISS_AVAILABLE, VectorSearchEngine
from artificial_memory.runtime.facade import RuntimeConfig


def test_lexical_recall_is_on_by_default() -> None:
    assert RuntimeConfig().lexical_recall is True


def test_conflict_surfacing_stays_off_by_default() -> None:
    # Product-path transfer is still partial (V4_PROGRAM 4.27), so this stays
    # opt-in even though the player-side arena result is a +6pt leader.
    assert RuntimeConfig().conflict_surfacing is False


def test_env_can_disable_lexical_recall(monkeypatch) -> None:
    monkeypatch.setenv("LETHE_LEXICAL_RECALL", "0")
    assert RuntimeConfig.from_env().lexical_recall is False

    monkeypatch.setenv("LETHE_LEXICAL_RECALL", "false")
    assert RuntimeConfig.from_env().lexical_recall is False


def test_env_can_enable_conflict_surfacing(monkeypatch) -> None:
    monkeypatch.setenv("LETHE_CONFLICT_SURFACING", "1")
    assert RuntimeConfig.from_env().conflict_surfacing is True


def test_env_accepts_overrides_and_ignores_empty_values(monkeypatch) -> None:
    monkeypatch.setenv("LETHE_LEXICAL_RECALL", "")
    config = RuntimeConfig.from_env(database_path="somewhere.db")
    assert config.lexical_recall is True
    assert config.database_path == "somewhere.db"


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
def test_vector_search_does_not_load_the_model_at_construction(tmp_path: Path) -> None:
    class _Store:
        pass

    engine = VectorSearchEngine(_Store(), index_path=tmp_path / "idx")
    assert engine._model is None
