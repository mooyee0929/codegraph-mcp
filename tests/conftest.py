from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from codegraph.indexer import index_repository
from codegraph.store import GraphStore

FIXTURE = Path(__file__).parent / "fixtures" / "sample_repo"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway copy of the fixture repo, initialised as a git repository."""
    target = tmp_path / "repo"
    shutil.copytree(FIXTURE, target)
    subprocess.run(["git", "init", "-q"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.name", "ci"], cwd=target, check=True)
    subprocess.run(["git", "add", "-A"], cwd=target, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=target, check=True)
    return target


@pytest.fixture
def store(repo: Path) -> Iterator[GraphStore]:
    with GraphStore.open(repo) as graph_store:
        index_repository(repo, graph_store)
        yield graph_store
