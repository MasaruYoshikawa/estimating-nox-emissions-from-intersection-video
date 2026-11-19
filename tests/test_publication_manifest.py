"""The release manifest is deterministic and uses repository-relative paths."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from scripts.publication_manifest import ACTIVE, build_manifest


def test_manifest_is_portable_and_preserves_receipt_identity(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.egg-info/\n")
    for name in ACTIVE.values():
        directory = tmp_path / name
        directory.mkdir(parents=True)
        (directory / "values.csv").write_text("value\n1\n")
    historical = tmp_path / "results/original/receipt.json"
    historical.parent.mkdir()
    historical.write_bytes(b'{"source": "historical receipt"}\n')
    cache = tmp_path / "src/example.egg-info"
    cache.mkdir(parents=True)
    (cache / "PKG-INFO").write_text("local installation metadata")
    first = build_manifest(tmp_path)
    assert first == build_manifest(tmp_path)
    assert first["preserved_historical_receipts"] == ["results/original/receipt.json"]
    assert (
        first["files"]["results/original/receipt.json"]
        == hashlib.sha256(historical.read_bytes()).hexdigest()
    )
    assert all(not Path(name).is_absolute() for name in first["files"])
    assert not any(".egg-info/" in name for name in first["files"])
    historical.write_text("changed")
    assert first != build_manifest(tmp_path)


def test_manifest_requires_the_active_bundles(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="active result bundle"):
        build_manifest(tmp_path)
