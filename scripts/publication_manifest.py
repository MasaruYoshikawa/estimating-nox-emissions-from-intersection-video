"""Bind the public package to portable, exact-byte file identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

MANIFEST = "results/publication_manifest_20261119.json"
ACTIVE = {
    "statistics": "results/monitor_quality_reanalysis_20261119",
    "quality_preparation": "results/monitor_quality_preparation_20261119",
    "receptor_quality": "results/monitor_quality_sources_20261119",
    "regional_quality": "results/setagaya_quality_sources_20261119",
    "supplements": "results/monitor_quality_supplements_20261119",
    "ordering": "results/monitor_quality_ordering_20261119",
    "figures": "results/monitor_quality_figures_20261119",
}
ROOT_FILES = (
    ".gitattributes",
    "README.md",
    "NOTICE.md",
    "LICENSE",
    "CITATION.cff",
    "Makefile",
    "pyproject.toml",
    "environment.yml",
    "requirements-dev.lock",
    ".gitignore",
)
SUBTREES = (".github", "archive", "data", "docs", "results", "scripts", "src", "tests")


def build_manifest(root: Path) -> dict[str, Any]:
    """Identify release files without changing historical receipts or results."""
    for name in ACTIVE.values():
        if not (root / name).is_dir():
            raise FileNotFoundError(f"active result bundle is missing: {name}")
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
        )
        .decode("utf-8")
        .split("\0")
    )
    candidates = [
        root / name
        for name in names
        if name
        and name != MANIFEST
        and (name in ROOT_FILES or name.split("/", 1)[0] in SUBTREES)
        and (root / name).is_file()
    ]
    files = {}
    for path in sorted(set(candidates)):
        if path.is_symlink():
            raise ValueError(f"release inputs must be ordinary files: {path.relative_to(root)}")
        with path.open("rb") as stream:
            files[path.relative_to(root).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    active_roots = set(ACTIVE.values())
    historical_receipts = [
        name
        for name in files
        if Path(name).name == "receipt.json" and Path(name).parent.as_posix() not in active_roots
    ]
    return {
        "schema_version": 1,
        "active_bundles": ACTIVE,
        "preserved_historical_receipts": historical_receipts,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / MANIFEST
    manifest = build_manifest(root)
    if args.check:
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("publication manifest differs from the current package")
        print(f"Verified {len(manifest['files'])} publication files")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"Recorded {len(manifest['files'])} publication files")


if __name__ == "__main__":
    main()
