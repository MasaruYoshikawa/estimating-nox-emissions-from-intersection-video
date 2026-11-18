from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

TRAINING_PROVENANCE_SCHEMA = "detector_training_provenance_v4"
CLASS_NAMES = ["bus", "car", "heavy truck", "light truck"]
TRAINING_PARAMETERS = {"epochs": 50, "imgsz": 640, "batch": 4, "deterministic": True}
DATASET_CONFIG = {
    "train": "train/images",
    "val": "valid/images",
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES,
}
CODE_PATHS = (
    "pyproject.toml",
    "src/vehicle_emission/detection/__init__.py",
    "src/vehicle_emission/detection/evaluate.py",
    "src/vehicle_emission/detection/lineage.py",
    "src/vehicle_emission/detection/train.py",
)
LIBRARY_NAMES = (
    "numpy",
    "pandas",
    "PyYAML",
    "pillow",
    "opencv-python",
    "torch",
    "torchvision",
    "ultralytics",
)


def detector_code_identity(repository_root: Path) -> dict[str, str]:
    """Hash the detector's executing source files and dependency descriptor."""

    root = repository_root.resolve()
    identities: dict[str, str] = {}
    for relative_path in CODE_PATHS:
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Detector code path escapes repository root: {relative_path}")
        identities[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identities


def detector_library_versions() -> dict[str, str]:
    """Read actual installed detector-library versions without importing a model."""

    versions = {"python": platform.python_version()}
    for name in LIBRARY_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required detector library is not installed: {name}") from exc
    return versions


def validate_library_versions(
    recorded: Any,
    repository_root: Path,
    *,
    revision: str | None = None,
) -> None:
    """Require complete versions satisfying their own recorded dependency descriptor."""

    required = {"python", *LIBRARY_NAMES}
    if not isinstance(recorded, dict) or set(recorded) != required:
        raise ValueError("Detector training library_versions fields are incomplete or unsupported")
    if any(not isinstance(value, str) or not value.strip() for value in recorded.values()):
        raise ValueError("Detector training library versions must be non-empty strings")
    if re.fullmatch(r"3\.11\.\d+", recorded["python"]) is None:
        raise ValueError("Detector training requires the pinned Python 3.11 runtime")
    descriptor_bytes = (
        code_at_revision(repository_root, revision, "pyproject.toml")
        if revision is not None
        else (repository_root / "pyproject.toml").read_bytes()
    )
    descriptor = tomllib.loads(descriptor_bytes.decode("utf-8"))
    dependencies = [
        *descriptor["project"]["dependencies"],
        *descriptor["project"]["optional-dependencies"]["vision"],
    ]
    pins = {
        name.lower(): pin
        for name, pin in (dependency.split("==", 1) for dependency in dependencies)
    }
    for name in LIBRARY_NAMES:
        if name.lower() not in pins or recorded[name].split("+", 1)[0] != pins[name.lower()]:
            raise ValueError(f"Detector training library version violates the project pin: {name}")


def validate_code_identity(
    recorded: Any,
    repository_root: Path,
    revision: str,
    *,
    injected_fixture: bool,
    require_current: bool = False,
) -> None:
    """Verify historical source bytes, and optionally require unchanged current execution."""

    if not isinstance(recorded, dict) or set(recorded) != set(CODE_PATHS):
        raise ValueError("Detector training code_files_sha256 fields are incomplete or unsupported")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in recorded.values()
    ):
        raise ValueError("Detector training code identities must be SHA-256 digests")
    if require_current and recorded != IMPORTED_CODE_IDENTITY:
        raise ValueError("Detector current source bytes do not match the imported code identity")
    if (require_current or injected_fixture) and detector_code_identity(
        repository_root
    ) != recorded:
        raise ValueError(
            "Detector training code identity does not match the executing source bytes"
        )
    if injected_fixture:
        return
    if (
        require_current
        and Path(__file__).resolve()
        != (repository_root / "src/vehicle_emission/detection/lineage.py").resolve()
    ):
        raise ValueError("Detector must execute the source recorded under repository_root")
    for relative_path, digest in recorded.items():
        if (
            hashlib.sha256(code_at_revision(repository_root, revision, relative_path)).hexdigest()
            != digest
        ):
            raise ValueError("Detector training code identity does not match its recorded revision")


def code_at_revision(repository_root: Path, revision: str, relative_path: str) -> bytes:
    """Read one source file from a required historical Git object without modifying the tree."""

    process = subprocess.run(
        ["git", "-C", str(repository_root), "show", f"{revision}:{relative_path}"],
        capture_output=True,
        timeout=15,
    )
    if process.returncode:
        raise ValueError("Detector training code revision or its source files are unavailable")
    return process.stdout


# Capture the source loaded by this process, not just the disk state when a run starts.
IMPORTED_CODE_IDENTITY = detector_code_identity(Path(__file__).resolve().parents[3])
