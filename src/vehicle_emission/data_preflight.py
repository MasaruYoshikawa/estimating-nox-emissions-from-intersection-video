"""Deterministic availability check for empirical reproduction inputs."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict

InputKind = Literal["file", "nonempty_directory", "registered_git_submodule"]


class InputRecord(TypedDict, total=False):
    path: str
    status: Literal["ready", "missing"]
    recovery: str
    size_bytes: int
    sha256: str
    expected_sha256_or_commit: str


class PreflightReport(TypedDict):
    schema_version: int
    status: Literal["ready", "missing"]
    required_inputs: list[InputRecord]


@dataclass(frozen=True)
class InputRequirement:
    """One reproduction input and its non-destructive recovery instruction."""

    path: str
    kind: InputKind
    recovery: str
    expected_sha256_or_commit: str | None = None


REQUIRED_INPUTS: Final[tuple[InputRequirement, ...]] = (
    InputRequirement(
        "data/airpollution_history/AirPollution_Sukiyabashi.csv",
        "file",
        "Restore the tracked source CSV from Git and verify its recorded provenance.",
    ),
    InputRequirement(
        "data/airpollution_history/AirPollution_Shinjuku.csv",
        "file",
        "Restore the tracked source CSV from Git; do not replace it with a tuned date subset.",
    ),
    InputRequirement(
        "data/airpollution_history/AirPollution_Setagaya.csv",
        "file",
        "Restore the tracked sensitivity-source CSV from Git.",
    ),
    InputRequirement(
        "data/weather_history/weather_history.csv",
        "file",
        "Restore the tracked JMA-derived CSV from Git and retain its source metadata.",
    ),
    InputRequirement(
        "data/results.db",
        "file",
        "Restore the exact out-of-band SQLite snapshot, or rebuild a new versioned database "
        "after the correctness and measurement gates pass.",
    ),
    InputRequirement(
        "data/manifests/video_asset_inventory_v2.json",
        "file",
        "Create a private-video inventory with site, date, duration, local path, and SHA-256; "
        "the raw recordings themselves remain ignored and out of band.",
    ),
    InputRequirement(
        "data/manifests/feature_asset_inventory_v2.json",
        "file",
        "Create a numeric-feature inventory with site, date, source-video hash, path, and "
        "SHA-256; feature files remain ignored and out of band.",
    ),
    InputRequirement(
        "data/manifests/model_asset_inventory_v2.json",
        "file",
        "Create a detector-model inventory with checkpoint path, hash, training run, and "
        "software/runtime identifiers; checkpoint files remain out of band.",
    ),
    InputRequirement(
        "data/features",
        "nonempty_directory",
        "Restore numeric per-vehicle feature exports from private storage; do not add raw frames.",
    ),
    InputRequirement(
        "data/raw_video",
        "nonempty_directory",
        "Restore the private source recordings locally under this ignored directory; do not commit them.",
    ),
    InputRequirement(
        "runs/rtdetr_finetune-4/weights/best.pt",
        "file",
        "Restore the exact detector checkpoint out of band and record its SHA-256 hash.",
    ),
    InputRequirement(
        "src/vehicle_emission/baselines/RT-DETR",
        "registered_git_submodule",
        "Restore the missing .gitmodules registration (path and URL) for the tracked "
        "RT-DETR gitlink, then initialise the submodule at its recorded commit.",
        "a21d516aca15da57e65f35c47659c7535ad2b6b3",
    ),
    InputRequirement(
        "data/calibration/site_calibration_v2.json",
        "file",
        "Create the reviewed two-site perspective-aware calibration artefact under T-125.",
    ),
    InputRequirement(
        "data/manifests/two_site_assets_v2.json",
        "file",
        "Create a two-site asset manifest with site, period, source, path, and SHA-256 fields.",
    ),
)


def _registered_git_submodule_is_ready(
    root: Path,
    relative_path: str,
    expected_commit: str | None,
) -> bool:
    """Return whether a registered submodule is checked out at the expected commit."""

    gitmodules_path = root / ".gitmodules"
    if not gitmodules_path.is_file() or gitmodules_path.stat().st_size == 0:
        return False

    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(gitmodules_path, encoding="utf-8")
    except configparser.Error:
        return False

    registered = any(
        section.startswith("submodule ")
        and config.get(section, "path", fallback="").strip() == relative_path
        and bool(config.get(section, "url", fallback="").strip())
        for section in config.sections()
    )
    checkout = root / relative_path
    if not registered or not checkout.is_dir():
        return False
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    observed_commit = completed.stdout.strip().lower()
    return bool(observed_commit) and (
        expected_commit is None or observed_commit == expected_commit.lower()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_preflight_report(
    root: Path,
    requirements: Sequence[InputRequirement] = REQUIRED_INPUTS,
) -> PreflightReport:
    """Inspect required inputs without downloading, altering, or substituting them."""

    records: list[InputRecord] = []
    for requirement in requirements:
        candidate = root / requirement.path
        if requirement.kind == "file":
            present = candidate.is_file() and candidate.stat().st_size > 0
        elif requirement.kind == "nonempty_directory":
            present = candidate.is_dir() and any(item.is_file() for item in candidate.rglob("*"))
        else:
            present = _registered_git_submodule_is_ready(
                root,
                requirement.path,
                requirement.expected_sha256_or_commit,
            )

        record: InputRecord = {
            "path": requirement.path,
            "status": "ready" if present else "missing",
            "recovery": requirement.recovery,
        }
        if present and requirement.kind == "file":
            record["size_bytes"] = candidate.stat().st_size
            record["sha256"] = _sha256(candidate)
        if requirement.expected_sha256_or_commit is not None:
            record["expected_sha256_or_commit"] = requirement.expected_sha256_or_commit
        records.append(record)

    return {
        "schema_version": 1,
        "status": "ready" if all(item["status"] == "ready" for item in records) else "missing",
        "required_inputs": records,
    }


def write_preflight_report(report: PreflightReport, output_path: Path) -> None:
    """Write a stable JSON report suitable for task evidence."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, default=Path("build/data_preflight.json"))
    args = parser.parse_args()

    report = build_preflight_report(args.root.resolve())
    write_preflight_report(report, args.report)
    missing = [
        str(item["path"]) for item in report["required_inputs"] if item["status"] == "missing"
    ]
    if missing:
        print(f"Empirical input preflight: missing {len(missing)} required item(s).")
        for path in missing:
            print(f"  - {path}")
        print(f"Machine-readable report: {args.report}")
        return 2

    print(f"Empirical input preflight: ready. Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
