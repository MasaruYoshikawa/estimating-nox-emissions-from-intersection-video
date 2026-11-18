"""Guarded, non-destructive orchestration helpers for project reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from vehicle_emission import database

REQUIRED_FREEZE_FIELDS = {
    "schema_version",
    "status",
    "approved_by",
    "approved_at_utc",
    "git_commit",
    "source_db_sha256",
    "soramame_hour_convention",
    "eligibility_spec_sha256",
    "inference_spec_sha256",
}

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SORAMAME_HOUR_CONVENTIONS = {"end-labelled-hour", "centre-labelled-hour"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_freeze_manifest(manifest: object) -> dict[str, object]:
    """Validate the human approval and immutable-input fields of a freeze."""

    if not isinstance(manifest, dict):
        raise ValueError("Analysis-freeze manifest must be a JSON object")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "accepted":
        raise ValueError(
            "Analysis-freeze manifest must have schema_version=1 and status='accepted'"
        )
    missing_fields = sorted(REQUIRED_FREEZE_FIELDS - manifest.keys())
    if missing_fields:
        raise ValueError(f"Analysis-freeze manifest is missing required fields: {missing_fields}")

    approved_by = manifest["approved_by"]
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("Analysis-freeze field 'approved_by' must be non-empty")
    approved_at = manifest["approved_at_utc"]
    if not isinstance(approved_at, str):
        raise ValueError("Analysis-freeze field 'approved_at_utc' must be canonical UTC")
    try:
        datetime.strptime(approved_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(
            "Analysis-freeze field 'approved_at_utc' must use YYYY-MM-DDTHH:MM:SSZ"
        ) from error

    convention = manifest["soramame_hour_convention"]
    if convention not in SORAMAME_HOUR_CONVENTIONS:
        raise ValueError(
            "Analysis-freeze field 'soramame_hour_convention' must name an approved convention"
        )
    commit = manifest["git_commit"]
    if not isinstance(commit, str) or GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("Analysis-freeze field 'git_commit' must be a full Git object ID")
    for key in (
        "source_db_sha256",
        "eligibility_spec_sha256",
        "inference_spec_sha256",
    ):
        value = manifest[key]
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"Analysis-freeze field {key!r} must be a SHA-256 hex digest")
    return manifest


def _verified_clean_git_state() -> tuple[Path, str]:
    """Return repository root and HEAD, refusing an uncommitted source state."""

    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    ).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    if status.strip():
        raise ValueError(
            "Repository has uncommitted or untracked files; an accepted freeze must name "
            "a clean, committed source state"
        )
    return root, commit


def prepare_working_database(
    source_db: Path,
    working_db: Path,
    freeze_manifest: Path,
) -> dict[str, object]:
    """Validate the method freeze and copy an empirical DB into ``build/``.

    Existing empirical databases and committed result files are never modified.
    The accepted freeze manifest is intentionally absent until the later
    inference-recovery gate, so a premature full run fails before any copy.
    """

    if not source_db.is_file():
        raise FileNotFoundError(f"Empirical database is missing: {source_db}")
    if not freeze_manifest.is_file():
        raise FileNotFoundError(
            "Accepted Phase 5 analysis-freeze manifest is missing: " f"{freeze_manifest}"
        )

    manifest = _validate_freeze_manifest(json.loads(freeze_manifest.read_text(encoding="utf-8")))
    source_hash = _sha256(source_db)
    if manifest["source_db_sha256"] != source_hash:
        raise ValueError("Source database hash does not match the accepted freeze manifest")
    project_root, current_commit = _verified_clean_git_state()
    if manifest["git_commit"] != current_commit:
        raise ValueError("Current Git commit does not match the accepted freeze manifest")

    resolved_working_db = working_db.resolve()
    protected_roots = [(project_root / name).resolve() for name in ("data", "results", "figures")]
    if any(resolved_working_db.is_relative_to(root) for root in protected_roots):
        raise ValueError("Working database must be written under build/, not data/results/figures")
    build_root = (project_root / "build").resolve()
    if not resolved_working_db.is_relative_to(build_root):
        raise ValueError("Working database must be written under build/")

    working_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_db, working_db)
    return manifest


def run_fixture_smoke(output_dir: Path) -> Path:
    """Run a deterministic database/count orchestration smoke test on fixtures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "fixture.db"
    if db_path.exists():
        db_path.unlink()
    database.init_db(str(db_path))

    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc, group_id
        ) VALUES (1, 'fixture', 'fixture.mp4', 'Sukiyabashi',
                  '2026-04-30T15:00:00Z', '2026-04-30', '15:00:00',
                  '2026-04-30T15:01:00Z', 'fixture-group')
        """
    )
    connection.commit()
    connection.close()
    database.insert_entering_counts(
        str(db_path),
        1,
        [
            {"direction": "entering_from_left", "class_id": None, "vehicle_count": 3},
            {"direction": "entering_from_left", "class_id": 1, "vehicle_count": 2},
            {"direction": "entering_from_left", "class_id": 2, "vehicle_count": 1},
        ],
    )

    connection = sqlite3.connect(db_path)
    rows, canonical_total, heavy_total = connection.execute(
        """
        SELECT
            COUNT(*),
            SUM(CASE WHEN class_id IS NULL THEN vehicle_count ELSE 0 END),
            SUM(CASE WHEN class_id IN (0, 2) THEN vehicle_count ELSE 0 END)
        FROM entering_counts
        WHERE video_id = 1
        """
    ).fetchone()
    connection.close()

    summary = {
        "schema_version": 1,
        "steps": ["initialise_database", "write_entering_counts", "verify_fixture"],
        "stored_rows": rows,
        "canonical_vehicle_count": canonical_total,
        "heavy_vehicle_count": heavy_total,
        "status": "ready",
    }
    summary_path = output_dir / "pipeline_smoke.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="copy inputs into a guarded build workspace")
    prepare.add_argument("--source-db", type=Path, required=True)
    prepare.add_argument("--working-db", type=Path, required=True)
    prepare.add_argument("--freeze-manifest", type=Path, required=True)

    smoke = subparsers.add_parser("smoke", help="run deterministic fixture orchestration")
    smoke.add_argument("--output-dir", type=Path, default=Path("build/pipeline_smoke"))

    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_working_database(args.source_db, args.working_db, args.freeze_manifest)
        runtime_path = args.working_db.parent / "accepted_freeze_runtime.json"
        runtime_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "soramame_hour_convention": manifest["soramame_hour_convention"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Prepared non-destructive working database: {args.working_db}")
        print(f"Prepared accepted runtime settings: {runtime_path}")
    else:
        result = run_fixture_smoke(args.output_dir)
        print(f"Fixture orchestration passed: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
