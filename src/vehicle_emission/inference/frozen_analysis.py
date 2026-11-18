"""Versioned analysis orchestration.

The fixed v4 numeric-export engine is available through ``execute-v4``.
The legacy v2 route remains disabled. Its safe
``preflight`` command validates all committed method/provenance gates before it
hashes the out-of-band database and returns a receipt containing no empirical
values.  ``execute`` has no engine hook and always stops after preflight. A
future database implementation would need a safe database lease and mandatory
result-validation postcondition before it can interpret empirical values.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from vehicle_emission.inference.manifest_validation import (
    FileSnapshot,
    LightweightValidationReceipt,
    ManifestValidationError,
    ResultValidationContext,
    build_result_validation_context,
    snapshot_to_dict,
    validate_lightweight_gates,
    verify_external_database,
)


class ExecutionDisabledError(RuntimeError):
    """Raised because the reviewed execution boundary does not exist yet."""


@dataclass(frozen=True)
class FrozenAnalysisPreflightReceipt:
    """Safe receipt created after all gates and the final database hash pass."""

    evidence_status: str
    freeze_id: str
    repository_root: str
    head_commit: str
    analysis_code_commit: str
    engine_id: str
    freeze_sha256: str
    decision_sha256: tuple[str, ...]
    manifest_sha256: tuple[str, ...]
    evidence_sha256: tuple[str, ...]
    database_snapshot: FileSnapshot
    result_validation_context: ResultValidationContext
    database_bytes_hashed: bool
    empirical_values_interpreted: bool
    legacy_analysis_invoked: bool

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready representation."""

        value = asdict(self)
        value["database_snapshot"] = snapshot_to_dict(self.database_snapshot)
        return value


def _make_receipt(
    lightweight: LightweightValidationReceipt,
    database_snapshot: FileSnapshot,
) -> FrozenAnalysisPreflightReceipt:
    method = lightweight.method_config
    engine = method["engine"]
    assert isinstance(engine, Mapping)
    return FrozenAnalysisPreflightReceipt(
        evidence_status="preflight_only_no_empirical_analysis",
        freeze_id=lightweight.freeze_id,
        repository_root=lightweight.repository_root,
        head_commit=lightweight.head_commit,
        analysis_code_commit=lightweight.analysis_code_commit,
        engine_id=str(engine["engine_id"]),
        freeze_sha256=lightweight.freeze_snapshot.sha256,
        decision_sha256=tuple(snapshot.sha256 for snapshot in lightweight.decision_snapshots),
        manifest_sha256=tuple(snapshot.sha256 for snapshot in lightweight.manifest_snapshots),
        evidence_sha256=tuple(snapshot.sha256 for snapshot in lightweight.evidence_snapshots),
        database_snapshot=database_snapshot,
        result_validation_context=build_result_validation_context(
            lightweight,
            database_snapshot,
        ),
        database_bytes_hashed=True,
        empirical_values_interpreted=False,
        legacy_analysis_invoked=False,
    )


def preflight_frozen_analysis(
    freeze_path: Path,
    *,
    repository_root: Path,
) -> FrozenAnalysisPreflightReceipt:
    """Validate committed gates, then and only then hash the empirical DB."""

    lightweight = validate_lightweight_gates(
        freeze_path,
        repository_root=repository_root,
    )
    database_snapshot = verify_external_database(
        lightweight,
    )
    return _make_receipt(lightweight, database_snapshot)


def run_frozen_analysis(
    freeze_path: Path,
    *,
    repository_root: Path,
) -> object:
    """Perform preflight and then stop: no executable engine hook exists yet.

    A later implementation must add a descriptor-preserving immutable database
    lease and make strict result-manifest validation a mandatory postcondition.
    That reviewed boundary does not exist in this module, so neither a caller
    nor a manifest can register or inject an engine today.
    """

    receipt = preflight_frozen_analysis(
        freeze_path,
        repository_root=repository_root,
    )
    raise ExecutionDisabledError(
        "no executable analysis-engine hook exists; preflight passed for "
        f"{receipt.engine_id!r}, but empirical execution remains disabled"
    )


def _default_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "execute", "execute-v4"))
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--repository-root", type=Path, default=_default_root())
    arguments = parser.parse_args(argv)
    if arguments.command == "execute-v4":
        if (
            arguments.config is None
            or arguments.output_directory is None
            or arguments.freeze is not None
        ):
            parser.error("execute-v4 requires --config and --output-directory, not --freeze")
        from vehicle_emission.inference.v4_pipeline import main as v4_main

        return v4_main(
            [
                "--config",
                str(arguments.config),
                "--output-directory",
                str(arguments.output_directory),
                "--repository-root",
                str(arguments.repository_root),
            ]
        )
    if (
        arguments.freeze is None
        or arguments.config is not None
        or arguments.output_directory is not None
    ):
        parser.error("legacy commands require --freeze, not v4 configuration/output arguments")
    try:
        if arguments.command == "preflight":
            receipt = preflight_frozen_analysis(
                arguments.freeze,
                repository_root=arguments.repository_root,
            )
            print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
        else:
            run_frozen_analysis(
                arguments.freeze,
                repository_root=arguments.repository_root,
            )
    except (ManifestValidationError, ExecutionDisabledError, FileNotFoundError) as error:
        parser.exit(2, f"frozen analysis blocked: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
