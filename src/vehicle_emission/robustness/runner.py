"""Hash-bound optional diagnostics on fresh build paths, with read-only replay."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_emission.robustness.speed import SpeedPerturbation, calculate_speed_sensitivity
from vehicle_emission.robustness.temporal import TemporalControlConfig, calculate_temporal_controls

SCHEMA_VERSION = 1
STATUS = "optional_diagnostics_only"
SCOPE = "locked_retrospective_two_site"
TEXT_SOURCE_COLUMNS = (
    "site",
    "source_id",
    "profile_id",
    "ts_utc",
    "monitor_start_utc",
    "monitor_end_utc",
    "local_calendar_date",
    "target_observation_status",
)
CODE_PATHS = (
    "src/vehicle_emission/robustness/__init__.py",
    "src/vehicle_emission/robustness/speed.py",
    "src/vehicle_emission/robustness/temporal.py",
    "src/vehicle_emission/robustness/runner.py",
    "src/vehicle_emission/emissions/movestar.py",
    "src/vehicle_emission/vehicle_classes.py",
    "src/vehicle_emission/time_utils.py",
    "src/vehicle_emission/emissions/VehicleSrcCoeff.csv",
    "src/vehicle_emission/emissions/EmsRate_1.csv",
    "src/vehicle_emission/emissions/EmsRate_2.csv",
    "src/vehicle_emission/emissions/EmsRate_6.csv",
    "src/vehicle_emission/emissions/EmsRate_13.csv",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Capture the actual imported calculator/coefficient tree, not an arbitrary
# caller-supplied root. Dirty source is allowed only when these exact bytes match.
_IMPORTED_ROOT = Path(__file__).resolve().parents[3]
_IMPORTED_CODE_SHA256 = {name: _sha256((_IMPORTED_ROOT / name).read_bytes()) for name in CODE_PATHS}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _inside_root(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        raise ValueError("receipt/config paths must be repository-relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("paths must stay inside the repository")
    return resolved


def _load_source(spec: Any, root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError("each source requires exactly path and sha256")
    if not isinstance(spec["path"], str) or not isinstance(spec["sha256"], str):
        raise ValueError("source path/hash must be strings")
    path = _inside_root(spec["path"], root)
    data = path.read_bytes()
    if _sha256(data) != spec["sha256"]:
        raise ValueError(f"source hash mismatch: {spec['path']}")
    # Identifiers are arbitrary strings: e.g. "001" and "NA" are not numbers
    # or missing identities. Only the numeric speed column uses pandas' default
    # missing-value vocabulary, from the same hash-verified immutable bytes.
    rows = pd.read_csv(
        io.BytesIO(data),
        keep_default_na=False,
        dtype=dict.fromkeys(TEXT_SOURCE_COLUMNS, str),
    )
    if "speed_mps" in rows:
        rows["speed_mps"] = pd.read_csv(io.BytesIO(data), usecols=["speed_mps"])["speed_mps"]
    return rows, {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(data),
    }


def _calculate(config: Any, root: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    required = {
        "schema_version",
        "analysis_scope",
        "status",
        "predeclaration_id",
        "speed",
        "temporal",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"configuration requires exactly {sorted(required)}")
    if (
        config["schema_version"] != SCHEMA_VERSION
        or config["status"] != STATUS
        or config["analysis_scope"] != SCOPE
    ):
        raise ValueError("unsupported robustness schema/scope/status")
    if not isinstance(config["predeclaration_id"], str) or not config["predeclaration_id"].strip():
        raise ValueError(
            "an explicit predeclaration_id is required; this is not evidence of acceptance"
        )
    tables: dict[str, pd.DataFrame] = {}
    sources: dict[str, object] = {}
    states: dict[str, str] = {"speed": "not_requested", "temporal": "not_requested"}
    speed = config["speed"]
    if speed is not None:
        if not isinstance(speed, dict) or set(speed) != {"profiles", "scenarios"}:
            raise ValueError("speed requires exactly profiles and scenarios")
        profiles, sources["profiles"] = _load_source(speed["profiles"], root)
        if not isinstance(speed["scenarios"], list) or not all(
            isinstance(item, dict) for item in speed["scenarios"]
        ):
            raise ValueError("scenarios must be a list of explicitly specified perturbations")
        scenarios = tuple(SpeedPerturbation(**item) for item in speed["scenarios"])
        calculated_speed = calculate_speed_sensitivity(profiles, scenarios)
        tables.update(
            {
                "speed_profiles.csv": calculated_speed.profiles,
                "speed_runs.csv": calculated_speed.runs,
                "speed_sensitivity.csv": calculated_speed.summary,
            }
        )
        states["speed"] = "calculated_profile_level"
    temporal = config["temporal"]
    if temporal is not None:
        if not isinstance(temporal, dict) or set(temporal) != {
            "hourly_rows",
            "calendar_start_date",
            "calendar_end_date",
            "day_offsets",
            "acf_lags_hours",
        }:
            raise ValueError(
                "temporal requires hourly_rows, full calendar bounds, day_offsets and acf_lags_hours"
            )
        hourly_rows, sources["hourly_rows"] = _load_source(temporal["hourly_rows"], root)
        settings = TemporalControlConfig(
            temporal["calendar_start_date"],
            temporal["calendar_end_date"],
            tuple(temporal["day_offsets"]),
            tuple(temporal["acf_lags_hours"]),
        )
        calculated_temporal = calculate_temporal_controls(hourly_rows, settings)
        tables.update(
            {
                "temporal_contrasts.csv": calculated_temporal.contrasts,
                "temporal_support.csv": calculated_temporal.support,
                "temporal_day_mapping.csv": calculated_temporal.day_mapping,
                "temporal_acf.csv": calculated_temporal.autocorrelation,
            }
        )
        states["temporal"] = "calculated_full_calendar_whole_day_circular_shifts"
    output = {
        name: frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()
        for name, frame in tables.items()
    }
    return output, {"sources": sources, "diagnostics": states}


def _code_hashes(root: Path) -> dict[str, str]:
    hashes = {name: _sha256((root / name).read_bytes()) for name in CODE_PATHS}
    current_imported = {name: _sha256((_IMPORTED_ROOT / name).read_bytes()) for name in CODE_PATHS}
    if hashes != _IMPORTED_CODE_SHA256 or current_imported != _IMPORTED_CODE_SHA256:
        raise ValueError("repository/code/coefficient bytes differ from the imported calculator")
    return hashes


def _check_fresh_target(root: Path, output_directory: Path) -> None:
    if output_directory.exists() or output_directory.is_symlink():
        raise ValueError("output directory already exists; diagnostics never overwrite")
    tracked = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            "HEAD",
            "--",
            output_directory.relative_to(root).as_posix(),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if tracked:
        raise ValueError("output target contains committed artifacts, even if locally deleted")


def _recheck_inputs(
    config_path: Path,
    config_data: bytes,
    root: Path,
    inputs: dict[str, object],
    code_hashes: dict[str, str],
) -> None:
    if config_path.read_bytes() != config_data:
        raise ValueError("configuration changed during calculation")
    sources = inputs["sources"]
    assert isinstance(sources, dict)
    for source in sources.values():
        if _sha256(_inside_root(source["path"], root).read_bytes()) != source["sha256"]:
            raise ValueError("source changed during calculation")
    if _code_hashes(root) != code_hashes:
        raise ValueError("code/coefficient bytes changed during calculation")


def run_robustness(
    config_path: Path, output_directory: Path, repository_root: Path
) -> dict[str, object]:
    """Calculate optional diagnostics, rejecting existing/non-build output paths."""

    root = repository_root.resolve()
    config_path = config_path.resolve()
    if output_directory.is_symlink():
        raise ValueError("output directory already exists as a symlink")
    output_directory = output_directory.resolve()
    if not config_path.is_relative_to(root):
        raise ValueError("configuration must be inside the repository for replay")
    if not output_directory.is_relative_to(root / "build") or output_directory == root / "build":
        raise ValueError("optional diagnostics must use a fresh subdirectory under build/")
    _check_fresh_target(root, output_directory)
    code_hashes = _code_hashes(root)
    config_data = config_path.read_bytes()
    config = json.loads(config_data)
    outputs, inputs = _calculate(config, root)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    outputs["config.json"] = config_data
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "analysis_scope": SCOPE,
        "config_source": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": _sha256(config_data),
        },
        "code_revision": revision,
        "code_sha256": code_hashes,
        **inputs,
        "output_sha256": {name: _sha256(data) for name, data in outputs.items()},
        "interpretation": "Optional descriptive diagnostics only; no empirical error bounds, confidence intervals, valid-null or p-value claims.",
        "temporal_transformation": "Shared whole-day circular shifts of both predictors; targets fixed; within-day order and moved missingness retained; no stochastic step or seed.",
        "speed_population": "Full one-second profiles; zeros retained; explicit missing speeds split runs; same support in every scenario; MOVEStar endpoints applied per run.",
    }
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".robustness-", dir=output_directory.parent
    ) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for name, data in outputs.items():
            (stage / name).write_bytes(data)
        (stage / "receipt.json").write_bytes(_json_bytes(receipt))
        _recheck_inputs(config_path, config_data, root, inputs, code_hashes)
        _check_fresh_target(root, output_directory)
        os.rename(stage, output_directory)
    return receipt


def replay_robustness(output_directory: Path, repository_root: Path) -> dict[str, object]:
    """Read-only hash verification and numerical replay against current source/code."""

    root = repository_root.resolve()
    output_directory = output_directory.resolve()
    if not output_directory.is_relative_to(root / "build"):
        raise ValueError("replay accepts only a build bundle inside the repository")
    receipt = json.loads((output_directory / "receipt.json").read_bytes())
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["status"] != STATUS
        or receipt["analysis_scope"] != SCOPE
    ):
        raise ValueError("unsupported robustness receipt")
    if receipt["code_sha256"] != _code_hashes(root):
        raise ValueError("generating code/coefficient hash mismatch")
    source_config = _inside_root(receipt["config_source"]["path"], root).read_bytes()
    config_data = (output_directory / "config.json").read_bytes()
    if _sha256(source_config) != receipt["config_source"]["sha256"] or config_data != source_config:
        raise ValueError("configuration hash mismatch")
    outputs, inputs = _calculate(json.loads(config_data), root)
    _recheck_inputs(
        _inside_root(receipt["config_source"]["path"], root),
        config_data,
        root,
        inputs,
        receipt["code_sha256"],
    )
    outputs["config.json"] = config_data
    if inputs["sources"] != receipt["sources"] or inputs["diagnostics"] != receipt["diagnostics"]:
        raise ValueError("source/diagnostic receipt mismatch")
    if receipt["output_sha256"] != {name: _sha256(data) for name, data in outputs.items()}:
        raise ValueError("calculated output hash mismatch")
    if {item.name for item in output_directory.iterdir()} != {*outputs, "receipt.json"}:
        raise ValueError("unexpected or missing bundle artifacts")
    for name, expected in outputs.items():
        if (output_directory / name).read_bytes() != expected:
            raise ValueError(f"output replay mismatch: {name}")
    return {"status": "replay_verified_read_only", "artifacts_verified": len(outputs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--output-directory", type=Path, required=True)
    execute.add_argument("--repository-root", type=Path, default=Path.cwd())
    replay = commands.add_parser("replay")
    replay.add_argument("--output-directory", type=Path, required=True)
    replay.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        if args.command == "execute":
            result = run_robustness(args.config, args.output_directory, args.repository_root)
            print(
                json.dumps(
                    {"status": result["status"], "diagnostics": result["diagnostics"]},
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps(
                    replay_robustness(args.output_directory, args.repository_root), sort_keys=True
                )
            )
    except (ValueError, TypeError, OSError, KeyError) as exc:
        parser.exit(2, f"robustness: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
