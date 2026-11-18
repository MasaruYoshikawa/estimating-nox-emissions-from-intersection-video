#!/usr/bin/env python3
"""Run prespecified synthetic precision scenarios for T-126 design review.

This entry point writes in-repository artifacts only beneath
``coordination/research``. The current task is protocol design, not empirical
analysis.
"""

# ruff: noqa: E402 -- the standalone entry point adds src/ before package imports.

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vehicle_emission.design.power_precision import (
    SimulationPlan,
    SimulationScenario,
    run_precision_simulation,
)
from vehicle_emission.design.two_site_precision import (
    TwoSiteSimulationPlan,
    TwoSiteSimulationScenario,
    run_two_site_precision_simulation,
)

DEFAULT_CONFIG = PROJECT_ROOT / "coordination/research/T-126-simulation-plan.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "coordination/research/T-126-simulation-design-output.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "coordination/research/T-126-simulation-design-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _same_existing_inode(first: Path, second: Path) -> bool:
    try:
        first_stat = first.stat()
        second_stat = second.stat()
    except FileNotFoundError:
        return False
    return (first_stat.st_dev, first_stat.st_ino) == (second_stat.st_dev, second_stat.st_ino)


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _validate_paths(config_path: Path, output_path: Path, manifest_path: Path) -> None:
    if not config_path.exists() or not config_path.is_file():
        raise ValueError("config must be an existing regular file")

    named = {
        "config": config_path,
        "output": output_path,
        "manifest": manifest_path,
    }
    resolved = {name: path.resolve() for name, path in named.items()}
    for left, right in (("config", "output"), ("config", "manifest"), ("output", "manifest")):
        if resolved[left] == resolved[right]:
            raise ValueError(f"{left} and {right} paths must be distinct")
        if _same_existing_inode(named[left], named[right]):
            raise ValueError(f"{left} and {right} must not identify the same existing inode")

    if output_path.suffix.lower() != ".csv":
        raise ValueError("output path must have .csv suffix")
    if manifest_path.suffix.lower() != ".json":
        raise ValueError("manifest path must have .json suffix")

    project = PROJECT_ROOT.resolve()
    allowed = (PROJECT_ROOT / "coordination/research").resolve()
    for name in ("output", "manifest"):
        lexical = named[name].absolute()
        destination = resolved[name]
        lexical_in_project = _is_within(lexical, project)
        resolved_in_project = _is_within(destination, project)
        if (lexical_in_project or resolved_in_project) and not (
            _is_within(lexical, allowed) and _is_within(destination, allowed)
        ):
            raise ValueError(
                f"in-repository {name} writes are allowed only beneath "
                "coordination/research; use an external temporary directory otherwise"
            )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("evidence_status") != "synthetic_design_simulation_only":
        raise ValueError("config must declare synthetic_design_simulation_only evidence status")
    has_single_site = bool(data.get("plans") and data.get("scenarios"))
    has_two_site = bool(data.get("two_site_plans") and data.get("two_site_scenarios"))
    if not has_single_site and not has_two_site:
        raise ValueError(
            "config must contain a complete single-site or two-site plan/scenario pair"
        )
    return data


def _expand_effect_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios_by_id = {
        str(scenario["scenario_id"]): scenario for scenario in config.get("two_site_scenarios", [])
    }
    expanded: list[dict[str, Any]] = []
    seen_ids = set(scenarios_by_id)
    for cell in config.get("two_site_effect_grid", []):
        if not isinstance(cell, dict):
            raise ValueError("each two_site_effect_grid cell must be an object")
        base_id = str(cell.get("base_scenario_id", ""))
        if base_id not in scenarios_by_id:
            raise ValueError(f"effect-grid base scenario does not exist: {base_id}")
        scenario_id = cell.get("scenario_id")
        design_axis_label = cell.get("design_axis_label")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("effect-grid scenario_id must be a non-empty string")
        if not isinstance(design_axis_label, str) or not design_axis_label.strip():
            raise ValueError("effect-grid design_axis_label must be a non-empty string")
        if scenario_id in seen_ids:
            raise ValueError(f"duplicate two-site scenario_id: {scenario_id}")
        seen_ids.add(scenario_id)
        scenario = copy.deepcopy(scenarios_by_id[base_id])
        scenario["scenario_id"] = scenario_id
        scenario["design_axis_label"] = design_axis_label
        scenario["dependence_label"] = "reference_dependence_effect_grid"
        overrides = cell.get("site_overrides")
        if not isinstance(overrides, dict) or set(overrides) != {
            "Sukiyabashi",
            "Shinjuku",
        }:
            raise ValueError("effect-grid site_overrides must name both sites")
        for site in scenario["sites"]:
            site_override = overrides[site["site"]]
            if not isinstance(site_override, dict):
                raise ValueError("each effect-grid site override must be an object")
            site.update(site_override)
        expanded.append(scenario)
    return expanded


def run(config_path: Path, output_path: Path, manifest_path: Path) -> pd.DataFrame:
    """Run a design configuration and write deterministic CSV plus metadata."""

    config_path = _resolved(config_path)
    output_path = _resolved(output_path)
    manifest_path = _resolved(manifest_path)
    _validate_paths(config_path, output_path, manifest_path)
    config = _load_config(config_path)

    plan_ids = [
        str(plan.get("plan_id", ""))
        for collection in ("plans", "two_site_plans", "two_site_effect_grid_plans")
        for plan in config.get(collection, [])
    ]
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("plan_id values must be unique across all plan collections")
    base_two_site_ids = [
        str(scenario.get("scenario_id", "")) for scenario in config.get("two_site_scenarios", [])
    ]
    if len(base_two_site_ids) != len(set(base_two_site_ids)):
        raise ValueError("two-site scenario_id values must be unique")

    frames: list[pd.DataFrame] = []
    for plan_data in config.get("plans", []):
        plan_values = dict(plan_data)
        plan_id = str(plan_values.pop("plan_id"))
        plan = SimulationPlan(**plan_values)
        for scenario_data in config.get("scenarios", []):
            scenario = SimulationScenario(**scenario_data)
            frame = run_precision_simulation(scenario, plan)
            frame.insert(1, "plan_id", plan_id)
            frames.append(frame)

    for plan_data in config.get("two_site_plans", []):
        two_site_plan = TwoSiteSimulationPlan(**plan_data)
        for scenario_data in config.get("two_site_scenarios", []):
            two_site_scenario = TwoSiteSimulationScenario.from_dict(scenario_data)
            frames.append(run_two_site_precision_simulation(two_site_scenario, two_site_plan))

    effect_scenarios = _expand_effect_grid(config)
    for plan_data in config.get("two_site_effect_grid_plans", []):
        two_site_plan = TwoSiteSimulationPlan(**plan_data)
        for scenario_data in effect_scenarios:
            effect_scenario = TwoSiteSimulationScenario.from_dict(scenario_data)
            frames.append(run_two_site_precision_simulation(effect_scenario, two_site_plan))

    output = pd.concat(frames, ignore_index=True)
    sort_columns = [
        column for column in ("plan_id", "scenario_id", "rq", "site") if column in output
    ]
    output = output.sort_values(sort_columns, kind="stable")
    output_bytes = output.to_csv(
        None, index=False, lineterminator="\n", float_format="%.10g"
    ).encode("utf-8")
    _atomic_write_bytes(output_path, output_bytes)

    module_path = PROJECT_ROOT / "src/vehicle_emission/design/power_precision.py"
    two_site_module_path = PROJECT_ROOT / "src/vehicle_emission/design/two_site_precision.py"
    script_path = Path(__file__).resolve()
    manifest = {
        "evidence_status": "synthetic_design_simulation_only",
        "interpretation": (
            "Implementation/design check only; not empirical evidence and not a final "
            "data-sufficiency decision."
        ),
        "config": _display_path(config_path),
        "config_sha256": _sha256(config_path),
        "output": _display_path(output_path),
        "output_sha256": _sha256(output_path),
        "module_sha256": _sha256(module_path),
        "two_site_module_sha256": _sha256(two_site_module_path),
        "script_sha256": _sha256(script_path),
        "rows": len(output),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scipy": version("scipy"),
        "canonical_runtime_satisfied": sys.version_info[:2] == (3, 11),
        "held_out_outcome_accessed": False,
    }
    _atomic_write_bytes(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    run(args.config, args.output, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
