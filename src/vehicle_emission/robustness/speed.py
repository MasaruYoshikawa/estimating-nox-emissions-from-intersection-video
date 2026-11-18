"""Profile-level speed perturbations through the actual MOVEStar implementation.

Terminology follows ``coordination/GLOSSARY.md``. Caller-supplied scenarios
are not error estimates or confidence intervals. No hourly index is rescaled.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vehicle_emission.emissions.movestar import MOVEStarModel
from vehicle_emission.vehicle_classes import emission_vehicle_type_for_project_class

PROFILE_KEY = ["site", "source_id", "profile_id"]
PROFILE_COLUMNS = {*PROFILE_KEY, "class_id", "second_index", "speed_mps"}
SITES = ("Sukiyabashi", "Shinjuku")
RUN_COLUMNS = [
    *PROFILE_KEY,
    "scenario_id",
    "run_number",
    "first_second_index",
    "end_second_index_exclusive",
    "run_seconds",
    "zero_speed_seconds",
    "speed_mps",
    "acceleration_mps2",
    "vsp_kw_per_tonne",
    "operating_mode_seconds",
    "nox_g",
]


@dataclass(frozen=True)
class SpeedPerturbation:
    """An explicit affine speed scenario, without an empirical error claim."""

    scenario_id: str
    multiplier: float
    additive_mps: float

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        for name in ("multiplier", "additive_mps"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and explicitly supplied")
        if self.multiplier < 0:
            raise ValueError("multiplier cannot be negative")


@dataclass(frozen=True)
class SpeedSensitivityTables:
    """Profile support, recalculated run diagnostics and profile/scenario totals."""

    profiles: pd.DataFrame
    runs: pd.DataFrame
    summary: pd.DataFrame


def _prepare_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.columns.duplicated().any() or not PROFILE_COLUMNS.issubset(profiles):
        raise ValueError(f"profiles require unique columns: {sorted(PROFILE_COLUMNS)}")
    if profiles.empty:
        raise ValueError("profiles must contain explicit second-level observations")
    rows = profiles.copy()
    for column in PROFILE_KEY:
        if not rows[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{column} must identify every profile observation")
    if not rows.site.isin(SITES).all():
        raise ValueError("profiles contain an unsupported site")
    for column in ("class_id", "second_index"):
        values = pd.to_numeric(rows[column], errors="raise")
        if (
            rows[column].map(lambda value: isinstance(value, bool | np.bool_)).any()
            or not (np.isfinite(values) & values.ge(0) & values.eq(np.floor(values))).all()
        ):
            raise ValueError(f"{column} must contain non-negative integers")
        rows[column] = values.astype("int64")
    for class_id in rows.class_id.unique():
        emission_vehicle_type_for_project_class(int(class_id))
    rows["speed_mps"] = pd.to_numeric(rows.speed_mps, errors="raise")
    if np.isinf(rows.speed_mps).any() or rows.speed_mps.dropna().lt(0).any():
        raise ValueError("baseline speeds must be non-negative, finite or explicitly missing")
    if (
        "duration_seconds" in rows
        and not pd.to_numeric(rows.duration_seconds, errors="raise").eq(1.0).all()
    ):
        raise ValueError("MOVEStar profiles require full one-second bins, not partial durations")
    rows = rows.sort_values([*PROFILE_KEY, "second_index"], kind="stable").reset_index(drop=True)
    if rows.duplicated([*PROFILE_KEY, "second_index"]).any():
        raise ValueError("duplicate profile seconds would double-count emissions")
    for _, profile in rows.groupby(PROFILE_KEY, sort=False):
        if profile.class_id.nunique() != 1:
            raise ValueError("a profile requires one fixed project class")
        if len(profile) > 1 and not (np.diff(profile.second_index.to_numpy()) == 1).all():
            raise ValueError("profile second indexes must be contiguous; split gaps into profiles")
        if "ts_utc" in profile:
            if (
                not profile.ts_utc.astype("string")
                .str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
                .fillna(False)
                .all()
            ):
                raise ValueError("ts_utc must identify explicit UTC-second boundaries")
            times = pd.to_datetime(profile.ts_utc, utc=True)
            if len(times) > 1 and not times.diff().iloc[1:].eq(pd.Timedelta(seconds=1)).all():
                raise ValueError("profile UTC seconds must be contiguous")
    return rows


def calculate_speed_sensitivity(
    profiles: pd.DataFrame, scenarios: tuple[SpeedPerturbation, ...]
) -> SpeedSensitivityTables:
    """Recalculate each scenario on the same profile/run population.

    Explicit missing speeds are retained in support and split contiguous
    valid runs. Acceleration never crosses a missing second or profile edge;
    MOVEStar's own endpoint convention is used separately on each valid run.
    Impossible negative perturbed speeds reject the entire calculation.
    """

    rows = _prepare_profiles(profiles)
    if not scenarios or len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise ValueError("supply non-empty, uniquely identified speed scenarios")
    model = MOVEStarModel()
    support: list[dict[str, object]] = []
    output: list[dict[str, object]] = []
    totals: list[dict[str, object]] = []
    for key, profile in rows.groupby(PROFILE_KEY, sort=False):
        identity = dict(zip(PROFILE_KEY, key, strict=True))
        speed = profile.speed_mps.to_numpy(dtype=float)
        valid = np.isfinite(speed)
        support_hash = hashlib.sha256(
            json.dumps(profile.loc[valid, "second_index"].tolist(), separators=(",", ":")).encode()
        ).hexdigest()
        padded = np.r_[False, valid, False].astype(int)
        starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
        class_id = int(profile.class_id.iloc[0])
        common = {
            **identity,
            "class_id": class_id,
            "profile_seconds": len(profile),
            "valid_seconds": int(valid.sum()),
            "missing_seconds": int((~valid).sum()),
            "valid_runs": len(starts),
            "support_sha256": support_hash,
        }
        support.append(common)
        native_type = emission_vehicle_type_for_project_class(class_id)
        baseline_nox = sum(
            float(model.estimate_emissions(native_type, speed[start:end])["Emission Rate"][0][2])
            for start, end in zip(starts, ends, strict=True)
        )
        for scenario in scenarios:
            perturbed = speed * scenario.multiplier + scenario.additive_mps
            if not np.isfinite(perturbed[valid]).all() or (perturbed[valid] < 0).any():
                raise ValueError(
                    f"scenario {scenario.scenario_id!r} produces invalid/negative speeds"
                )
            scenario_nox = 0.0
            for run_number, (start, end) in enumerate(zip(starts, ends, strict=True)):
                selected = perturbed[start:end]
                acceleration = model.calculate_acceleration(selected)
                vsp = model.calculate_vsp(native_type, selected, acceleration)
                if not np.isfinite(acceleration).all() or not np.isfinite(vsp).all():
                    raise ValueError("perturbed profile exceeds finite MOVEStar numerical support")
                modes = model.calculate_operating_modes(selected, acceleration, vsp)
                nox = float(model.estimate_emissions(native_type, selected)["Emission Rate"][0][2])
                scenario_nox += nox
                output.append(
                    {
                        **identity,
                        "scenario_id": scenario.scenario_id,
                        "run_number": run_number,
                        "first_second_index": int(profile.second_index.iloc[start]),
                        "end_second_index_exclusive": int(profile.second_index.iloc[end - 1]) + 1,
                        "run_seconds": end - start,
                        "zero_speed_seconds": int((selected == 0).sum()),
                        "speed_mps": json.dumps(selected.tolist(), separators=(",", ":")),
                        "acceleration_mps2": json.dumps(
                            acceleration.tolist(), separators=(",", ":")
                        ),
                        "vsp_kw_per_tonne": json.dumps(vsp.tolist(), separators=(",", ":")),
                        "operating_mode_seconds": json.dumps(modes.tolist(), separators=(",", ":")),
                        "nox_g": nox,
                    }
                )
            totals.append(
                {
                    **common,
                    "scenario_id": scenario.scenario_id,
                    "baseline_nox_g": baseline_nox if valid.any() else math.nan,
                    "scenario_nox_g": scenario_nox if valid.any() else math.nan,
                    "difference_nox_g": scenario_nox - baseline_nox if valid.any() else math.nan,
                    "status": (
                        "calculated_optional_diagnostic" if valid.any() else "no_valid_seconds"
                    ),
                }
            )
    # Per-profile totals retain all-missing profiles rather than turning them into zeros.
    return SpeedSensitivityTables(
        pd.DataFrame(support), pd.DataFrame(output, columns=RUN_COLUMNS), pd.DataFrame(totals)
    )
