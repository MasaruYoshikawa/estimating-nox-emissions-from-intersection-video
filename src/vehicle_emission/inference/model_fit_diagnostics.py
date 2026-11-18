"""Construction-point and fixed-state diagnostics for the continuous comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vehicle_emission.baselines.vtcpfm import VTCPFMModel
from vehicle_emission.emissions.movestar import BASE_DIR

VSP_POINTS = {
    12: 1.5,
    13: 4.5,
    14: 7.5,
    15: 10.5,
    16: 15.0,
    22: 1.5,
    23: 4.5,
    24: 7.5,
    25: 10.5,
    27: 15.0,
    28: 21.0,
    29: 27.0,
    30: 35.0,
    33: 3.0,
    35: 9.0,
    37: 15.0,
    38: 21.0,
    39: 27.0,
    40: 35.0,
}
CLASS_NAMES = {1: "Passenger car", 2: "Light truck", 6: "Bus", 13: "Heavy truck"}
GRID = np.arange(-5.0, 80.25, 0.25)
SENSITIVITY_POINTS = (0.0, 1.5, 10.0, 15.0, 30.0, 35.0, 50.0, 80.0)


def construction_points(table: pd.DataFrame) -> pd.DataFrame:
    """Recover the twenty equally weighted NOx points used by the existing fit."""
    if table.shape != (23, 8):
        raise ValueError("rate table must have 23 operating modes and eight columns")
    try:
        values = table.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("rate table must contain finite numeric values") from exc
    if not np.isfinite(values).all() or (values[:, 1:] < 0).any():
        raise ValueError("rate table must contain finite non-negative rates")
    required = {0, 1, 11, 21, *VSP_POINTS}
    if len(set(values[:, 0])) != 23 or set(values[:, 0]) != required:
        raise ValueError("rate table has missing, duplicate or invalid operating modes")
    rates = dict(zip(values[:, 0].astype(int), values[:, 3], strict=True))
    return pd.DataFrame(
        {
            "opmode_id": [1, *VSP_POINTS],
            "vsp_kw_per_tonne": [0.0, *VSP_POINTS.values()],
            "reference_nox_g_per_hour": [rates[1], *(rates[mode] for mode in VSP_POINTS)],
        }
    )


def _implemented_rate(vsp: np.ndarray, coefficients: np.ndarray, idle: float) -> np.ndarray:
    return np.maximum(np.where(vsp <= 0, idle, np.polyval(coefficients, vsp)), 0.0)


def _roots(coefficients: np.ndarray) -> list[float]:
    return sorted(
        float(root.real)
        for root in np.roots(coefficients)
        if abs(root.imag) < 1e-10 and root.real > 0
    )


def calculate(model: VTCPFMModel | None = None) -> dict[str, pd.DataFrame]:
    """Return deterministic model-construction diagnostics, not observed-state shares."""
    model = model or VTCPFMModel()
    if set(model.ems_tables) != set(CLASS_NAMES):
        raise ValueError("all four rate tables are required")
    for table in model.ems_tables.values():
        construction_points(table)
    scenario_tables = [
        ("original", vehicle_type, table.copy())
        for vehicle_type, table in sorted(model.ems_tables.items())
    ]
    for recipient, donor in ((6, 13), (13, 6)):
        table = model.ems_tables[recipient].copy()
        donor_value = float(model.ems_tables[donor].loc[lambda frame: frame[0].eq(40), 3].iloc[0])
        table.loc[table[0].eq(40), 3] = donor_value
        scenario_tables.append((f"mode40_from_type_{donor}", recipient, table))

    points, summaries, responses, roots, sensitivities = [], [], [], [], []
    for scenario, vehicle_type, table in scenario_tables:
        point = construction_points(table)
        vsp = point.vsp_kw_per_tonne.to_numpy()
        reference = point.reference_nox_g_per_hour.to_numpy()
        coefficients = np.polyfit(vsp, reference, 2)
        if scenario == "original" and not np.allclose(
            coefficients, model.coefficients[vehicle_type][3], rtol=1e-12, atol=1e-12
        ):
            raise ValueError("reconstructed fit disagrees with the existing implementation")
        idle = float(reference[0])
        point["raw_fitted_nox_g_per_hour"] = np.polyval(coefficients, vsp)
        point["implemented_nox_g_per_hour"] = _implemented_rate(vsp, coefficients, idle)
        for name in ("raw_fitted", "implemented"):
            point[f"{name}_residual_g_per_hour"] = point[f"{name}_nox_g_per_hour"] - reference
        identity: dict[str, str | int] = {
            "scenario": scenario,
            "vehicle_type": vehicle_type,
            "vehicle_class": CLASS_NAMES[vehicle_type],
        }
        point = point.assign(**identity)
        points.append(point)
        summary: dict[str, str | int | float] = {
            **identity,
            "n_construction_points": len(point),
            "fit_min_vsp_kw_per_tonne": float(vsp.min()),
            "fit_max_vsp_kw_per_tonne": float(vsp.max()),
            "quadratic": float(coefficients[0]),
            "linear": float(coefficients[1]),
            "constant": float(coefficients[2]),
            "idle_nox_g_per_hour": idle,
            "reference_mean_nox_g_per_hour": float(reference.mean()),
            "raw_negative_construction_points": int((np.polyval(coefficients, vsp) < 0).sum()),
        }
        variance_sum = float(np.square(reference - reference.mean()).sum())
        for name in ("raw_fitted", "implemented"):
            residual = point[f"{name}_residual_g_per_hour"].to_numpy()
            summary.update(
                {
                    f"{name}_rmse_g_per_hour": float(np.sqrt(np.mean(residual**2))),
                    f"{name}_mae_g_per_hour": float(np.mean(np.abs(residual))),
                    f"{name}_max_absolute_error_g_per_hour": float(np.max(np.abs(residual))),
                    f"{name}_r_squared": (
                        1 - float(np.square(residual).sum()) / variance_sum
                        if variance_sum > 0
                        else np.nan
                    ),
                }
            )
        summaries.append(summary)
        raw_grid = np.polyval(coefficients, GRID)
        responses.append(
            pd.DataFrame(
                {
                    **identity,
                    "vsp_kw_per_tonne": GRID,
                    "raw_polynomial_nox_g_per_hour": raw_grid,
                    "implemented_nox_g_per_hour": _implemented_rate(GRID, coefficients, idle),
                    "above_fit_range": vsp.max() < GRID,
                    "positive_power_clipped": (GRID > 0) & (raw_grid < 0),
                }
            )
        )
        for root in _roots(coefficients):
            dynamics_type = {1: 2, 2: 3}.get(vehicle_type, vehicle_type)
            dynamics = model.df_coeff.loc[model.df_coeff.VehicleType.eq(dynamics_type)].iloc[0]
            speed_roots = _roots(
                np.array([dynamics.C, dynamics.B, dynamics.A, -root * dynamics.f], dtype=float)
            )
            roots.append(
                {
                    **identity,
                    "positive_polynomial_root_kw_per_tonne": root,
                    "steady_speed_at_root_kmh": (speed_roots[0] * 3.6 if speed_roots else np.nan),
                    "root_above_fit_range": root > vsp.max(),
                }
            )
        if scenario != "original":
            fixed = np.array(SENSITIVITY_POINTS)
            original = _implemented_rate(
                fixed, model.coefficients[vehicle_type][3], model.idle_rates[vehicle_type][3]
            )
            changed = _implemented_rate(fixed, coefficients, idle)
            sensitivities.append(
                pd.DataFrame(
                    {
                        **identity,
                        "vsp_kw_per_tonne": fixed,
                        "original_nox_g_per_hour": original,
                        "substituted_nox_g_per_hour": changed,
                        "difference_nox_g_per_hour": changed - original,
                        "relative_change": np.divide(
                            changed - original,
                            original,
                            out=np.full(len(original), np.nan),
                            where=original > 0,
                        ),
                    }
                )
            )

    bus = model.ems_tables[6].set_index(0).sort_index()
    truck = model.ems_tables[13].set_index(0).sort_index()
    comparison = pd.DataFrame(
        {
            "opmode_id": bus.index,
            "bus_nox_g_per_hour": bus[3].to_numpy(),
            "heavy_truck_nox_g_per_hour": truck[3].to_numpy(),
            "different_rate_cells": (bus != truck).sum(axis=1).to_numpy(),
        }
    )
    return {
        "fit_points.csv": pd.concat(points, ignore_index=True),
        "fit_summary.csv": pd.DataFrame(summaries),
        "fixed_vsp_response.csv": pd.concat(responses, ignore_index=True),
        "polynomial_roots.csv": pd.DataFrame(roots),
        "mode40_substitution_response.csv": pd.concat(sensitivities, ignore_index=True),
        "bus_truck_table_comparison.csv": comparison,
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_directory: Path) -> None:
    """Write diagnostics to a fresh directory with input, code and output hashes."""
    if output_directory.exists() or output_directory.is_symlink():
        raise ValueError("output directory must be fresh")
    source_directory = Path(BASE_DIR)
    inputs = [source_directory / f"EmsRate_{vehicle_type}.csv" for vehicle_type in CLASS_NAMES]
    inputs.append(source_directory / "VehicleSrcCoeff.csv")
    implementation = Path(__file__).resolve()
    code = [
        implementation,
        source_directory / "movestar.py",
        source_directory.parent / "baselines" / "vtcpfm.py",
    ]
    input_hashes = {str(path): _digest(path) for path in inputs}
    code_hashes = {str(path): _digest(path) for path in code}
    tables = calculate()
    output_directory.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(output_directory / name, index=False, float_format="%.17g")
    if any(
        _digest(Path(path)) != digest for path, digest in {**input_hashes, **code_hashes}.items()
    ):
        raise ValueError("source or calculator changed during diagnostics")
    receipt = {
        "status": "constructed_state_model_diagnostics",
        "inputs": input_hashes,
        "calculator_sha256": code_hashes,
        "outputs": {name: _digest(output_directory / name) for name in tables},
        "fit": "equal-weight quadratic on idle plus nineteen positive-VSP construction points",
        "grid_vsp_kw_per_tonne": {"start": -5, "stop_inclusive": 80, "step": 0.25},
        "counterfactual": "replace only mode-40 NOx with the other heavy-class value; refit",
        "interpretation": "constructed rates, not empirical operating-state frequencies or emissions",
    }
    (output_directory / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_directory)


if __name__ == "__main__":
    main()
