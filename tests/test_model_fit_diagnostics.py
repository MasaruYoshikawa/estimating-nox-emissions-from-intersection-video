"""Independent fit calculations, response boundaries and fresh-output checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from vehicle_emission.baselines.vtcpfm import VTCPFMModel
from vehicle_emission.inference.model_fit_diagnostics import (
    CLASS_NAMES,
    GRID,
    VSP_POINTS,
    calculate,
    construction_points,
    run,
)


def test_construction_has_twenty_equal_weight_points_with_duplicate_powers() -> None:
    model = VTCPFMModel()
    points = construction_points(model.ems_tables[1])
    assert len(points) == 20
    assert points.opmode_id.tolist() == [1, *VSP_POINTS]
    assert points.vsp_kw_per_tonne.tolist() == [0.0, *VSP_POINTS.values()]
    assert points.vsp_kw_per_tonne.eq(1.5).sum() == 2
    assert points.vsp_kw_per_tonne.eq(35).sum() == 2
    assert points.iloc[0].reference_nox_g_per_hour == model.idle_rates[1][3]


def test_original_fit_and_metrics_match_independent_least_squares() -> None:
    model = VTCPFMModel()
    tables = calculate(model)
    summary = tables["fit_summary.csv"].query("scenario == 'original'").set_index("vehicle_type")
    for vehicle_type in CLASS_NAMES:
        points = construction_points(model.ems_tables[vehicle_type])
        x = points.vsp_kw_per_tonne.to_numpy()
        y = points.reference_nox_g_per_hour.to_numpy()
        matrix = np.column_stack([x**2, x, np.ones(len(x))])
        coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
        row = summary.loc[vehicle_type]
        assert np.allclose(coefficients, row[["quadratic", "linear", "constant"]].astype(float))
        raw = matrix @ coefficients
        actual = np.maximum(np.where(x <= 0, y[0], raw), 0)
        for name, predicted in (("raw_fitted", raw), ("implemented", actual)):
            difference = predicted - y
            assert row[f"{name}_rmse_g_per_hour"] == pytest.approx(np.sqrt(np.mean(difference**2)))
            assert row[f"{name}_mae_g_per_hour"] == pytest.approx(np.abs(difference).mean())
            assert row[f"{name}_max_absolute_error_g_per_hour"] == pytest.approx(
                np.abs(difference).max()
            )
            assert row[f"{name}_r_squared"] == pytest.approx(
                1 - np.sum(difference**2) / np.sum((y - y.mean()) ** 2)
            )


def test_implemented_fit_has_idle_branch_and_positive_power_clipping() -> None:
    tables = calculate()
    curves = tables["fixed_vsp_response.csv"]
    car = curves.query("scenario == 'original' and vehicle_type == 1").set_index("vsp_kw_per_tonne")
    assert car.loc[0, "implemented_nox_g_per_hour"] > 0
    assert car.loc[0.25, "implemented_nox_g_per_hour"] == 0
    assert car.loc[2, "positive_power_clipped"]
    assert not car.loc[2.25, "positive_power_clipped"]
    assert (
        car.loc[car.index <= 0, "implemented_nox_g_per_hour"]
        == car.loc[0, "implemented_nox_g_per_hour"]
    ).all()
    assert not car.loc[0, "positive_power_clipped"]


def test_grid_range_and_extrapolation_are_explicit_without_observed_shares() -> None:
    curves = calculate()["fixed_vsp_response.csv"]
    assert len(curves) == 6 * len(GRID)
    assert len(GRID) == 341
    for _, group in curves.groupby(["scenario", "vehicle_type"]):
        assert group.vsp_kw_per_tonne.tolist() == list(GRID)
        assert group.above_fit_range.eq(group.vsp_kw_per_tonne.gt(35)).all()
    assert "observed_frequency" not in curves
    bus = curves.query("scenario == 'original' and vehicle_type == 6").set_index("vsp_kw_per_tonne")
    assert bus.loc[67.75, "implemented_nox_g_per_hour"] > 0
    assert bus.loc[68, "implemented_nox_g_per_hour"] == 0
    assert bus.loc[80, "positive_power_clipped"]


def test_root_steady_speed_reconstructs_vsp_at_zero_acceleration() -> None:
    model = VTCPFMModel()
    roots = calculate(model)["polynomial_roots.csv"]
    for row in roots.itertuples():
        if row.scenario != "original":
            continue
        coefficients = model.coefficients[row.vehicle_type][3]
        assert np.polyval(coefficients, row.positive_polynomial_root_kw_per_tonne) == pytest.approx(
            0, abs=1e-10
        )
        vsp = model.calculate_vsp(
            row.vehicle_type, np.array([row.steady_speed_at_root_kmh / 3.6]), np.array([0])
        )[0]
        assert vsp == pytest.approx(row.positive_polynomial_root_kw_per_tonne)
    car = roots.query("scenario == 'original' and vehicle_type == 1").iloc[0]
    assert car.steady_speed_at_root_kmh == pytest.approx(43.2169, abs=0.001)
    bus = roots.query("scenario == 'original' and vehicle_type == 6").iloc[0]
    assert bus.positive_polynomial_root_kw_per_tonne == pytest.approx(67.771, abs=0.001)
    assert bus.root_above_fit_range


def test_mode40_substitution_is_symmetric_and_changes_the_whole_fit() -> None:
    model = VTCPFMModel()
    original_tables = {key: value.copy(deep=True) for key, value in model.ems_tables.items()}
    tables = calculate(model)
    fit = tables["fit_summary.csv"].set_index(["scenario", "vehicle_type"])
    points = tables["fit_points.csv"]
    responses = tables["mode40_substitution_response.csv"]
    for recipient, donor in ((6, 13), (13, 6)):
        scenario = f"mode40_from_type_{donor}"
        original = construction_points(model.ems_tables[recipient])
        changed = points.query("scenario == @scenario and vehicle_type == @recipient")
        difference = (
            changed.reference_nox_g_per_hour.to_numpy()
            - original.reference_nox_g_per_hour.to_numpy()
        )
        assert np.flatnonzero(difference).tolist() == [19]
        assert (
            changed.iloc[19].reference_nox_g_per_hour
            == original_tables[donor].loc[lambda frame: frame[0].eq(40), 3].iloc[0]
        )
        x = original.vsp_kw_per_tonne.to_numpy()
        design = np.column_stack([x**2, x, np.ones(len(x))])
        expected_change = np.linalg.pinv(design) @ difference
        for position, column in enumerate(("quadratic", "linear", "constant")):
            observed = (
                fit.loc[(scenario, recipient), column] - fit.loc[("original", recipient), column]
            )
            assert observed == pytest.approx(expected_change[position], abs=1e-11)
        response = responses.query("scenario == @scenario").set_index("vsp_kw_per_tonne")
        assert response.loc[0, "difference_nox_g_per_hour"] == 0
        assert response.loc[1.5, "difference_nox_g_per_hour"] != 0
        assert response.loc[15, "difference_nox_g_per_hour"] != 0
    for vehicle_type in model.ems_tables:
        assert_frame_equal(model.ems_tables[vehicle_type], original_tables[vehicle_type])


def test_bus_and_truck_table_difference_is_one_no_x_cell() -> None:
    comparison = calculate()["bus_truck_table_comparison.csv"]
    assert len(comparison) == 23
    assert comparison.different_rate_cells.sum() == 1
    different = comparison.query("different_rate_cells > 0").iloc[0]
    assert different.opmode_id == 40
    assert different.bus_nox_g_per_hour == pytest.approx(7.990540027618408)
    assert different.heavy_truck_nox_g_per_hour == pytest.approx(29.86359977722168)


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "invalid_mode", "negative", "infinite", "text"]
)
def test_reject_malformed_tables(defect: str) -> None:
    table = VTCPFMModel().ems_tables[1].copy()
    if defect == "missing":
        table = table.iloc[:-1]
    elif defect == "duplicate":
        table.iloc[-1, 0] = table.iloc[0, 0]
    elif defect == "invalid_mode":
        table.iloc[-1, 0] = 41
    elif defect == "negative":
        table.iloc[0, 3] = -1
    elif defect == "infinite":
        table.iloc[0, 3] = np.inf
    else:
        table = table.astype(object)
        table.iloc[0, 3] = "unavailable"
    with pytest.raises(ValueError):
        construction_points(table)


def test_reject_missing_class_and_stale_coefficients() -> None:
    model = VTCPFMModel()
    model.ems_tables.pop(13)
    with pytest.raises(ValueError, match="all four"):
        calculate(model)
    model = VTCPFMModel()
    model.coefficients[1][3][0] += 0.1
    with pytest.raises(ValueError, match="disagrees"):
        calculate(model)


def test_constant_reference_keeps_undefined_r_squared() -> None:
    model = VTCPFMModel()
    for table in model.ems_tables.values():
        table[3] = 2.0
    model._calibrate()
    summary = calculate(model)["fit_summary.csv"]
    assert summary.raw_fitted_r_squared.isna().all()
    assert summary.implemented_r_squared.isna().all()
    assert np.allclose(summary.implemented_rmse_g_per_hour, 0, atol=1e-12)


def test_fresh_output_replay_and_receipt_hashes(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    run(first)
    run(second)
    assert sorted(path.name for path in first.iterdir()) == sorted(
        path.name for path in second.iterdir()
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    receipt = json.loads((first / "receipt.json").read_text())
    for name, expected in receipt["outputs"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == expected
    for section in ("inputs", "calculator_sha256"):
        for path, expected in receipt[section].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    with pytest.raises(ValueError, match="fresh"):
        run(first)


def test_dangling_symlink_is_not_a_fresh_output(tmp_path: Path) -> None:
    link = tmp_path / "output"
    link.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ValueError, match="fresh"):
        run(link)
