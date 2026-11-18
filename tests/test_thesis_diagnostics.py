"""Source-count and explicit model-state diagnostic tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vehicle_emission.baselines.vtcpfm import VTCPFMModel
from vehicle_emission.emissions.movestar import MOVEStarModel
from vehicle_emission.measurement.thesis_diagnostics import (
    count_records,
    count_summary,
    model_parameters,
    quadratic_identity,
    read_sheet_cells,
    response_grid,
    run,
    worked_hour,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "site,manual,automated,absolute",
    [
        ("Sukiyabashi", 1512, 1488, 48),
        ("Shinjuku", 1043, 1065, 46),
    ],
)
def test_entering_detail_totals(site: str, manual: int, automated: int, absolute: int) -> None:
    records = count_records(
        ROOT / "results" / site / "manual_vs_cv_counting_validation.xlsx", site, "entering"
    )
    summary = count_summary(records, ["site", "kind"]).iloc[0]
    assert summary.manual_count == manual
    assert summary.automated_count == automated
    assert summary.absolute_difference == absolute
    assert summary.agreement_percent == pytest.approx(100 * (1 - absolute / manual))
    assert abs(summary.signed_difference) < summary.absolute_difference


@pytest.mark.parametrize(
    "site,manual,automated,absolute",
    [
        ("Sukiyabashi", 1500, 1511, 69),
        ("Shinjuku", 1041, 1064, 45),
    ],
)
def test_classification_detail_totals(
    site: str, manual: int, automated: int, absolute: int
) -> None:
    records = count_records(
        ROOT / "results" / site / "manual_vs_cv_counting_validation.xlsx", site, "classification"
    )
    summary = count_summary(records, ["site", "kind"]).iloc[0]
    assert summary.manual_count == manual
    assert summary.automated_count == automated
    assert summary.absolute_difference == absolute
    assert "Total" not in set(records.category)


def test_count_breakdowns_reconcile() -> None:
    records = count_records(
        ROOT / "results/Shinjuku/manual_vs_cv_counting_validation.xlsx",
        "Shinjuku",
        "classification",
    )
    by_clip = count_summary(records, ["site", "kind", "clip"])
    by_class = count_summary(records, ["site", "kind", "category"])
    for column in ["manual_count", "automated_count", "absolute_difference", "signed_difference"]:
        assert by_clip[column].sum() == by_class[column].sum()
    assert len(by_clip) == 5
    assert len(by_class) == 4


def test_zero_manual_is_not_perfect_agreement() -> None:
    records = pd.DataFrame(
        [
            {
                "site": "s",
                "clip": 1,
                "manual_count": 0,
                "automated_count": 2,
                "signed_difference": 2,
                "absolute_difference": 2,
            }
        ]
    )
    assert np.isnan(count_summary(records, ["site"]).agreement_percent.iloc[0])


def test_wrong_kind_and_sheet_fail() -> None:
    path = ROOT / "results/Shinjuku/manual_vs_cv_counting_validation.xlsx"
    with pytest.raises(ValueError, match="kind"):
        count_records(path, "Shinjuku", "unknown")
    with pytest.raises(ValueError, match="expected one"):
        read_sheet_cells(path, "missing")


def test_parameters_use_actual_class_adapter_and_coefficients() -> None:
    parameters = model_parameters().set_index("class_id")
    assert parameters.native_vehicle_type.to_dict() == {0: 6, 1: 1, 2: 13, 3: 2}
    assert parameters.moves_source_type_id.to_dict() == {0: 42, 1: 21, 2: 62, 3: 31}
    assert parameters.loc[1, "A"] == pytest.approx(0.156461)
    assert parameters.loc[0, "power_divisor_tonnes"] == pytest.approx(17.1)
    assert parameters.loc[1, "alpha_2_g_per_hour_per_vsp_squared"] > 0
    assert parameters.loc[0, "alpha_2_g_per_hour_per_vsp_squared"] < 0


def test_explicit_response_matches_actual_rate_rules() -> None:
    grid = response_grid(np.array([-1.0, 0.0, 1.0, 5.0, 15.0]), (25.0, 60.0))
    model = VTCPFMModel()
    discrete = MOVEStarModel()
    assert len(grid) == 40
    assert grid.continuous_nox_mg_per_second.ge(0).all()
    for row in grid.itertuples():
        native = {0: 6, 1: 1, 2: 13, 3: 2}[row.class_id]
        vsp = discrete.calculate_vsp(
            native, np.array([row.speed_kmh / 3.6]), np.array([row.acceleration_mps2])
        )
        assert vsp[0] == pytest.approx(row.vsp_kw_per_tonne)
        coeff = model.coefficients[native][3]
        expected = (
            model.idle_rates[native][3]
            if row.vsp_kw_per_tonne <= 0
            else max(0, np.polyval(coeff, row.vsp_kw_per_tonne))
        )
        assert row.continuous_nox_mg_per_second == pytest.approx(expected / 3.6)
    assert (
        grid.loc[
            (grid.class_id == 1) & (grid.vsp_kw_per_tonne == 1), "continuous_nox_mg_per_second"
        ]
        .eq(0)
        .all()
    )


def test_response_rejects_invalid_states() -> None:
    with pytest.raises(ValueError, match="finite"):
        response_grid(np.array([np.nan]), (25.0,))
    with pytest.raises(ValueError, match="positive"):
        response_grid(np.array([5.0]), (0.0,))


def test_quadratic_identity_and_branch_guards() -> None:
    result = quadratic_identity(np.array([0.4, 2.0, 1.0]), np.array([3.0, 5.0, 7.0]))
    assert result["curvature_variance_g_per_hour"] == pytest.approx(0.4 * 8 / 3)
    assert abs(result["identity_residual_g_per_hour"]) < 1e-12
    with pytest.raises(ValueError, match="positive, unclipped"):
        quadratic_identity(np.array([1.0, 0.0, -5.0]), np.array([1.0, 3.0]))


def test_existing_output_is_preserved(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        run(ROOT, tmp_path, tmp_path / "missing", tmp_path)


def test_worked_hour_accepts_mixed_iso_precision_and_replays(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "source_site": "Sukiyabashi",
                "video_id": 1,
                "recording_datetime_utc": "2026-05-01T04:00:00.000001Z",
            },
            {
                "source_site": "Sukiyabashi",
                "video_id": 2,
                "recording_datetime_utc": "2026-05-01T04:10:00Z",
            },
        ]
    ).to_csv(tmp_path / "video_master.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_site": "Sukiyabashi",
                "video_id": video,
                "estimation_method": method,
                "vehicle_count": 3,
                "average_speed_kmh": 10.0,
                "nox_grams": 2.0,
                "nox_hybrid_adjusted_grams": 1.5,
            }
            for video in [1, 2]
            for method in ["MOVEStar", "VT-CPFM"]
        ]
    ).to_csv(tmp_path / "video_emissions.csv", index=False)
    panel = tmp_path / "panel.csv"
    pd.DataFrame(
        [
            {
                "site": "Sukiyabashi",
                "monitor_start_utc": "2026-05-01T04:00:00Z",
                "movestar_nox_grams": 3.0,
                "vt_cpfm_nox_grams": 3.0,
                "vehicle_count": 6,
                "ambient_nox_ppm": 0.02,
                "source_interval_precipitation_mm": 0,
            }
        ]
    ).to_csv(panel, index=False)
    clips, summary = worked_hour(tmp_path, panel)
    assert len(clips) == 4
    assert summary.recorded_clips.eq(2).all()
    assert summary.recorded_profiles.eq(6).all()
    assert summary.raw_nox_index_grams.eq(4).all()
    assert summary.adjusted_nox_index_grams.eq(3).all()
