from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from vehicle_emission.inference.panel_replay import load_panel, replay_panel, validate_panel
from vehicle_emission.inference.recorded_hourly_extension import (
    CLASS_COLUMNS,
    ExtensionConfig,
    infer_extensions,
    reference_class_weights,
)


@pytest.fixture
def panel_frame() -> pd.DataFrame:
    weights = reference_class_weights().set_index("class_id").normalised_weight
    records = []
    for day in range(12):
        for hour in range(4):
            start = pd.Timestamp("2026-05-01T00:00:00Z") + pd.Timedelta(days=day, hours=hour)
            local = start.tz_convert("Asia/Tokyo")
            for number, site in enumerate(("Sukiyabashi", "Shinjuku")):
                classes = [day % 3 + 1, 20 + day + hour, hour % 3 + 1, day % 4 + hour]
                total = sum(classes)
                records.append(
                    {
                        "site": site,
                        "monitor_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "monitor_end_utc": (start + pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "local_calendar_date": local.strftime("%Y-%m-%d"),
                        "hour_jst": local.hour,
                        "weekday_jst": local.dayofweek,
                        "ambient_nox_ppm": (day % 5 + hour + number) / 1000,
                        "movestar_nox_grams": 1 + day + hour / 4 + number,
                        "vt_cpfm_nox_grams": 2 + day + hour / 2 + (day % 3) / 2 + number,
                        "vehicle_count": 5 * total,
                        "mean_speed_mps": 2 + day / 10 + hour / 4,
                        "captured_clip_count": 6,
                        "entering_known": True,
                        "heavy_known": True,
                        "class_counts_known": True,
                        "captured_clip_entering_count": total,
                        "captured_clip_bus_heavy_count": classes[0] + classes[2],
                        **dict(zip(CLASS_COLUMNS, classes, strict=True)),
                        "fixed_rate_weighted_entries": sum(
                            classes[i] * weights.loc[i] for i in range(4)
                        ),
                        "bus_share": classes[0] / total,
                        "heavy_truck_share": classes[2] / total,
                        "light_truck_share": classes[3] / total,
                        "between_clip_speed_cv": (1 + day % 4 + hour) / 10,
                        "movestar_nox_grams_raw": 1.1 + day + hour / 4 + number,
                        "vt_cpfm_nox_grams_raw": 2.1 + day + hour / 2 + (day % 3) / 2 + number,
                        "temperature": 12 + day + hour / 10,
                        "wind_speed": 1 + hour / 4,
                        "setagaya_nox_ppm": (day % 3 + hour + 1) / 1000,
                        "residual_complete": True,
                    }
                )
    return pd.DataFrame(records).sort_values(["site", "monitor_start_utc"]).reset_index(drop=True)


def test_current_published_panel_passes_without_changing_support() -> None:
    source = (
        Path(__file__).parents[1]
        / "results/recorded_hourly_extension_20261109/extension_hourly_panel.csv"
    )
    prepared = load_panel(source)
    assert prepared.groupby("site").size().to_dict() == {"Shinjuku": 555, "Sukiyabashi": 593}
    assert prepared.groupby("site").residual_complete.sum().to_dict() == {
        "Shinjuku": 553,
        "Sukiyabashi": 590,
    }


def test_missing_regional_values_change_only_residual_mask_and_zero_is_valid(panel_frame) -> None:
    panel_frame.loc[0, "setagaya_nox_ppm"] = np.nan
    panel_frame.loc[1, "setagaya_nox_ppm"] = 0
    panel_frame.loc[2, "temperature"] = -2
    prepared = validate_panel(panel_frame)
    assert len(prepared) == len(panel_frame)
    assert not prepared.loc[0, "residual_complete"]
    assert prepared.loc[1, "residual_complete"]
    assert prepared.loc[1, "setagaya_nox_ppm"] == 0
    assert prepared.loc[2, "residual_complete"]
    assert prepared.ambient_nox_ppm.eq(0).any()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("ambient_nox_ppm", np.nan, "ambient_nox_ppm"),
        ("ambient_nox_ppm", -0.001, "ambient_nox_ppm"),
        ("wind_speed", np.inf, "wind_speed"),
        ("movestar_nox_grams", -1, "movestar_nox_grams"),
        ("vehicle_count", 2.2, "integer counts"),
        ("captured_clip_count", 0, "captured clips"),
        ("entering_known", "False", "boolean"),
        ("bus_share", 0.9, "bus_share"),
        ("fixed_rate_weighted_entries", 2, "weighted entries"),
        ("hour_jst", 23, "calendar fields"),
        ("local_calendar_date", "2026-04-30", "calendar fields"),
        ("monitor_start_utc", "2026-05-01T00:00:00", "canonical UTC"),
        ("monitor_end_utc", "2026-05-01T00:00:00Z", "whole calendar"),
    ],
)
def test_bad_panel_values_rejected(panel_frame, column, value, message) -> None:
    panel_frame[column] = panel_frame[column].astype(object)
    panel_frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_panel(panel_frame)


def test_duplicate_keys_and_headers_are_rejected(panel_frame, tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate panel"):
        validate_panel(pd.concat([panel_frame, panel_frame.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="distinct columns"):
        validate_panel(pd.concat([panel_frame, panel_frame[["site"]]], axis=1))
    path = tmp_path / "duplicates.csv"
    path.write_text("site,site\nShinjuku,Shinjuku\n")
    with pytest.raises(ValueError, match="distinct CSV header"):
        load_panel(path)


def test_unknown_class_counts_remain_unavailable(panel_frame) -> None:
    panel_frame.loc[
        0,
        [
            *CLASS_COLUMNS,
            "fixed_rate_weighted_entries",
            "bus_share",
            "heavy_truck_share",
            "light_truck_share",
            "captured_clip_bus_heavy_count",
        ],
    ] = np.nan
    panel_frame.loc[0, ["class_counts_known", "heavy_known"]] = False
    prepared = validate_panel(panel_frame)
    assert pd.isna(prepared.loc[0, CLASS_COLUMNS[0]])
    assert pd.isna(prepared.loc[0, "fixed_rate_weighted_entries"])
    assert not prepared.loc[0, "residual_complete"]
    panel_frame.loc[0, "class_counts_known"] = True
    with pytest.raises(ValueError, match="class-count availability"):
        validate_panel(panel_frame)


def test_stored_float_round_trip(panel_frame, tmp_path) -> None:
    path = tmp_path / "panel.csv"
    panel_frame.to_csv(path, index=False, float_format="%.17g")
    actual = load_panel(path)
    assert (
        actual.movestar_nox_grams.to_numpy().tobytes()
        == panel_frame.movestar_nox_grams.to_numpy().tobytes()
    )


def test_replay_is_fresh_deterministic_and_agrees_with_existing_estimator(
    panel_frame, tmp_path
) -> None:
    source = tmp_path / "published" / "panel.csv"
    source.parent.mkdir()
    panel_frame.to_csv(source, index=False, float_format="%.17g")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    config = ExtensionConfig(19, 2, 42)
    first, second = tmp_path / "first", tmp_path / "second"
    summary = replay_panel(panel=source, output_directory=first, config=config)
    replay_panel(panel=source, output_directory=second, config=config, expected_bundle=first)
    direct = infer_extensions(load_panel(source), config)
    for name, expected in direct.items():
        actual = pd.read_csv(first / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(
            actual,
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert summary["sites"]["Shinjuku"]["core_hours"] == 48
    core = pd.read_csv(first / "core_inference.csv")
    row = core.loc[core.site.eq("Shinjuku") & core.estimand.eq("vt_cpfm_nox_grams")].iloc[0]
    subset = panel_frame.loc[panel_frame.site.eq("Shinjuku")]
    assert row.estimate == pytest.approx(
        spearmanr(subset.vt_cpfm_nox_grams, subset.ambient_nox_ppm).statistic
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    receipt = json.loads((first / "receipt.json").read_text())
    assert receipt["core_family_size"] == 10
    assert receipt["input_sha256"]["extension_hourly_panel.csv"] == source_hash
    assert all(not Path(name).is_absolute() for name in receipt["calculator_sha256"])
    assert all(
        hashlib.sha256((first / name).read_bytes()).hexdigest() == digest
        for name, digest in receipt["outputs"].items()
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        replay_panel(panel=source, output_directory=first, config=config)
    with pytest.raises(ValueError, match="separate"):
        replay_panel(panel=source, output_directory=source.parent / "new", config=config)


def test_expected_table_difference_prevents_publication(panel_frame, tmp_path) -> None:
    source = tmp_path / "input" / "panel.csv"
    source.parent.mkdir()
    panel_frame.to_csv(source, index=False)
    expected = tmp_path / "expected"
    expected.mkdir()
    (expected / "core_inference.csv").write_text("site,estimate\nShinjuku,123\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="differs from expected"):
        replay_panel(
            panel=source,
            output_directory=output,
            config=ExtensionConfig(19, 2),
            expected_bundle=expected,
        )
    assert not output.exists()
