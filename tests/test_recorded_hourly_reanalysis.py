from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest
from scipy.stats import spearmanr

from vehicle_emission.inference.recorded_hourly_reanalysis import (
    PREDICTORS,
    STATUS,
    _config,
    _entry_inference,
    _inference,
    _load,
    _partial_rank_points,
    _recorded_counts,
    review_recorded_panel,
)


@pytest.fixture
def recorded_inputs(tmp_path: Path) -> dict[str, Path]:
    panel, ambient, weather = [], [], []
    for day in range(8):
        for hour in (0, 1):
            start = pd.Timestamp("2026-05-01T00:00:00Z") + pd.Timedelta(days=day, hours=hour)
            weather.append(
                {
                    "timestamp_utc": start + pd.Timedelta(hours=1),
                    "precipitation": int(day == 2 and hour == 0),
                }
            )
            for number, site in enumerate(("Sukiyabashi", "Shinjuku")):
                panel.append(
                    {
                        "site": site,
                        "hour_timestamp_utc": start,
                        "date_local_jst": start.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d"),
                        "movestar_nox_grams": 1 + day * 2 + hour,
                        "vt_cpfm_nox_grams": 3 + (day + 2) * (hour + 1),
                        "vehicle_count": 20 + day * 3 + hour,
                        "mean_speed_mps": 2 + day / 5 + hour / 10,
                        "nox_ppm": 999,
                        "precipitation_mm": 0,
                    }
                )
                ambient.append(
                    {
                        "monitoring_place": site,
                        "timestamp_utc": start,
                        "nox_ppm": (1 + day * 2 + hour * 3 + number) / 1000,
                    }
                )
    files = {}
    for name, values in (("panel", panel), ("ambient", ambient), ("weather", weather)):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame(values).to_csv(path, index=False)
        files[name] = path
    return files


def _run(files: dict[str, Path], directory: Path, **settings: object) -> dict[str, object]:
    return review_recorded_panel(
        **files,
        output_directory=directory,
        resamples=19,
        seed=42,
        block_lengths=(1, 2),
        day_offsets=(-1, 1),
        **settings,
    )


def test_actual_same_hour_target_replaces_legacy_target_and_rain_uses_interval_end(
    recorded_inputs: dict[str, Path],
) -> None:
    rows = _load(**recorded_inputs)
    assert rows.ambient_nox_ppm.max() < 1
    assert rows.source_interval_precipitation_mm.gt(0).sum() == 2
    assert rows.monitor_end_utc.ne(rows.monitor_start_utc).all()
    assert rows.local_calendar_date.nunique() == 8


def test_dry_filter_shared_draws_paired_differences_and_no_source_acceptance(
    recorded_inputs: dict[str, Path], tmp_path: Path
) -> None:
    before = {name: path.read_bytes() for name, path in recorded_inputs.items()}
    output = tmp_path / "review"
    summary = _run(recorded_inputs, output, rainfall_mode="dry_only")
    assert summary["status"] == STATUS
    assert summary["distinct_observed_calendar_dates"] == 8
    assert summary["sites"]["Sukiyabashi"]["recorded_hours"] == 15
    assert summary["sites"]["Shinjuku"]["recorded_hours"] == 15
    assert summary["sites"]["Shinjuku"]["rainy_source_intervals"] == 0
    rows = pd.read_csv(output / "recorded_hourly_panel.csv")
    result = pd.read_csv(output / "association_and_paired_contrasts.csv")
    assert result.groupby("block_length_days").resample_sha256.nunique().eq(1).all()
    assert result.resamples_valid.eq(19).all()
    for site, group in rows.groupby("site"):
        point = result.loc[result.site.eq(site) & result.block_length_days.eq(1)].set_index(
            "estimand"
        )
        for predictor in PREDICTORS:
            assert point.loc[predictor, "estimate"] == pytest.approx(
                spearmanr(group[predictor], group.ambient_nox_ppm).statistic
            )
        assert point.loc["continuous_minus_movestar", "estimate"] == pytest.approx(
            point.loc["vt_cpfm_nox_grams", "estimate"] - point.loc["movestar_nox_grams", "estimate"]
        )
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["source_validation"] == "not_assessed_by_this_reanalysis"
    assert receipt["recorded_hour_mask"] == "dry_at_interval_end"
    assert len(receipt["focal_estimands_per_site"]) == 5
    assert all(item["primary_family_size"] == 10 for item in receipt["inference"])
    assert {name: path.read_bytes() for name, path in recorded_inputs.items()} == before
    for name, digest in receipt["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert not (output / "analysis_manifest_v4.json").exists()
    assert not (output / "video_seconds.csv").exists()


def test_replay_is_deterministic_and_temporal_controls_are_descriptive(
    recorded_inputs: dict[str, Path], tmp_path: Path
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _run(recorded_inputs, first)
    _run(recorded_inputs, second)
    for name in (
        "association_and_paired_contrasts.csv",
        "temporal_controls.csv",
        "condition_point_estimates.csv",
        "summary.json",
        "parameters.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    controls = pd.read_csv(first / "temporal_controls.csv")
    assert controls.groupby("day_offset").shared_transformation_sha256.nunique().eq(1).all()
    assert controls.retained_hours.le(controls.eligible_hours).all()
    assert controls.status.eq("descriptive_optional_diagnostic").all()
    assert not any("p_value" in name or "ci_" in name for name in controls)
    pdt.assert_frame_equal(
        pd.read_csv(first / "recorded_hourly_panel.csv"),
        pd.read_csv(second / "recorded_hourly_panel.csv"),
    )


@pytest.mark.parametrize(
    "damage", ["duplicate", "date", "negative", "missing_target", "missing_weather"]
)
def test_invalid_recorded_inputs_remain_visible(
    recorded_inputs: dict[str, Path], tmp_path: Path, damage: str
) -> None:
    if damage in {"duplicate", "date", "negative"}:
        path = recorded_inputs["panel"]
        frame = pd.read_csv(path)
        if damage == "duplicate":
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        elif damage == "date":
            frame.loc[0, "date_local_jst"] = "2026-04-30"
        else:
            frame.loc[0, "movestar_nox_grams"] = -1
        frame.to_csv(path, index=False)
    else:
        name = "ambient" if damage == "missing_target" else "weather"
        frame = pd.read_csv(recorded_inputs[name]).iloc[1:]
        frame.to_csv(recorded_inputs[name], index=False)
    if damage == "missing_weather":
        rows = _load(**recorded_inputs)
        assert rows.source_interval_precipitation_mm.isna().sum() == 2
    else:
        with pytest.raises(ValueError):
            _run(recorded_inputs, tmp_path / "invalid")
        assert not (tmp_path / "invalid").exists()


def test_existing_destination_is_not_overwritten(
    recorded_inputs: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "keep"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        _run(recorded_inputs, output)
    assert sentinel.read_text() == "keep"


def test_raw_counts_require_predictor_replay_match(
    recorded_inputs: dict[str, Path], tmp_path: Path
) -> None:
    rows = _load(**recorded_inputs)
    source = tmp_path / "raw"
    source.mkdir()
    pd.DataFrame(
        {
            "source_site": rows.site,
            "video_id": range(len(rows)),
            "recording_datetime_utc": rows.monitor_start_utc,
        }
    ).to_csv(source / "video_master.csv", index=False)
    pd.DataFrame(
        {
            "source_site": rows.site,
            "video_id": range(len(rows)),
            "estimation_method": "MOVEStar",
            "nox_hybrid_adjusted_grams": 999,
            "vehicle_count": rows.vehicle_count,
        }
    ).to_csv(source / "video_emissions.csv", index=False)
    pd.DataFrame(columns=["source_site", "video_id", "recorded_entering_count"]).to_csv(
        source / "entering_count_audit.csv", index=False
    )
    pd.DataFrame(
        columns=["source_site", "video_id", "direction", "class_id", "vehicle_count"]
    ).to_csv(source / "entering_counts.csv", index=False)
    with pytest.raises(ValueError, match="do not reproduce"):
        _recorded_counts(rows, source)


def _raw_fixture(rows: pd.DataFrame, source: Path) -> None:
    source.mkdir()
    videos, emissions, audit, classes = [], [], [], []
    for video, row in rows.iterrows():
        identity = {"source_site": row.site, "video_id": video}
        videos.append(
            {
                **identity,
                "recording_datetime_utc": row.monitor_start_utc,
                "video_path": f"data/production/{video}.mp4",
            }
        )
        for method, column in (("MOVEStar", PREDICTORS[0]), ("VT-CPFM", PREDICTORS[1])):
            emissions.append(
                {
                    **identity,
                    "estimation_method": method,
                    "nox_hybrid_adjusted_grams": row[column],
                    "vehicle_count": row.vehicle_count,
                }
            )
        total, heavy = 10 + video % 3, video % 3
        audit.append(
            {
                **identity,
                "recorded_entering_count": total,
                "count_status": "total_and_classes_agree",
            }
        )
        for class_id, count in ((1, total - heavy), (2, heavy)):
            classes.append(
                {**identity, "direction": "entering", "class_id": class_id, "vehicle_count": count}
            )
    for name, records in (
        ("video_master", videos),
        ("video_emissions", emissions),
        ("entering_count_audit", audit),
        ("entering_counts", classes),
    ):
        pd.DataFrame(records).to_csv(source / f"{name}.csv", index=False)


def test_known_entry_total_does_not_invent_missing_class_counts(
    recorded_inputs: dict[str, Path], tmp_path: Path
) -> None:
    rows = _load(**recorded_inputs)
    source = tmp_path / "raw"
    _raw_fixture(rows, source)
    audit = pd.read_csv(source / "entering_count_audit.csv")
    audit.loc[audit.video_id.eq(0), "count_status"] = "direction_total_only"
    audit.to_csv(source / "entering_count_audit.csv", index=False)
    classes = pd.read_csv(source / "entering_counts.csv")
    classes.loc[classes.video_id.ne(0)].to_csv(source / "entering_counts.csv", index=False)
    prepared, _ = _recorded_counts(rows, source)
    assert prepared.entering_known.all()
    assert prepared.captured_clip_entering_count.notna().all()
    assert prepared.captured_clip_bus_heavy_count.isna().sum() == 1
    assert prepared.heavy_known.sum() == len(rows) - 1


def test_optional_count_absence_does_not_abort_primary_statistics(
    recorded_inputs: dict[str, Path],
) -> None:
    rows = _load(**recorded_inputs)
    config = _config(19, 42, 1)
    primary = _inference(rows, config)
    rows["captured_clip_entering_count"] = float("nan")
    rows["captured_clip_bus_heavy_count"] = float("nan")
    optional = _entry_inference(rows, config)
    assert optional.status.eq("not_estimable").all()
    assert optional.estimate.isna().all()
    assert optional.resamples_valid.eq(0).all()
    assert optional.interval_family.eq("two_models_two_sites_captured_clip_count_comparisons").all()
    assert set(optional.resample_sha256) == set(primary.resample_sha256)
    assert _inference(rows, config).equals(primary)


def test_partial_rank_diagnostic_has_no_unsupported_significance(
    recorded_inputs: dict[str, Path],
) -> None:
    rows = _load(**recorded_inputs)
    table = _partial_rank_points(rows)
    assert len(table) == 4
    assert set(table.controlled_predictor) == {"vehicle_count"}
    assert set(table.n_hours) == {16}
    assert not any("p_value" in column or "ci_" in column for column in table)


def test_explicit_validation_material_excludes_its_whole_site_hour_only(
    recorded_inputs: dict[str, Path], tmp_path: Path
) -> None:
    rows = _load(**recorded_inputs)
    source = tmp_path / "raw"
    _raw_fixture(rows, source)
    videos = pd.read_csv(source / "video_master.csv")
    videos.loc[videos.video_id.eq(0), "video_path"] = "data/counting_test1/count1.mp4"
    videos.to_csv(source / "video_master.csv", index=False)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    (source / "manifest.json").write_text(json.dumps({"outputs": hashes}))
    output = tmp_path / "selected"
    summary = _run(
        recorded_inputs,
        output,
        rainfall_mode="dry_only",
        raw_source_directory=source,
    )
    assert summary["sites"]["Shinjuku"]["recorded_hours"] == 14
    assert summary["sites"]["Sukiyabashi"]["recorded_hours"] == 15
    audit = pd.read_csv(output / "excluded_recorded_source_hours.csv")
    assert len(audit) == 1
    assert audit.video_id.iloc[0] == 0
    assert audit.site.iloc[0] == "Shinjuku"
    primary = pd.read_csv(output / "association_and_paired_contrasts.csv")
    optional = pd.read_csv(output / "captured_clip_count_comparisons.csv")
    assert (
        primary.groupby("block_length_days")
        .resample_sha256.first()
        .equals(optional.groupby("block_length_days").resample_sha256.first())
    )


def test_source_manifest_mismatch_does_not_publish(
    recorded_inputs: dict[str, Path], tmp_path: Path
) -> None:
    source = tmp_path / "raw"
    _raw_fixture(_load(**recorded_inputs), source)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    (source / "manifest.json").write_text(json.dumps({"outputs": hashes}))
    (source / "entering_counts.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="source hash differs"):
        _run(recorded_inputs, tmp_path / "invalid", raw_source_directory=source)
    assert not (tmp_path / "invalid").exists()
