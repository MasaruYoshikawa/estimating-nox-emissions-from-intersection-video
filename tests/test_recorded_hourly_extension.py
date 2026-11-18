from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from vehicle_emission.inference.recorded_hourly_extension import (
    CLASS_COLUMNS,
    ExtensionConfig,
    _cross_site,
    _interval,
    descriptive_tables,
    extension_values,
    infer_extensions,
    prepare_extension_panel,
    rank_residual_values,
    reference_class_weights,
    run_extension,
)


@pytest.fixture
def extension_inputs(tmp_path: Path) -> tuple[pd.DataFrame, Path, Path]:
    source = tmp_path / "source"
    base = tmp_path / "base"
    source.mkdir()
    base.mkdir()
    videos, emissions, audits, classes, core, weather, ambient = [], [], [], [], [], [], []
    video_id = 0
    for day in range(12):
        for hour in range(6):
            start = pd.Timestamp("2026-05-01T00:00:00Z") + pd.Timedelta(days=day, hours=hour)
            stamp = start.strftime("%Y-%m-%dT%H:%M:%SZ")
            weather.append(
                {
                    "timestamp_utc": stamp,
                    "temperature": 12 + day + hour / 10,
                    "wind_speed": 1 + hour / 4,
                }
            )
            ambient.append(
                {
                    "timestamp_utc": stamp,
                    "monitoring_place": "Setagaya",
                    "nox_ppm": (day % 4 + hour + 1) / 1000,
                }
            )
            for site_number, site in enumerate(("Sukiyabashi", "Shinjuku")):
                totals, speed = {}, []
                profile_volume, entry, heavy = 0, 0, 0
                for clip in range(2):
                    video_id += 1
                    identity = {"source_site": site, "video_id": video_id}
                    videos.append(
                        {
                            **identity,
                            "recording_datetime_utc": (
                                start + pd.Timedelta(minutes=clip * 10)
                            ).isoformat(),
                        }
                    )
                    values = [
                        day % 3 + clip,
                        20 + day + hour + clip,
                        (day + hour + clip) % 5,
                        hour % 3 + clip,
                    ]
                    speed.append(8 + day / 10 + hour / 5 + clip * (1 + day / 20))
                    entry += sum(values)
                    heavy += values[0] + values[2]
                    profiles = 5 * sum(values)
                    profile_volume += profiles
                    for method, multiplier in (("MOVEStar", 1.0), ("VT-CPFM", 1.3)):
                        adjusted = (1 + day + hour * (1 + clip) / 4 + site_number) * multiplier
                        emissions.append(
                            {
                                **identity,
                                "estimation_method": method,
                                "nox_grams": adjusted * (1.03 + day / 100),
                                "nox_hybrid_adjusted_grams": adjusted,
                                "average_speed_kmh": speed[-1],
                                "vehicle_count": profiles,
                            }
                        )
                        totals[method] = totals.get(method, 0) + adjusted
                    for direction in ("top", "bottom", "left", "right"):
                        direction_name = f"entering_from_{direction}"
                        selected = values if direction == "bottom" else [0] * 4
                        total = sum(selected)
                        audits.append(
                            {
                                **identity,
                                "direction": direction_name,
                                "direction_total": total,
                                "class_row_sum": total,
                                "recorded_entering_count": total,
                                "count_status": "total_and_classes_agree",
                            }
                        )
                        for class_id, count in enumerate(selected):
                            classes.append(
                                {
                                    **identity,
                                    "direction": direction_name,
                                    "class_id": class_id,
                                    "vehicle_count": count,
                                }
                            )
                core.append(
                    {
                        "site": site,
                        "monitor_start_utc": stamp,
                        "monitor_end_utc": (start + pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "local_calendar_date": start.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d"),
                        "ambient_nox_ppm": (1 + day % 5 + hour * 2 + site_number) / 1000,
                        "movestar_nox_grams": totals["MOVEStar"],
                        "vt_cpfm_nox_grams": totals["VT-CPFM"],
                        "vehicle_count": profile_volume,
                        "mean_speed_mps": np.mean(speed) / 3.6,
                        "captured_clip_count": 2,
                        "captured_clip_entering_count": entry,
                        "captured_clip_bus_heavy_count": heavy,
                    }
                )
    frame = pd.DataFrame(core).sort_values(["site", "monitor_start_utc"]).reset_index(drop=True)
    for name, values in (
        ("video_master", videos),
        ("video_emissions", emissions),
        ("entering_count_audit", audits),
        ("entering_counts", classes),
        ("weather_history", weather),
        ("air_pollution_history", ambient),
    ):
        pd.DataFrame(values).to_csv(source / f"{name}.csv", index=False)
    pd.DataFrame(core).to_csv(base / "recorded_hourly_panel.csv", index=False)
    pd.DataFrame(
        {
            "day_offset": [-1, -7],
            "site": ["Sukiyabashi"] * 2,
            "retained_hours": [60, 60],
            "retained_wrapped_hours": [3, 4],
        }
    ).to_csv(base / "temporal_controls.csv", index=False)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (source / "manifest.json").write_text(
        json.dumps({"outputs": {p.name: digest(p) for p in source.glob("*.csv")}})
    )
    bindings = {
        "actual_ambient": "air_pollution_history.csv",
        "weather": "weather_history.csv",
        "raw_manifest": "manifest.json",
        **{
            name: f"{name}.csv"
            for name in (
                "video_master",
                "video_emissions",
                "entering_counts",
                "entering_count_audit",
            )
        },
    }
    (base / "receipt.json").write_text(
        json.dumps(
            {
                "outputs": {p.name: digest(p) for p in base.glob("*.csv")},
                "input_sha256": {key: digest(source / name) for key, name in bindings.items()},
            }
        )
    )
    return frame, source, base


def test_reference_weights_are_outcome_independent_and_canonical() -> None:
    table = reference_class_weights().set_index("class_id")
    assert table.loc[1, "normalised_weight"] == 1
    assert table.native_vehicle_type.to_dict() == {0: 6, 1: 1, 2: 13, 3: 2}
    assert table.reference_speed_kmh.eq(25).all()
    assert (table.reference_nox_g_per_second > 0).all()


def test_source_reconstruction_class_totals_raw_indices_and_clip_cv(extension_inputs) -> None:
    core, source, _ = extension_inputs
    prepared = prepare_extension_panel(core, source, reference_class_weights())
    assert len(prepared) == len(core)
    assert prepared.residual_complete.all()
    assert prepared[list(CLASS_COLUMNS)].sum(axis=1).eq(prepared.captured_clip_entering_count).all()
    assert prepared.between_clip_speed_cv.gt(0).all()
    assert prepared.movestar_nox_grams_raw.gt(prepared.movestar_nox_grams).all()
    first = prepared.iloc[0]
    speeds = [8, 9]
    assert first.between_clip_speed_cv == pytest.approx(np.std(speeds, ddof=1) / np.mean(speeds))


@pytest.mark.parametrize("damage", ["duplicate", "negative", "replay"])
def test_source_damage_is_rejected(extension_inputs, damage) -> None:
    core, source, _ = extension_inputs
    path = source / "video_emissions.csv"
    frame = pd.read_csv(path)
    if damage == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif damage == "negative":
        frame.loc[0, "average_speed_kmh"] = -1
    else:
        frame.loc[0, "nox_hybrid_adjusted_grams"] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        prepare_extension_panel(core, source, reference_class_weights())


def test_unknown_classes_affect_only_their_counts_and_residual_population(extension_inputs) -> None:
    core, source, _ = extension_inputs
    path = source / "entering_count_audit.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.index[0], "count_status"] = "direction_total_only"
    frame.to_csv(path, index=False)
    prepared = prepare_extension_panel(core, source, reference_class_weights())
    assert len(prepared) == len(core)
    assert prepared.fixed_rate_weighted_entries.isna().sum() == 1
    assert (~prepared.residual_complete).sum() == 1
    assert prepared.between_clip_speed_cv.notna().all()


def test_zero_clip_speed_mean_is_undefined_not_zero_volatility(extension_inputs) -> None:
    core, source, _ = extension_inputs
    path = source / "video_emissions.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.video_id.isin([1, 2]), "average_speed_kmh"] = 0
    frame.to_csv(path, index=False)
    core.loc[
        core.site.eq("Sukiyabashi") & core.monitor_start_utc.eq("2026-05-01T00:00:00Z"),
        "mean_speed_mps",
    ] = 0
    prepared = prepare_extension_panel(core, source, reference_class_weights())
    assert prepared.between_clip_speed_cv.isna().sum() == 1
    assert len(prepared) == len(core)


def test_rank_residual_marginal_matches_spearman_and_common_support(extension_inputs) -> None:
    core, source, _ = extension_inputs
    prepared = prepare_extension_panel(core, source, reference_class_weights())
    group = prepared.loc[prepared.site.eq("Sukiyabashi")]
    values = rank_residual_values(group)
    assert values["marginal:movestar_nox_grams"] == pytest.approx(
        spearmanr(group.movestar_nox_grams, group.ambient_nox_ppm).statistic
    )
    group = group.copy()
    group.loc[group.index[0], "residual_complete"] = False
    altered = rank_residual_values(group)
    subset = group.loc[group.residual_complete]
    assert altered["marginal:movestar_nox_grams"] == pytest.approx(
        spearmanr(subset.movestar_nox_grams, subset.ambient_nox_ppm).statistic
    )


def test_count_contrast_sign_and_descriptive_volatility_support(extension_inputs) -> None:
    core, source, _ = extension_inputs
    prepared = prepare_extension_panel(core, source, reference_class_weights())
    values = extension_values(prepared)
    group = prepared.loc[prepared.site.eq("Shinjuku")]
    expected = (
        spearmanr(group.vt_cpfm_nox_grams, group.ambient_nox_ppm).statistic
        - spearmanr(group.fixed_rate_weighted_entries, group.ambient_nox_ppm).statistic
    )
    assert values[
        "count", "Shinjuku", "vt_cpfm_nox_grams:fixed_rate_weighted_entries"
    ] == pytest.approx(expected)
    tables = descriptive_tables(prepared)
    assert tables["volatility_support.csv"].groupby("site").n_hours.sum().eq(72).all()
    assert (
        _cross_site(
            {
                (s, x): (1 if s == "Shinjuku" else 0)
                for s in ("Shinjuku", "Sukiyabashi")
                for x in (
                    "movestar_nox_grams",
                    "vt_cpfm_nox_grams",
                    "movestar_minus_recorded_volume",
                    "continuous_minus_recorded_volume",
                    "continuous_minus_movestar",
                )
            }
        )["continuous_minus_movestar"]
        == 1
    )


def test_shared_calendar_draws_and_fixed_prefixes(extension_inputs) -> None:
    core, source, _ = extension_inputs
    prepared = prepare_extension_panel(core, source, reference_class_weights())
    config = ExtensionConfig(19, 5, 42, (1, 2))
    tables = infer_extensions(prepared, config)
    assert (
        tables["core_inference.csv"]
        .groupby("block_length_days")
        .resample_sha256.nunique()
        .eq(1)
        .all()
    )
    assert tables["core_inference.csv"].family_size.isin([1, 10]).all()
    assert (
        tables["residual_comparisons.csv"]
        .groupby("block_length_days")
        .resample_sha256.nunique()
        .eq(1)
        .all()
    )
    assert tables["residual_comparisons.csv"].n_hours.eq(72).all()
    assert tables["core_interval_convergence.csv"].resamples.eq(19).all()
    assert tables["volatility_comparisons.csv"].resampling_population_hours.eq(72).all()
    lower = tables["volatility_comparisons.csv"].loc[
        lambda value: value.estimand.str.startswith("lower:")
    ]
    assert lower.point_stratum_hours.lt(lower.n_hours).all()
    assert np.isnan(_interval([1, np.nan])[0])


def test_existing_destination_is_never_overwritten(extension_inputs, tmp_path) -> None:
    _, source, base = extension_inputs
    target = tmp_path / "keep"
    target.mkdir()
    with pytest.raises(FileExistsError):
        run_extension(
            base_bundle=base,
            source_directory=source,
            output_directory=target,
            config=ExtensionConfig(19, 2),
        )


def test_raw_hash_tampering_fails_before_output(extension_inputs, tmp_path) -> None:
    _, source, base = extension_inputs
    with (source / "weather_history.csv").open("a") as stream:
        stream.write("\n")
    target = tmp_path / "reject"
    with pytest.raises(ValueError, match="source hash"):
        run_extension(
            base_bundle=base,
            source_directory=source,
            output_directory=target,
            config=ExtensionConfig(19, 2),
        )
    assert not target.exists()


def test_end_to_end_fresh_bundle_receipts_and_deterministic_replay(
    extension_inputs, tmp_path
) -> None:
    _, source, base = extension_inputs
    first, second = tmp_path / "first", tmp_path / "second"
    config = ExtensionConfig(19, 19, 42, (1, 2))
    result = run_extension(
        base_bundle=base, source_directory=source, output_directory=first, config=config
    )
    run_extension(base_bundle=base, source_directory=source, output_directory=second, config=config)
    assert result["sites"]["Sukiyabashi"]["core_hours"] == 72
    receipt = json.loads((first / "receipt.json").read_text())
    assert receipt["source_validation"] == "not_assessed_by_this_statistical_extension"
    assert receipt["core_family_size"] == 10
    offsets = pd.read_csv(first / "offset_control_contrasts.csv")
    assert offsets.weekday_preserved_for_nonwrapped_pairs.tolist() == [False, True]
    assert offsets.retained_weekday_preserving_hours.tolist() == [0, 56]
    for name, digest in receipt["outputs"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_internally_valid_different_source_snapshot_does_not_match_base(
    extension_inputs, tmp_path
) -> None:
    _, source, base = extension_inputs
    path = source / "weather_history.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "temperature"] += 1
    frame.to_csv(path, index=False)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["outputs"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="base/source snapshot"):
        run_extension(
            base_bundle=base,
            source_directory=source,
            output_directory=tmp_path / "reject",
            config=ExtensionConfig(19, 2),
        )


@pytest.mark.parametrize("direction", ["unrecognised", "entering_from_top"])
def test_class_direction_requires_canonical_matching_audit(extension_inputs, direction) -> None:
    core, source, _ = extension_inputs
    class_path = source / "entering_counts.csv"
    classes = pd.read_csv(class_path)
    row = classes.iloc[0].copy()
    row["direction"] = direction
    if direction == "entering_from_top":
        audit_path = source / "entering_count_audit.csv"
        audit = pd.read_csv(audit_path)
        audit.loc[~(audit.video_id.eq(row.video_id) & audit.direction.eq(direction))].to_csv(
            audit_path, index=False
        )
        classes = classes.loc[
            ~(classes.video_id.eq(row.video_id) & classes.direction.eq(direction))
        ]
    classes = pd.concat([classes, pd.DataFrame([row])], ignore_index=True)
    classes.to_csv(class_path, index=False)
    with pytest.raises(ValueError):
        prepare_extension_panel(core, source, reference_class_weights())


def test_positive_uniform_scaling_cannot_create_a_hybrid_rank_effect(extension_inputs) -> None:
    core, source, _ = extension_inputs
    prepared = prepare_extension_panel(core, source, reference_class_weights())
    for model in ("movestar_nox_grams", "vt_cpfm_nox_grams"):
        prepared[f"{model}_raw"] = prepared[model] * 2
    values = extension_values(prepared)
    assert all(
        abs(value) < 1e-12
        for (package, _, estimand), value in values.items()
        if package == "hybrid" and estimand.endswith("adjusted_minus_raw")
    )


def test_volatility_median_is_recomputed_after_calendar_resampling(extension_inputs) -> None:
    core, source, _ = extension_inputs
    prepared = prepare_extension_panel(core, source, reference_class_weights())
    skewed = pd.concat([prepared, prepared.loc[prepared.hour_jst.le(10)]] * 2, ignore_index=True)
    values = extension_values(skewed)
    subset = skewed.loc[skewed.site.eq("Sukiyabashi")]
    lower = subset.loc[subset.between_clip_speed_cv.le(subset.between_clip_speed_cv.median())]
    expected = spearmanr(lower.movestar_nox_grams, lower.ambient_nox_ppm).statistic
    assert values["volatility", "Sukiyabashi", "lower:movestar_nox_grams"] == pytest.approx(
        expected
    )
