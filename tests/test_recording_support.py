from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from vehicle_emission.inference import recording_support as support
from vehicle_emission.inference.paired_inference import joint_calendar_block_draws
from vehicle_emission.inference.recorded_hourly_reanalysis import _config


@pytest.fixture
def source_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel, videos, emissions = [], [], []
    identity = 0
    for day in range(12):
        if day == 4:
            continue
        for site_index, site in enumerate(support.SITES):
            for hour in range(3):
                start = pd.Timestamp("2026-05-01T00:00:00Z") + pd.Timedelta(days=day, hours=hour)
                minutes = [0, 10, 20, 30, 40, 50]
                if (site_index, day, hour) == (0, 0, 0):
                    minutes.append(55)
                elif (site_index, day, hour) == (0, 2, 0):
                    minutes.pop()
                elif (site_index, day, hour) == (1, 1, 0):
                    minutes[-1] = 40
                elif (site_index, day, hour) == (1, 3, 0):
                    minutes[-1] = 41
                totals = {"MOVEStar": 0.0, "VT-CPFM": 0.0}
                profile_count = 0
                for clip, minute in enumerate(minutes):
                    identity += 1
                    key = {"source_site": site, "video_id": identity}
                    videos.append(
                        {
                            **key,
                            "recording_datetime_utc": (
                                start + pd.Timedelta(minutes=minute)
                            ).isoformat(),
                        }
                    )
                    count = 8 + (day * 2 + hour + clip + site_index) % 13
                    profile_count += count
                    for method in totals:
                        value = (
                            (day + hour + 1) * (1 + clip / 10)
                            if method == "MOVEStar"
                            else (day % 4 + 2 * hour + 1) * (1 + clip / 15)
                        )
                        totals[method] += value
                        emissions.append(
                            {
                                **key,
                                "estimation_method": method,
                                "nox_hybrid_adjusted_grams": value,
                                "vehicle_count": count,
                            }
                        )
                panel.append(
                    {
                        "site": site,
                        "monitor_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "monitor_end_utc": (start + pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "local_calendar_date": start.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d"),
                        "captured_clip_count": len(minutes),
                        "ambient_nox_ppm": (2 + day % 5 + hour * 2 + site_index * (day % 3)) / 1000,
                        "movestar_nox_grams": totals["MOVEStar"],
                        "vt_cpfm_nox_grams": totals["VT-CPFM"],
                        "vehicle_count": profile_count,
                        "captured_clip_entering_count": 20 + day * 3 + hour,
                        "captured_clip_bus_heavy_count": 2 + (day + hour) % 5,
                        "fixed_rate_weighted_entries": 30 + day + hour * 4,
                    }
                )
    return pd.DataFrame(panel), pd.DataFrame(videos), pd.DataFrame(emissions)


@pytest.fixture
def input_bundles(source_frames, tmp_path: Path) -> tuple[Path, Path]:
    panel, videos, emissions = source_frames
    source, extension = tmp_path / "source", tmp_path / "extension"
    source.mkdir()
    extension.mkdir()
    videos.to_csv(source / "video_master.csv", index=False)
    emissions.to_csv(source / "video_emissions.csv", index=False)
    panel.to_csv(extension / "extension_hourly_panel.csv", index=False)
    manifest = {
        "outputs": {
            name: support._hash(source / name)
            for name in ("video_master.csv", "video_emissions.csv")
        }
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    receipt = {
        "outputs": {
            "extension_hourly_panel.csv": support._hash(extension / "extension_hourly_panel.csv")
        },
        "input_sha256": {
            name: support._hash(source / filename)
            for name, filename in (
                ("source_manifest", "manifest.json"),
                ("video_master", "video_master.csv"),
                ("video_emissions", "video_emissions.csv"),
            )
        },
    }
    (extension / "receipt.json").write_text(json.dumps(receipt))
    return source, extension


def prepared(source_frames) -> pd.DataFrame:
    return support.prepare_recording_support(*source_frames)[0]


def oracle_rho(group: pd.DataFrame, column: str) -> float:
    return float(spearmanr(group[column], group.ambient_nox_ppm).statistic)


def test_metadata_rule_distinguishes_record_count_unique_starts_and_clock_bins(
    source_frames,
) -> None:
    panel, clips = support.prepare_recording_support(*source_frames)
    assert len(panel) == 66
    assert len(clips) == 396
    excluded = panel.loc[~panel.restricted_support]
    assert len(excluded) == 4
    assert sorted(excluded.source_clip_count) == [5, 6, 6, 7]
    assert sorted(excluded.distinct_clip_starts) == [5, 5, 6, 7]
    assert sorted(excluded.distinct_ten_minute_bins) == [5, 5, 5, 6]
    assert not panel.duplicated(support.KEYS).any()


def test_support_selection_is_independent_of_outcome(source_frames) -> None:
    panel, videos, emissions = source_frames
    original = prepared(source_frames)
    panel = panel.copy()
    panel["ambient_nox_ppm"] = panel.ambient_nox_ppm.iloc[::-1].to_numpy()
    changed, _ = support.prepare_recording_support(panel, videos, emissions)
    pd.testing.assert_series_equal(original.restricted_support, changed.restricted_support)


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate_video",
        "duplicate_method",
        "missing_method",
        "clip_count",
        "index",
        "profiles",
        "missing_hour",
        "bad_timestamp",
        "naive_timestamp",
        "hour_alias",
        "bad_date",
        "bad_end",
        "nan_primary",
        "infinite_entry",
        "fractional_count",
    ],
)
def test_invalid_source_or_panel_is_rejected(source_frames, damage: str) -> None:
    panel, videos, emissions = (frame.copy() for frame in source_frames)
    if damage == "duplicate_video":
        videos = pd.concat([videos, videos.iloc[[0]]], ignore_index=True)
    elif damage == "duplicate_method":
        emissions = pd.concat([emissions, emissions.iloc[[0]]], ignore_index=True)
    elif damage == "missing_method":
        emissions = emissions.iloc[1:]
    elif damage == "clip_count":
        panel.loc[0, "captured_clip_count"] += 1
    elif damage == "index":
        panel.loc[0, "vt_cpfm_nox_grams"] += 1
    elif damage == "profiles":
        emissions.loc[0, "vehicle_count"] += 1
    elif damage == "missing_hour":
        videos = videos.loc[videos.video_id.gt(7)]
    elif damage == "bad_timestamp":
        videos.loc[0, "recording_datetime_utc"] = "not a timestampZ"
    elif damage == "naive_timestamp":
        videos.loc[0, "recording_datetime_utc"] = "2026-05-01T00:00:00"
    elif damage == "hour_alias":
        panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
        panel.loc[len(panel) - 1, "monitor_start_utc"] = "2026-05-01T00:00:00+00:00"
    elif damage == "bad_date":
        panel.loc[0, "local_calendar_date"] = "2026-04-30"
    elif damage == "bad_end":
        panel.loc[0, "monitor_end_utc"] = "2026-05-01T02:00:00Z"
    elif damage == "nan_primary":
        panel.loc[0, "ambient_nox_ppm"] = np.nan
    elif damage == "infinite_entry":
        panel["fixed_rate_weighted_entries"] = panel.fixed_rate_weighted_entries.astype(float)
        panel.loc[0, "fixed_rate_weighted_entries"] = np.inf
    elif damage == "fractional_count":
        panel["captured_clip_count"] = panel.captured_clip_count.astype(float)
        panel.loc[0, "captured_clip_count"] = 6.5
    with pytest.raises(ValueError):
        support.prepare_recording_support(panel, videos, emissions)


def test_count_pairing_and_contrast_directions_match_independent_spearman(source_frames) -> None:
    source_frames[0].loc[0, "captured_clip_bus_heavy_count"] = np.nan
    panel = prepared(source_frames)
    values = support.recording_support_values(panel)
    for site in support.SITES:
        for cohort in ("current", "restricted"):
            group = panel.loc[panel.site.eq(site)]
            if cohort == "restricted":
                group = group.loc[group.restricted_support]
            assert values[
                site, cohort, "recorded_core", "continuous_minus_movestar"
            ] == pytest.approx(
                oracle_rho(group, "vt_cpfm_nox_grams") - oracle_rho(group, "movestar_nox_grams")
            )
            assert values[
                site, cohort, "recorded_core", "continuous_minus_recorded_volume"
            ] == pytest.approx(
                oracle_rho(group, "vt_cpfm_nox_grams") - oracle_rho(group, "vehicle_count")
            )
            group = group.loc[group.entry_counts_complete]
            assert values[
                site,
                cohort,
                "complete_entry_counts",
                "continuous_minus_fixed_rate_weighted_entries",
            ] == pytest.approx(
                oracle_rho(group, "vt_cpfm_nox_grams")
                - oracle_rho(group, "fixed_rate_weighted_entries")
            )
        for estimand, population in support.ESTIMANDS:
            assert values[site, "restricted_minus_current", population, estimand] == pytest.approx(
                values[site, "restricted", population, estimand]
                - values[site, "current", population, estimand]
            )
    suk = panel.loc[panel.site.eq("Sukiyabashi")]
    assert len(suk.loc[suk.entry_counts_complete]) == len(suk) - 1


def test_joint_full_calendar_draws_and_direct_interval_replay(source_frames) -> None:
    panel = prepared(source_frames)
    config = support.RecordingSupportConfig(resamples=19, block_lengths=(1, 7))
    result = support.infer_recording_support(panel, config)
    assert set(result.calendar_days) == {12}
    assert result.groupby("block_length_days").resample_sha256.nunique().eq(1).all()
    assert set(result.interval_family) == {"pointwise_descriptive"}
    for block in config.block_lengths:
        rng = np.random.default_rng(config.seed)
        direct = []
        digest = hashlib.sha256(b"recording-support-joint-calendar-draws-v1\0")
        for selected, indices, _ in joint_calendar_block_draws(
            panel, _config(config.resamples, config.seed, block)
        ):
            expected = []
            while len(expected) < 12:
                start = int(rng.integers(0, 12 - block + 1))
                expected.extend(range(start, start + block))
            np.testing.assert_array_equal(selected, expected[:12])
            expected_dates = [f"2026-05-{position + 1:02d}" for position in selected]
            expected_indices = np.concatenate(
                [
                    panel.index[panel.local_calendar_date.eq(date)].to_numpy()
                    for date in expected_dates
                ]
            )
            np.testing.assert_array_equal(indices, expected_indices)
            digest.update(selected.astype("<i8").tobytes())
            digest.update(indices.astype("<i8").tobytes())
            sample = panel.iloc[indices]
            estimates = []
            for restriction in (False, True):
                group = sample.loc[sample.site.eq("Sukiyabashi")]
                if restriction:
                    group = group.loc[group.restricted_support]
                estimates.append(
                    spearmanr(group.vt_cpfm_nox_grams, group.ambient_nox_ppm).statistic
                    - spearmanr(group.movestar_nox_grams, group.ambient_nox_ppm).statistic
                )
            direct.append(estimates[1] - estimates[0])
        row = result.loc[
            result.site.eq("Sukiyabashi")
            & result.cohort.eq("restricted_minus_current")
            & result.estimand.eq("continuous_minus_movestar")
            & result.block_length_days.eq(block)
        ].iloc[0]
        np.testing.assert_allclose(
            [row.ci_low, row.ci_high], np.quantile(direct, [0.025, 0.975]), atol=1e-14
        )
        assert row.resample_sha256 == digest.hexdigest()
        assert row.resamples_valid == config.resamples
        assert row.resamples_invalid == 0


def test_invalid_sparse_draws_are_counted_without_conditional_intervals(source_frames) -> None:
    panel = prepared(source_frames)
    panel["restricted_support"] = panel.local_calendar_date.eq("2026-05-01")
    result = support.infer_recording_support(panel, support.RecordingSupportConfig(79, 42, (1,)))
    restricted = result.loc[result.cohort.ne("current")]
    assert restricted.resamples_invalid.gt(0).all()
    assert restricted.resamples_valid.gt(0).all()
    assert restricted.ci_low.isna().all()
    assert restricted.ci_high.isna().all()
    assert restricted.interval_status.eq("not_estimable").all()
    assert (restricted.resamples_valid + restricted.resamples_invalid).eq(79).all()


def test_empty_secondary_and_constant_predictor_are_not_filled(source_frames) -> None:
    panel = prepared(source_frames)
    panel["entry_counts_complete"] = False
    panel["vehicle_count"] = 1
    result = support.infer_recording_support(panel, support.RecordingSupportConfig(19, 42, (1,)))
    unavailable = result.loc[
        result.population.eq("complete_entry_counts")
        | result.estimand.isin(["vehicle_count", "continuous_minus_recorded_volume"])
    ]
    assert unavailable.estimate.isna().all()
    assert unavailable.ci_low.isna().all()
    assert unavailable.resamples_valid.eq(0).all()


@pytest.mark.parametrize(
    "config",
    [
        support.RecordingSupportConfig(18),
        support.RecordingSupportConfig(seed=-1),
        support.RecordingSupportConfig(block_lengths=()),
        support.RecordingSupportConfig(block_lengths=(1, 1)),
        support.RecordingSupportConfig(block_lengths=(True,)),
    ],
)
def test_invalid_configs_fail(config) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_oversized_block_fails(source_frames) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        support.infer_recording_support(
            prepared(source_frames), support.RecordingSupportConfig(19, 42, (13,))
        )


def test_fresh_receipted_bundle_is_deterministic_and_preserves_inputs(
    input_bundles, tmp_path: Path
) -> None:
    source, extension = input_bundles
    before = {
        path: path.read_bytes() for directory in (source, extension) for path in directory.iterdir()
    }
    config = support.RecordingSupportConfig(19, 42, (1, 7))
    first, second = tmp_path / "first", tmp_path / "second"
    receipt = support.run_recording_support(
        source_directory=source, extension_bundle=extension, output_directory=first, config=config
    )
    support.run_recording_support(
        source_directory=source, extension_bundle=extension, output_directory=second, config=config
    )
    assert receipt["calendar_days"] == 12
    assert receipt["support_rule"] == support.SUPPORT_RULE
    assert all(path.read_bytes() == expected for path, expected in before.items())
    assert {path.name for path in first.iterdir()} == {path.name for path in second.iterdir()}
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    for name, expected in receipt["outputs"].items():
        assert support._hash(first / name) == expected
    for name, expected in receipt["input_sha256"].items():
        assert support._hash(Path(receipt["input_paths"][name])) == expected
    for name, expected in receipt["calculator_sha256"].items():
        assert support._hash(Path(name)) == expected


@pytest.mark.parametrize(
    "kind", ["existing", "broken_symlink", "source_nested", "extension_nested"]
)
def test_existing_or_nested_output_is_rejected(input_bundles, tmp_path: Path, kind: str) -> None:
    source, extension = input_bundles
    target = tmp_path / "target"
    if kind == "existing":
        target.mkdir()
        (target / "keep.txt").write_text("unchanged")
    elif kind == "broken_symlink":
        target.symlink_to(tmp_path / "missing")
    else:
        target = (source if kind == "source_nested" else extension) / "nested"
    with pytest.raises((FileExistsError, ValueError)):
        support.run_recording_support(
            source_directory=source, extension_bundle=extension, output_directory=target
        )
    if kind == "existing":
        assert (target / "keep.txt").read_text() == "unchanged"


@pytest.mark.parametrize(
    "changed", ["panel", "video_master", "emissions", "manifest", "different_snapshot"]
)
def test_receipt_mismatch_is_rejected_before_output(
    input_bundles, tmp_path: Path, changed: str
) -> None:
    source, extension = input_bundles
    files = {
        "panel": extension / "extension_hourly_panel.csv",
        "video_master": source / "video_master.csv",
        "emissions": source / "video_emissions.csv",
        "manifest": source / "manifest.json",
        "different_snapshot": source / "video_master.csv",
    }
    files[changed].write_bytes(files[changed].read_bytes() + b"\n")
    if changed == "different_snapshot":
        manifest = json.loads((source / "manifest.json").read_text())
        manifest["outputs"]["video_master.csv"] = support._hash(files[changed])
        (source / "manifest.json").write_text(json.dumps(manifest))
    target = tmp_path / "reject"
    with pytest.raises(ValueError):
        support.run_recording_support(
            source_directory=source, extension_bundle=extension, output_directory=target
        )
    assert not target.exists()


def test_input_mutation_during_calculation_is_rejected(
    input_bundles, tmp_path: Path, monkeypatch
) -> None:
    source, extension = input_bundles
    original = support.infer_recording_support

    def mutate(panel, config):
        result = original(panel, config)
        path = source / "video_master.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(support, "infer_recording_support", mutate)
    target = tmp_path / "reject"
    with pytest.raises(ValueError, match="changed during"):
        support.run_recording_support(
            source_directory=source,
            extension_bundle=extension,
            output_directory=target,
            config=support.RecordingSupportConfig(19, 42, (1,)),
        )
    assert not target.exists()
