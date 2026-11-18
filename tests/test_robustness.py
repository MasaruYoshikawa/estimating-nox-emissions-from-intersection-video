from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from scipy.stats import spearmanr

from vehicle_emission.emissions.movestar import MOVEStarModel
from vehicle_emission.robustness import runner
from vehicle_emission.robustness.runner import CODE_PATHS, main, replay_robustness, run_robustness
from vehicle_emission.robustness.speed import SpeedPerturbation, calculate_speed_sensitivity
from vehicle_emission.robustness.temporal import TemporalControlConfig, calculate_temporal_controls
from vehicle_emission.speed.production import second_speed_exports


def _profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": "Sukiyabashi",
            "source_id": "fixture-video",
            "profile_id": "vehicle-a",
            "class_id": 1,
            "second_index": range(9),
            "speed_mps": [0.0, 1.0, 2.0, 4.0, np.nan, 12.0, 13.0, 15.0, 0.0],
            "duration_seconds": 1.0,
        }
    )


def _hourly() -> pd.DataFrame:
    rows = []
    # Full day 2 is absent, and a site-specific hour is absent on day 4.
    for site in ("Sukiyabashi", "Shinjuku"):
        for day in (0, 2, 3, 4):
            for hour in (0, 1, 2, 3):
                if site == "Shinjuku" and day == 3 and hour == 2:
                    continue
                start = pd.Timestamp("2025-06-01", tz="Asia/Tokyo") + pd.Timedelta(
                    days=day, hours=hour
                )
                rows.append(
                    {
                        "site": site,
                        "monitor_start_utc": start.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "monitor_end_utc": (start + pd.Timedelta(hours=1))
                        .tz_convert("UTC")
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "local_calendar_date": start.strftime("%Y-%m-%d"),
                        "ambient_nox_ppm": (day * 7 + hour * 3 + 1) / 1000,
                        "relative_nox_emission_index": day * 5 + hour**2 + 1,
                        "count_only_index": (day + 1) * (4 - hour),
                        "primary_measurement_eligible": True,
                        "target_observation_status": "actual_observation",
                    }
                )
    return pd.DataFrame(rows)


def _temporal_config() -> TemporalControlConfig:
    return TemporalControlConfig("2025-06-01", "2025-06-05", (1, -1), (1, 24, 48, 200))


def test_actual_movestar_recalculation_zero_identity_and_nonlinearity() -> None:
    scenarios = (SpeedPerturbation("zero", 1.0, 0.0), SpeedPerturbation("double", 2.0, 0.0))
    result = calculate_speed_sensitivity(_profiles(), scenarios)
    zero, double = result.summary.iloc[0], result.summary.iloc[1]
    assert zero.difference_nox_g == 0.0
    assert result.summary.support_sha256.nunique() == 1
    assert result.profiles.valid_seconds.iloc[0] == 8
    assert result.profiles.missing_seconds.iloc[0] == 1
    assert result.profiles.valid_runs.iloc[0] == 2
    model = MOVEStarModel()
    expected = sum(
        model.estimate_emissions(1, profile)["Emission Rate"][0][2]
        for profile in ([0.0, 2.0, 4.0, 8.0], [24.0, 26.0, 30.0, 0.0])
    )
    assert double.scenario_nox_g == expected
    assert double.scenario_nox_g != 2 * double.baseline_nox_g
    assert result.runs.zero_speed_seconds.sum() == 4
    assert result.runs.first_second_index.tolist() == [0, 5, 0, 5]
    assert json.loads(result.runs.acceleration_mps2.iloc[0]) == [0.0, 1.0, 1.5, 0.0]
    assert sum(json.loads(result.runs.operating_mode_seconds.iloc[0])) == 4
    pdt.assert_frame_equal(
        result.summary,
        calculate_speed_sensitivity(_profiles().sample(frac=1, random_state=6), scenarios).summary,
    )


@pytest.mark.parametrize(
    "change", ["gap", "duplicate", "class", "negative", "infinite", "partial", "timestamp"]
)
def test_invalid_profile_contracts(change: str) -> None:
    profiles = _profiles()
    if change == "gap":
        profiles.loc[8, "second_index"] = 10
    elif change == "duplicate":
        profiles.loc[8, "second_index"] = 7
    elif change == "class":
        profiles.loc[8, "class_id"] = 3
    elif change == "negative":
        profiles.loc[0, "speed_mps"] = -1
    elif change == "infinite":
        profiles.loc[0, "speed_mps"] = np.inf
    elif change == "partial":
        profiles.loc[0, "duration_seconds"] = 0.5
    else:
        profiles["ts_utc"] = "2025-06-01T00:00:00Z"
    with pytest.raises(ValueError):
        calculate_speed_sensitivity(profiles, (SpeedPerturbation("zero", 1, 0),))


def test_negative_perturbation_rejected_without_clipping_and_missing_retained() -> None:
    with pytest.raises(ValueError, match="negative speeds"):
        calculate_speed_sensitivity(_profiles(), (SpeedPerturbation("impossible", 1, -1),))
    profiles = _profiles()
    profiles["speed_mps"] = np.nan
    result = calculate_speed_sensitivity(profiles, (SpeedPerturbation("zero", 1, 0),))
    assert result.summary.status.iloc[0] == "no_valid_seconds"
    assert pd.isna(result.summary.baseline_nox_g.iloc[0])
    assert result.runs.empty
    assert "nox_g" in result.runs


def test_production_complete_profiles_directly_feed_movestar_adapter() -> None:
    identity = {
        "site": "Sukiyabashi",
        "source_id": "fixture-video",
        "video_id": 1,
        "track_id": 4,
        "class_id": 1,
        "camera_epoch": "fixture-epoch",
        "calibration_id": "fixture-homography",
        "image_zone": "road-plane",
    }
    steps = pd.DataFrame(
        [
            {
                **identity,
                "start_utc": "2025-06-01T00:00:00.500Z",
                "end_utc": "2025-06-01T00:00:01Z",
                "speed_mps": 8.0,
            },
            {
                **identity,
                "start_utc": "2025-06-01T00:00:01Z",
                "end_utc": "2025-06-01T00:00:03Z",
                "speed_mps": 0.0,
            },
            {
                **identity,
                "start_utc": "2025-06-01T00:00:04Z",
                "end_utc": "2025-06-01T00:00:06Z",
                "speed_mps": 12.0,
            },
        ]
    )
    seconds, profiles = second_speed_exports(steps)
    assert seconds.duration_seconds.iloc[0] == 0.5
    assert profiles.duration_seconds.eq(1.0).all()
    assert profiles.ts_utc.str.endswith("Z").all()
    assert profiles.profile_id.nunique() == 2
    result = calculate_speed_sensitivity(profiles, (SpeedPerturbation("zero", 1, 0),))
    assert result.summary.difference_nox_g.eq(0).all()
    assert result.summary.valid_seconds.sum() == 4
    assert result.runs.zero_speed_seconds.sum() == 2
    with pytest.raises(ValueError, match="one-second bins"):
        calculate_speed_sensitivity(
            seconds.assign(profile_id="partial", second_index=range(len(seconds))),
            (SpeedPerturbation("zero", 1, 0),),
        )


def test_full_calendar_shared_day_shift_missingness_sign_and_common_masks() -> None:
    result = calculate_temporal_controls(_hourly(), _temporal_config())
    assert len(result.support) == 2 * 2 * 5 * 24
    assert result.contrasts.calendar_hours.eq(120).all()
    assert result.contrasts.groupby("day_offset").shared_transformation_sha256.nunique().eq(1).all()
    forward = result.support.loc[
        (result.support.site == "Sukiyabashi") & (result.support.day_offset == 1)
    ]
    # The absent second day moves to day 3; compressing available rows cannot satisfy this.
    third_day = forward.loc[forward.monitor_start_utc.str.startswith("2025-06-02T15:")].iloc[0]
    assert not third_day.transformed_predictors_available
    assert third_day.source_predictor_start_utc == "2025-06-01T15:00:00Z"
    first = forward.iloc[0]
    assert first.wrapped
    assert first.source_predictor_start_utc == "2025-06-04T15:00:00Z"
    assert (
        forward.source_predictor_start_utc.str[11:13].tolist()
        == forward.monitor_start_utc.str[11:13].tolist()
    )
    observed_rows = _hourly().set_index(["site", "monitor_start_utc"])
    for _, contrast in result.contrasts.iterrows():
        pairs = result.support.loc[
            (result.support.site == contrast.site)
            & (result.support.day_offset == contrast.day_offset)
            & result.support.retained_pair
        ]
        observed = observed_rows.loc[[(contrast.site, key) for key in pairs.monitor_start_utc]]
        control = observed_rows.loc[
            [(contrast.site, key) for key in pairs.source_predictor_start_utc]
        ]
        expected_observed = (
            spearmanr(observed.relative_nox_emission_index, observed.ambient_nox_ppm).statistic
            - spearmanr(observed.count_only_index, observed.ambient_nox_ppm).statistic
        )
        expected_control = (
            spearmanr(control.relative_nox_emission_index, observed.ambient_nox_ppm).statistic
            - spearmanr(control.count_only_index, observed.ambient_nox_ppm).statistic
        )
        assert contrast.observed_index_count_contrast == pytest.approx(expected_observed)
        assert contrast.control_index_count_contrast == pytest.approx(expected_control)
        assert contrast.retained_hours == len(pairs)
    pdt.assert_frame_equal(
        result.contrasts,
        calculate_temporal_controls(
            _hourly().sample(frac=1, random_state=8), _temporal_config()
        ).contrasts,
    )


def test_acf_counts_exact_calendar_lags_not_adjacent_compressed_rows() -> None:
    result = calculate_temporal_controls(_hourly(), _temporal_config()).autocorrelation
    for site in ("Sukiyabashi", "Shinjuku"):
        starts = pd.to_datetime(
            _hourly().loc[_hourly().site.eq(site), "monitor_start_utc"], utc=True
        )
        for lag in (1, 24, 48, 200):
            expected = sum((value + pd.Timedelta(hours=lag)) in set(starts) for value in starts)
            actual = result.loc[result.site.eq(site) & result.lag_hours.eq(lag)]
            assert actual.exact_calendar_pairs.eq(expected).all()
    assert result.loc[result.lag_hours.eq(200), "pearson_autocorrelation"].isna().all()


@pytest.mark.parametrize(
    "change", ["derived", "excluded", "interval", "duplicate", "outside", "negative"]
)
def test_temporal_rejects_nonactual_or_invalid_hourly_inputs(change: str) -> None:
    rows = _hourly()
    if change == "derived":
        rows.loc[0, "target_observation_status"] = "interpolated"
    elif change == "excluded":
        rows.loc[0, "primary_measurement_eligible"] = False
    elif change == "interval":
        rows.loc[0, "monitor_end_utc"] = rows.loc[0, "monitor_start_utc"]
    elif change == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]])
    elif change == "outside":
        with pytest.raises(ValueError, match="outside"):
            calculate_temporal_controls(
                rows, TemporalControlConfig("2025-06-02", "2025-06-06", (1,), (24,))
            )
        return
    else:
        rows.loc[0, "count_only_index"] = -1
    with pytest.raises(ValueError):
        calculate_temporal_controls(rows, _temporal_config())


def _cli_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repository"
    root.mkdir()
    actual_root = Path(__file__).resolve().parents[1]
    for name in CODE_PATHS:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual_root / name, root / name)
    # The fixture is a real repo because receipts record the actual revision.
    import subprocess

    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    source_specs = {}
    for name, rows in (("profiles", _profiles()), ("hourly", _hourly())):
        path = root / f"{name}.csv"
        rows.to_csv(path, index=False)
        source_specs[name] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    config = {
        "schema_version": 1,
        "analysis_scope": "locked_retrospective_two_site",
        "status": "optional_diagnostics_only",
        "predeclaration_id": "controlled_fixture_not_empirical_evidence",
        "speed": {
            "profiles": source_specs["profiles"],
            "scenarios": [
                {"scenario_id": "zero", "multiplier": 1.0, "additive_mps": 0.0},
                {"scenario_id": "double", "multiplier": 2.0, "additive_mps": 0.0},
            ],
        },
        "temporal": {
            "hourly_rows": source_specs["hourly"],
            "calendar_start_date": "2025-06-01",
            "calendar_end_date": "2025-06-05",
            "day_offsets": [1, -1],
            "acf_lags_hours": [1, 24, 48],
        },
    }
    path = root / "robustness.json"
    path.write_text(json.dumps(config))
    return root, path


def test_cli_deterministic_fresh_hash_bound_read_only_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, config = _cli_fixture(tmp_path)
    output = root / "build" / "first"
    assert (
        main(
            [
                "execute",
                "--config",
                str(config),
                "--output-directory",
                str(output),
                "--repository-root",
                str(root),
            ]
        )
        == 0
    )
    capsys.readouterr()
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    assert replay_robustness(output, root)["status"] == "replay_verified_read_only"
    assert before == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()
    }
    second = root / "build" / "second"
    run_robustness(config, second, root)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    with pytest.raises(ValueError, match="already exists"):
        run_robustness(config, output, root)
    with pytest.raises(ValueError, match="fresh subdirectory"):
        run_robustness(config, root / "results" / "forbidden", root)


@pytest.mark.parametrize(
    "target",
    [
        "profiles.csv",
        "robustness.json",
        "build/first/speed_sensitivity.csv",
        "src/vehicle_emission/emissions/EmsRate_1.csv",
    ],
)
def test_replay_rejects_source_config_artifact_and_coefficient_tampering(
    tmp_path: Path, target: str
) -> None:
    root, config = _cli_fixture(tmp_path)
    output = root / "build" / "first"
    run_robustness(config, output, root)
    with (root / target).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError):
        replay_robustness(output, root)


def test_absent_optional_inputs_recorded_without_invented_outputs(tmp_path: Path) -> None:
    root, config = _cli_fixture(tmp_path)
    settings = json.loads(config.read_text())
    settings.update(speed=None, temporal=None)
    config.write_text(json.dumps(settings))
    receipt = run_robustness(config, root / "build" / "absent", root)
    assert receipt["diagnostics"] == {"speed": "not_requested", "temporal": "not_requested"}
    output_hashes = receipt["output_sha256"]
    assert isinstance(output_hashes, dict)
    assert set(output_hashes) == {"config.json"}
    assert replay_robustness(root / "build" / "absent", root)["artifacts_verified"] == 1


def test_all_missing_profile_cli_retains_empty_run_schema_and_replays(tmp_path: Path) -> None:
    root, config = _cli_fixture(tmp_path)
    profiles = _profiles()
    profiles["speed_mps"] = np.nan
    source = root / "profiles.csv"
    profiles.to_csv(source, index=False)
    settings = json.loads(config.read_text())
    settings["speed"]["profiles"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    settings["temporal"] = None
    config.write_text(json.dumps(settings))
    output = root / "build" / "missing"
    run_robustness(config, output, root)
    runs = pd.read_csv(output / "speed_runs.csv")
    assert runs.empty and {"nox_g", "speed_mps", "run_seconds"}.issubset(runs.columns)
    summary = pd.read_csv(output / "speed_sensitivity.csv")
    assert summary.status.eq("no_valid_seconds").all()
    assert summary.baseline_nox_g.isna().all()
    assert replay_robustness(output, root)["status"] == "replay_verified_read_only"


@pytest.mark.parametrize("identifier", ["001", "NA"])
@pytest.mark.parametrize("missing_speed", ["", "NA"])
def test_csv_profile_ids_roundtrip_without_losing_numeric_speed_missingness(
    tmp_path: Path,
    identifier: str,
    missing_speed: str,
) -> None:
    root, config = _cli_fixture(tmp_path)
    profiles = _profiles()
    profiles["source_id"] = identifier
    profiles["profile_id"] = identifier
    source = root / "profiles.csv"
    profiles.to_csv(source, index=False, na_rep=missing_speed)
    settings = json.loads(config.read_text())
    spec = settings["speed"]["profiles"]
    spec["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    config.write_text(json.dumps(settings))
    parsed, _ = runner._load_source(spec, root)
    assert parsed.source_id.eq(identifier).all()
    assert parsed.profile_id.eq(identifier).all()
    assert parsed.speed_mps.isna().sum() == 1
    output = root / "build" / "literal-identities"
    run_robustness(config, output, root)
    support = pd.read_csv(
        output / "speed_profiles.csv",
        keep_default_na=False,
        dtype={"source_id": str, "profile_id": str},
    )
    assert support.source_id.tolist() == [identifier]
    assert support.profile_id.tolist() == [identifier]
    assert support.valid_runs.tolist() == [2]
    assert support.valid_seconds.tolist() == [8]
    assert replay_robustness(output, root)["status"] == "replay_verified_read_only"


def test_csv_temporal_identity_date_and_status_fields_remain_literal_strings(
    tmp_path: Path,
) -> None:
    root, _ = _cli_fixture(tmp_path)
    source = root / "literal-strings.csv"
    fields = {name: ["001", "NA"] for name in runner.TEXT_SOURCE_COLUMNS}
    pd.DataFrame(fields).to_csv(source, index=False)
    parsed, _ = runner._load_source(
        {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}, root
    )
    for name in runner.TEXT_SOURCE_COLUMNS:
        assert parsed[name].tolist() == ["001", "NA"]


@pytest.mark.parametrize(
    "target", ["profiles.csv", "robustness.json", "src/vehicle_emission/emissions/EmsRate_1.csv"]
)
def test_changes_during_calculation_abort_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    root, config = _cli_fixture(tmp_path)
    output = root / "build" / "interrupted"
    calculate = runner._calculate

    def mutate_after_calculation(settings, repository):
        results = calculate(settings, repository)
        with (root / target).open("ab") as handle:
            handle.write(b"\n")
        return results

    monkeypatch.setattr(runner, "_calculate", mutate_after_calculation)
    with pytest.raises(ValueError):
        run_robustness(config, output, root)
    assert not output.exists()
    assert not list(output.parent.glob(".robustness-*"))


def test_stage_write_failure_leaves_no_partial_published_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _cli_fixture(tmp_path)
    output = root / "build" / "interrupted"
    write = Path.write_bytes

    def fail_receipt(path, data):
        if path.name == "receipt.json":
            raise OSError("controlled fixture write failure")
        return write(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_receipt)
    with pytest.raises(OSError, match="controlled fixture"):
        run_robustness(config, output, root)
    assert not output.exists()
    assert not list(output.parent.glob(".robustness-*"))


def test_copied_repository_cannot_claim_different_calculator_bytes(tmp_path: Path) -> None:
    root, config = _cli_fixture(tmp_path)
    coefficient = root / "src/vehicle_emission/emissions/EmsRate_1.csv"
    with coefficient.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="imported calculator"):
        run_robustness(config, root / "build" / "different", root)


def test_locally_deleted_committed_target_is_not_a_fresh_path(tmp_path: Path) -> None:
    import subprocess

    root, config = _cli_fixture(tmp_path)
    output = root / "build" / "historical"
    output.mkdir(parents=True)
    historical = output / "preserved.csv"
    historical.write_bytes(b"fixture historical output\n")
    subprocess.run(["git", "add", "build/historical/preserved.csv"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture historical output",
        ],
        cwd=root,
        check=True,
    )
    historical.unlink()
    output.rmdir()
    with pytest.raises(ValueError, match="committed artifacts"):
        run_robustness(config, output, root)
    assert not output.exists()
