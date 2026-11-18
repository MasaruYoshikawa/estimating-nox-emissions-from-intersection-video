"""Regression checks for computed-row plotting and immutable artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import vehicle_emission.inference.generate_v3_figures as figure_module
from vehicle_emission.inference.generate_v3_figures import (
    EVIDENCE_STATUS,
    FIGURE_NAMES,
    RECEIPT_NAME,
    SCOPE,
    FigureInputError,
    build_figures,
    generate_figures,
    load_figure_inputs,
    main,
)
from vehicle_emission.inference.paired_inference import (
    PairedInferenceConfig,
    compute_paired_inference,
    write_inference_artifacts,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records() -> list[dict[str, str]]:
    records = []
    for site, index in (("Sukiyabashi", 0.2), ("Shinjuku", 0.25)):
        for estimand, estimate in (
            ("rq1_index", index),
            ("rq1_count", 0.1),
            ("rq3_paired_advantage", index - 0.1),
            ("rq2_wind_contrast", -0.3),
        ):
            records.append(
                {
                    "site": site,
                    "estimand": estimand,
                    "estimate": str(estimate),
                    "ci_low": str(estimate - 0.1),
                    "ci_high": str(estimate + 0.1),
                    "n_hours": "12",
                    "n_days": "3",
                    "resamples_requested": "100",
                    "resamples_valid": "95",
                    "interval_method": "union_calendar_moving_block_percentile",
                    "interval_adjustment": "bonferroni",
                    "interval_family": "primary",
                    "confidence_level": "0.95",
                    "family_size": "4",
                    "classification": "inconclusive",
                    "analysis_scope": SCOPE,
                    "evidence_status": EVIDENCE_STATUS,
                    "observation_sha256": ("a" if site == "Sukiyabashi" else "b") * 64,
                    "resample_sha256": "c" * 64,
                    "failure_reason": "",
                }
            )
    cross = records[0].copy()
    cross.update(
        site="cross_site",
        estimand="cross_site_index_heterogeneity",
        estimate="0.05",
        ci_low="-0.1",
        ci_high="0.2",
        classification="descriptive",
    )
    records.append(cross)
    return records


def _write_inputs(tmp_path: Path, records: list[dict[str, str]] | None = None) -> tuple[Path, Path]:
    records = _records() if records is None else records
    csv_path = tmp_path / "computed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    config = {"seed": 42, "block_days": 2}
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    inputs = [{"path": "synthetic_hourly.csv", "sha256": "d" * 64}]
    input_bytes = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    receipt = {
        "schema_version": 1,
        "artifact_kind": "paired_inference_component",
        "results_sha256": _digest(csv_path),
        "row_count": len(records),
        "analysis_scope": SCOPE,
        "evidence_status": EVIDENCE_STATUS,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "inputs": inputs,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": config,
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return csv_path, receipt_path


def test_changed_rows_change_plotted_points_and_support(tmp_path: Path) -> None:
    csv_path, receipt = _write_inputs(tmp_path)
    figures = build_figures(load_figure_inputs(csv_path, receipt))
    original = figures[FIGURE_NAMES[1]].axes[0].lines[1].get_xdata()[0]
    assert original == pytest.approx(0.1)
    assert "12 hours; 3 days" in figures[FIGURE_NAMES[1]].axes[0].get_yticklabels()[0].get_text()
    assert "provisional analysis components" in figures[FIGURE_NAMES[1]].axes[0].get_title()
    assert "98.75% Bonferroni interval" in figures[FIGURE_NAMES[1]].axes[0].texts[0].get_text()
    for figure in figures.values():
        plt.close(figure)

    records = _records()
    records[0]["estimate"] = "-0.25"
    records[0].update(ci_low="-0.35", ci_high="-0.15")
    records[2].update(estimate="-0.35", ci_low="-0.45", ci_high="-0.25")
    records[8].update(estimate="0.5", ci_low="0.3", ci_high="0.7")
    csv_path, receipt = _write_inputs(tmp_path, records)
    figures = build_figures(load_figure_inputs(csv_path, receipt))
    assert figures[FIGURE_NAMES[1]].axes[0].lines[1].get_xdata()[0] == pytest.approx(-0.35)
    assert figures[FIGURE_NAMES[3]].axes[0].lines[1].get_xdata()[0] == pytest.approx(0.5)
    assert "Favourable minus opposing" in figures[FIGURE_NAMES[2]].axes[0].get_xlabel()
    for figure in figures.values():
        plt.close(figure)


def test_not_estimable_omits_points_and_intervals(tmp_path: Path) -> None:
    records = _records()
    for record in (records[3], records[7]):
        record.update(
            classification="not_estimable",
            failure_reason="insufficient_wind_days",
            ci_low="",
            ci_high="",
            resamples_valid="0",
        )
    csv_path, receipt = _write_inputs(tmp_path, records)
    figures = build_figures(load_figure_inputs(csv_path, receipt))
    axis = figures[FIGURE_NAMES[2]].axes[0]
    assert len(axis.lines) == 1  # Zero reference only.
    assert all("Not estimable" in text.get_text() for text in axis.texts)
    for figure in figures.values():
        plt.close(figure)


@pytest.mark.parametrize("field", ["results_sha256", "input_sha256", "config_sha256"])
@pytest.mark.parametrize("replacement", [None, "not-a-hash"])
def test_missing_or_malformed_receipt_hash_is_rejected(
    tmp_path: Path, field: str, replacement: str | None
) -> None:
    csv_path, receipt_path = _write_inputs(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = replacement
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(FigureInputError, match="SHA-256"):
        load_figure_inputs(csv_path, receipt_path)


def test_changed_results_or_config_fail_identity_check(tmp_path: Path) -> None:
    csv_path, receipt = _write_inputs(tmp_path)
    csv_path.write_text(csv_path.read_text() + "\n")
    with pytest.raises(FigureInputError, match="result bytes"):
        load_figure_inputs(csv_path, receipt)
    csv_path, receipt_path = _write_inputs(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["config"]["seed"] = 99
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(FigureInputError, match="config_sha256"):
        load_figure_inputs(csv_path, receipt_path)


@pytest.mark.parametrize(
    ("row", "field", "value", "message"),
    [
        (0, "n_hours", "0", "n_days cannot exceed"),
        (0, "n_days", "0", "require support"),
        (2, "n_hours", "13", "identical paired support"),
        (2, "observation_sha256", "e" * 64, "identical paired support"),
        (2, "resample_sha256", "e" * 64, "shared resampling"),
        (2, "estimate", "0.2", "index minus count"),
        (8, "estimate", "-0.05", "Shinjuku minus Sukiyabashi"),
        (0, "ci_low", "0.9", "bounds are reversed"),
        (0, "ci_high", "inf", "must be finite"),
        (0, "evidence_status", "verified", "provisional retrospective"),
        (0, "classification", "confirmed", "classified as confirmed"),
    ],
)
def test_invalid_rows_are_rejected(
    tmp_path: Path, row: int, field: str, value: str, message: str
) -> None:
    records = _records()
    records[row][field] = value
    csv_path, receipt = _write_inputs(tmp_path, records)
    with pytest.raises(FigureInputError, match=message):
        load_figure_inputs(csv_path, receipt)


def test_missing_and_duplicate_rq_rows_fail(tmp_path: Path) -> None:
    records = _records()
    csv_path, receipt = _write_inputs(tmp_path, records[:-1])
    with pytest.raises(FigureInputError, match="all nine"):
        load_figure_inputs(csv_path, receipt)
    records[-1] = records[0].copy()
    csv_path, receipt = _write_inputs(tmp_path, records)
    with pytest.raises(FigureInputError, match="duplicate result"):
        load_figure_inputs(csv_path, receipt)


def test_no_legacy_fallback_or_output_creation_for_missing_input(tmp_path: Path) -> None:
    (tmp_path / "phase5_rq1_two_site_v2.csv").write_text("historical values")
    output = tmp_path / "new_figures"
    with pytest.raises(FigureInputError, match="explicit results"):
        generate_figures(tmp_path / "absent.csv", tmp_path / "absent.json", output)
    assert not output.exists()


def test_existing_artifacts_are_preserved(tmp_path: Path) -> None:
    csv_path, receipt = _write_inputs(tmp_path)
    output = tmp_path / "figures"
    output.mkdir()
    historical = output / "fig_phase7_rq3_baseline_ablation.png"
    historical.write_bytes(b"committed v3 artifact")
    before = _digest(historical)
    with pytest.raises(FigureInputError, match="existing artifacts"):
        generate_figures(csv_path, receipt, output)
    assert _digest(historical) == before
    assert list(output.iterdir()) == [historical]


def test_tracked_destination_is_rejected_even_if_missing(tmp_path: Path) -> None:
    csv_path, receipt = _write_inputs(tmp_path)
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    output = repository / "build" / "tracked_figures"
    output.mkdir(parents=True)
    tracked = output / FIGURE_NAMES[0]
    tracked.write_bytes(b"tracked")
    subprocess.run(["git", "add", "build"], cwd=repository, check=True)
    tracked.unlink()
    output.rmdir()
    with pytest.raises(FigureInputError, match="tracked artifacts"):
        generate_figures(csv_path, receipt, output)


def test_new_figures_have_output_hashes_and_cannot_be_replaced(tmp_path: Path) -> None:
    csv_path, receipt = _write_inputs(tmp_path)
    output = tmp_path / "new_figures"
    paths = generate_figures(csv_path, receipt, output)
    assert {path.name for path in paths} == set(FIGURE_NAMES)
    figure_receipt = json.loads((output / RECEIPT_NAME).read_text())
    assert figure_receipt["results_sha256"] == _digest(csv_path)
    assert figure_receipt["analysis_receipt_sha256"] == _digest(receipt)
    assert figure_receipt["evidence_status"] == EVIDENCE_STATUS
    for record in figure_receipt["artifacts"]:
        assert record["sha256"] == _digest(output / record["path"])
    before = {path.name: _digest(path) for path in paths}
    with pytest.raises(FigureInputError, match="existing artifacts"):
        generate_figures(csv_path, receipt, output)
    assert {path.name: _digest(path) for path in paths} == before


def test_cli_requires_explicit_input_receipt_and_output() -> None:
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_input_change_during_plotting_fails_before_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, receipt = _write_inputs(tmp_path)
    original = build_figures

    def changed_input(rows: tuple[figure_module.FigureRow, ...]) -> dict[str, plt.Figure]:
        figures = original(rows)
        csv_path.write_text(csv_path.read_text() + "\n")
        return figures

    monkeypatch.setattr(figure_module, "build_figures", changed_input)
    output = tmp_path / "new_figures"
    with pytest.raises(FigureInputError, match="bytes changed"):
        generate_figures(csv_path, receipt, output)
    assert not output.exists()


def test_actual_component_writer_feeds_figures(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    config = PairedInferenceConfig(
        resampling_method="union_calendar_moving_block",
        block_length_days=2,
        minimum_site_hours=3,
        minimum_site_days=2,
        minimum_stratum_hours=3,
        minimum_stratum_days=2,
        resamples=49,
        minimum_valid_resamples=19,
        seed=83,
        interval_method="percentile",
        interval_adjustment="bonferroni",
        confidence_level=0.95,
        primary_interval_family="two_site_rq1_and_rq3",
        primary_family_size=4,
        wind_interval_family="two_site_conditional_rq2",
        wind_family_size=2,
        wind_mode="unavailable",
        wind_definition_id=None,
        favourable_min_mps=None,
        opposing_max_mps=None,
    )
    records = []
    for site in ("Sukiyabashi", "Shinjuku"):
        for day in range(6):
            for hour in range(2):
                start = pd.Timestamp("2025-06-01T00:00:00Z") + pd.Timedelta(days=day, hours=hour)
                records.append(
                    {
                        "site": site,
                        "monitor_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "monitor_end_utc": (start + pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "local_calendar_date": start.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d"),
                        "ambient_nox_ppm": ((day * 7 + hour * 3) % 11) / 1000,
                        "relative_nox_emission_index": day * 2 + hour + 1,
                        "count_only_index": (day + hour) % 5,
                        "primary_measurement_eligible": True,
                        "target_observation_status": "actual_observation",
                    }
                )
    frame = pd.DataFrame(records)
    input_path = tmp_path / "synthetic_hourly.csv"
    frame.to_csv(input_path, index=False)
    results = compute_paired_inference(frame, config)
    analysis_dir = tmp_path / "build" / "inference"
    receipt = write_inference_artifacts(
        results,
        config,
        inputs=[{"path": input_path.name, "sha256": _digest(input_path)}],
        output_directory=analysis_dir,
        repository_root=tmp_path,
        code_revision="synthetic_test",
    )
    rows = load_figure_inputs(analysis_dir / "paired_inference.csv", receipt)
    for row in rows:
        computed = results.loc[
            (results.site == row.site) & (results.estimand == row.estimand)
        ].iloc[0]
        if row.classification != "not_estimable":
            assert row.estimate == pytest.approx(computed.estimate)
    assert sum(row.classification == "not_estimable" for row in rows) == 2
    paths = generate_figures(
        analysis_dir / "paired_inference.csv", receipt, tmp_path / "build" / "plots"
    )
    assert len(paths) == 4
