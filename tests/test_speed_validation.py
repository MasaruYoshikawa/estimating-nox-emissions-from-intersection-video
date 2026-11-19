from __future__ import annotations

from pathlib import Path

import pytest

from vehicle_emission.speed.evaluate import evaluate_speed, main


def test_speed_validation_rejects_preentered_summary(tmp_path: Path) -> None:
    output = tmp_path / "phase3_speed_validation_v2.csv"
    with pytest.raises(ValueError, match="Pre-entered speed summary"):
        evaluate_speed(
            calibration_path=Path("tests/fixtures/legacy/site_calibration_v2.json"),
            manifest_path=Path("tests/fixtures/legacy/speed_validation_v2.csv"),
            output_path=output,
        )
    assert not output.exists()
    assert not list(tmp_path.iterdir())


def test_evaluate_speed_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="calibration file not found"):
        evaluate_speed(
            calibration_path=tmp_path / "absent.json",
            manifest_path=Path("tests/fixtures/legacy/speed_validation_v2.csv"),
            output_path=tmp_path / "out.csv",
        )


def test_legacy_cli_flags_cannot_copy_summary(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--calibration",
                "tests/fixtures/legacy/site_calibration_v2.json",
                "--manifest",
                "tests/fixtures/legacy/speed_validation_v2.csv",
                "--output",
                str(tmp_path / "out.csv"),
            ]
        )
    assert not list(tmp_path.iterdir())
