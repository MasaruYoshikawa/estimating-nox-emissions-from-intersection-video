from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_fixed_measurement_writer_fails_without_creating_results(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scratch/generate_measurement_validation_v3.py"
    completed = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert completed.returncode != 0
    assert "writer is retired" in completed.stderr
    assert "independent reference observations" in completed.stderr
    assert list(tmp_path.iterdir()) == []
