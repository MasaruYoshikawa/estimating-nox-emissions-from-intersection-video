"""Smoke test so `pytest` collects at least one test at bootstrap time."""

from vehicle_emission import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"
