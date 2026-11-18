"""Unit and validation tests for VTCPFMModel."""

import numpy as np
import pytest

from vehicle_emission.baselines.vtcpfm import VTCPFMModel
from vehicle_emission.emissions.movestar import MOVEStarModel


def test_vtcpfm_calibration() -> None:
    """Verify that VTCPFMModel calibrates its coefficients correctly during init."""
    model = VTCPFMModel()

    # All vehicle types must be present in the calibrated dictionaries
    assert set(model.coefficients.keys()) == {1, 2, 6, 13}
    assert set(model.idle_rates.keys()) == {1, 2, 6, 13}

    # Verify coefficients and idle rates shape
    for vt in [1, 2, 6, 13]:
        for col_idx in range(1, 8):
            coeffs = model.coefficients[vt][col_idx]
            assert isinstance(coeffs, np.ndarray)
            assert len(coeffs) == 3  # Quadratic fit [a2, a1, a0]
            assert isinstance(model.idle_rates[vt][col_idx], float)


def test_vtcpfm_estimate_emissions() -> None:
    """Verify emissions estimation structure and constraint handling."""
    model = VTCPFMModel()
    speed_mps = [0.0, 5.0, 10.0, 12.0, 10.0, 5.0, 0.0]

    # Verify normal execution
    res = model.estimate_emissions(veh_type=1, speed_mps=speed_mps)
    assert "Emission Rate" in res
    assert "Emission Factor" in res

    rates = res["Emission Rate"][0]
    factors = res["Emission Factor"][0]

    assert len(rates) == 9  # CO, HC, NOx, PM2.5_Ele, PM2.5_Org, Energy, CO2, Fuel, TT
    assert len(factors) == 9

    # NOx (index 2) and Energy (index 5) must be positive and non-zero
    assert rates[2] > 0.0
    assert rates[5] > 0.0

    # HC (index 1), PM2.5_Org (index 4), CO2 (index 6), and Fuel (index 7) are non-negative
    assert rates[1] >= 0.0
    assert rates[4] >= 0.0
    assert rates[6] >= 0.0
    assert rates[7] >= 0.0

    # For heavy vehicles (Transit Bus, type 6), they should be strictly 0.0
    # due to the MOVES SQL query structure
    bus_rates = model.estimate_emissions(veh_type=6, speed_mps=speed_mps)["Emission Rate"][0]
    assert bus_rates[1] == 0.0
    assert bus_rates[4] == 0.0
    assert bus_rates[6] == 0.0
    assert bus_rates[7] == 0.0

    # Duration should match speed profile length
    assert rates[8] == float(len(speed_mps))

    # Corner cases
    empty_res = model.estimate_emissions(veh_type=1, speed_mps=[])
    assert empty_res == {}

    zero_speed_res = model.estimate_emissions(veh_type=1, speed_mps=[0.0, 0.0, 0.0])
    assert "Emission Rate" in zero_speed_res
    assert "Emission Factor" not in zero_speed_res  # Cannot divide by zero distance


def test_vtcpfm_class_mapping() -> None:
    """Verify project category ID mapping matches expectations."""
    model = VTCPFMModel()
    speed_mps = [3.0, 4.0, 5.0]

    # 0 (Bus) -> veh_type = 6
    assert model.estimate_class_emissions(
        class_id=0, speed_mps=speed_mps
    ) == model.estimate_emissions(veh_type=6, speed_mps=speed_mps)

    # 1 (Car) -> veh_type = 1
    assert model.estimate_class_emissions(
        class_id=1, speed_mps=speed_mps
    ) == model.estimate_emissions(veh_type=1, speed_mps=speed_mps)

    # 2 (Heavy Truck) -> veh_type = 13
    assert model.estimate_class_emissions(
        class_id=2, speed_mps=speed_mps
    ) == model.estimate_emissions(veh_type=13, speed_mps=speed_mps)

    # 3 (Light Truck) -> veh_type = 2
    assert model.estimate_class_emissions(
        class_id=3, speed_mps=speed_mps
    ) == model.estimate_emissions(veh_type=2, speed_mps=speed_mps)

    with pytest.raises(ValueError):
        model.estimate_class_emissions(class_id=99, speed_mps=speed_mps)


def test_vtcpfm_vs_movestar() -> None:
    """Compare VT-CPFM outputs directly against MOVEStar.

    Ensure that they track close to each other but vary due to the continuous
    fit vs. discrete operating mode binning.
    """
    movestar = MOVEStarModel()
    vtcpfm = VTCPFMModel()

    speed_mps = [0.0, 2.0, 5.0, 8.0, 12.0, 15.0, 10.0, 4.0, 0.0]

    # Run for Passenger Car (veh_type = 1)
    ms_res = movestar.estimate_emissions(veh_type=1, speed_mps=speed_mps)
    cp_res = vtcpfm.estimate_emissions(veh_type=1, speed_mps=speed_mps)

    ms_rates = ms_res["Emission Rate"][0]
    cp_rates = cp_res["Emission Rate"][0]

    # Compare NOx (index 2) and Energy (index 5)
    # They should be comparable in scale (within 75% due to continuous approximation)
    assert pytest.approx(ms_rates[2], rel=0.75) == cp_rates[2]
    assert pytest.approx(ms_rates[5], rel=0.75) == cp_rates[5]
