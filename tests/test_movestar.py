"""Unit and replication tests for MOVEStarModel."""

import os

import pandas as pd
import pytest

from vehicle_emission.emissions.movestar import BASE_DIR, MOVEStarModel


def test_movestar_replication() -> None:
    """Verify that our MOVEStar implementation replicates the MATLAB reference values.

    The test uses `test.csv` speed profile for Passenger Car (veh_type = 1).
    """
    model = MOVEStarModel()

    # Load test.csv from emissions directory
    csv_path = os.path.join(BASE_DIR, "test.csv")
    df = pd.read_csv(csv_path)

    # In test.csv, speed is in mph or m/s?
    # Let's check: in the original movestar.py, it reads test.csv and has:
    # "Read the data % Unit: speed -- m/s"
    # So speed is in m/s.
    speed_mps = df["Speed"].values

    # Run the model for Passenger Car (veh_type = 1)
    res = model.estimate_emissions(veh_type=1, speed_mps=speed_mps)

    assert "Emission Rate" in res
    rates = res["Emission Rate"][0]

    # Reference values from MATLAB output for test.csv with vehType = 1:
    # CO: 0.551277
    # HC: 0.00685283
    # NOx: 0.0325122
    # PM2.5_Ele: 0.000262
    # PM2.5_Org: 0.00120169
    # Energy: 2544.57
    # CO2: 181.041
    # Fuel: 56.6988
    # TT: 30

    expected = {
        "CO": 0.551277,
        "HC": 0.00685283,
        "NOx": 0.0325122,
        "PM2.5_Ele": 0.000262,
        "PM2.5_Org": 0.00120169,
        "Energy": 2544.57,
        "CO2": 181.041,
        "Fuel": 56.6988,
        "TT": 30.0,
    }

    # Assert all values are within 0.1% of expected MATLAB values
    assert pytest.approx(rates[0], rel=1e-3) == expected["CO"]
    assert pytest.approx(rates[1], rel=1e-3) == expected["HC"]
    assert pytest.approx(rates[2], rel=1e-3) == expected["NOx"]
    assert pytest.approx(rates[3], rel=1e-3) == expected["PM2.5_Ele"]
    assert pytest.approx(rates[4], rel=1e-3) == expected["PM2.5_Org"]
    assert pytest.approx(rates[5], rel=1e-3) == expected["Energy"]
    assert pytest.approx(rates[6], rel=1e-3) == expected["CO2"]
    assert pytest.approx(rates[7], rel=1e-3) == expected["Fuel"]
    assert pytest.approx(rates[8], rel=1e-3) == expected["TT"]


def test_invalid_vehicle_type() -> None:
    """Verify that ValueError is raised on invalid vehicle type."""
    model = MOVEStarModel()
    with pytest.raises(ValueError, match="Vehicle type must be 1"):
        model.estimate_emissions(veh_type=3, speed_mps=[0.0, 1.0])


def test_empty_trajectory() -> None:
    """Verify empty trajectory returns empty dict."""
    model = MOVEStarModel()
    assert model.estimate_emissions(veh_type=1, speed_mps=[]) == {}


def test_all_zeros_trajectory() -> None:
    """Verify all zeros trajectory does not produce Emission Factor."""
    model = MOVEStarModel()
    res = model.estimate_emissions(veh_type=1, speed_mps=[0.0, 0.0, 0.0])
    assert "Emission Rate" in res
    assert "Emission Factor" not in res


def test_estimate_class_emissions() -> None:
    """Verify that project class IDs map directly to correct native vehicle types."""
    model = MOVEStarModel()

    speed_mps = [10.0, 12.0, 11.0]

    # Map project class ID to native MOVEStar veh_type
    # - 0: bus -> 6 (Transit Bus)
    # - 1: car -> 1 (Passenger Car)
    # - 2: heavy truck -> 13 (Combination Truck)
    # - 3: light truck -> 2 (Passenger Truck)

    bus_res = model.estimate_class_emissions(class_id=0, speed_mps=speed_mps)
    car_res = model.estimate_class_emissions(class_id=1, speed_mps=speed_mps)
    h_truck_res = model.estimate_class_emissions(class_id=2, speed_mps=speed_mps)
    l_truck_res = model.estimate_class_emissions(class_id=3, speed_mps=speed_mps)

    assert bus_res == model.estimate_emissions(veh_type=6, speed_mps=speed_mps)
    assert car_res == model.estimate_emissions(veh_type=1, speed_mps=speed_mps)
    assert h_truck_res == model.estimate_emissions(veh_type=13, speed_mps=speed_mps)
    assert l_truck_res == model.estimate_emissions(veh_type=2, speed_mps=speed_mps)


def test_heavy_vehicles() -> None:
    """Verify that Transit Bus (veh_type = 6) and Heavy Truck (veh_type = 13)
    execute successfully.
    """
    model = MOVEStarModel()
    speed_mps = [5.0, 6.0, 7.0, 8.0, 7.0, 6.0, 5.0]

    bus_res = model.estimate_emissions(veh_type=6, speed_mps=speed_mps)
    truck_res = model.estimate_emissions(veh_type=13, speed_mps=speed_mps)

    assert "Emission Rate" in bus_res
    assert "Emission Rate" in truck_res

    # Check that NOx (index 2) is calculated and non-zero
    assert bus_res["Emission Rate"][0][2] > 0.0
    assert truck_res["Emission Rate"][0][2] > 0.0
