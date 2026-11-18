"""MOVEStar vehicle fuel and emission model based on USEPA MOVES.

Reference:
    ziranw/MOVESTAR-Fuel-and-Emission-Model (v1.4)
    Wang, Z., Wu, G., and Scora, G. (2020)
"""

import os
from typing import Any

import numpy as np
import pandas as pd

from vehicle_emission.vehicle_classes import emission_vehicle_type_for_project_class

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MOVEStarModel:
    """MOVEStar emissions model implementation.

    Provides calculation of instantaneous/total criteria pollutants and fuel consumption
    for passenger cars and light-duty trucks using second-by-second speed profiles.
    """

    def __init__(self) -> None:
        """Initialise the model and load coefficient and emission rate files."""
        self.coeff_path = os.path.join(BASE_DIR, "VehicleSrcCoeff.csv")
        self.df_coeff = pd.read_csv(self.coeff_path)

        # Cache the emission rate tables
        self.ems_tables: dict[int, pd.DataFrame] = {}
        for vt in [1, 2, 6, 13]:
            ems_path = os.path.join(BASE_DIR, f"EmsRate_{vt}.csv")
            if os.path.exists(ems_path):
                self.ems_tables[vt] = pd.read_csv(ems_path, header=None)

    def calculate_acceleration(self, speed_mps: np.ndarray) -> np.ndarray:
        """Calculate acceleration from speed using central difference.

        Args:
            speed_mps: Second-by-second speed profile in m/s.

        Returns:
            Acceleration profile in m/s^2.
        """
        n = len(speed_mps)
        acc = np.zeros(n)
        if n < 2:
            return acc

        for i in range(1, n - 1):
            acc[i] = (speed_mps[i + 1] - speed_mps[i - 1]) / 2.0
        return acc

    def calculate_vsp(
        self, veh_type: int, speed_mps: np.ndarray, acc_mps2: np.ndarray
    ) -> np.ndarray:
        """Calculate Vehicle Specific Power (VSP) in kW/tonne.

        Args:
            veh_type: 1 for Passenger Car, 2 for Passenger Truck.
            speed_mps: Second-by-second speed profile in m/s.
            acc_mps2: Second-by-second acceleration profile in m/s^2.

        Returns:
            VSP profile in kW/tonne.
        """
        # Correct the off-by-one mapping bug in the UCR Python script:
        # veh_type = 1 (Passenger Car) -> VehicleType = 2 in VehicleSrcCoeff.csv
        # veh_type = 2 (Passenger Truck) -> VehicleType = 3 in VehicleSrcCoeff.csv
        # veh_type = 6 (Transit Bus) -> VehicleType = 6 in VehicleSrcCoeff.csv
        # veh_type = 13 (Combination Long-haul Truck) -> VehicleType = 13 in VehicleSrcCoeff.csv
        if veh_type == 1:
            csv_veh_type = 2
        elif veh_type == 2:
            csv_veh_type = 3
        else:
            csv_veh_type = veh_type

        row = self.df_coeff[self.df_coeff["VehicleType"] == csv_veh_type]
        if row.empty:
            raise ValueError(f"Invalid vehicle type mapping for type: {veh_type}")

        a_coeff = row["A"].values[0]
        b_coeff = row["B"].values[0]
        c_coeff = row["C"].values[0]
        m_coeff = row["M"].values[0]
        f_coeff = row["f"].values[0]

        vsp = (
            a_coeff * speed_mps
            + b_coeff * (speed_mps**2)
            + c_coeff * (speed_mps**3)
            + m_coeff * speed_mps * acc_mps2
        )
        return vsp / f_coeff

    def calculate_operating_modes(
        self, speed_mps: np.ndarray, acc_mps2: np.ndarray, vsp: np.ndarray
    ) -> np.ndarray:
        """Assign each second to one of the 23 MOVES operating modes.

        Args:
            speed_mps: Second-by-second speed profile in m/s.
            acc_mps2: Second-by-second acceleration profile in m/s^2.
            vsp: Second-by-second VSP profile in kW/tonne.

        Returns:
            An array of size 23 containing the frequency of each operating mode.
        """
        c1 = 2.23693629  # m/s to mph conversion factor
        size_bin = np.zeros(23)

        if len(vsp) == 0:
            return size_bin

        max_vsp = np.max(vsp) + 1.0

        for i in range(len(speed_mps)):
            sc1 = speed_mps[i] * c1
            bin_selector = 0

            # Acceleration check for braking/deceleration bin
            if (
                acc_mps2[i] * c1 <= -2.0
                or i >= 3
                and acc_mps2[i - 2] * c1 < -1.0
                and acc_mps2[i - 1] * c1 < -1.0
            ):
                bin_selector = 1
            # Idle check
            elif sc1 < 1.0:
                bin_selector = 2
            # Low, medium, high speed range bin assignments
            else:
                if sc1 < 25.0:
                    vsp_map = [0.0, 3.0, 6.0, 9.0, 12.0, max_vsp]
                    bin_offset = 3
                elif sc1 < 50.0:
                    vsp_map = [0.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 30.0, max_vsp]
                    bin_offset = 9
                else:
                    vsp_map = [6.0, 12.0, 18.0, 24.0, 30.0, max_vsp]
                    bin_offset = 18

                for index, vsp_cur in enumerate(vsp_map):
                    if vsp[i] < vsp_cur:
                        bin_selector = bin_offset + index
                        break

            # 1-based to 0-based index
            bin_idx = bin_selector - 1
            size_bin[bin_idx] += 1.0

        return size_bin

    def estimate_emissions(
        self, veh_type: int, speed_mps: list[float] | np.ndarray
    ) -> dict[str, Any]:
        """Estimate the emissions for a given speed trajectory.

        Args:
            veh_type: 1 for Passenger Car, 2 for Passenger Truck.
            speed_mps: list or array of second-by-second speed values in m/s.

        Returns:
            Dictionary containing 'Emission Rate' (total grams and Energy/Fuel) and
            'Emission Factor' (normalized per mile).
        """
        if veh_type not in [1, 2, 6, 13]:
            raise ValueError(
                "Vehicle type must be 1 (Car), 2 (Truck), " "6 (Bus), or 13 (Combination Truck)"
            )

        speed = np.array(speed_mps, dtype=float)
        n_seconds = len(speed)

        if n_seconds == 0:
            return {}

        acc = self.calculate_acceleration(speed)
        vsp = self.calculate_vsp(veh_type, speed, acc)
        size_bin = self.calculate_operating_modes(speed, acc, vsp)

        # Load appropriate emission rate table
        df_ems = self.ems_tables[veh_type]
        # Divide by 3600 to convert from hourly rates to per-second rates
        df_sec = df_ems / 3600.0

        # Perform dot product of operating mode counts with emission tables
        bin_counts = size_bin.reshape(-1, 1)
        # Select columns 1 through 7 (excluding Operating Mode ID column at index 0)
        ems_totals = np.dot(bin_counts.T, df_sec.loc[:, 1:])
        ems_totals = ems_totals[0]  # Shape: (7,)

        # Fuel rate calculation from CO2
        # Fuel = CO2 * 12 / 44 * 13.78 / 12 = CO2 * 13.78 / 44
        co2_total = ems_totals[6]
        fuel_total = co2_total * 13.78 / 44.0

        # Build output structure matching original MOVESTAR outputs:
        # CO(g), HC(g), NOx(g), PM2.5_Ele(g), PM2.5_Org(g), Energy(KJ), CO2(g), Fuel(g), TT(s)
        total_vals = np.append(ems_totals, [fuel_total, float(n_seconds)])

        output_res: dict[str, Any] = {"Emission Rate": [total_vals.tolist()]}

        # Corner case: all zeros or negative speeds
        if all(v == 0.0 for v in speed) or any(v < 0.0 for v in speed):
            return output_res

        # Travel distance in meters
        td_meters = np.sum(speed)
        c1 = 1609.34  # 1 mile = 1609.34 meters

        # Normalized to grams/mi or KJ/mi
        factor_vals = np.append(total_vals[:8] * c1 / td_meters, [td_meters / c1])
        output_res["Emission Factor"] = [factor_vals.tolist()]

        return output_res

    def estimate_class_emissions(
        self, class_id: int, speed_mps: list[float] | np.ndarray
    ) -> dict[str, Any]:
        """Estimate emissions for a specific project class ID.

        Maps project class ID directly to native MOVEStar vehicle type:
        - 0 (Bus) -> veh_type = 6 (Transit Bus)
        - 1 (Car) -> veh_type = 1 (Passenger Car)
        - 2 (Heavy Truck) -> veh_type = 13 (Combination Truck)
        - 3 (Light Truck) -> veh_type = 2 (Passenger Truck)

        Args:
            class_id: Project class ID (0: bus, 1: car, 2: heavy truck, 3: light truck).
            speed_mps: Second-by-second speed profile in m/s.

        Returns:
            Dictionary containing emissions.
        """
        veh_type = emission_vehicle_type_for_project_class(class_id)
        return self.estimate_emissions(veh_type=veh_type, speed_mps=speed_mps)
