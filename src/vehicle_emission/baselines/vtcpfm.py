"""Virginia Tech Comprehensive Power-Based Fuel Consumption and Emission Model (VT-CPFM).

Developed as a baseline model to compare against MOVEStar. Calibrates a
quadratic polynomial to MOVES database emission rates over VSP ranges.
"""

from typing import Any

import numpy as np

from vehicle_emission.emissions.movestar import MOVEStarModel
from vehicle_emission.vehicle_classes import emission_vehicle_type_for_project_class


class VTCPFMModel(MOVEStarModel):
    """Virginia Tech Comprehensive Power-Based Fuel and Emission Model (VT-CPFM).

    Inherits from MOVEStarModel to reuse vehicle dynamics (VSP calculation) and
    vendored database tables. Calibrates quadratic coefficients during init.
    """

    def __init__(self) -> None:
        """Initialise the model and run the self-calibration logic."""
        super().__init__()
        self.coefficients: dict[int, dict[int, np.ndarray]] = {}
        self.idle_rates: dict[int, dict[int, float]] = {}
        self._calibrate()

    def _calibrate(self) -> None:
        """Fit a second-order polynomial model to the positive VSP bins in ems_tables.

        Uses opModeID = 1 (idle) as the default rate for VSP <= 0.
        """
        # Map opModeID to representative VSP values for fitting
        opmode_vsp_map = {
            12: 1.5,  # Low speed VSP [0, 3)
            13: 4.5,  # Low speed VSP [3, 6)
            14: 7.5,  # Low speed VSP [6, 9)
            15: 10.5,  # Low speed VSP [9, 12)
            16: 15.0,  # Low speed VSP >= 12
            22: 1.5,  # Med speed VSP [0, 3)
            23: 4.5,  # Med speed VSP [3, 6)
            24: 7.5,  # Med speed VSP [6, 9)
            25: 10.5,  # Med speed VSP [9, 12)
            27: 15.0,  # Med speed VSP [12, 18)
            28: 21.0,  # Med speed VSP [18, 24)
            29: 27.0,  # Med speed VSP [24, 30)
            30: 35.0,  # Med speed VSP >= 30
            33: 3.0,  # High speed VSP < 6
            35: 9.0,  # High speed VSP [6, 12)
            37: 15.0,  # High speed VSP [12, 18)
            38: 21.0,  # High speed VSP [18, 24)
            39: 27.0,  # High speed VSP [24, 30)
            40: 35.0,  # High speed VSP >= 30
        }

        for vt, df in self.ems_tables.items():
            self.coefficients[vt] = {}
            self.idle_rates[vt] = {}

            # Retrieve the idle row (opModeID = 1)
            idle_row = df[df[0] == 1]
            if idle_row.empty:
                idle_row = df.iloc[1:2]

            # Fit a quadratic model for columns 1 to 7
            # (CO, HC, NOx, PM2.5_Ele, PM2.5_Org, Energy, CO2)
            for col_idx in range(1, 8):
                idle_rate = float(idle_row[col_idx].values[0])
                self.idle_rates[vt][col_idx] = idle_rate

                # Gather points for VSP > 0
                vsp_points = [0.0]
                rate_points = [idle_rate]

                for opmode, vsp in opmode_vsp_map.items():
                    mode_row = df[df[0] == opmode]
                    if not mode_row.empty:
                        rate_val = float(mode_row[col_idx].values[0])
                        vsp_points.append(vsp)
                        rate_points.append(rate_val)

                # Fit quadratic polynomial: rate = a2 * vsp^2 + a1 * vsp + a0
                coeffs = np.polyfit(vsp_points, rate_points, deg=2)
                self.coefficients[vt][col_idx] = coeffs

    def estimate_emissions(
        self, veh_type: int, speed_mps: list[float] | np.ndarray
    ) -> dict[str, Any]:
        """Estimate emissions for a given speed trajectory using VT-CPFM.

        Args:
            veh_type: 1 (Car), 2 (Truck), 6 (Bus), or 13 (Combination Truck).
            speed_mps: second-by-second speed values in m/s.

        Returns:
            Dictionary containing 'Emission Rate' and 'Emission Factor'.
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

        # Initialize array for total emissions (7 columns)
        total_vals = np.zeros(7)

        for col_idx in range(1, 8):
            coeffs = self.coefficients[veh_type][col_idx]  # [a2, a1, a0]
            idle_val = self.idle_rates[veh_type][col_idx]

            # VSP <= 0 -> constant idle rate. VSP > 0 -> quadratic polynomial.
            # Convert hourly rates from table to per-second rates (divide by 3600)
            sec_rates = np.where(
                vsp <= 0.0,
                idle_val / 3600.0,
                (coeffs[0] * (vsp**2) + coeffs[1] * vsp + coeffs[2]) / 3600.0,
            )
            # Clip negative rates to 0 to prevent physical impossibilities
            sec_rates = np.maximum(sec_rates, 0.0)
            total_vals[col_idx - 1] = np.sum(sec_rates)

        # Fuel rate calculation from CO2
        # Fuel = CO2 * 13.78 / 44
        co2_total = total_vals[6]
        fuel_total = co2_total * 13.78 / 44.0

        # Match exact structure:
        # CO(g), HC(g), NOx(g), PM2.5_Ele(g), PM2.5_Org(g), Energy(KJ), CO2(g), Fuel(g), TT(s)
        total_vals = np.append(total_vals, [fuel_total, float(n_seconds)])

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
        """Estimate emissions for a specific project category ID.

        Maps category ID directly to native VT-CPFM vehicle type:
        - 0 (Bus) -> veh_type = 6 (Transit Bus)
        - 1 (Car) -> veh_type = 1 (Passenger Car)
        - 2 (Heavy Truck) -> veh_type = 13 (Combination Truck)
        - 3 (Light Truck) -> veh_type = 2 (Passenger Truck)

        Args:
            class_id: Category ID (0: bus, 1: car, 2: heavy truck, 3: light truck).
            speed_mps: second-by-second speed profile in m/s.

        Returns:
            Dictionary containing emissions.
        """
        veh_type = emission_vehicle_type_for_project_class(class_id)
        return self.estimate_emissions(veh_type=veh_type, speed_mps=speed_mps)
