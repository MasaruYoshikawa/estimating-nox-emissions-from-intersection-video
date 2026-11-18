"""Outcome-neutral study-design utilities.

This package contains simulation and design-checking code.  It must not read
held-out ambient outcomes or write empirical result files.
"""

from vehicle_emission.design.power_precision import (
    BootstrapIntervalAttempts,
    IntervalResult,
    SimulationPlan,
    SimulationScenario,
    cluster_bootstrap_interval_attempts,
    cluster_bootstrap_intervals,
    run_precision_simulation,
    simulate_site_series,
)
from vehicle_emission.design.two_site_precision import (
    JointResamplePlan,
    SiteDesign,
    SyntheticTwoSiteData,
    TwoSiteSimulationPlan,
    TwoSiteSimulationScenario,
    make_joint_resample_plan,
    run_two_site_precision_simulation,
    simulate_two_site_series,
)

__all__ = [
    "BootstrapIntervalAttempts",
    "IntervalResult",
    "SimulationPlan",
    "SimulationScenario",
    "JointResamplePlan",
    "SiteDesign",
    "SyntheticTwoSiteData",
    "TwoSiteSimulationPlan",
    "TwoSiteSimulationScenario",
    "cluster_bootstrap_interval_attempts",
    "cluster_bootstrap_intervals",
    "make_joint_resample_plan",
    "run_precision_simulation",
    "run_two_site_precision_simulation",
    "simulate_site_series",
    "simulate_two_site_series",
]
