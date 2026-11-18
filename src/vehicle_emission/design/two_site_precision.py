"""Synthetic two-site operating-characteristic simulations for T-126.

Formal estimands and terminology live in ``coordination/GLOSSARY.md``.  This
module is deliberately isolated from empirical loaders: it generates only
seeded synthetic hourly observations and cannot write research results.

The implementation compares candidate design rules.  In particular, coverage
thresholds, lag handling, and resampling settings are inputs rather than
frozen project choices while Q-012 and Q-016 remain open.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm, rankdata

CoverageRule = Literal["complete_only", "threshold_normalize"]
MissingnessPattern = Literal["uniform", "concentrated_by_day"]
LagRule = Literal["fixed", "select_max"]
DayResampling = Literal["iid_days", "moving_block_days"]
LagEdgePolicy = Literal["within_date_no_wrap_candidate_specific"]

SITES = ("Sukiyabashi", "Shinjuku")


@dataclass(frozen=True)
class SiteDesign:
    """Prespecified synthetic marginal design for one named site.

    Every ``rho_*`` input is a Pearson correlation in the latent Gaussian
    generator.  It is not treated as the resulting Spearman estimand.  The
    latter is estimated independently after ties, coverage, wind mixing, and
    any lag-selection functional have been applied.
    """

    site: str
    unmatched_days: int
    rho_emission_target_favourable: float
    rho_emission_target_opposing: float
    rho_count_target: float
    rho_emission_count: float
    favourable_wind_probability: float
    true_lag_hours: int
    target_tie_probabilities: tuple[float, ...] | None = None
    count_tie_probabilities: tuple[float, ...] | None = None

    def validate(self, candidate_lags: tuple[int, ...]) -> None:
        if self.site not in SITES:
            raise ValueError(f"site must be one of {SITES}")
        if self.unmatched_days < 0:
            raise ValueError("unmatched_days cannot be negative")
        for name in (
            "rho_emission_target_favourable",
            "rho_emission_target_opposing",
            "rho_count_target",
            "rho_emission_count",
        ):
            if not -0.90 <= float(getattr(self, name)) <= 0.90:
                raise ValueError(f"{name} must be in [-0.90, 0.90]")
        if not 0.0 < self.favourable_wind_probability < 1.0:
            raise ValueError("favourable_wind_probability must be in (0, 1)")
        if self.true_lag_hours not in candidate_lags:
            raise ValueError("true_lag_hours must be one of candidate_lags")
        _validate_probabilities(self.target_tie_probabilities, "target_tie_probabilities")
        _validate_probabilities(self.count_tie_probabilities, "count_tie_probabilities")


@dataclass(frozen=True)
class TwoSiteSimulationScenario:
    """A fully prespecified synthetic two-site data-generating scenario."""

    scenario_id: str
    dependence_label: str
    matched_days: int
    hours_per_day: int
    sites: tuple[SiteDesign, SiteDesign]
    cross_site_correlation: float
    between_day_phi: float
    within_day_phi: float
    day_effect_fraction: float
    missing_day_rate: float
    partial_hour_rate: float
    partial_coverage_min: float
    missingness_pattern: MissingnessPattern
    coverage_rule: CoverageRule
    coverage_threshold: float
    candidate_lags: tuple[int, ...]
    fixed_lag_hours: int
    lag_rule: LagRule
    min_site_days: int
    min_stratum_hours: int
    min_stratum_days: int
    design_axis_label: str = "unspecified_synthetic_design_axis"
    lag_edge_policy: LagEdgePolicy = "within_date_no_wrap_candidate_specific"

    def validate(self) -> None:
        if (
            not self.scenario_id.strip()
            or not self.dependence_label.strip()
            or not self.design_axis_label.strip()
        ):
            raise ValueError(
                "scenario_id, dependence_label, and design_axis_label must be non-empty"
            )
        if self.matched_days < 4 or self.hours_per_day < 2:
            raise ValueError("at least four matched days and two hours per day are required")
        if tuple(site.site for site in self.sites) != SITES:
            raise ValueError(f"sites must be ordered exactly as {SITES}")
        if not self.candidate_lags or len(set(self.candidate_lags)) != len(self.candidate_lags):
            raise ValueError("candidate_lags must be non-empty and unique")
        if self.fixed_lag_hours not in self.candidate_lags:
            raise ValueError("fixed_lag_hours must be one of candidate_lags")
        for site in self.sites:
            site.validate(self.candidate_lags)
            for base_rho in (
                site.rho_emission_target_favourable,
                site.rho_emission_target_opposing,
            ):
                _emission_coefficients(
                    base_rho,
                    site.rho_count_target,
                    site.rho_emission_count,
                )
        if not 0.0 <= self.cross_site_correlation <= 0.90:
            raise ValueError("cross_site_correlation must be in [0, 0.90]")
        for name in ("between_day_phi", "within_day_phi"):
            if not 0.0 <= float(getattr(self, name)) < 0.95:
                raise ValueError(f"{name} must be in [0, 0.95)")
        if not 0.0 <= self.day_effect_fraction < 1.0:
            raise ValueError("day_effect_fraction must be in [0, 1)")
        if not 0.0 <= self.missing_day_rate < 1.0:
            raise ValueError("missing_day_rate must be in [0, 1)")
        if not 0.0 <= self.partial_hour_rate < 1.0:
            raise ValueError("partial_hour_rate must be in [0, 1)")
        if not 0.0 < self.partial_coverage_min <= 1.0:
            raise ValueError("partial_coverage_min must be in (0, 1]")
        if self.missingness_pattern not in ("uniform", "concentrated_by_day"):
            raise ValueError("unsupported missingness_pattern")
        if self.coverage_rule not in ("complete_only", "threshold_normalize"):
            raise ValueError("unsupported coverage_rule")
        if not 0.0 < self.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be in (0, 1]")
        if self.lag_rule not in ("fixed", "select_max"):
            raise ValueError("unsupported lag_rule")
        if self.lag_edge_policy != "within_date_no_wrap_candidate_specific":
            raise ValueError("unsupported lag_edge_policy")
        if self.min_site_days < 4:
            raise ValueError("site-level support requires at least four days")
        if self.min_stratum_hours < 3 or self.min_stratum_days < 2:
            raise ValueError("RQ2 support minima require >=3 hours and >=2 days")

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> TwoSiteSimulationScenario:
        data = dict(values)
        raw_sites = data.pop("sites")
        if not isinstance(raw_sites, list):
            raise ValueError("sites must be a list")
        sites: list[SiteDesign] = []
        for raw_site in raw_sites:
            if not isinstance(raw_site, dict):
                raise ValueError("each site configuration must be an object")
            site_values = dict(raw_site)
            for name in ("target_tie_probabilities", "count_tie_probabilities"):
                if site_values.get(name) is not None:
                    site_values[name] = tuple(site_values[name])
            sites.append(SiteDesign(**site_values))  # type: ignore[arg-type]
        data["sites"] = tuple(sites)
        data["candidate_lags"] = tuple(data["candidate_lags"])  # type: ignore[arg-type]
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TwoSiteSimulationPlan:
    """Candidate Monte Carlo and joint day-resampling settings."""

    plan_id: str
    simulations: int = 100
    bootstrap_resamples: int = 199
    confidence_level: float = 0.95
    multiplicity_family_size: int = 1
    day_resampling: DayResampling = "iid_days"
    day_block_length: int = 1
    reference_draws: int = 20_000
    rq1_practical_threshold: float = 0.40
    rq2_practical_threshold: float = 0.15
    rq3_practical_margin: float = 0.10
    seed: int = 0

    def validate(self) -> None:
        if not self.plan_id.strip():
            raise ValueError("plan_id must be non-empty")
        if self.simulations < 1 or self.bootstrap_resamples < 19:
            raise ValueError("simulations must be positive and bootstrap_resamples >= 19")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
        if self.multiplicity_family_size < 1:
            raise ValueError("multiplicity_family_size must be positive")
        if self.day_resampling not in ("iid_days", "moving_block_days"):
            raise ValueError("unsupported day_resampling")
        if self.day_block_length < 1:
            raise ValueError("day_block_length must be positive")
        if self.day_resampling == "iid_days" and self.day_block_length != 1:
            raise ValueError("iid_days requires day_block_length=1")
        if self.reference_draws < 1_000:
            raise ValueError("reference_draws must be at least 1000")

    @property
    def adjusted_alpha(self) -> float:
        return (1.0 - self.confidence_level) / self.multiplicity_family_size


@dataclass(frozen=True)
class SyntheticTwoSiteData:
    """Generated eligible hourly rows plus outcome-neutral design commitments."""

    frame: pd.DataFrame
    coverage_mask_sha256: str
    wind_mask_sha256: str
    design_mask_sha256: str
    coverage_rule: str
    missingness_pattern: str


@dataclass(frozen=True)
class JointResamplePlan:
    """One bootstrap draw preserving matched and unmatched date semantics."""

    indices_by_site: dict[str, NDArray[np.int64]]
    cluster_labels_by_site: dict[str, NDArray[np.object_]]
    matched_dates_selected: tuple[str, ...]
    matched_presence_by_site: dict[str, tuple[bool, ...]]
    unmatched_dates_selected: dict[str, tuple[str, ...]]
    failures_by_site: dict[str, str]
    sha256: str


@dataclass(frozen=True)
class _PointResult:
    values: dict[tuple[str, str], float]
    failures: dict[tuple[str, str], str]
    selected_lags: dict[str, int]
    stratum_support: dict[str, dict[str, int]]
    lag_support: dict[str, dict[int, dict[str, int]]]


@dataclass(frozen=True)
class _ReferenceResult:
    point: _PointResult
    mcse: dict[tuple[str, str], float]
    valid_batches: dict[tuple[str, str], int]
    support_by_site: dict[str, dict[str, float]]
    batches_requested: int


def _validate_probabilities(probabilities: tuple[float, ...] | None, name: str) -> None:
    if probabilities is None:
        return
    if len(probabilities) < 3 or any(value <= 0.0 for value in probabilities):
        raise ValueError(f"{name} must contain at least three positive probabilities")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-10):
        raise ValueError(f"{name} must sum to one")


def _seed(root: int, scenario_id: str, label: str) -> int:
    digest = hashlib.sha256(f"{root}|{scenario_id}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _hash_records(records: Sequence[tuple[object, ...]]) -> str:
    encoded = json.dumps(records, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _quantise_probabilities(
    values: NDArray[np.float64], probabilities: tuple[float, ...] | None
) -> NDArray[np.float64]:
    if probabilities is None:
        return values.copy()
    thresholds = np.cumsum(np.asarray(probabilities, dtype=float))[:-1]
    return np.searchsorted(thresholds, norm.cdf(values), side="right").astype(float)


def _rank_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    ranked_x = rankdata(x, method="average")
    ranked_y = rankdata(y, method="average")
    ranked_x -= ranked_x.mean()
    ranked_y -= ranked_y.mean()
    denominator = math.sqrt(float(ranked_x @ ranked_x) * float(ranked_y @ ranked_y))
    if denominator <= 0.0:
        return math.nan
    return float((ranked_x @ ranked_y) / denominator)


def _cross_site_normals(
    rng: np.random.Generator, rows: int, columns: int, correlation: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    first = rng.normal(size=(rows, columns))
    independent = rng.normal(size=(rows, columns))
    second = correlation * first + math.sqrt(1.0 - correlation**2) * independent
    return first, second


def _day_ar(innovations: NDArray[np.float64], phi: float) -> NDArray[np.float64]:
    states = np.empty_like(innovations)
    states[0] = innovations[0]
    scale = math.sqrt(1.0 - phi**2)
    for day in range(1, len(innovations)):
        states[day] = phi * states[day - 1] + scale * innovations[day]
    return states


def _emission_coefficients(
    rho_emission_target: float,
    rho_count_target: float,
    rho_emission_count: float,
) -> tuple[float, float]:
    """Return residual coefficients realising all three latent correlations."""

    count_residual_scale = math.sqrt(1.0 - rho_count_target**2)
    shared_residual = (
        rho_emission_count - rho_emission_target * rho_count_target
    ) / count_residual_scale
    independent_variance = 1.0 - rho_emission_target**2 - shared_residual**2
    if independent_variance <= 1e-10:
        raise ValueError("site target/emission/count correlations are not positive definite")
    return shared_residual, math.sqrt(independent_variance)


def _concentrated_partial_probabilities(rate: float) -> tuple[float, float, float]:
    """Return bad-day prevalence plus high/low rates with marginal ``rate``."""

    if rate == 0.0:
        return 0.0, 0.0, 0.0
    low_probability = 0.1 * rate
    initial_bad_day_probability = min(0.95, 2.0 * rate)
    minimum_bad_day_probability = (rate - low_probability) / (1.0 - low_probability)
    bad_day_probability = max(
        initial_bad_day_probability,
        minimum_bad_day_probability,
    )
    high_probability = (rate - (1.0 - bad_day_probability) * low_probability) / bad_day_probability
    if not 0.0 <= high_probability <= 1.0:
        raise ValueError("cannot construct concentrated partial-hour probabilities")
    return bad_day_probability, high_probability, low_probability


def _design_rows(scenario: TwoSiteSimulationScenario, rng: np.random.Generator) -> pd.DataFrame:
    """Generate wind and coverage before any synthetic outcome value."""

    rows: list[dict[str, object]] = []
    bad_day_probability, high_partial_rate, low_partial_rate = _concentrated_partial_probabilities(
        scenario.partial_hour_rate
    )
    for site_design in scenario.sites:
        date_specs = [(f"M{day:04d}", True) for day in range(scenario.matched_days)]
        site_prefix = "S" if site_design.site == "Sukiyabashi" else "J"
        date_specs.extend(
            (f"{site_prefix}U{day:04d}", False) for day in range(site_design.unmatched_days)
        )
        missing_days = rng.random(len(date_specs)) < scenario.missing_day_rate
        concentrated = rng.random(len(date_specs)) < bad_day_probability
        for day_index, ((date_id, matched), missing_day) in enumerate(
            zip(date_specs, missing_days, strict=True)
        ):
            for hour in range(scenario.hours_per_day):
                favourable = bool(rng.random() < site_design.favourable_wind_probability)
                if scenario.missingness_pattern == "uniform":
                    partial = bool(rng.random() < scenario.partial_hour_rate)
                else:
                    partial_probability = (
                        high_partial_rate if concentrated[day_index] else low_partial_rate
                    )
                    partial = bool(rng.random() < partial_probability)
                coverage = (
                    float(rng.uniform(scenario.partial_coverage_min, 0.999999)) if partial else 1.0
                )
                target_available = not bool(missing_day)
                retained = target_available
                if scenario.coverage_rule == "complete_only":
                    retained = retained and not partial
                else:
                    retained = retained and coverage >= scenario.coverage_threshold
                rows.append(
                    {
                        "site": site_design.site,
                        "date_id": date_id,
                        "matched_date": matched,
                        "day_order": day_index,
                        "hour": hour,
                        "wind_stratum": "favourable" if favourable else "opposing",
                        "coverage_fraction": coverage,
                        "covered_seconds": int(math.floor(3600.0 * coverage)),
                        "missing_day": bool(missing_day),
                        "target_available": target_available,
                        "retained": retained,
                    }
                )
    return pd.DataFrame(rows)


def simulate_two_site_series(
    scenario: TwoSiteSimulationScenario, *, seed: int
) -> SyntheticTwoSiteData:
    """Generate paired synthetic site-hours with auditable design-mask hashes.

    A dedicated RNG stream creates wind, missing-day, and coverage fields
    before a second independent stream generates target/predictor values.
    Shared matched dates receive prespecified cross-site dependence; unmatched
    dates are generated from independent site streams.
    """

    scenario.validate()
    mask_rng = np.random.default_rng(_seed(seed, scenario.scenario_id, "design_masks"))
    value_rng = np.random.default_rng(_seed(seed, scenario.scenario_id, "synthetic_values"))
    design = _design_rows(scenario, mask_rng)

    coverage_records = [
        (
            row.site,
            row.date_id,
            int(row.hour),
            round(float(row.coverage_fraction), 12),
            bool(row.missing_day),
            bool(row.retained),
        )
        for row in design.itertuples(index=False)
    ]
    wind_records = [
        (row.site, row.date_id, int(row.hour), row.wind_stratum)
        for row in design.itertuples(index=False)
    ]
    coverage_hash = _hash_records(coverage_records)
    wind_hash = _hash_records(wind_records)
    design_hash = hashlib.sha256(
        bytes.fromhex(coverage_hash) + bytes.fromhex(wind_hash)
    ).hexdigest()

    # Exactly one latent emission innovation is generated at each site/date/hour.
    # Lag candidates below are shifts of the resulting observed emission series;
    # they are never independently generated predictor columns.
    columns = 3  # target, count innovation, emission innovation
    matched_rows = scenario.matched_days * scenario.hours_per_day
    matched_a, matched_b = _cross_site_normals(
        value_rng, matched_rows, columns, scenario.cross_site_correlation
    )
    matched_latent = {
        SITES[0]: matched_a.reshape(scenario.matched_days, scenario.hours_per_day, columns),
        SITES[1]: matched_b.reshape(scenario.matched_days, scenario.hours_per_day, columns),
    }
    for site in SITES:
        for day in range(scenario.matched_days):
            matched_latent[site][day] = _day_ar(matched_latent[site][day], scenario.within_day_phi)

    # Correlated between-day effects are generated only for overlapping dates.
    day_a, day_b = _cross_site_normals(
        value_rng, scenario.matched_days, columns, scenario.cross_site_correlation
    )
    matched_days = {
        SITES[0]: _day_ar(day_a, scenario.between_day_phi),
        SITES[1]: _day_ar(day_b, scenario.between_day_phi),
    }
    within_scale = math.sqrt(1.0 - scenario.day_effect_fraction)
    day_scale = math.sqrt(scenario.day_effect_fraction)
    site_lookup = {site.site: site for site in scenario.sites}
    value_frames: list[pd.DataFrame] = []

    for site in SITES:
        site_design = site_lookup[site]
        site_rows = design.loc[design["site"] == site].copy()
        unmatched_dates = sorted(
            site_rows.loc[~site_rows["matched_date"], "date_id"].unique().tolist()
        )
        unmatched_innovations = value_rng.normal(size=(len(unmatched_dates), columns))
        unmatched_days = (
            _day_ar(unmatched_innovations, scenario.between_day_phi)
            if unmatched_dates
            else np.empty((0, columns))
        )
        unmatched_lookup = {date: index for index, date in enumerate(unmatched_dates)}
        unmatched_within = value_rng.normal(
            size=(len(unmatched_dates), scenario.hours_per_day, columns)
        )
        for day in range(len(unmatched_dates)):
            unmatched_within[day] = _day_ar(unmatched_within[day], scenario.within_day_phi)
        site_value_rows: list[dict[str, object]] = []
        for row in site_rows.itertuples(index=False):
            if row.matched_date:
                day_index = int(str(row.date_id)[1:])
                raw = matched_latent[site][day_index, int(row.hour)]
                day_effect = matched_days[site][day_index]
            else:
                day_index = unmatched_lookup[str(row.date_id)]
                raw = unmatched_within[day_index, int(row.hour)]
                day_effect = unmatched_days[day_index]
            base = within_scale * raw + day_scale * day_effect
            latent_target = float(base[0])
            count_innovation = float(base[1])
            coverage = float(row.coverage_fraction)
            generated_target = float(
                _quantise_probabilities(
                    np.asarray([latent_target]), site_design.target_tie_probabilities
                )[0]
            )
            target = generated_target if bool(row.target_available) else math.nan
            output: dict[str, object] = {
                "site": site,
                "date_id": row.date_id,
                "matched_date": bool(row.matched_date),
                "day_order": int(row.day_order),
                "hour": int(row.hour),
                "wind_stratum": row.wind_stratum,
                "coverage_fraction": float(row.coverage_fraction),
                "covered_seconds": int(row.covered_seconds),
                "target": target,
                "latent_target_internal": latent_target,
                "count_innovation_internal": count_innovation,
                "emission_innovation_internal": float(base[2]),
                "retained_internal": bool(row.retained),
                "target_available_internal": bool(row.target_available),
            }
            site_value_rows.append(output)

        site_values = pd.DataFrame(site_value_rows).sort_values(
            ["matched_date", "day_order", "date_id", "hour"], kind="stable"
        )
        source_full: dict[tuple[str, int], float] = {}
        source_raw: dict[tuple[str, int], float] = {}
        source_observed: dict[tuple[str, int], float] = {}
        count_source_full: dict[tuple[str, int], float] = {}
        count_source_raw: dict[tuple[str, int], float] = {}
        count_source_observed: dict[tuple[str, int], float] = {}
        source_coverage: dict[tuple[str, int], float] = {}
        source_covered_seconds: dict[tuple[str, int], int] = {}
        by_key = {
            (str(row.date_id), int(row.hour)): row for row in site_values.itertuples(index=False)
        }
        for source in site_values.itertuples(index=False):
            source_key = (str(source.date_id), int(source.hour))
            target_key = (
                str(source.date_id),
                int(source.hour) + site_design.true_lag_hours,
            )
            target_row = by_key.get(target_key)
            if target_row is None:
                # The trailing/leading part of the source process that cannot
                # map to the true lag remains a genuine process innovation.
                # It is still available to other actual shift candidates.
                full_emission = float(source.emission_innovation_internal)
                full_count = float(source.count_innovation_internal)
            else:
                base_rho = (
                    site_design.rho_emission_target_favourable
                    if target_row.wind_stratum == "favourable"
                    else site_design.rho_emission_target_opposing
                )
                shared_residual, independent_residual = _emission_coefficients(
                    base_rho,
                    site_design.rho_count_target,
                    site_design.rho_emission_count,
                )
                full_emission = (
                    base_rho * float(target_row.latent_target_internal)
                    + shared_residual * float(source.count_innovation_internal)
                    + independent_residual * float(source.emission_innovation_internal)
                )
                full_count = site_design.rho_count_target * float(
                    target_row.latent_target_internal
                ) + math.sqrt(1.0 - site_design.rho_count_target**2) * float(
                    source.count_innovation_internal
                )
            full_count = float(
                _quantise_probabilities(
                    np.asarray([full_count]), site_design.count_tie_probabilities
                )[0]
            )
            coverage = float(source.coverage_fraction)
            if scenario.coverage_rule == "threshold_normalize":
                raw_emission = coverage * full_emission + (1.0 - coverage) * float(
                    value_rng.normal()
                )
                emission = raw_emission / coverage
                raw_count = max(
                    coverage * full_count + (1.0 - coverage) * float(value_rng.normal()),
                    0.0,
                )
                count = raw_count / coverage
            else:
                raw_emission = full_emission
                emission = raw_emission
                raw_count = full_count
                count = raw_count
            source_full[source_key] = full_emission
            count_source_full[source_key] = full_count
            source_coverage[source_key] = coverage
            source_covered_seconds[source_key] = int(source.covered_seconds)
            if bool(source.retained_internal):
                source_raw[source_key] = raw_emission
                source_observed[source_key] = emission
                count_source_raw[source_key] = raw_count
                count_source_observed[source_key] = count
            else:
                source_raw[source_key] = math.nan
                source_observed[source_key] = math.nan
                count_source_raw[source_key] = math.nan
                count_source_observed[source_key] = math.nan

        site_values["emission_source_full_hour_synthetic"] = [
            source_full[(str(row.date_id), int(row.hour))]
            for row in site_values.itertuples(index=False)
        ]
        site_values["emission_source_raw_covered"] = [
            source_raw[(str(row.date_id), int(row.hour))]
            for row in site_values.itertuples(index=False)
        ]
        site_values["emission_source"] = [
            source_observed[(str(row.date_id), int(row.hour))]
            for row in site_values.itertuples(index=False)
        ]
        site_values["count_source_full_hour_synthetic"] = [
            count_source_full[(str(row.date_id), int(row.hour))]
            for row in site_values.itertuples(index=False)
        ]
        site_values["count_source_raw_covered"] = [
            count_source_raw[(str(row.date_id), int(row.hour))]
            for row in site_values.itertuples(index=False)
        ]
        site_values["count_source"] = [
            count_source_observed[(str(row.date_id), int(row.hour))]
            for row in site_values.itertuples(index=False)
        ]
        for lag in scenario.candidate_lags:
            lag_raw: list[float] = []
            lag_observed: list[float] = []
            count_lag_raw: list[float] = []
            count_lag_observed: list[float] = []
            lag_source_coverage: list[float] = []
            lag_source_covered_seconds: list[float] = []
            for target_row in site_values.itertuples(index=False):
                source_key = (str(target_row.date_id), int(target_row.hour) - lag)
                lag_raw.append(source_raw.get(source_key, math.nan))
                lag_observed.append(source_observed.get(source_key, math.nan))
                count_lag_raw.append(count_source_raw.get(source_key, math.nan))
                count_lag_observed.append(count_source_observed.get(source_key, math.nan))
                lag_source_coverage.append(source_coverage.get(source_key, math.nan))
                lag_source_covered_seconds.append(
                    float(source_covered_seconds[source_key])
                    if source_key in source_covered_seconds
                    else math.nan
                )
            site_values[f"emission_lag_{lag}_raw_covered"] = lag_raw
            site_values[f"emission_lag_{lag}"] = lag_observed
            site_values[f"count_lag_{lag}_raw_covered"] = count_lag_raw
            site_values[f"count_lag_{lag}"] = count_lag_observed
            site_values[f"emission_lag_{lag}_source_coverage_fraction"] = lag_source_coverage
            site_values[f"predictor_lag_{lag}_source_covered_seconds"] = lag_source_covered_seconds
        value_frames.append(site_values)

    frame = pd.concat(value_frames, ignore_index=True)
    # Retain unavailable planned dates as an explicit calendar/presence mask
    # for the joint moving-block timeline. Their target is NaN, so no analysis
    # can use them; each lag column already applies its own shifted source rule.
    frame = frame.drop(
        columns=[
            "latent_target_internal",
            "count_innovation_internal",
            "emission_innovation_internal",
            "retained_internal",
            "target_available_internal",
        ]
    )
    frame = frame.sort_values(["site", "matched_date", "day_order", "hour"], kind="stable")
    frame = frame.reset_index(drop=True)
    return SyntheticTwoSiteData(
        frame=frame,
        coverage_mask_sha256=coverage_hash,
        wind_mask_sha256=wind_hash,
        design_mask_sha256=design_hash,
        coverage_rule=scenario.coverage_rule,
        missingness_pattern=scenario.missingness_pattern,
    )


def _sample_dates(
    dates: tuple[str, ...],
    *,
    rng: np.random.Generator,
    method: DayResampling,
    block: int,
    pool_label: str,
) -> tuple[str, ...]:
    if not dates:
        return ()
    if method == "iid_days":
        selected = rng.choice(np.asarray(dates, dtype=object), size=len(dates), replace=True)
        return tuple(str(value) for value in selected)
    if block > len(dates):
        raise ValueError(
            f"configured moving block length {block} exceeds {pool_label} pool length "
            f"{len(dates)}; block length is never shortened"
        )
    chosen: list[str] = []
    maximum_start = len(dates) - block
    while len(chosen) < len(dates):
        start = int(rng.integers(0, maximum_start + 1))
        chosen.extend(dates[start : start + block])
    return tuple(chosen[: len(dates)])


def make_joint_resample_plan(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    method: DayResampling,
    block_length: int,
) -> JointResamplePlan:
    """Draw a union matched timeline jointly and declared unmatched dates independently.

    A planned matched date observed at only one site remains in the shared
    timeline.  The same selected positions are applied at both sites through
    explicit presence masks; it is never relabelled as a site-independent day.
    """

    matched_by_site = {
        site: set(
            frame.loc[
                (frame["site"] == site) & frame["matched_date"] & frame["target"].notna(),
                "date_id",
            ].unique()
        )
        for site in SITES
    }
    # The same planned union timeline is the cluster universe for IID and
    # moving-block candidates. Missing-both dates may therefore produce an
    # empty selected position at both sites; that explicit presence mask keeps
    # the compared resampling rules from changing their target universe.
    matched_dates = tuple(sorted(frame.loc[frame["matched_date"], "date_id"].unique().tolist()))
    selected_matched = _sample_dates(
        matched_dates,
        rng=rng,
        method=method,
        block=block_length,
        pool_label="union matched-date",
    )
    selected_unmatched: dict[str, tuple[str, ...]] = {}
    failures_by_site: dict[str, str] = {}
    matched_presence: dict[str, tuple[bool, ...]] = {}
    indices_by_site: dict[str, NDArray[np.int64]] = {}
    cluster_labels_by_site: dict[str, NDArray[np.object_]] = {}
    digest = hashlib.sha256(b"two-site-union-timeline-day-resample-v2\0")
    digest.update(json.dumps(selected_matched, separators=(",", ":")).encode())
    for site in SITES:
        site_matched_dates = matched_by_site[site]
        presence = tuple(date in site_matched_dates for date in selected_matched)
        matched_presence[site] = presence
        site_unmatched = tuple(
            sorted(
                frame.loc[
                    (frame["site"] == site) & ~frame["matched_date"],
                    "date_id",
                ].unique()
            )
        )
        try:
            selected = _sample_dates(
                site_unmatched,
                rng=rng,
                method=method,
                block=block_length,
                pool_label=f"{site} declared-unmatched-date",
            )
        except ValueError as error:
            selected = ()
            failures_by_site[site] = str(error)
        selected_unmatched[site] = selected
        digest.update(site.encode() + b"\0")
        digest.update(json.dumps(presence, separators=(",", ":")).encode())
        digest.update(json.dumps(selected, separators=(",", ":")).encode())
        digest.update(failures_by_site.get(site, "").encode())
        pieces: list[NDArray[np.int64]] = []
        cluster_labels: list[str] = []
        for draw_number, (date, is_present) in enumerate(
            zip(selected_matched, presence, strict=True)
        ):
            if not is_present:
                continue
            member_indices = frame.index[
                (frame["site"] == site) & (frame["date_id"] == date)
            ].to_numpy(dtype=np.int64)
            pieces.append(member_indices)
            cluster_labels.extend([f"BM{draw_number:04d}:{date}"] * len(member_indices))
        for unmatched_number, date in enumerate(selected):
            member_indices = frame.index[
                (frame["site"] == site) & (frame["date_id"] == date)
            ].to_numpy(dtype=np.int64)
            pieces.append(member_indices)
            cluster_labels.extend([f"BU{unmatched_number:04d}:{site}:{date}"] * len(member_indices))
        indices_by_site[site] = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        cluster_labels_by_site[site] = np.asarray(cluster_labels, dtype=object)
    return JointResamplePlan(
        indices_by_site=indices_by_site,
        cluster_labels_by_site=cluster_labels_by_site,
        matched_dates_selected=selected_matched,
        matched_presence_by_site=matched_presence,
        unmatched_dates_selected=selected_unmatched,
        failures_by_site=failures_by_site,
        sha256=digest.hexdigest(),
    )


def _complete_lag_rows(site_frame: pd.DataFrame, lag: int) -> pd.DataFrame:
    return site_frame.dropna(subset=["target", f"count_lag_{lag}", f"emission_lag_{lag}"])


def _choose_lag(site_frame: pd.DataFrame, scenario: TwoSiteSimulationScenario) -> tuple[int, float]:
    if scenario.lag_rule == "fixed":
        lag = scenario.fixed_lag_hours
        complete = _complete_lag_rows(site_frame, lag)
        return lag, _rank_correlation(
            complete[f"emission_lag_{lag}"].to_numpy(dtype=float),
            complete["target"].to_numpy(dtype=float),
        )
    correlations = {
        lag: _rank_correlation(
            _complete_lag_rows(site_frame, lag)[f"emission_lag_{lag}"].to_numpy(dtype=float),
            _complete_lag_rows(site_frame, lag)["target"].to_numpy(dtype=float),
        )
        for lag in scenario.candidate_lags
    }
    finite = {lag: value for lag, value in correlations.items() if math.isfinite(value)}
    if not finite:
        return scenario.candidate_lags[0], math.nan
    # Stable tie break is prespecified by candidate_lags order.
    lag = max(scenario.candidate_lags, key=lambda value: finite.get(value, -math.inf))
    return lag, finite[lag]


def _rq2_support(site_frame: pd.DataFrame) -> tuple[dict[str, int], str | None]:
    support: dict[str, int] = {}
    for stratum in ("favourable", "opposing"):
        part = site_frame.loc[site_frame["wind_stratum"] == stratum]
        support[f"{stratum}_hours"] = len(part)
        support[f"{stratum}_days"] = int(part["date_id"].nunique())
    return support, None


def _point_results(
    frames: dict[str, pd.DataFrame], scenario: TwoSiteSimulationScenario
) -> _PointResult:
    values: dict[tuple[str, str], float] = {}
    failures: dict[tuple[str, str], str] = {}
    selected_lags: dict[str, int] = {}
    supports: dict[str, dict[str, int]] = {}
    lag_supports: dict[str, dict[int, dict[str, int]]] = {}
    site_rq1: dict[str, float] = {}
    for site in SITES:
        part = frames[site]
        site_lag_support: dict[int, dict[str, int]] = {}
        for candidate_lag in scenario.candidate_lags:
            complete = _complete_lag_rows(part, candidate_lag)
            available_target_hours = int(part["target"].notna().sum())
            site_lag_support[candidate_lag] = {
                "planned_target_hours": len(part),
                "available_target_hours": available_target_hours,
                "target_unavailable_hours": len(part) - available_target_hours,
                "complete_pair_hours": len(complete),
                "lost_edge_or_unobserved_source_hours": (available_target_hours - len(complete)),
                "complete_pair_days": int(complete["date_id"].nunique()),
            }
        lag_supports[site] = site_lag_support
        lag, rq1 = _choose_lag(part, scenario)
        selected_lags[site] = lag
        analysis = _complete_lag_rows(part, lag)
        support, _ = _rq2_support(analysis)
        supports[site] = support
        observed_days = int(analysis["date_id"].nunique()) if not analysis.empty else 0
        if observed_days < scenario.min_site_days:
            reason = f"site_days={observed_days}<minimum_site_days={scenario.min_site_days}"
            for rq in ("RQ1", "RQ2", "RQ3"):
                failures[(rq, site)] = reason
            continue
        if not math.isfinite(rq1):
            failures[("RQ1", site)] = "selected lag association is undefined"
        else:
            values[("RQ1", site)] = rq1
            site_rq1[site] = rq1
        target = analysis["target"].to_numpy(dtype=float)
        count_rho = _rank_correlation(analysis[f"count_lag_{lag}"].to_numpy(dtype=float), target)
        delta = rq1 - count_rho
        if not math.isfinite(delta):
            failures[("RQ3", site)] = "paired emission/count association is undefined"
        else:
            values[("RQ3", site)] = delta
        minimum_failure: list[str] = []
        for stratum in ("favourable", "opposing"):
            hours = support[f"{stratum}_hours"]
            days = support[f"{stratum}_days"]
            if hours < scenario.min_stratum_hours:
                minimum_failure.append(f"{stratum}_hours={hours}<{scenario.min_stratum_hours}")
            if days < scenario.min_stratum_days:
                minimum_failure.append(f"{stratum}_days={days}<{scenario.min_stratum_days}")
        if minimum_failure:
            failures[("RQ2", site)] = ";".join(minimum_failure)
        else:
            favourable = analysis.loc[analysis["wind_stratum"] == "favourable"]
            opposing = analysis.loc[analysis["wind_stratum"] == "opposing"]
            favourable_rho = _rank_correlation(
                favourable[f"emission_lag_{lag}"].to_numpy(dtype=float),
                favourable["target"].to_numpy(dtype=float),
            )
            opposing_rho = _rank_correlation(
                opposing[f"emission_lag_{lag}"].to_numpy(dtype=float),
                opposing["target"].to_numpy(dtype=float),
            )
            delta_wind = favourable_rho - opposing_rho
            if math.isfinite(delta_wind):
                values[("RQ2", site)] = delta_wind
            else:
                failures[("RQ2", site)] = "wind-stratum association is degenerate"
    heterogeneity_key = ("H_E", "cross_site")
    if all(site in site_rq1 and math.isfinite(site_rq1[site]) for site in SITES):
        values[heterogeneity_key] = site_rq1["Shinjuku"] - site_rq1["Sukiyabashi"]
    else:
        failures[heterogeneity_key] = "one or both site RQ1 estimates are undefined"
    return _PointResult(values, failures, selected_lags, supports, lag_supports)


def _reference_results(
    scenario: TwoSiteSimulationScenario, plan: TwoSiteSimulationPlan
) -> _ReferenceResult:
    reference_days = max(50, math.ceil(plan.reference_draws / scenario.hours_per_day))
    reference_sites = tuple(replace(site, unmatched_days=0) for site in scenario.sites)
    reference_scenario = replace(
        scenario,
        matched_days=reference_days,
        sites=reference_sites,  # type: ignore[arg-type]
        missing_day_rate=0.0,
    )
    replicate_points: list[_PointResult] = []
    support_samples: dict[str, dict[str, list[float]]] = {
        site: {
            "available_target_hours": [],
            "selected_complete_pair_hours": [],
            "selected_complete_pair_days": [],
        }
        for site in SITES
    }
    # Four independently seeded, full-size references make the finite-reference
    # uncertainty visible.  This is still a smoke diagnostic, not proof that
    # the configured reference size is decision-quality.
    batches_requested = 4
    for batch in range(batches_requested):
        synthetic = simulate_two_site_series(
            reference_scenario,
            seed=_seed(plan.seed, scenario.scenario_id, f"reference-batch-{batch}"),
        )
        frames = {
            site: synthetic.frame.loc[synthetic.frame["site"] == site].copy() for site in SITES
        }
        point = _point_results(frames, reference_scenario)
        replicate_points.append(point)
        for site in SITES:
            selected = point.selected_lags.get(site, reference_scenario.fixed_lag_hours)
            selected_support = point.lag_support[site][selected]
            support_samples[site]["available_target_hours"].append(
                float(selected_support["available_target_hours"])
            )
            support_samples[site]["selected_complete_pair_hours"].append(
                float(selected_support["complete_pair_hours"])
            )
            support_samples[site]["selected_complete_pair_days"].append(
                float(selected_support["complete_pair_days"])
            )

    keys = [(rq, site) for site in SITES for rq in ("RQ1", "RQ2", "RQ3")] + [("H_E", "cross_site")]
    reference_values: dict[tuple[str, str], float] = {}
    mcse: dict[tuple[str, str], float] = {}
    valid_batches: dict[tuple[str, str], int] = {}
    for key in keys:
        values = np.asarray(
            [point.values[key] for point in replicate_points if key in point.values],
            dtype=float,
        )
        valid_batches[key] = len(values)
        if len(values):
            reference_values[key] = float(np.mean(values))
        mcse[key] = (
            float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) >= 2 else math.nan
        )
    representative = replace(replicate_points[0], values=reference_values)
    support_by_site: dict[str, dict[str, float]] = {}
    for site in SITES:
        support_by_site[site] = {
            "planned_days_per_reference_batch": float(reference_days),
            "planned_hours_per_reference_batch": float(
                reference_days * reference_scenario.hours_per_day
            ),
            **{
                f"median_{name}_per_reference_batch": float(np.median(values))
                for name, values in support_samples[site].items()
            },
        }
    return _ReferenceResult(
        point=representative,
        mcse=mcse,
        valid_batches=valid_batches,
        support_by_site=support_by_site,
        batches_requested=batches_requested,
    )


def _counter_json(counter: Counter[str]) -> str:
    return json.dumps(dict(sorted(counter.items())), separators=(",", ":"))


def _probability(count: int, denominator: int) -> float:
    return count / denominator


def _mcse(probability: float, denominator: int) -> float:
    return math.sqrt(probability * (1.0 - probability) / denominator)


def classify_interval_decision(
    rq: str,
    *,
    lower: float,
    upper: float,
    samples: NDArray[np.float64],
    adjusted_alpha: float,
    rq1_threshold: float,
    rq2_threshold: float,
    rq3_margin: float,
) -> str:
    """Apply candidate design labels, including explicit open boundaries."""

    if rq == "RQ1":
        if lower > rq1_threshold:
            return "practically_strong"
        if lower > 0.0:
            return "positive"
        if upper <= 0.0:
            return "non_positive"
        return "inconclusive"
    if rq == "RQ2":
        if lower > rq2_threshold:
            return "meaningful_moderation"
        if upper < rq2_threshold:
            return "proposed_magnitude_ruled_out"
        return "inconclusive"
    if rq == "RQ3":
        if lower > rq3_margin:
            return "meaningful_added_value"
        if upper < -rq3_margin:
            return "meaningful_count_advantage"
        equivalence_lower, equivalence_upper = np.quantile(
            samples, [adjusted_alpha, 1.0 - adjusted_alpha]
        )
        if equivalence_lower > -rq3_margin and equivalence_upper < rq3_margin:
            return "candidate_equivalence"
        return "inconclusive"
    return "descriptive_interval_estimable"


def run_two_site_precision_simulation(
    scenario: TwoSiteSimulationScenario, plan: TwoSiteSimulationPlan
) -> pd.DataFrame:
    """Evaluate all two-site estimands with unconditional failure accounting."""

    scenario.validate()
    plan.validate()
    references = _reference_results(scenario, plan)
    keys: list[tuple[str, str]] = [(rq, site) for site in SITES for rq in ("RQ1", "RQ2", "RQ3")] + [
        ("H_E", "cross_site")
    ]
    if any(key not in references.point.values for key in keys):
        missing = [key for key in keys if key not in references.point.values]
        raise ValueError(f"synthetic reference estimands are undefined: {missing}")

    estimates: dict[tuple[str, str], list[float]] = {key: [] for key in keys}
    widths: dict[tuple[str, str], list[float]] = {key: [] for key in keys}
    valid_resample_counts: dict[tuple[str, str], list[int]] = {key: [] for key in keys}
    successes: Counter[tuple[str, str]] = Counter()
    covered: Counter[tuple[str, str]] = Counter()
    failures: dict[tuple[str, str], Counter[str]] = {key: Counter() for key in keys}
    decisions: dict[tuple[str, str], Counter[str]] = {key: Counter() for key in keys}
    point_lag_counts: dict[str, Counter[str]] = {site: Counter() for site in SITES}
    bootstrap_lag_counts: dict[str, Counter[str]] = {site: Counter() for site in SITES}
    rq2_invalid_resamples: dict[str, Counter[str]] = {site: Counter() for site in SITES}
    support_values: dict[str, dict[str, list[float]]] = {
        site: {
            name: []
            for name in ("favourable_hours", "favourable_days", "opposing_hours", "opposing_days")
        }
        for site in SITES
    }
    lag_support_values: dict[str, dict[int, dict[str, list[float]]]] = {
        site: {
            lag: {
                name: []
                for name in (
                    "planned_target_hours",
                    "available_target_hours",
                    "target_unavailable_hours",
                    "complete_pair_hours",
                    "lost_edge_or_unobserved_source_hours",
                    "complete_pair_days",
                )
            }
            for lag in scenario.candidate_lags
        }
        for site in SITES
    }
    mask_hashes: list[tuple[str, str, str]] = []
    draw_digest = hashlib.sha256(b"two-site-all-replication-draws-v1\0")
    generated_draws = 0
    matched_observed: list[int] = []
    matched_planned_empty: list[int] = []
    unmatched_observed: dict[str, list[int]] = {site: [] for site in SITES}
    unmatched_planned_empty: dict[str, list[int]] = {site: [] for site in SITES}
    partial_coverage_values: dict[str, list[float]] = {site: [] for site in SITES}
    partial_hours_per_replication: dict[str, list[int]] = {site: [] for site in SITES}
    target_clock_partial_hours_per_replication: dict[str, list[int]] = {site: [] for site in SITES}

    for replication in range(plan.simulations):
        synthetic = simulate_two_site_series(
            scenario,
            seed=_seed(plan.seed, scenario.scenario_id, f"replication-{replication}"),
        )
        mask_hashes.append(
            (
                synthetic.coverage_mask_sha256,
                synthetic.wind_mask_sha256,
                synthetic.design_mask_sha256,
            )
        )
        frame = synthetic.frame
        frames = {site: frame.loc[frame["site"] == site].copy() for site in SITES}
        common_dates = set(
            frames[SITES[0]]
            .loc[
                frames[SITES[0]]["matched_date"] & frames[SITES[0]]["target"].notna(),
                "date_id",
            ]
            .unique()
        ) & set(
            frames[SITES[1]]
            .loc[
                frames[SITES[1]]["matched_date"] & frames[SITES[1]]["target"].notna(),
                "date_id",
            ]
            .unique()
        )
        matched_observed.append(len(common_dates))
        matched_planned_empty.append(scenario.matched_days - len(common_dates))
        for site in SITES:
            declared_unmatched = frames[site].loc[
                ~frames[site]["matched_date"] & frames[site]["target"].notna(),
                "date_id",
            ]
            unmatched_observed[site].append(int(declared_unmatched.nunique()))
            planned_site_only_dates = (
                frames[site].loc[~frames[site]["matched_date"], "date_id"].nunique()
            )
            unmatched_planned_empty[site].append(
                int(planned_site_only_dates - declared_unmatched.nunique())
            )
            target_clock_partial_hours_per_replication[site].append(
                int(frames[site]["coverage_fraction"].lt(1.0).sum())
            )
        points = _point_results(frames, scenario)
        for site, lag in points.selected_lags.items():
            point_lag_counts[site][str(lag)] += 1
            selected_source_coverage = frames[site].loc[
                frames[site][f"emission_lag_{lag}"].notna(),
                f"emission_lag_{lag}_source_coverage_fraction",
            ]
            partial_coverages = selected_source_coverage.loc[
                selected_source_coverage < 1.0
            ].to_numpy(dtype=float)
            partial_coverage_values[site].extend(partial_coverages.tolist())
            partial_hours_per_replication[site].append(len(partial_coverages))
        for site, stratum_support in points.stratum_support.items():
            for name, support_value in stratum_support.items():
                support_values[site][name].append(float(support_value))
        for site, by_lag in points.lag_support.items():
            for lag, lag_support in by_lag.items():
                for name, support_value in lag_support.items():
                    lag_support_values[site][lag][name].append(float(support_value))

        bootstrap_values: dict[tuple[str, str], list[float]] = {key: [] for key in keys}
        bootstrap_component_failures: dict[tuple[str, str], Counter[str]] = {
            key: Counter() for key in keys
        }
        rng = np.random.default_rng(
            _seed(plan.seed, scenario.scenario_id, f"bootstrap-{replication}")
        )
        try:
            for draw in range(plan.bootstrap_resamples):
                draw_plan = make_joint_resample_plan(
                    frame,
                    rng=rng,
                    method=plan.day_resampling,
                    block_length=plan.day_block_length,
                )
                generated_draws += 1
                draw_digest.update(f"{replication}|{draw}|{draw_plan.sha256}\n".encode())
                sampled_frames: dict[str, pd.DataFrame] = {}
                for site in SITES:
                    sampled_frame = frame.loc[draw_plan.indices_by_site[site]].copy()
                    sampled_frame["date_id"] = draw_plan.cluster_labels_by_site[site]
                    sampled_frames[site] = sampled_frame
                sampled = _point_results(sampled_frames, scenario)
                for site, lag in sampled.selected_lags.items():
                    bootstrap_lag_counts[site][str(lag)] += 1
                for site in SITES:
                    if site in draw_plan.failures_by_site:
                        reason = draw_plan.failures_by_site[site]
                        for rq in ("RQ1", "RQ2", "RQ3"):
                            sampled.failures[(rq, site)] = reason
                            sampled.values.pop((rq, site), None)
                            bootstrap_component_failures[(rq, site)][reason] += 1
                        sampled.values.pop(("H_E", "cross_site"), None)
                        cross_reason = f"site-specific resampling failure at {site}: {reason}"
                        sampled.failures[("H_E", "cross_site")] = cross_reason
                        bootstrap_component_failures[("H_E", "cross_site")][cross_reason] += 1
                    rq2_key = ("RQ2", site)
                    if rq2_key in sampled.failures:
                        rq2_invalid_resamples[site][sampled.failures[rq2_key]] += 1
                for key, statistic_value in sampled.values.items():
                    bootstrap_values[key].append(statistic_value)
        except ValueError as error:
            reason = str(error)
            draw_digest.update(f"{replication}|not_generated|{reason}\n".encode())
            for key in keys:
                failures[key][reason] += 1
            continue

        alpha = plan.adjusted_alpha
        for key in keys:
            point = points.values.get(key)
            if point is None:
                failures[key][points.failures.get(key, "point estimand unavailable")] += 1
                continue
            samples = np.asarray(bootstrap_values[key], dtype=float)
            samples = samples[np.isfinite(samples)]
            valid_resample_counts[key].append(len(samples))
            if len(samples) < 19:
                if bootstrap_component_failures[key]:
                    detail = _counter_json(bootstrap_component_failures[key])
                    failures[key][
                        f"fewer than 19 valid joint bootstrap estimates ({len(samples)}); "
                        f"component_failures={detail}"
                    ] += 1
                else:
                    failures[key][
                        f"fewer than 19 valid joint bootstrap estimates ({len(samples)})"
                    ] += 1
                continue
            lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
            lower = float(lower)
            upper = float(upper)
            successes[key] += 1
            estimates[key].append(point)
            widths[key].append(upper - lower)
            covered[key] += int(lower <= references.point.values[key] <= upper)
            rq, _ = key
            category = classify_interval_decision(
                rq,
                lower=lower,
                upper=upper,
                samples=samples,
                adjusted_alpha=alpha,
                rq1_threshold=plan.rq1_practical_threshold,
                rq2_threshold=plan.rq2_practical_threshold,
                rq3_margin=plan.rq3_practical_margin,
            )
            decisions[key][category] += 1

    simulations = plan.simulations
    mask_digest = hashlib.sha256(
        json.dumps(mask_hashes, separators=(",", ":")).encode()
    ).hexdigest()
    coverage_mask_digest = hashlib.sha256(
        json.dumps([values[0] for values in mask_hashes], separators=(",", ":")).encode()
    ).hexdigest()
    wind_mask_digest = hashlib.sha256(
        json.dumps([values[1] for values in mask_hashes], separators=(",", ":")).encode()
    ).hexdigest()
    scenario_hash = hashlib.sha256(
        json.dumps(asdict(scenario), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan_hash = hashlib.sha256(
        json.dumps(asdict(plan), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rows: list[dict[str, object]] = []
    total_bootstrap_requested = simulations * plan.bootstrap_resamples
    for key in keys:
        rq, site = key
        successful = successes[key]
        failed = simulations - successful
        coverage = _probability(covered[key], simulations)
        not_estimable = _probability(failed, simulations)
        row: dict[str, object] = {
            "evidence_status": "synthetic_design_simulation_only",
            "design_scope": "two_site_candidate_design_not_method_freeze",
            "plan_id": plan.plan_id,
            "scenario_id": scenario.scenario_id,
            "dependence_label": scenario.dependence_label,
            "design_axis_label": scenario.design_axis_label,
            "rq": rq,
            "site": site,
            "estimand": {
                "RQ1": "site_specific_emission_index_association",
                "RQ2": "wind_moderation_contrast",
                "RQ3": "association_advantage_over_counts",
                "H_E": "cross_site_association_heterogeneity",
            }[rq],
            "reference_estimand": references.point.values[key],
            "reference_estimand_mcse": references.mcse[key],
            "reference_valid_batches": references.valid_batches[key],
            "reference_batches_requested": references.batches_requested,
            "reference_uncertainty_method": (
                "standard_error_across_independent_full_size_synthetic_reference_batches"
            ),
            "reference_support_by_site_json": json.dumps(
                references.support_by_site,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "reference_precision_status": "smoke_only_not_decision_quality",
            "simulations_requested": simulations,
            "simulations_successful": successful,
            "simulations_failed": failed,
            "probability_denominator": "all_requested_replications",
            "failure_reasons_json": _counter_json(failures[key]),
            "not_estimable_probability": not_estimable,
            "not_estimable_probability_mcse": _mcse(not_estimable, simulations),
            "median_estimate": float(np.median(estimates[key])) if estimates[key] else math.nan,
            "median_ci_width": float(np.median(widths[key])) if widths[key] else math.nan,
            "median_valid_bootstrap_resamples": (
                float(np.median(valid_resample_counts[key])) if valid_resample_counts[key] else 0.0
            ),
            "coverage_probability": coverage,
            "coverage_probability_mcse": _mcse(coverage, simulations),
            "decision_probabilities_json": json.dumps(
                {
                    **{name: count / simulations for name, count in sorted(decisions[key].items())},
                    "not_estimable": not_estimable,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            "matched_days_planned": scenario.matched_days,
            "median_matched_days_observed_both_sites": float(np.median(matched_observed)),
            "median_matched_planned_empty_presence_positions": float(
                np.median(matched_planned_empty)
            ),
            "sukiyabashi_unmatched_days_planned": scenario.sites[0].unmatched_days,
            "shinjuku_unmatched_days_planned": scenario.sites[1].unmatched_days,
            "median_sukiyabashi_unmatched_days_observed": float(
                np.median(unmatched_observed["Sukiyabashi"])
            ),
            "median_shinjuku_unmatched_days_observed": float(
                np.median(unmatched_observed["Shinjuku"])
            ),
            "median_sukiyabashi_unmatched_planned_empty_positions": float(
                np.median(unmatched_planned_empty["Sukiyabashi"])
            ),
            "median_shinjuku_unmatched_planned_empty_positions": float(
                np.median(unmatched_planned_empty["Shinjuku"])
            ),
            "cross_site_correlation": scenario.cross_site_correlation,
            "between_day_phi": scenario.between_day_phi,
            "within_day_phi": scenario.within_day_phi,
            "day_effect_fraction": scenario.day_effect_fraction,
            "missing_day_rate": scenario.missing_day_rate,
            "partial_hour_rate": scenario.partial_hour_rate,
            "partial_coverage_min": scenario.partial_coverage_min,
            "day_resampling": plan.day_resampling,
            "day_block_length": plan.day_block_length,
            "moving_block_short_pool_policy": (
                "site_estimand_not_estimable_no_shortening_no_fallback"
            ),
            "day_cluster_universe": (
                "planned_local_date_timeline_including_explicit_empty_presence_positions"
            ),
            "cross_site_draw_sha256": draw_digest.hexdigest(),
            "joint_resample_draws_generated": generated_draws,
            "joint_resample_draws_requested": total_bootstrap_requested,
            "bootstrap_resamples_per_replication": plan.bootstrap_resamples,
            "interval_method_requested": "joint_day_cluster_percentile",
            "fallback_policy": "error_no_fallback",
            "confidence_level": plan.confidence_level,
            "multiplicity_method_candidate": (
                "bonferroni" if plan.multiplicity_family_size > 1 else "none"
            ),
            "multiplicity_family_size": plan.multiplicity_family_size,
            "adjusted_alpha": plan.adjusted_alpha,
            "reference_draws_candidate": plan.reference_draws,
            "coverage_rule": scenario.coverage_rule,
            "coverage_threshold_candidate": scenario.coverage_threshold,
            "missingness_pattern": scenario.missingness_pattern,
            "predictor_coverage_normalization": (
                "raw_covered_divided_by_same_coverage_fraction_for_count_and_emission"
                if scenario.coverage_rule == "threshold_normalize"
                else "complete_hours_only_no_normalization"
            ),
            "coverage_wind_mask_commitment_sha256": mask_digest,
            "coverage_mask_commitment_sha256": coverage_mask_digest,
            "wind_mask_commitment_sha256": wind_mask_digest,
            "mask_generation_order": "coverage_and_wind_before_independent_value_rng",
            "lag_rule_candidate": scenario.lag_rule,
            "fixed_lag_hours_candidate": scenario.fixed_lag_hours,
            "candidate_lags_json": json.dumps(scenario.candidate_lags, separators=(",", ":")),
            "lag_edge_policy": scenario.lag_edge_policy,
            "lag_candidates_constructed_from": (
                "one_observed_source_process_per_site_date_shifted_within_date"
            ),
            "rq3_predictor_window_alignment": (
                "emission_and_count_share_shifted_source_key_coverage_and_covered_seconds"
            ),
            "selection_functional": (
                "fixed_prespecified_lag"
                if scenario.lag_rule == "fixed"
                else "max_spearman_reselected_in_every_joint_bootstrap_draw"
            ),
            "observational_unit": "one_synthetic_actual_hourly_monitor_observation",
            "target_interpolation_used": False,
            "correlation_input_scale": "latent_gaussian_pearson",
            "equivalence_rule_candidate": (
                "family_adjusted_tost_interval_same_joint_draws"
                if rq == "RQ3"
                else "not_applicable"
            ),
            "min_stratum_hours_candidate": scenario.min_stratum_hours,
            "min_stratum_days_candidate": scenario.min_stratum_days,
            "min_site_days_candidate": scenario.min_site_days,
            "point_lag_selection_frequencies_json": (
                _counter_json(point_lag_counts[site])
                if site in SITES
                else json.dumps(
                    {name: dict(sorted(point_lag_counts[name].items())) for name in SITES},
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ),
            "bootstrap_lag_selection_frequencies_json": (
                _counter_json(bootstrap_lag_counts[site])
                if site in SITES
                else json.dumps(
                    {name: dict(sorted(bootstrap_lag_counts[name].items())) for name in SITES},
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ),
            "target_tie_probabilities_json": (
                json.dumps(
                    next(
                        design.target_tie_probabilities
                        for design in scenario.sites
                        if design.site == site
                    ),
                    separators=(",", ":"),
                )
                if site in SITES
                else "{}"
            ),
            "count_tie_probabilities_json": (
                json.dumps(
                    next(
                        design.count_tie_probabilities
                        for design in scenario.sites
                        if design.site == site
                    ),
                    separators=(",", ":"),
                )
                if site in SITES
                else "{}"
            ),
            "scenario_sha256": scenario_hash,
            "plan_sha256": plan_hash,
            "seed": plan.seed,
        }
        if site in SITES:
            site_design = next(design for design in scenario.sites if design.site == site)
            support_summary = support_values[site]
            row.update(
                {
                    f"median_{name}": float(np.median(values)) if values else math.nan
                    for name, values in support_summary.items()
                }
            )
            row["support_summary_probability_denominator"] = "all_requested_replications"
            row["lag_support_summary_json"] = json.dumps(
                {
                    str(lag): {
                        f"median_{name}": float(np.median(values)) if values else math.nan
                        for name, values in by_measure.items()
                    }
                    for lag, by_measure in lag_support_values[site].items()
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            invalid_count = sum(rq2_invalid_resamples[site].values())
            row["rq2_invalid_stratum_resample_probability"] = (
                invalid_count / total_bootstrap_requested if total_bootstrap_requested else math.nan
            )
            row["rq2_invalid_stratum_probability_denominator"] = (
                "all_requested_joint_bootstrap_draws"
            )
            row["rq2_invalid_stratum_reasons_json"] = _counter_json(rq2_invalid_resamples[site])
            row["median_retained_partial_coverage"] = (
                float(np.median(partial_coverage_values[site]))
                if partial_coverage_values[site]
                else math.nan
            )
            row["median_retained_partial_hours_per_replication"] = float(
                np.median(partial_hours_per_replication[site])
            )
            row["median_target_clock_partial_hours_per_replication"] = float(
                np.median(target_clock_partial_hours_per_replication[site])
            )
            row["partial_coverage_summary_window"] = "selected_lag_shifted_predictor_source_window"
            row["latent_pearson_rho_emission_target_favourable"] = (
                site_design.rho_emission_target_favourable
            )
            row["latent_pearson_rho_emission_target_opposing"] = (
                site_design.rho_emission_target_opposing
            )
            row["latent_pearson_rho_count_target"] = site_design.rho_count_target
            row["latent_pearson_rho_emission_count"] = site_design.rho_emission_count
        else:
            for name in (
                "favourable_hours",
                "favourable_days",
                "opposing_hours",
                "opposing_days",
            ):
                row[f"median_{name}"] = math.nan
            row["rq2_invalid_stratum_resample_probability"] = math.nan
            row["rq2_invalid_stratum_reasons_json"] = "{}"
            row["rq2_invalid_stratum_probability_denominator"] = "not_applicable"
            row["median_retained_partial_coverage"] = math.nan
            row["median_retained_partial_hours_per_replication"] = math.nan
            row["median_target_clock_partial_hours_per_replication"] = math.nan
            row["partial_coverage_summary_window"] = "not_applicable"
            row["latent_pearson_rho_emission_target_favourable"] = math.nan
            row["latent_pearson_rho_emission_target_opposing"] = math.nan
            row["latent_pearson_rho_count_target"] = math.nan
            row["latent_pearson_rho_emission_count"] = math.nan
            row["support_summary_probability_denominator"] = "not_applicable"
            row["lag_support_summary_json"] = "{}"
        rows.append(row)
    return pd.DataFrame(rows)
