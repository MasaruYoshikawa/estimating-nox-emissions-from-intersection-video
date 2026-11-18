"""Seeded design simulations for RQ1 and RQ3 precision.

The formal estimands and terminology are recorded in ``coordination/GLOSSARY.md``.
This module evaluates operating characteristics under prespecified synthetic
scenarios.  It neither loads ambient-monitor data nor produces empirical
research results.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm, rankdata

IntervalMethod = Literal["percentile", "bca"]
FallbackPolicy = Literal["error", "percentile"]
MultiplicityMethod = Literal["none", "bonferroni"]


@dataclass(frozen=True)
class SimulationScenario:
    """One prespecified synthetic data-generating scenario.

    Correlation inputs are latent Spearman correlations before target/count
    quantisation.  The simulation separately estimates the post-quantisation
    population estimands used for coverage calculations.
    """

    scenario_id: str
    days: int
    hours_per_day: int
    rho_emission_target: float
    rho_count_target: float
    rho_emission_count: float
    ar1_phi: float
    day_effect_fraction: float
    missing_day_rate: float
    missing_hour_rate: float
    target_levels: int | None
    count_levels: int | None

    def validate(self) -> None:
        """Raise ``ValueError`` when a scenario is internally inconsistent."""

        if not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        if self.days < 4:
            raise ValueError("at least four planned day clusters are required")
        if self.hours_per_day < 2:
            raise ValueError("hours_per_day must be at least two")
        for name in (
            "rho_emission_target",
            "rho_count_target",
            "rho_emission_count",
        ):
            value = getattr(self, name)
            if not -0.95 <= value <= 0.95:
                raise ValueError(f"{name} must be in [-0.95, 0.95]")
        if not 0.0 <= self.ar1_phi < 0.99:
            raise ValueError("ar1_phi must be in [0, 0.99)")
        if not 0.0 <= self.day_effect_fraction < 1.0:
            raise ValueError("day_effect_fraction must be in [0, 1)")
        for name in ("missing_day_rate", "missing_hour_rate"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("target_levels", "count_levels"):
            value = getattr(self, name)
            if value is not None and value < 3:
                raise ValueError(f"{name} must be None or at least three")
        _latent_correlation_matrix(self)


@dataclass(frozen=True)
class SimulationPlan:
    """Monte Carlo and inferential settings for a design evaluation."""

    simulations: int = 500
    bootstrap_resamples: int = 999
    confidence_level: float = 0.95
    interval_method: IntervalMethod = "percentile"
    fallback_policy: FallbackPolicy = "error"
    multiplicity_method: MultiplicityMethod = "none"
    family_size: int = 1
    rq1_practical_threshold: float = 0.40
    rq3_practical_margin: float = 0.10
    reference_draws: int = 200_000
    seed: int = 0

    def validate(self) -> None:
        """Raise ``ValueError`` when simulation settings are unsafe."""

        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if self.bootstrap_resamples < 19:
            raise ValueError("bootstrap_resamples must be at least 19")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
        if self.interval_method not in ("percentile", "bca"):
            raise ValueError("unsupported interval_method")
        if self.fallback_policy not in ("error", "percentile"):
            raise ValueError("unsupported fallback_policy")
        if self.multiplicity_method not in ("none", "bonferroni"):
            raise ValueError("unsupported multiplicity_method")
        if self.family_size < 1:
            raise ValueError("family_size must be positive")
        if self.multiplicity_method == "none" and self.family_size != 1:
            raise ValueError("family_size must be one when multiplicity_method is none")
        if self.rq1_practical_threshold <= 0:
            raise ValueError("rq1_practical_threshold must be positive")
        if self.rq3_practical_margin <= 0:
            raise ValueError("rq3_practical_margin must be positive")
        if self.reference_draws < 1_000:
            raise ValueError("reference_draws must be at least 1000")

    @property
    def nominal_alpha(self) -> float:
        """Return the two-sided nominal type-I error probability."""

        return 1.0 - self.confidence_level

    @property
    def adjusted_alpha(self) -> float:
        """Return the per-estimand alpha for the candidate procedure."""

        if self.multiplicity_method == "bonferroni":
            return self.nominal_alpha / self.family_size
        return self.nominal_alpha


@dataclass(frozen=True)
class IntervalResult:
    """Point estimate and explicit cluster-bootstrap interval metadata."""

    estimate: float
    lower: float
    upper: float
    method_requested: str
    method_used: str
    fallback_used: bool
    fallback_reason: str | None
    valid_resamples: int
    requested_resamples: int
    cluster_count: int
    resample_index_sha256: str


@dataclass(frozen=True)
class BootstrapIntervalAttempts:
    """Per-estimand interval outcomes from one shared bootstrap draw plan.

    Common design errors (for example, fewer than four observed day clusters)
    still raise ``ValueError``.  Once shared resamples can be generated, an
    undefined interval is recorded only against the affected estimand so that
    one failed comparator cannot erase another valid result.
    """

    intervals: dict[str, IntervalResult]
    failure_reasons: dict[str, str]
    resample_index_sha256: str
    cluster_count: int


def _canonical_interval_method(method: IntervalMethod) -> str:
    return f"cluster_{method}"


def _spearman_to_gaussian(rho: float) -> float:
    """Convert a Gaussian-copula Spearman correlation to latent Pearson."""

    return 2.0 * math.sin(math.pi * rho / 6.0)


def _latent_correlation_matrix(scenario: SimulationScenario) -> NDArray[np.float64]:
    correlations = np.array(
        [
            _spearman_to_gaussian(scenario.rho_emission_target),
            _spearman_to_gaussian(scenario.rho_count_target),
            _spearman_to_gaussian(scenario.rho_emission_count),
        ],
        dtype=float,
    )
    matrix = np.array(
        [
            [1.0, correlations[0], correlations[1]],
            [correlations[0], 1.0, correlations[2]],
            [correlations[1], correlations[2], 1.0],
        ],
        dtype=float,
    )
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
    if minimum_eigenvalue <= 1e-10:
        raise ValueError(
            "latent pairwise correlations do not form a positive-definite matrix "
            f"(minimum eigenvalue={minimum_eigenvalue:.6g})"
        )
    return matrix


def _quantise_latent(values: NDArray[np.float64], levels: int | None) -> NDArray[np.float64]:
    if levels is None:
        return values.copy()
    uniform = norm.cdf(values)
    # norm.cdf may round to exactly one for an extreme draw.
    return np.minimum(np.floor(uniform * levels), levels - 1).astype(float)


def _correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    ranked_x = rankdata(x, method="average")
    ranked_y = rankdata(y, method="average")
    ranked_x -= ranked_x.mean()
    ranked_y -= ranked_y.mean()
    denominator = math.sqrt(float(ranked_x @ ranked_x) * float(ranked_y @ ranked_y))
    if denominator == 0.0:
        return math.nan
    return float((ranked_x @ ranked_y) / denominator)


def _statistics(
    target: NDArray[np.float64],
    emission: NDArray[np.float64],
    count: NDArray[np.float64],
) -> NDArray[np.float64]:
    rho_emission = _correlation(emission, target)
    rho_count = _correlation(count, target)
    return np.array([rho_emission, rho_emission - rho_count], dtype=float)


def simulate_site_series(
    scenario: SimulationScenario,
    *,
    seed: int,
) -> pd.DataFrame:
    """Generate paired hourly synthetic data with independent day clusters.

    The three series share a contemporaneous Gaussian-copula dependence
    structure.  Within each day, a stationary AR(1) component and a correlated
    day effect induce dependence.  Missingness is drawn without reference to
    any generated value and the same retained rows are used by all estimators.
    """

    scenario.validate()
    rng = np.random.default_rng(seed)
    correlation_matrix = _latent_correlation_matrix(scenario)
    cholesky = np.linalg.cholesky(correlation_matrix)
    rows: list[tuple[int, int, float, float, float]] = []
    ar_scale = math.sqrt(1.0 - scenario.ar1_phi**2)
    within_scale = math.sqrt(1.0 - scenario.day_effect_fraction)
    day_scale = math.sqrt(scenario.day_effect_fraction)

    for day in range(scenario.days):
        innovations = rng.normal(size=(scenario.hours_per_day, 3)) @ cholesky.T
        within = np.empty_like(innovations)
        within[0] = innovations[0]
        for hour in range(1, scenario.hours_per_day):
            within[hour] = scenario.ar1_phi * within[hour - 1] + ar_scale * innovations[hour]
        day_effect = rng.normal(size=3) @ cholesky.T
        latent = within_scale * within + day_scale * day_effect
        for hour in range(scenario.hours_per_day):
            rows.append((day, hour, latent[hour, 0], latent[hour, 1], latent[hour, 2]))

    frame = pd.DataFrame(rows, columns=["day", "hour", "target", "emission", "count"])
    frame["target"] = _quantise_latent(frame["target"].to_numpy(), scenario.target_levels)
    frame["count"] = _quantise_latent(frame["count"].to_numpy(), scenario.count_levels)

    missing_days = rng.random(scenario.days) < scenario.missing_day_rate
    missing_hours = rng.random(len(frame)) < scenario.missing_hour_rate
    retained = ~missing_days[frame["day"].to_numpy(dtype=int)] & ~missing_hours
    return frame.loc[retained].reset_index(drop=True)


def _cluster_resample_indices(
    clusters: NDArray[np.int64],
    *,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[list[NDArray[np.int64]], str]:
    unique_clusters = np.unique(clusters)
    members = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    indices: list[NDArray[np.int64]] = []
    digest = hashlib.sha256()
    for _ in range(resamples):
        selected = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sampled = np.concatenate([members[cluster] for cluster in selected]).astype(np.int64)
        indices.append(sampled)
        digest.update(np.asarray(selected, dtype="<i8").tobytes())
    return indices, digest.hexdigest()


def _adjusted_percentiles(
    point: float,
    bootstrap_values: NDArray[np.float64],
    jackknife_values: NDArray[np.float64],
    alpha: float,
) -> tuple[float, float]:
    """Return BCa endpoints or raise when acceleration is not estimable."""

    valid_bootstrap = bootstrap_values[np.isfinite(bootstrap_values)]
    valid_jackknife = jackknife_values[np.isfinite(jackknife_values)]
    if valid_bootstrap.size < 19:
        raise ValueError("fewer than 19 finite bootstrap estimates")
    if valid_jackknife.size < 4:
        raise ValueError("fewer than four finite delete-one-cluster estimates")

    below = int(np.count_nonzero(valid_bootstrap < point))
    equal = int(np.count_nonzero(valid_bootstrap == point))
    probability = (below + 0.5 * equal) / valid_bootstrap.size
    continuity = 0.5 / valid_bootstrap.size
    probability = float(np.clip(probability, continuity, 1.0 - continuity))
    bias_correction = float(norm.ppf(probability))

    jackknife_mean = float(valid_jackknife.mean())
    differences = jackknife_mean - valid_jackknife
    squared_sum = float(np.sum(differences**2))
    if squared_sum <= np.finfo(float).eps:
        raise ValueError("delete-one-cluster acceleration is degenerate")
    acceleration = float(np.sum(differences**3) / (6.0 * math.pow(squared_sum, 1.5)))

    endpoints: list[float] = []
    for probability_endpoint in (alpha / 2.0, 1.0 - alpha / 2.0):
        z_alpha = float(norm.ppf(probability_endpoint))
        denominator = 1.0 - acceleration * (bias_correction + z_alpha)
        if abs(denominator) <= 1e-12:
            raise ValueError("BCa endpoint denominator is zero")
        adjusted = norm.cdf(bias_correction + (bias_correction + z_alpha) / denominator)
        if not 0.0 < adjusted < 1.0:
            raise ValueError("BCa adjusted percentile lies outside (0, 1)")
        endpoints.append(float(adjusted))
    return endpoints[0], endpoints[1]


def _interval_from_samples(
    *,
    point: float,
    bootstrap_values: NDArray[np.float64],
    jackknife_values: NDArray[np.float64] | None,
    alpha: float,
    method: IntervalMethod,
    fallback_policy: FallbackPolicy,
) -> tuple[float, float, str, bool, str | None]:
    valid = bootstrap_values[np.isfinite(bootstrap_values)]
    if valid.size < 19:
        raise ValueError("fewer than 19 finite bootstrap estimates")
    if method == "percentile":
        lower, upper = np.quantile(valid, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lower), float(upper), "cluster_percentile", False, None

    try:
        if jackknife_values is None:
            raise ValueError("BCa requires delete-one-cluster estimates")
        lower_probability, upper_probability = _adjusted_percentiles(
            point,
            valid,
            jackknife_values,
            alpha,
        )
        lower, upper = np.quantile(valid, [lower_probability, upper_probability])
        return float(lower), float(upper), "cluster_bca", False, None
    except ValueError as error:
        if fallback_policy == "error":
            raise
        lower, upper = np.quantile(valid, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lower), float(upper), "cluster_percentile", True, str(error)


def cluster_bootstrap_interval_attempts(
    frame: pd.DataFrame,
    *,
    resamples: int,
    confidence_level: float,
    method: IntervalMethod,
    fallback_policy: FallbackPolicy,
    multiplicity_method: MultiplicityMethod = "none",
    family_size: int = 1,
    seed: int = 0,
) -> BootstrapIntervalAttempts:
    """Attempt RQ1 and paired RQ3 intervals using shared day resamples."""

    required = {"day", "target", "emission", "count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
    complete = frame.dropna(subset=sorted(required)).copy()
    clusters = complete["day"].to_numpy(dtype=np.int64)
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 4:
        raise ValueError("at least four observed day clusters are required")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0.5, 1)")
    if resamples < 19:
        raise ValueError("resamples must be at least 19")
    if method not in ("percentile", "bca"):
        raise ValueError("unsupported interval method")
    if fallback_policy not in ("error", "percentile"):
        raise ValueError("unsupported fallback policy")
    if multiplicity_method == "none":
        if family_size != 1:
            raise ValueError("family_size must be one without multiplicity adjustment")
        adjusted_alpha = 1.0 - confidence_level
    elif multiplicity_method == "bonferroni":
        if family_size < 1:
            raise ValueError("family_size must be positive")
        adjusted_alpha = (1.0 - confidence_level) / family_size
    else:
        raise ValueError("unsupported multiplicity_method")

    target = complete["target"].to_numpy(dtype=float)
    emission = complete["emission"].to_numpy(dtype=float)
    count = complete["count"].to_numpy(dtype=float)
    points = _statistics(target, emission, count)

    rng = np.random.default_rng(seed)
    resample_indices, index_hash = _cluster_resample_indices(
        clusters,
        resamples=resamples,
        rng=rng,
    )
    bootstrap = np.array(
        [_statistics(target[index], emission[index], count[index]) for index in resample_indices]
    )

    jackknife: NDArray[np.float64] | None = None
    if method == "bca":
        jackknife = np.array(
            [
                _statistics(
                    target[clusters != cluster],
                    emission[clusters != cluster],
                    count[clusters != cluster],
                )
                for cluster in unique_clusters
            ]
        )

    results: dict[str, IntervalResult] = {}
    failure_reasons: dict[str, str] = {}
    for column, estimand in enumerate(("rq1_rho", "rq3_delta")):
        if not math.isfinite(float(points[column])):
            if estimand == "rq1_rho":
                reason = "point estimate is undefined because target or emission is degenerate"
            else:
                reason = (
                    "point estimate is undefined because target, emission, or count is degenerate"
                )
            failure_reasons[estimand] = reason
            continue
        jackknife_values = None if jackknife is None else jackknife[:, column]
        try:
            lower, upper, method_used, fallback_used, fallback_reason = _interval_from_samples(
                point=float(points[column]),
                bootstrap_values=bootstrap[:, column],
                jackknife_values=jackknife_values,
                alpha=adjusted_alpha,
                method=method,
                fallback_policy=fallback_policy,
            )
        except ValueError as error:
            failure_reasons[estimand] = str(error)
            continue
        results[estimand] = IntervalResult(
            estimate=float(points[column]),
            lower=lower,
            upper=upper,
            method_requested=_canonical_interval_method(method),
            method_used=method_used,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            valid_resamples=int(np.isfinite(bootstrap[:, column]).sum()),
            requested_resamples=resamples,
            cluster_count=len(unique_clusters),
            resample_index_sha256=index_hash,
        )
    return BootstrapIntervalAttempts(
        intervals=results,
        failure_reasons=failure_reasons,
        resample_index_sha256=index_hash,
        cluster_count=len(unique_clusters),
    )


def cluster_bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    resamples: int,
    confidence_level: float,
    method: IntervalMethod,
    fallback_policy: FallbackPolicy,
    multiplicity_method: MultiplicityMethod = "none",
    family_size: int = 1,
    seed: int = 0,
) -> dict[str, IntervalResult]:
    """Estimate both intervals or fail with estimand-specific reasons.

    ``cluster_bootstrap_interval_attempts`` is used by design simulations that
    must retain valid results when only one estimand is undefined.  This
    convenience API preserves the historical all-estimands-or-error contract.
    """

    attempts = cluster_bootstrap_interval_attempts(
        frame,
        resamples=resamples,
        confidence_level=confidence_level,
        method=method,
        fallback_policy=fallback_policy,
        multiplicity_method=multiplicity_method,
        family_size=family_size,
        seed=seed,
    )
    if attempts.failure_reasons:
        details = "; ".join(
            f"{estimand}: {reason}" for estimand, reason in sorted(attempts.failure_reasons.items())
        )
        raise ValueError(f"one or more estimand intervals are not estimable: {details}")
    return attempts.intervals


def _reference_estimands(
    scenario: SimulationScenario,
    *,
    draws: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    latent = rng.multivariate_normal(
        mean=np.zeros(3),
        cov=_latent_correlation_matrix(scenario),
        size=draws,
    )
    target = _quantise_latent(latent[:, 0], scenario.target_levels)
    count = _quantise_latent(latent[:, 2], scenario.count_levels)
    estimands = _statistics(target, latent[:, 1], count)

    # Independent chunks quantify the numerical error of the synthetic
    # reference instead of treating a finite Monte Carlo value as exact.
    chunk_count = min(20, draws // 1_000)
    chunk_size = draws // chunk_count
    chunk_estimates = np.array(
        [
            _statistics(
                target[start : start + chunk_size],
                latent[start : start + chunk_size, 1],
                count[start : start + chunk_size],
            )
            for start in range(0, chunk_count * chunk_size, chunk_size)
        ]
    )
    if chunk_count > 1:
        reference_mcse = np.std(chunk_estimates, axis=0, ddof=1) / math.sqrt(chunk_count)
    else:
        reference_mcse = np.full(2, math.nan)
    return estimands, reference_mcse


def _equivalence_interval_attempts(
    frame: pd.DataFrame,
    *,
    resamples: int,
    alpha: float,
    method: IntervalMethod,
    fallback_policy: FallbackPolicy,
    seed: int,
) -> BootstrapIntervalAttempts:
    # TOST at one-sided alpha is equivalent to checking a 1 - 2*alpha
    # two-sided interval against both margins.
    confidence_level = 1.0 - 2.0 * alpha
    if confidence_level <= 0.5:
        raise ValueError("adjusted equivalence confidence level must exceed 0.5")
    return cluster_bootstrap_interval_attempts(
        frame,
        resamples=resamples,
        confidence_level=confidence_level,
        method=method,
        fallback_policy=fallback_policy,
        seed=seed,
    )


def _mcse(probability: float, replications: int) -> float:
    if replications <= 0 or not math.isfinite(probability):
        return math.nan
    return math.sqrt(probability * (1.0 - probability) / replications)


def _safe_probability(values: Sequence[bool]) -> float:
    if not values:
        return math.nan
    return float(np.mean(values))


def _count_probability(count: int, replications: int) -> float:
    if replications <= 0:
        return math.nan
    return count / replications


def _counter_json(counter: Counter[str]) -> str:
    return json.dumps(dict(sorted(counter.items())), separators=(",", ":"))


def _finite_median(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    return float(np.median(finite))


def _lag1_and_effective_hours(frame: pd.DataFrame) -> tuple[float, float]:
    """Return a diagnostic rank lag-1 correlation and AR(1) effective hours."""

    previous: list[float] = []
    current: list[float] = []
    for _, day in frame.sort_values(["day", "hour"]).groupby("day", sort=False):
        values = day["target"].to_numpy(dtype=float)
        hours = day["hour"].to_numpy(dtype=int)
        adjacent = np.diff(hours) == 1
        previous.extend(values[:-1][adjacent])
        current.extend(values[1:][adjacent])
    lag1 = _correlation(np.asarray(previous), np.asarray(current))
    if not math.isfinite(lag1):
        return math.nan, math.nan
    lag1 = float(np.clip(lag1, -0.99, 0.99))
    effective = len(frame) * (1.0 - lag1) / (1.0 + lag1)
    return lag1, float(np.clip(effective, 1.0, len(frame)))


def _scenario_seed(plan_seed: int, scenario_id: str, stream: int) -> int:
    payload = f"{plan_seed}|{scenario_id}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def run_precision_simulation(
    scenario: SimulationScenario,
    plan: SimulationPlan,
) -> pd.DataFrame:
    """Evaluate design operating characteristics for one synthetic scenario.

    Returned rows are simulation-design evidence only.  They do not estimate
    either site's empirical association.
    """

    scenario.validate()
    plan.validate()
    reference, reference_mcse = _reference_estimands(
        scenario,
        draws=plan.reference_draws,
        seed=_scenario_seed(plan.seed, scenario.scenario_id, -1),
    )
    if not np.all(np.isfinite(reference)):
        raise ValueError("reference estimand is undefined")

    coverage_metrics: dict[str, list[float]] = {
        "observed_days": [],
        "observed_hours": [],
        "coverage_fraction": [],
        "lag1_rank_autocorrelation": [],
        "effective_hours_diagnostic": [],
    }
    estimand_metrics: dict[str, dict[str, list[float]]] = {
        "RQ1": {"estimate": [], "lower": [], "upper": []},
        "RQ3": {"estimate": [], "lower": [], "upper": []},
    }
    primary_successes: Counter[str] = Counter()
    covered: Counter[str] = Counter()
    fallback_replications: Counter[str] = Counter()
    primary_failures: dict[str, Counter[str]] = {
        "RQ1": Counter(),
        "RQ3": Counter(),
    }
    decisions: dict[str, Counter[str]] = {
        "RQ1": Counter(),
        "RQ3": Counter(),
    }
    decision_failures: dict[str, Counter[str]] = {
        "RQ1": Counter(),
        "RQ3": Counter(),
    }
    equivalence_successes = 0
    equivalence_failures: Counter[str] = Counter()
    equivalence_resample_replications_verified = 0

    scenario_hash = hashlib.sha256(
        json.dumps(asdict(scenario), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan_hash = hashlib.sha256(
        json.dumps(asdict(plan), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    pairing_digest = hashlib.sha256()
    pairing_digest.update(b"vehicle-emission-paired-day-bootstrap-v1\0")
    pairing_digest.update(bytes.fromhex(scenario_hash))
    pairing_digest.update(bytes.fromhex(plan_hash))
    paired_resample_replications_generated = 0

    for replication in range(plan.simulations):
        data_seed = _scenario_seed(plan.seed, scenario.scenario_id, 2 * replication)
        bootstrap_seed = _scenario_seed(plan.seed, scenario.scenario_id, 2 * replication + 1)
        frame = simulate_site_series(scenario, seed=data_seed)

        observed_days = int(frame["day"].nunique())
        observed_hours = len(frame)
        coverage_metrics["observed_days"].append(float(observed_days))
        coverage_metrics["observed_hours"].append(float(observed_hours))
        coverage_metrics["coverage_fraction"].append(
            observed_hours / (scenario.days * scenario.hours_per_day)
        )
        lag1, effective_hours = _lag1_and_effective_hours(frame)
        coverage_metrics["lag1_rank_autocorrelation"].append(lag1)
        coverage_metrics["effective_hours_diagnostic"].append(effective_hours)

        try:
            attempts = cluster_bootstrap_interval_attempts(
                frame,
                resamples=plan.bootstrap_resamples,
                confidence_level=plan.confidence_level,
                method=plan.interval_method,
                fallback_policy=plan.fallback_policy,
                multiplicity_method=plan.multiplicity_method,
                family_size=plan.family_size,
                seed=bootstrap_seed,
            )
        except ValueError as error:
            reason = str(error)
            pairing_digest.update(
                f"{replication}|{bootstrap_seed}|not_generated|{reason}\n".encode()
            )
            for rq in ("RQ1", "RQ3"):
                primary_failures[rq][reason] += 1
                decision_failures[rq][reason] += 1
            equivalence_failures[f"primary RQ3 interval not estimable: {reason}"] += 1
            continue

        paired_resample_replications_generated += 1
        pairing_digest.update(
            (
                f"{replication}|{bootstrap_seed}|generated|" f"{attempts.resample_index_sha256}\n"
            ).encode()
        )

        rq1 = attempts.intervals.get("rq1_rho")
        if rq1 is None:
            reason = attempts.failure_reasons["rq1_rho"]
            primary_failures["RQ1"][reason] += 1
            decision_failures["RQ1"][reason] += 1
        else:
            primary_successes["RQ1"] += 1
            estimand_metrics["RQ1"]["estimate"].append(rq1.estimate)
            estimand_metrics["RQ1"]["lower"].append(rq1.lower)
            estimand_metrics["RQ1"]["upper"].append(rq1.upper)
            covered["RQ1"] += int(rq1.lower <= reference[0] <= rq1.upper)
            fallback_replications["RQ1"] += int(rq1.fallback_used)
            if rq1.lower > plan.rq1_practical_threshold:
                decisions["RQ1"]["practically_strong"] += 1
            elif 0.0 < rq1.lower <= plan.rq1_practical_threshold:
                decisions["RQ1"]["positive"] += 1
            elif rq1.upper <= 0.0:
                decisions["RQ1"]["non_positive"] += 1
            else:
                decisions["RQ1"]["inconclusive"] += 1

        rq3 = attempts.intervals.get("rq3_delta")
        if rq3 is None:
            reason = attempts.failure_reasons["rq3_delta"]
            primary_failures["RQ3"][reason] += 1
            decision_failures["RQ3"][reason] += 1
            equivalence_failures[f"primary RQ3 interval not estimable: {reason}"] += 1
            continue

        primary_successes["RQ3"] += 1
        estimand_metrics["RQ3"]["estimate"].append(rq3.estimate)
        estimand_metrics["RQ3"]["lower"].append(rq3.lower)
        estimand_metrics["RQ3"]["upper"].append(rq3.upper)
        covered["RQ3"] += int(rq3.lower <= reference[1] <= rq3.upper)
        rq3_fallback = rq3.fallback_used

        equivalence: IntervalResult | None = None
        equivalence_failure_reason: str | None = None
        try:
            equivalence_attempts = _equivalence_interval_attempts(
                frame,
                resamples=plan.bootstrap_resamples,
                alpha=plan.adjusted_alpha,
                method=plan.interval_method,
                fallback_policy=plan.fallback_policy,
                seed=bootstrap_seed,
            )
        except ValueError as error:
            equivalence_failure_reason = str(error)
        else:
            if equivalence_attempts.resample_index_sha256 != attempts.resample_index_sha256:
                raise RuntimeError(
                    "equivalence and primary intervals did not share the paired "
                    "bootstrap draw plan"
                )
            equivalence_resample_replications_verified += 1
            equivalence = equivalence_attempts.intervals.get("rq3_delta")
            if equivalence is None:
                equivalence_failure_reason = equivalence_attempts.failure_reasons["rq3_delta"]

        if equivalence is None:
            assert equivalence_failure_reason is not None
            equivalence_failures[equivalence_failure_reason] += 1
        else:
            equivalence_successes += 1
            rq3_fallback = rq3_fallback or equivalence.fallback_used
        fallback_replications["RQ3"] += int(rq3_fallback)

        if rq3.lower > plan.rq3_practical_margin:
            decisions["RQ3"]["practically_strong"] += 1
        elif rq3.upper < -plan.rq3_practical_margin:
            decisions["RQ3"]["non_positive"] += 1
        elif equivalence is None:
            decision_failures["RQ3"][
                f"equivalence interval not estimable: {equivalence_failure_reason}"
            ] += 1
        elif (
            equivalence.lower > -plan.rq3_practical_margin
            and equivalence.upper < plan.rq3_practical_margin
        ):
            decisions["RQ3"]["equivalence"] += 1
        else:
            decisions["RQ3"]["inconclusive"] += 1

    simulations = plan.simulations
    decision_successes = {
        rq: int(sum(category_counts.values())) for rq, category_counts in decisions.items()
    }
    if decision_successes["RQ1"] + sum(decision_failures["RQ1"].values()) != simulations:
        raise RuntimeError("RQ1 simulation accounting does not equal requested replications")
    if decision_successes["RQ3"] + sum(decision_failures["RQ3"].values()) != simulations:
        raise RuntimeError("RQ3 simulation accounting does not equal requested replications")
    if equivalence_successes + sum(equivalence_failures.values()) != simulations:
        raise RuntimeError("RQ3 equivalence accounting does not equal requested replications")

    common = {
        "evidence_status": "synthetic_design_simulation_only",
        "scenario_id": scenario.scenario_id,
        "planned_days": scenario.days,
        "hours_per_day": scenario.hours_per_day,
        "median_observed_days": _finite_median(coverage_metrics["observed_days"]),
        "median_observed_hours": _finite_median(coverage_metrics["observed_hours"]),
        "mean_coverage_fraction": float(np.mean(coverage_metrics["coverage_fraction"])),
        "median_lag1_rank_autocorrelation": _finite_median(
            coverage_metrics["lag1_rank_autocorrelation"]
        ),
        "median_effective_hours_diagnostic": _finite_median(
            coverage_metrics["effective_hours_diagnostic"]
        ),
        "ar1_phi": scenario.ar1_phi,
        "day_effect_fraction": scenario.day_effect_fraction,
        "missing_day_rate": scenario.missing_day_rate,
        "missing_hour_rate": scenario.missing_hour_rate,
        "target_levels": scenario.target_levels,
        "count_levels": scenario.count_levels,
        "latent_rho_emission_target": scenario.rho_emission_target,
        "latent_rho_count_target": scenario.rho_count_target,
        "latent_rho_emission_count": scenario.rho_emission_count,
        "simulations_requested": simulations,
        "probability_denominator": "all_requested_replications",
        "estimable_summary_conditioning": "estimable_replications_only",
        "bootstrap_resamples": plan.bootstrap_resamples,
        "interval_method_requested": _canonical_interval_method(plan.interval_method),
        "fallback_policy": plan.fallback_policy,
        "confidence_level": plan.confidence_level,
        "multiplicity_method": plan.multiplicity_method,
        "family_size": plan.family_size,
        "adjusted_alpha": plan.adjusted_alpha,
        "reference_draws": plan.reference_draws,
        "seed": plan.seed,
        "paired_resample_index_sha256": pairing_digest.hexdigest(),
        "paired_resample_hash_protocol": (
            "sha256_v1_ordered_replication_index_hashes_with_not_generated_sentinel"
        ),
        "paired_resample_replications_generated": paired_resample_replications_generated,
        "paired_resample_replications_not_generated": (
            simulations - paired_resample_replications_generated
        ),
        "equivalence_resample_replications_verified": (equivalence_resample_replications_verified),
        "scenario_sha256": scenario_hash,
        "plan_sha256": plan_hash,
    }

    def row_common(rq: str) -> dict[str, object]:
        successful = primary_successes[rq]
        failed = simulations - successful
        decision_successful = decision_successes[rq]
        decision_failed = simulations - decision_successful
        not_estimable = _count_probability(failed, simulations)
        decision_not_estimable = _count_probability(decision_failed, simulations)
        coverage_unconditional = _count_probability(covered[rq], simulations)
        coverage_conditional = _count_probability(covered[rq], successful)
        return {
            **common,
            "simulations_successful": successful,
            "simulations_failed": failed,
            "failure_reasons_json": _counter_json(primary_failures[rq]),
            "not_estimable_probability": not_estimable,
            "not_estimable_probability_mcse": _mcse(not_estimable, simulations),
            "decision_simulations_successful": decision_successful,
            "decision_simulations_failed": decision_failed,
            "decision_failure_reasons_json": _counter_json(decision_failures[rq]),
            "decision_not_estimable_probability": decision_not_estimable,
            "decision_not_estimable_probability_mcse": _mcse(decision_not_estimable, simulations),
            "fallback_rate": _count_probability(fallback_replications[rq], simulations),
            "fallback_rate_given_estimable": _count_probability(
                fallback_replications[rq], successful
            ),
            "median_estimate": _finite_median(estimand_metrics[rq]["estimate"]),
            "median_ci_width": _finite_median(
                [
                    upper - lower
                    for lower, upper in zip(
                        estimand_metrics[rq]["lower"],
                        estimand_metrics[rq]["upper"],
                        strict=True,
                    )
                ]
            ),
            "coverage_probability": coverage_unconditional,
            "coverage_probability_mcse": _mcse(coverage_unconditional, simulations),
            "coverage_probability_given_estimable": coverage_conditional,
            "coverage_probability_given_estimable_mcse": _mcse(coverage_conditional, successful),
        }

    rq1_strong = _count_probability(decisions["RQ1"]["practically_strong"], simulations)
    rq1_positive = _count_probability(decisions["RQ1"]["positive"], simulations)
    rq1_non_positive = _count_probability(decisions["RQ1"]["non_positive"], simulations)
    rq1_inconclusive = _count_probability(decisions["RQ1"]["inconclusive"], simulations)
    rq1_row = {
        **row_common("RQ1"),
        "rq": "RQ1",
        "estimand": "site_specific_emission_index_association",
        "reference_estimand": float(reference[0]),
        "reference_estimand_mcse": float(reference_mcse[0]),
        "equivalence_simulations_successful": math.nan,
        "equivalence_simulations_failed": math.nan,
        "equivalence_failure_reasons_json": "{}",
        "equivalence_not_estimable_probability": math.nan,
        "equivalence_not_estimable_probability_mcse": math.nan,
        "practically_strong_probability": rq1_strong,
        "positive_probability": rq1_positive,
        "non_positive_probability": rq1_non_positive,
        "equivalence_probability": math.nan,
        "inconclusive_probability": rq1_inconclusive,
        "practically_strong_probability_mcse": _mcse(rq1_strong, simulations),
        "positive_probability_mcse": _mcse(rq1_positive, simulations),
        "non_positive_probability_mcse": _mcse(rq1_non_positive, simulations),
        "equivalence_probability_mcse": math.nan,
        "inconclusive_probability_mcse": _mcse(rq1_inconclusive, simulations),
    }

    rq3_added = _count_probability(decisions["RQ3"]["practically_strong"], simulations)
    rq3_count = _count_probability(decisions["RQ3"]["non_positive"], simulations)
    rq3_equivalent = _count_probability(decisions["RQ3"]["equivalence"], simulations)
    rq3_inconclusive = _count_probability(decisions["RQ3"]["inconclusive"], simulations)
    equivalence_not_estimable = _count_probability(simulations - equivalence_successes, simulations)
    rq3_row = {
        **row_common("RQ3"),
        "rq": "RQ3",
        "estimand": "association_advantage_over_counts",
        "reference_estimand": float(reference[1]),
        "reference_estimand_mcse": float(reference_mcse[1]),
        "equivalence_simulations_successful": equivalence_successes,
        "equivalence_simulations_failed": simulations - equivalence_successes,
        "equivalence_failure_reasons_json": _counter_json(equivalence_failures),
        "equivalence_not_estimable_probability": equivalence_not_estimable,
        "equivalence_not_estimable_probability_mcse": _mcse(equivalence_not_estimable, simulations),
        "practically_strong_probability": rq3_added,
        "positive_probability": math.nan,
        "non_positive_probability": rq3_count,
        "equivalence_probability": rq3_equivalent,
        "inconclusive_probability": rq3_inconclusive,
        "practically_strong_probability_mcse": _mcse(rq3_added, simulations),
        "positive_probability_mcse": math.nan,
        "non_positive_probability_mcse": _mcse(rq3_count, simulations),
        "equivalence_probability_mcse": _mcse(rq3_equivalent, simulations),
        "inconclusive_probability_mcse": _mcse(rq3_inconclusive, simulations),
    }
    return pd.DataFrame([rq1_row, rq3_row])
