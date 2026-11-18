"""Full-calendar whole-day controls and exact-calendar autocorrelation.

These are predeclared descriptive diagnostics, not validated null generators
or significance tests. Formal terms follow ``coordination/GLOSSARY.md``.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from vehicle_emission.time_utils import TOKYO_TIMEZONE

SITES = ("Sukiyabashi", "Shinjuku")
VALUES = ("ambient_nox_ppm", "relative_nox_emission_index", "count_only_index")
UTC = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"


@dataclass(frozen=True)
class TemporalControlConfig:
    """Explicit inclusive local-date bounds and signed whole-day offsets.

    Positive offsets pair the target interval [t,t+1h) with predictors from
    [t-offset*24h,t+1h-offset*24h), modulo the complete declared calendar.
    Offsets do not estimate or change the primary physical lag.
    """

    calendar_start_date: str
    calendar_end_date: str
    day_offsets: tuple[int, ...]
    acf_lags_hours: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("calendar_start_date", "calendar_end_date"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}", getattr(self, name)
            ):
                raise ValueError("calendar bounds must be explicit ISO local dates")
        start = pd.Timestamp(self.calendar_start_date)
        end = pd.Timestamp(self.calendar_end_date)
        days = (end - start).days + 1
        if days < 2:
            raise ValueError("temporal controls require at least two full calendar days")
        if not self.day_offsets or any(
            isinstance(offset, bool) or not isinstance(offset, int) or not 0 < abs(offset) < days
            for offset in self.day_offsets
        ):
            raise ValueError("declare nonzero whole-day offsets smaller than the calendar")
        if len({offset % days for offset in self.day_offsets}) != len(self.day_offsets):
            raise ValueError("day_offsets must identify distinct circular transformations")
        if (
            not self.acf_lags_hours
            or len(set(self.acf_lags_hours)) != len(self.acf_lags_hours)
            or any(
                isinstance(lag, bool) or not isinstance(lag, int) or lag < 1
                for lag in self.acf_lags_hours
            )
        ):
            raise ValueError("declare unique positive integer ACF calendar-hour lags")


@dataclass(frozen=True)
class TemporalControlTables:
    contrasts: pd.DataFrame
    support: pd.DataFrame
    day_mapping: pd.DataFrame
    autocorrelation: pd.DataFrame


def _calendar_rows(rows: pd.DataFrame, config: TemporalControlConfig) -> pd.DataFrame:
    required = {
        "site",
        "monitor_start_utc",
        "monitor_end_utc",
        "local_calendar_date",
        "primary_measurement_eligible",
        "target_observation_status",
        *VALUES,
    }
    if rows.columns.duplicated().any() or not required.issubset(rows):
        raise ValueError(f"eligible hourly rows require unique columns: {sorted(required)}")
    frame = rows.copy()
    if frame.empty or not frame.site.isin(SITES).all():
        raise ValueError("hourly input must contain explicit supported sites")
    for column in ("monitor_start_utc", "monitor_end_utc"):
        if not frame[column].astype("string").str.fullmatch(UTC).fillna(False).all():
            raise ValueError("monitor intervals require canonical UTC timestamps")
    starts = pd.to_datetime(frame.monitor_start_utc, utc=True)
    ends = pd.to_datetime(frame.monitor_end_utc, utc=True)
    if (
        not (ends - starts).eq(pd.Timedelta(hours=1)).all()
        or not (starts.dt.minute.eq(0) & starts.dt.second.eq(0)).all()
    ):
        raise ValueError("monitor observations require half-open whole calendar-hour intervals")
    if not frame.local_calendar_date.eq(
        starts.dt.tz_convert(TOKYO_TIMEZONE).dt.strftime("%Y-%m-%d")
    ).all():
        raise ValueError("local dates must agree with explicit monitor UTC intervals")
    if (
        not frame.primary_measurement_eligible.map(
            lambda value: isinstance(value, bool | np.bool_) and bool(value)
        ).all()
        or not frame.target_observation_status.eq("actual_observation").all()
    ):
        raise ValueError("consume only actual, audited eligible hourly observations")
    if "target_is_derived" in frame and not frame.target_is_derived.eq(False).all():
        raise ValueError("derived/interpolated targets are not actual observations")
    for column in VALUES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not (np.isfinite(frame[column]) & frame[column].ge(0)).all():
            raise ValueError(f"{column} must be finite and non-negative on eligible rows")
    frame["_time"] = starts
    if frame.duplicated(["site", "_time"]).any():
        raise ValueError("duplicate eligible site-hour rows")
    calendar = pd.date_range(
        pd.Timestamp(config.calendar_start_date, tz=TOKYO_TIMEZONE),
        pd.Timestamp(config.calendar_end_date, tz=TOKYO_TIMEZONE) + pd.Timedelta(hours=23),
        freq="h",
    ).tz_convert("UTC")
    if not starts.isin(calendar).all():
        raise ValueError("eligible rows lie outside the explicitly declared full common calendar")
    index = pd.MultiIndex.from_product([SITES, calendar], names=["site", "_time"])
    return frame.set_index(["site", "_time"])[list(VALUES)].reindex(index)


def _correlation(x: np.ndarray, y: np.ndarray, *, ranks: bool = False) -> float:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return math.nan
    if ranks:
        x, y = rankdata(x), rankdata(y)
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))


def calculate_temporal_controls(
    rows: pd.DataFrame, config: TemporalControlConfig
) -> TemporalControlTables:
    """Move complete predictor days jointly; keep targets and hour order fixed.

    Both observed and control index/count contrasts use the intersection of
    observed and transformed complete-pair support, separately for each
    control/site. No row compression, filling, IID shuffling or p-value occurs.
    ACF is pairwise Pearson correlation at exact, non-wrapping calendar lags.
    """

    calendar_rows = _calendar_rows(rows, config)
    calendar = calendar_rows.loc[SITES[0]].index
    n_hours = len(calendar)
    n_days = n_hours // 24
    contrasts: list[dict[str, object]] = []
    support: list[pd.DataFrame] = []
    mapping: list[dict[str, object]] = []
    acf: list[dict[str, object]] = []
    for offset in config.day_offsets:
        source_day = (np.arange(n_days) - offset) % n_days
        for destination, source in enumerate(source_day):
            mapping.append(
                {
                    "day_offset": offset,
                    "destination_local_date": calendar[destination * 24]
                    .tz_convert(TOKYO_TIMEZONE)
                    .strftime("%Y-%m-%d"),
                    "source_local_date": calendar[source * 24]
                    .tz_convert(TOKYO_TIMEZONE)
                    .strftime("%Y-%m-%d"),
                    "wrapped": bool(destination - offset < 0 or destination - offset >= n_days),
                }
            )
        source_hour = (np.arange(n_hours) - offset * 24) % n_hours
        wrapped = (np.arange(n_hours) - offset * 24 < 0) | (
            np.arange(n_hours) - offset * 24 >= n_hours
        )
        transform_hash = hashlib.sha256(source_hour.astype("<i8").tobytes()).hexdigest()
        for site in SITES:
            observed = calendar_rows.loc[site].to_numpy(dtype=float)
            control = observed[source_hour]
            observed_valid = np.isfinite(observed).all(axis=1)
            transformed_valid = np.isfinite(control[:, 1:]).all(axis=1)
            retained = observed_valid & transformed_valid
            keys = calendar[retained].strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
            mask_hash = hashlib.sha256("\n".join(keys).encode()).hexdigest()
            target = observed[retained, 0]
            observed_index = _correlation(observed[retained, 1], target, ranks=True)
            observed_count = _correlation(observed[retained, 2], target, ranks=True)
            control_index = _correlation(control[retained, 1], target, ranks=True)
            control_count = _correlation(control[retained, 2], target, ranks=True)
            contrasts.append(
                {
                    "site": site,
                    "day_offset": offset,
                    "calendar_hours": n_hours,
                    "calendar_days": n_days,
                    "eligible_hours": int(observed_valid.sum()),
                    "retained_hours": int(retained.sum()),
                    "dropped_observed_hours": int((observed_valid & ~retained).sum()),
                    "wrapped_calendar_hours": int(wrapped.sum()),
                    "retained_wrapped_hours": int((retained & wrapped).sum()),
                    "pair_mask_sha256": mask_hash,
                    "shared_transformation_sha256": transform_hash,
                    "observed_index_rho": observed_index,
                    "observed_count_rho": observed_count,
                    "observed_index_count_contrast": observed_index - observed_count,
                    "control_index_rho": control_index,
                    "control_count_rho": control_count,
                    "control_index_count_contrast": control_index - control_count,
                    "status": (
                        "descriptive_optional_diagnostic"
                        if all(
                            math.isfinite(value)
                            for value in (
                                observed_index,
                                observed_count,
                                control_index,
                                control_count,
                            )
                        )
                        else "not_estimable"
                    ),
                }
            )
            support.append(
                pd.DataFrame(
                    {
                        "site": site,
                        "day_offset": offset,
                        "monitor_start_utc": calendar.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "monitor_end_utc": (calendar + pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "source_predictor_start_utc": calendar[source_hour].strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "observed_available": observed_valid,
                        "transformed_predictors_available": transformed_valid,
                        "retained_pair": retained,
                        "wrapped": wrapped,
                    }
                )
            )
    for site in SITES:
        for column in VALUES:
            values = calendar_rows.loc[site, column].to_numpy(dtype=float)
            for lag in config.acf_lags_hours:
                earlier, later = values[:-lag], values[lag:]
                valid = np.isfinite(earlier) & np.isfinite(later)
                acf.append(
                    {
                        "site": site,
                        "series": column,
                        "lag_hours": lag,
                        "calendar_candidate_pairs": max(n_hours - lag, 0),
                        "exact_calendar_pairs": int(valid.sum()),
                        "pearson_autocorrelation": _correlation(earlier[valid], later[valid]),
                    }
                )
    return TemporalControlTables(
        pd.DataFrame(contrasts),
        pd.concat(support, ignore_index=True),
        pd.DataFrame(mapping),
        pd.DataFrame(acf),
    )
