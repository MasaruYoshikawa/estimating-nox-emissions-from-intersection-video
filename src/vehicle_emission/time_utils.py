"""Timezone-boundary helpers for the project's persisted UTC timestamps."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TOKYO_TIMEZONE = ZoneInfo("Asia/Tokyo")
SoramameHourConvention = Literal["end-labelled-hour", "centre-labelled-hour"]
SORAMAME_HOUR_CONVENTIONS = ("end-labelled-hour", "centre-labelled-hour")


def format_utc(timestamp: datetime) -> str:
    """Format an aware timestamp as canonical ISO-8601 UTC with ``Z``."""

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Timestamp must include an explicit timezone offset")
    return timestamp.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_aware_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and reject timezone-naive values."""

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"Timestamp must include an explicit timezone offset: {value!r}")
    return timestamp


def normalize_utc_timestamp(value: str) -> str:
    """Normalize an aware ISO-8601 timestamp to canonical UTC text."""

    return format_utc(parse_aware_timestamp(value))


def local_timestamp_to_utc(
    value: str,
    *,
    input_format: str,
    timezone_name: str = "Asia/Tokyo",
) -> str:
    """Interpret a naive source timestamp in a named timezone and return UTC."""

    try:
        source_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown IANA timezone: {timezone_name!r}") from exc
    try:
        local_timestamp = datetime.strptime(value, input_format).replace(tzinfo=source_timezone)
    except ValueError as exc:
        raise ValueError(
            f"Timestamp {value!r} does not match {input_format!r} in {timezone_name}"
        ) from exc
    return format_utc(local_timestamp)


def soramame_hour_interval_utc(
    date_text: str,
    hour_label: int,
    *,
    convention: SoramameHourConvention,
) -> tuple[str, str]:
    """Return the explicitly selected half-open interval for a 1--24 label.

    ``end-labelled-hour`` makes label 1 cover 00:00--01:00 JST and label 24
    cover 23:00--24:00 JST. ``centre-labelled-hour`` makes the clock label the
    midpoint of a one-hour support interval. The project must select the
    convention from station-specific evidence; there is deliberately no
    default.
    """

    if hour_label not in range(1, 25):
        raise ValueError(f"Soramamekun hour label must be in 1..24, got {hour_label}")
    try:
        local_date = datetime.strptime(date_text, "%d/%m/%Y").date()
    except ValueError as exc:
        raise ValueError(f"Invalid Soramamekun date: {date_text!r}") from exc

    local_midnight = datetime.combine(local_date, time.min, tzinfo=TOKYO_TIMEZONE)
    label_time = local_midnight + timedelta(hours=hour_label)
    if convention == "end-labelled-hour":
        interval_end = label_time
        interval_start = interval_end - timedelta(hours=1)
    elif convention == "centre-labelled-hour":
        interval_start = label_time - timedelta(minutes=30)
        interval_end = label_time + timedelta(minutes=30)
    else:
        raise ValueError(
            f"Unsupported Soramamekun hour convention: {convention!r}; "
            f"expected one of {SORAMAME_HOUR_CONVENTIONS}"
        )
    return format_utc(interval_start), format_utc(interval_end)


def compact_local_filename_timestamp_utc(
    date_text: str,
    time_text: str,
    *,
    timezone_name: str,
) -> str:
    """Convert a ``YYYYMMDD``/``HHMM`` filename timestamp to UTC.

    A timezone is deliberately mandatory because capture-host filenames have
    historically represented a local VM clock rather than UTC.
    """

    return local_timestamp_to_utc(
        f"{date_text}_{time_text}",
        input_format="%Y%m%d_%H%M",
        timezone_name=timezone_name,
    )


def utc_date_and_time(value: str) -> tuple[date, time]:
    """Return the UTC calendar date and time represented by an aware value."""

    timestamp = parse_aware_timestamp(value).astimezone(UTC)
    return timestamp.date(), timestamp.time().replace(tzinfo=None)


__all__ = [
    "TOKYO_TIMEZONE",
    "SORAMAME_HOUR_CONVENTIONS",
    "SoramameHourConvention",
    "compact_local_filename_timestamp_utc",
    "format_utc",
    "local_timestamp_to_utc",
    "normalize_utc_timestamp",
    "parse_aware_timestamp",
    "soramame_hour_interval_utc",
    "utc_date_and_time",
]
