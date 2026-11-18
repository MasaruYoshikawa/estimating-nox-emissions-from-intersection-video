from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

Point = tuple[float, float]


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


@dataclass(frozen=True)
class LineOfInterest:
    start: Point
    end: Point

    def signed_distance(self, point: Point) -> float:
        x1, y1 = self.start
        x2, y2 = self.end
        x, y = point
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def side(self, point: Point) -> int:
        return _sign(self.signed_distance(point))

    def draw(
        self, frame: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2
    ) -> None:
        cv2.line(
            frame,
            (int(self.start[0]), int(self.start[1])),
            (int(self.end[0]), int(self.end[1])),
            color,
            thickness,
        )


class LineCrossingCounter:
    def __init__(self, line: LineOfInterest, direction: str | None = None):
        self.line = line
        self.direction = direction
        self.last_side: dict[int, int] = {}
        self.crossed_ids: set[int] = set()
        self.events: list[dict[str, object]] = []

    @property
    def total_count(self) -> int:
        return len(self.crossed_ids)

    @property
    def count_by_direction(self) -> dict[str, int]:
        return {
            "negative_to_positive": sum(
                1 for event in self.events if event["direction"] == "negative_to_positive"
            ),
            "positive_to_negative": sum(
                1 for event in self.events if event["direction"] == "positive_to_negative"
            ),
        }

    @property
    def count_by_class_and_direction(self) -> dict[object, dict[str, int]]:
        summary: dict[object, dict[str, int]] = {}
        for event in self.events:
            vehicle_class = event.get("class", "unknown")
            direction = event["direction"]
            if vehicle_class not in summary:
                summary[vehicle_class] = {"negative_to_positive": 0, "positive_to_negative": 0}
            summary[vehicle_class][direction] += 1
        return summary

    def _direction_label(self, previous_side: int, current_side: int) -> str:
        if previous_side < current_side:
            return "negative_to_positive"
        if previous_side > current_side:
            return "positive_to_negative"
        return "none"

    def _should_count(self, previous_side: int, current_side: int) -> bool:
        if self.direction is None:
            return True
        label = self._direction_label(previous_side, current_side)
        return label == self.direction

    def update(self, frame_id: int, tracks: Iterable[dict[str, object]]) -> None:
        for track in tracks:
            track_id = int(track["id"])
            center = track["center"]
            current_side = self.line.side(center)

            if current_side == 0:
                continue

            previous_side = self.last_side.get(track_id)
            self.last_side[track_id] = current_side

            if previous_side is None or previous_side == 0 or previous_side == current_side:
                continue

            if track_id in self.crossed_ids:
                continue

            if not self._should_count(previous_side, current_side):
                continue

            self.crossed_ids.add(track_id)
            event = {
                "frame_id": frame_id,
                "vehicle_id": track_id,
                "from_side": previous_side,
                "to_side": current_side,
                "direction": self._direction_label(previous_side, current_side),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
            }
            vehicle_class = track.get("class")
            if vehicle_class is not None:
                event["class"] = vehicle_class

            self.events.append(event)

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Iterable[dict[str, object]] | None = None,
        count_override: int | None = None,
        show_tracks: bool = False,
    ) -> None:
        """Draw the line, a count label, and optional per-track ID markers.

        If `count_override` is provided, it will be shown instead of the
        internal running total. This allows post-hoc video generation to
        display the per-frame count recorded during processing.
        """
        self.line.draw(frame)
        label = None
        if count_override is not None:
            if isinstance(count_override, dict):
                if count_override["negative_to_positive"] or count_override["positive_to_negative"]:
                    label = (
                        f"neg->pos: {count_override['negative_to_positive']} "
                        f"pos->neg: {count_override['positive_to_negative']}"
                    )
            else:
                label = f"Line Count: {int(count_override)}"
        elif self.direction is None:
            direction_counts = self.count_by_direction
            if direction_counts["negative_to_positive"] or direction_counts["positive_to_negative"]:
                label = (
                    f"neg->pos: {direction_counts['negative_to_positive']} "
                    f"pos->neg: {direction_counts['positive_to_negative']}"
                )
        else:
            label = f"Line Count: {self.total_count}"

        if label is not None:
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if tracks is None or not show_tracks:
            return

        for track in tracks:
            track_id = int(track["id"])
            center = track["center"]
            text = f"ID:{track_id}"
            cv2.putText(
                frame,
                text,
                (int(center[0]) + 4, int(center[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )


class MultiLineCrossingCounter:
    def __init__(self, line_specs: list[tuple[str, LineOfInterest, str | None]]):
        self.line_specs = line_specs
        self.counters = [
            LineCrossingCounter(line=line, direction=direction) for _, line, direction in line_specs
        ]

    def update(self, frame_id: int, tracks: Iterable[dict[str, object]]) -> None:
        for counter in self.counters:
            counter.update(frame_id, tracks)

    def totals(self) -> dict[str, int]:
        return {
            name: counter.total_count
            for (name, _, _), counter in zip(self.line_specs, self.counters, strict=False)
        }

    def directional_totals(self) -> dict[str, dict[str, int]]:
        return {
            name: counter.count_by_direction
            for (name, _, _), counter in zip(self.line_specs, self.counters, strict=False)
        }

    def class_direction_totals(self) -> dict[str, dict[object, dict[str, int]]]:
        return {
            name: counter.count_by_class_and_direction
            for (name, _, _), counter in zip(self.line_specs, self.counters, strict=False)
        }

    def straight_path_counts(self) -> dict[str, int]:
        # Pattern mapping follows the project's JOURNAL specification.
        # For vertical (up/down) patterns we infer the image ordering
        # from the line geometry so the mapping adapts to different videos.
        # Compute a vertical center for each exit line when available.
        centers_y: dict[str, float] = {}
        for name, line, _ in self.line_specs:
            cy = (line.start[1] + line.end[1]) / 2.0
            centers_y[name] = cy

        # If exit1 is above exit3 (smaller y in image coords) then
        # Up->Down should be exit1 then exit3 (neg->pos). Otherwise reverse.
        if centers_y.get("exit1", 0) < centers_y.get("exit3", 0):
            up_pair = ("exit1", "negative_to_positive")
            down_pair_first = ("exit3", "positive_to_negative")
            up_pair_second = ("exit1", "positive_to_negative")
        else:
            up_pair = ("exit3", "negative_to_positive")
            down_pair_first = ("exit1", "positive_to_negative")
            up_pair_second = ("exit3", "positive_to_negative")

        patterns = {
            "right_to_left": [("exit2", "negative_to_positive"), ("exit4", "negative_to_positive")],
            "left_to_right": [("exit4", "positive_to_negative"), ("exit2", "positive_to_negative")],
            "up_to_down": [
                up_pair,
                (
                    ("exit3", "negative_to_positive")
                    if up_pair[0] == "exit1"
                    else ("exit1", "negative_to_positive")
                ),
            ],
            "down_to_up": [down_pair_first, up_pair_second],
        }

        events_by_name: dict[str, list[dict[str, object]]] = {
            name: counter.events
            for (name, _, _), counter in zip(self.line_specs, self.counters, strict=False)
        }

        # Build the ordered crossing events for each vehicle across all lines.
        vehicle_history: dict[int, list[tuple[int, str, str, int]]] = {}
        for name, events in events_by_name.items():
            for event in events:
                vehicle_id = int(event["vehicle_id"])
                vehicle_history.setdefault(vehicle_id, []).append(
                    (event["frame_id"], name, event["direction"], vehicle_id)
                )

        for history in vehicle_history.values():
            history.sort(key=lambda item: item[0])

        summary = {key: 0 for key in patterns}
        for _, history in vehicle_history.items():
            sequence = [(line, direction) for _, line, direction, _ in history]
            for label, pattern in patterns.items():
                # Count each vehicle at most once for a pattern.
                if all(pair in sequence for pair in pattern):
                    # Ensure ordered appearance of the pattern.
                    first_index = sequence.index(pattern[0])
                    second_index = sequence.index(pattern[1])
                    if first_index < second_index:
                        summary[label] += 1
        return summary

    def entering_path_counts(self) -> dict[str, int]:
        centers_y: dict[str, float] = {}
        for name, line, _ in self.line_specs:
            cy = (line.start[1] + line.end[1]) / 2.0
            centers_y[name] = cy

        # Determine which exit is top vs bottom
        if centers_y.get("exit1", 0) < centers_y.get("exit3", 0):
            top_line = "exit1"
            bottom_line = "exit3"
        else:
            top_line = "exit3"
            bottom_line = "exit1"

        entering_directions = {
            "entering_from_left": ("exit4", "positive_to_negative"),
            "entering_from_right": ("exit2", "negative_to_positive"),
            "entering_from_top": (top_line, "negative_to_positive"),
            "entering_from_bottom": (bottom_line, "positive_to_negative"),
        }

        events_by_name: dict[str, list[dict[str, object]]] = {
            name: counter.events
            for (name, _, _), counter in zip(self.line_specs, self.counters, strict=False)
        }

        # Build the ordered crossing events for each vehicle across all lines.
        vehicle_history: dict[int, list[tuple[int, str, str]]] = {}
        for name, events in events_by_name.items():
            for event in events:
                vehicle_id = int(event["vehicle_id"])
                vehicle_history.setdefault(vehicle_id, []).append(
                    (event["frame_id"], name, event["direction"])
                )

        for history in vehicle_history.values():
            history.sort(key=lambda item: item[0])

        summary = {key: 0 for key in entering_directions}
        for _, history in vehicle_history.items():
            # Find the first event that matches any entering direction
            for _, line, direction in history:
                matched = False
                for entry_key, (entry_line, entry_dir) in entering_directions.items():
                    if line == entry_line and direction == entry_dir:
                        summary[entry_key] += 1
                        matched = True
                        break
                if matched:
                    break
        return summary

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Iterable[dict[str, object]] | None = None,
        count_overrides: list[int | None] | None = None,
    ) -> None:
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 165, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        for idx, ((name, line, _), counter) in enumerate(
            zip(self.line_specs, self.counters, strict=False)
        ):
            color = colors[idx % len(colors)]
            line.draw(frame, color=color, thickness=2)
            if (
                count_overrides is not None
                and idx < len(count_overrides)
                and count_overrides[idx] is not None
            ):
                override = count_overrides[idx]
                if isinstance(override, dict):
                    label = (
                        f"{name}: neg->pos {override['negative_to_positive']} "
                        f"pos->neg {override['positive_to_negative']}"
                    )
                else:
                    label = f"{name}: {override}"
            elif counter.direction is None:
                direction_counts = counter.count_by_direction
                label = (
                    f"{name}: neg->pos {direction_counts['negative_to_positive']} "
                    f"pos->neg {direction_counts['positive_to_negative']}"
                )
            else:
                label = f"{name}: {counter.total_count}"
            text_pos = (int(line.start[0]) + 4, int(line.start[1]) - 12 - idx * 18)
            cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if tracks is not None:
            for track in tracks:
                track_id = int(track["id"])
                center = track["center"]
                text = f"ID:{track_id}"
                cv2.putText(
                    frame,
                    text,
                    (int(center[0]) + 4, int(center[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )


__all__ = ["LineOfInterest", "LineCrossingCounter", "MultiLineCrossingCounter"]
