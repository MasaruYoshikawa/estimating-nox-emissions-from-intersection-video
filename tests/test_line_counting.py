from vehicle_emission.line_counting import (
    LineCrossingCounter,
    LineOfInterest,
    MultiLineCrossingCounter,
)


def test_line_crossing_counter_counts_when_tracks_cross_line() -> None:
    line = LineOfInterest(start=(0.0, 10.0), end=(10.0, 10.0))
    counter = LineCrossingCounter(line=line)

    tracks_frame_0 = [
        {"id": 1, "center": (5.0, 5.0)},
        {"id": 2, "center": (2.0, 12.0)},
    ]
    tracks_frame_1 = [
        {"id": 1, "center": (5.0, 15.0)},
        {"id": 2, "center": (2.0, 8.0)},
    ]

    counter.update(frame_id=0, tracks=tracks_frame_0)
    counter.update(frame_id=1, tracks=tracks_frame_1)

    assert counter.total_count == 2
    assert counter.events[0]["vehicle_id"] == 1
    assert counter.events[1]["vehicle_id"] == 2


def test_line_crossing_counter_directional_filtering() -> None:
    line = LineOfInterest(start=(0.0, 10.0), end=(10.0, 10.0))
    counter = LineCrossingCounter(line=line, direction="positive_to_negative")

    tracks_frame_0 = [{"id": 3, "center": (5.0, 12.0)}]
    tracks_frame_1 = [{"id": 3, "center": (5.0, 8.0)}]

    counter.update(frame_id=0, tracks=tracks_frame_0)
    counter.update(frame_id=1, tracks=tracks_frame_1)

    assert counter.total_count == 1
    assert counter.events[0]["direction"] == "positive_to_negative"


def test_line_crossing_counter_does_not_double_count_same_track() -> None:
    line = LineOfInterest(start=(0.0, 10.0), end=(10.0, 10.0))
    counter = LineCrossingCounter(line=line)

    frames = [
        [{"id": 4, "center": (5.0, 5.0)}],
        [{"id": 4, "center": (5.0, 15.0)}],
        [{"id": 4, "center": (5.0, 5.0)}],
    ]

    for frame_id, frame_tracks in enumerate(frames):
        counter.update(frame_id=frame_id, tracks=frame_tracks)

    assert counter.total_count == 1


def test_multi_line_crossing_counter_counts_multiple_exits() -> None:
    line_specs = [
        ("exit1", LineOfInterest(start=(0.0, 10.0), end=(10.0, 10.0)), None),
        ("exit2", LineOfInterest(start=(10.0, 0.0), end=(10.0, 10.0)), None),
    ]
    multi_counter = MultiLineCrossingCounter(line_specs)

    tracks_frame_0 = [
        {"id": 5, "center": (5.0, 5.0)},
        {"id": 6, "center": (5.0, 5.0)},
    ]
    tracks_frame_1 = [
        {"id": 5, "center": (5.0, 15.0)},
        {"id": 6, "center": (15.0, 5.0)},
    ]

    multi_counter.update(frame_id=0, tracks=tracks_frame_0)
    multi_counter.update(frame_id=1, tracks=tracks_frame_1)

    totals = multi_counter.totals()
    assert totals["exit1"] == 1
    assert totals["exit2"] == 1


def test_line_crossing_counter_directional_and_class_counts() -> None:
    line = LineOfInterest(start=(0.0, 10.0), end=(10.0, 10.0))
    counter = LineCrossingCounter(line=line)

    tracks_frame_0 = [
        {"id": 7, "center": (5.0, 12.0), "class": 1},
        {"id": 8, "center": (5.0, 8.0), "class": 2},
    ]
    tracks_frame_1 = [
        {"id": 7, "center": (5.0, 8.0), "class": 1},
        {"id": 8, "center": (5.0, 12.0), "class": 2},
    ]

    counter.update(frame_id=0, tracks=tracks_frame_0)
    counter.update(frame_id=1, tracks=tracks_frame_1)

    assert counter.total_count == 2
    assert counter.count_by_direction == {
        "negative_to_positive": 1,
        "positive_to_negative": 1,
    }
    assert counter.count_by_class_and_direction[1]["positive_to_negative"] == 1
    assert counter.count_by_class_and_direction[2]["negative_to_positive"] == 1


def test_multi_line_crossing_counter_directional_summary() -> None:
    line_specs = [
        ("exit1", LineOfInterest(start=(0.0, 10.0), end=(10.0, 10.0)), None),
        ("exit2", LineOfInterest(start=(10.0, 0.0), end=(10.0, 10.0)), None),
    ]
    multi_counter = MultiLineCrossingCounter(line_specs)

    tracks_frame_0 = [
        {"id": 5, "center": (5.0, 5.0), "class": 1},
        {"id": 6, "center": (5.0, 5.0), "class": 2},
    ]
    tracks_frame_1 = [
        {"id": 5, "center": (5.0, 15.0), "class": 1},
        {"id": 6, "center": (15.0, 5.0), "class": 2},
    ]

    multi_counter.update(frame_id=0, tracks=tracks_frame_0)
    multi_counter.update(frame_id=1, tracks=tracks_frame_1)

    totals = multi_counter.directional_totals()
    assert totals["exit1"]["negative_to_positive"] == 1
    assert totals["exit2"]["positive_to_negative"] == 1
    assert multi_counter.class_direction_totals()["exit1"][1]["negative_to_positive"] == 1
    assert multi_counter.class_direction_totals()["exit2"][2]["positive_to_negative"] == 1


def test_multi_line_crossing_counter_straight_path_counts() -> None:
    line_specs = [
        ("exit1", LineOfInterest(start=(0.0, 10.0), end=(10.0, 10.0)), None),
        ("exit2", LineOfInterest(start=(10.0, 0.0), end=(10.0, 10.0)), None),
        ("exit3", LineOfInterest(start=(0.0, 0.0), end=(10.0, 0.0)), None),
        ("exit4", LineOfInterest(start=(0.0, 0.0), end=(0.0, 10.0)), None),
    ]
    multi_counter = MultiLineCrossingCounter(line_specs)

    # Vehicle 9 crosses exit1 then exit3 moving from below to above (down_to_up path)
    multi_counter.update(frame_id=0, tracks=[{"id": 9, "center": (5.0, 15.0)}])
    multi_counter.update(frame_id=1, tracks=[{"id": 9, "center": (5.0, 5.0)}])
    multi_counter.update(frame_id=2, tracks=[{"id": 9, "center": (5.0, -5.0)}])

    # Vehicle 10 crosses exit2 then exit4 moving from right to left (right_to_left path)
    multi_counter.update(frame_id=0, tracks=[{"id": 10, "center": (15.0, 5.0)}])
    multi_counter.update(frame_id=1, tracks=[{"id": 10, "center": (5.0, 5.0)}])
    multi_counter.update(frame_id=2, tracks=[{"id": 10, "center": (-5.0, 5.0)}])

    straight_counts = multi_counter.straight_path_counts()
    assert straight_counts["down_to_up"] == 1
    assert straight_counts["right_to_left"] == 1
    assert straight_counts["left_to_right"] == 0
    assert straight_counts["up_to_down"] == 0
