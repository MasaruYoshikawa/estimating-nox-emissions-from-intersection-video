from vehicle_emission.line_counting import LineOfInterest, MultiLineCrossingCounter


def test_multi_line_crossing_counter_entering_path_counts() -> None:
    # Set up 4 exits representing Top, Right, Bottom, Left
    # exit1: Top (centered horizontally at Y=0)
    # exit2: Right (centered vertically at X=10)
    # exit3: Bottom (centered horizontally at Y=10)
    # exit4: Left (centered vertically at X=0)
    line_specs = [
        ("exit1", LineOfInterest(start=(0.0, 0.0), end=(10.0, 0.0)), None),
        ("exit2", LineOfInterest(start=(10.0, 0.0), end=(10.0, 10.0)), None),
        ("exit3", LineOfInterest(start=(0.0, 10.0), end=(10.0, 10.0)), None),
        ("exit4", LineOfInterest(start=(0.0, 0.0), end=(0.0, 10.0)), None),
    ]
    multi_counter = MultiLineCrossingCounter(line_specs)

    # 1. Vehicle 1 enters from Left (exit4: positive_to_negative crossing)
    # Start: outside Left (X = -5, Y = 5) -> inside (X = 5, Y = 5)
    multi_counter.update(frame_id=0, tracks=[{"id": 1, "center": (-5.0, 5.0)}])
    multi_counter.update(frame_id=1, tracks=[{"id": 1, "center": (5.0, 5.0)}])

    # 2. Vehicle 1 exits via Right (exit2: positive_to_negative crossing)
    # Start: inside (X = 5, Y = 5) -> outside Right (X = 15, Y = 5)
    # Crossing exit2 from positive_to_negative. This is an exit crossing
    # (entering for Right is negative_to_positive).
    # It should not count as entering from Right.
    multi_counter.update(frame_id=2, tracks=[{"id": 1, "center": (15.0, 5.0)}])

    # 3. Vehicle 2 enters from Right (exit2: negative_to_positive crossing)
    # Start: outside Right (X = 15, Y = 5) -> inside (X = 5, Y = 5)
    multi_counter.update(frame_id=0, tracks=[{"id": 2, "center": (15.0, 5.0)}])
    multi_counter.update(frame_id=1, tracks=[{"id": 2, "center": (5.0, 5.0)}])

    # 4. Vehicle 3 enters from Top (exit1: negative_to_positive crossing)
    # Start: outside Top (X = 5, Y = -5) -> inside (X = 5, Y = 5)
    multi_counter.update(frame_id=0, tracks=[{"id": 3, "center": (5.0, -5.0)}])
    multi_counter.update(frame_id=1, tracks=[{"id": 3, "center": (5.0, 5.0)}])

    # 5. Vehicle 4 enters from Bottom (exit3: positive_to_negative crossing)
    # Start: outside Bottom (X = 5, Y = 15) -> inside (X = 5, Y = 5)
    multi_counter.update(frame_id=0, tracks=[{"id": 4, "center": (5.0, 15.0)}])
    multi_counter.update(frame_id=1, tracks=[{"id": 4, "center": (5.0, 5.0)}])

    entering_counts = multi_counter.entering_path_counts()
    assert entering_counts["entering_from_left"] == 1
    assert entering_counts["entering_from_right"] == 1
    assert entering_counts["entering_from_top"] == 1
    assert entering_counts["entering_from_bottom"] == 1
