from __future__ import annotations

import cv2
import numpy as np
import pytest


def test_polygon_mask_application() -> None:
    # Set up a dummy HxWx3 frame (all ones)
    height, width = 100, 100
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Define a mask polygon (square in the center)
    # x1 y1 x2 y2 x3 y3 x4 y4
    mask_polygon = [10.0, 10.0, 90.0, 10.0, 90.0, 90.0, 10.0, 90.0]

    # Validate coordinate pairs
    if len(mask_polygon) % 2 != 0 or len(mask_polygon) < 6:
        raise ValueError("Invalid mask polygon.")

    # Parse and generate binary mask
    mask_pts = np.array(mask_polygon, dtype=np.int32).reshape((-1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [mask_pts], 255)

    # Apply the mask: black out unwanted internal region
    frame[mask == 255] = 0

    # Assert that pixels inside the mask are blacked out
    assert np.all(frame[50, 50] == 0)
    assert np.all(frame[20, 20] == 0)

    # Assert that pixels outside the mask are preserved
    assert np.all(frame[0, 0] == 255)
    assert np.all(frame[5, 5] == 255)
    assert np.all(frame[95, 95] == 255)


def test_invalid_polygon_validation() -> None:
    # Odd number of coordinates should raise error/fail validation
    invalid_coords_odd = [10.0, 10.0, 20.0]
    with pytest.raises(ValueError):
        if len(invalid_coords_odd) % 2 != 0 or len(invalid_coords_odd) < 6:
            raise ValueError("Invalid polygon.")

    # Too few coordinates (less than 3 vertices)
    invalid_coords_few = [10.0, 10.0, 20.0, 20.0]
    with pytest.raises(ValueError):
        if len(invalid_coords_few) % 2 != 0 or len(invalid_coords_few) < 6:
            raise ValueError("Invalid polygon.")


def test_multiple_polygon_mask_application() -> None:
    # Set up a dummy HxWx3 frame (all ones)
    height, width = 100, 100
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Define two mask polygons (two separate squares)
    poly1 = [10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]
    poly2 = [60.0, 60.0, 90.0, 60.0, 90.0, 90.0, 60.0, 90.0]
    polygons = [poly1, poly2]

    # Parse and generate binary mask
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        if len(poly) % 2 != 0 or len(poly) < 6:
            raise ValueError("Invalid polygon.")
        mask_pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [mask_pts], 255)

    # Apply the mask: black out unwanted internal regions
    frame[mask == 255] = 0

    # Outside both polygons should be preserved
    assert np.all(frame[0, 0] == 255)
    assert np.all(frame[50, 50] == 255)

    # Inside polygon 1 should be blacked out
    assert np.all(frame[20, 20] == 0)

    # Inside polygon 2 should be blacked out
    assert np.all(frame[70, 70] == 0)
