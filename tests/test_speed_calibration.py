from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vehicle_emission.speed.calibration import (
    compute_homography_matrix,
    project_image_to_world,
    validate_calibration_dict,
)


def test_legacy_calibration_example_schema() -> None:
    calib_path = Path("tests/fixtures/legacy/site_calibration_v2.json")
    assert calib_path.is_file()

    with calib_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert {"sukiyabashi", "shinjuku"} <= set(data)
    for site in ("sukiyabashi", "shinjuku"):
        item = data[site]
        assert item["route"] in {"homography", "bounded_zones"}
        if item["route"] == "homography":
            assert len(item["image_points"]) >= 4
            assert len(item["world_points_metres"]) >= 4
        else:
            assert item["zones"]
            assert all(
                {"id", "image_polygon", "scale_metres_per_pixel"} <= set(zone)
                for zone in item["zones"]
            )

    validate_calibration_dict(data)


def test_homography_matrix_computation_and_projection() -> None:
    img_pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    world_pts = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

    H = compute_homography_matrix(img_pts, world_pts)
    assert H.shape == (3, 3)

    proj = project_image_to_world(img_pts, H)
    np.testing.assert_allclose(proj, world_pts, rtol=1e-3, atol=1e-8)


def test_non_affine_homography_projects_unseen_points() -> None:
    expected_matrix = np.array([[0.2, 0.01, 2], [0.02, 0.1, -1], [0.001, 0.002, 1]])
    controls = [[0, 0], [100, 0], [120, 100], [0, 100], [40, 60]]
    world_controls = project_image_to_world(controls, expected_matrix)
    calculated_matrix = compute_homography_matrix(controls, world_controls)
    unseen = [[30, 20], [70, 80]]
    np.testing.assert_allclose(
        project_image_to_world(unseen, calculated_matrix),
        project_image_to_world(unseen, expected_matrix),
        rtol=1e-9,
        atol=1e-9,
    )


def test_projection_is_invariant_to_homography_scale() -> None:
    matrix = np.array([[0.2, 0.01, 2], [0.02, 0.1, -1], [0.001, 0.002, 1]])
    points = [[30, 20], [70, 80]]
    np.testing.assert_allclose(
        project_image_to_world(points, matrix * -1e-9),
        project_image_to_world(points, matrix),
    )


def test_validate_calibration_dict_rejects_missing_sites() -> None:
    with pytest.raises(ValueError, match="calibration must include sites"):
        validate_calibration_dict({"sukiyabashi": {"route": "homography"}})


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[0, 0], [0, 0], [1, 1], [0, 1]],
        [[0, 0], [1, 0], [1, float("nan")], [0, 1]],
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    ],
)
def test_invalid_homography_controls_fail(points: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        compute_homography_matrix(points, [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_mismatched_homography_controls_fail() -> None:
    with pytest.raises(ValueError, match="counts must match"):
        compute_homography_matrix(
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]],
        )


def test_horizon_is_rejected_instead_of_returning_raw_numerators() -> None:
    matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -1]], dtype=float)
    with pytest.raises(ValueError, match="horizon"):
        project_image_to_world([[1, 2]], matrix)


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.eye(2), np.full((3, 3), np.nan)])
def test_invalid_projection_matrix_fails(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match="matrix"):
        project_image_to_world([[1, 2]], matrix)


@pytest.mark.parametrize("scale", [0, -1, True, float("inf")])
def test_invalid_bounded_zone_scale_fails(scale: float) -> None:
    item = {
        "route": "bounded_zones",
        "zones": [
            {
                "id": "zone",
                "image_polygon": [[0, 0], [1, 0], [0, 1]],
                "scale_metres_per_pixel": scale,
            }
        ],
    }
    with pytest.raises(ValueError, match="scale"):
        validate_calibration_dict({"sukiyabashi": item, "shinjuku": item})
