"""Perspective-aware homography and bounded-zone speed calibration logic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np


def _points(value: object, name: str, minimum: int) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric Nx2 coordinates") from error
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} Nx2 coordinates")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} coordinates must be finite")
    return points


class CalibrationItem(TypedDict):
    site: str
    route: str
    camera_epoch: str
    scale_factor_pixels_per_metre: float
    image_points: list[list[float]]
    world_points_metres: list[list[float]]
    description: str


def validate_calibration_dict(data: dict[str, Any]) -> None:
    """Validate that calibration specification contains valid two-site parameters."""
    required_sites = {"sukiyabashi", "shinjuku"}
    if not required_sites.issubset(set(data)):
        raise ValueError(f"calibration must include sites: {required_sites}")

    for site_key in ("sukiyabashi", "shinjuku"):
        item = data[site_key]
        route = item.get("route")
        if route not in {"homography", "bounded_zones"}:
            raise ValueError(f"invalid calibration route for {site_key}: {route}")

        if route == "homography":
            img_pts = item.get("image_points", [])
            world_pts = item.get("world_points_metres", [])
            if len(img_pts) < 4 or len(world_pts) < 4:
                raise ValueError(f"homography route for {site_key} requires >= 4 points")
            if len(img_pts) != len(world_pts):
                raise ValueError(
                    f"image_points and world_points_metres count mismatch for {site_key}"
                )
            compute_homography_matrix(img_pts, world_pts)
        else:
            zones = item.get("zones", [])
            if not zones:
                raise ValueError(f"bounded_zones route for {site_key} requires zones")
            required_keys = {"id", "image_polygon", "scale_metres_per_pixel"}
            for zone in zones:
                if not required_keys.issubset(set(zone)):
                    raise ValueError(f"zone in {site_key} missing required keys: {required_keys}")
                polygon = _points(zone["image_polygon"], "image_polygon", 3)
                area = np.sum(
                    polygon[:, 0] * np.roll(polygon[:, 1], -1)
                    - polygon[:, 1] * np.roll(polygon[:, 0], -1)
                )
                if len(np.unique(polygon, axis=0)) != len(polygon) or abs(area) <= 1e-12:
                    raise ValueError("bounded-zone polygon must be non-degenerate")
                scale = zone["scale_metres_per_pixel"]
                if (
                    isinstance(scale, bool)
                    or not isinstance(scale, int | float)
                    or not np.isfinite(scale)
                    or scale <= 0
                ):
                    raise ValueError("bounded-zone scale must be finite and positive")


def compute_homography_matrix(
    image_points: Sequence[Sequence[float]], world_points: Sequence[Sequence[float]]
) -> np.ndarray:
    """Compute 3x3 perspective transformation matrix mapping image pixels to world metres."""
    src = _points(image_points, "image_points", 4)
    dst = _points(world_points, "world_points", 4)
    if len(src) != len(dst):
        raise ValueError("homography point correspondence counts must match")
    for points in (src, dst):
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("homography control points must be unique")
        if np.linalg.matrix_rank(np.column_stack((points, np.ones(len(points))))) < 3:
            raise ValueError("homography control points must not be collinear")

    # Solve DLT system for H
    a_matrix = []
    for i in range(len(src)):
        x, y = src[i][0], src[i][1]
        u, v = dst[i][0], dst[i][1]
        a_matrix.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        a_matrix.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    design = np.asarray(a_matrix)
    if np.linalg.matrix_rank(design) < 8:
        raise ValueError("homography correspondences do not define a unique transform")
    _, _, vh_mat = np.linalg.svd(design)
    h_matrix = vh_mat[-1].reshape(3, 3)
    if abs(h_matrix[2, 2]) > 1e-9:
        h_matrix = h_matrix / h_matrix[2, 2]
    if not np.isfinite(h_matrix).all() or np.linalg.matrix_rank(h_matrix) != 3:
        raise ValueError("homography must be finite and non-singular")
    return h_matrix


def project_image_to_world(
    points_pixel: np.ndarray | Sequence[Sequence[float]], h_matrix: np.ndarray
) -> np.ndarray:
    """Project N x 2 image pixel coordinates to N x 2 ground-plane metres using homography matrix."""
    if np.asarray(points_pixel).size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts_arr = _points(points_pixel, "points_pixel", 1)
    matrix = np.asarray(h_matrix, dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) != 3
    ):
        raise ValueError("homography matrix must be finite and non-singular 3x3")
    matrix = matrix / np.max(np.abs(matrix))
    pts = np.hstack([pts_arr, np.ones((len(pts_arr), 1), dtype=np.float64)])
    proj = (matrix @ pts.T).T
    if np.any(np.abs(proj[:, 2]) <= 1e-12):
        raise ValueError("image point is on the homography horizon")
    proj[:, :2] /= proj[:, 2, None]
    if not np.isfinite(proj[:, :2]).all():
        raise ValueError("projected ground-plane coordinates must be finite")
    return proj[:, :2]
