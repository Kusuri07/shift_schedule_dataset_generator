"""Shared polygon and homography helpers."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


Point = tuple[float, float]


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    return abs(sum(
        float(points[i][0]) * float(points[(i + 1) % len(points)][1])
        - float(points[(i + 1) % len(points)][0]) * float(points[i][1])
        for i in range(len(points))
    )) / 2.0


def aabb(points: Sequence[Sequence[float]]) -> list[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def is_convex_quad(points: Sequence[Sequence[float]]) -> bool:
    if len(points) != 4 or polygon_area(points) <= 1e-6:
        return False
    signs = []
    for index in range(4):
        a, b, c = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
        cross = (float(b[0]) - float(a[0])) * (float(c[1]) - float(b[1])) - (
            float(b[1]) - float(a[1])
        ) * (float(c[0]) - float(b[0]))
        if abs(cross) > 1e-6:
            signs.append(cross > 0)
    return bool(signs) and all(sign == signs[0] for sign in signs)


def quad_is_plausible(
    points: Sequence[Sequence[float]], image_width: int, image_height: int,
    *, min_area_ratio: float = 1e-5, max_aspect_ratio: float = 30.0,
) -> bool:
    if not is_convex_quad(points):
        return False
    area = polygon_area(points)
    if area / max(1.0, image_width * image_height) < min_area_ratio:
        return False
    lengths = [
        math.hypot(
            float(points[(i + 1) % 4][0]) - float(points[i][0]),
            float(points[(i + 1) % 4][1]) - float(points[i][1]),
        )
        for i in range(4)
    ]
    if min(lengths) <= 1e-3 or max(lengths) / min(lengths) > max_aspect_ratio:
        return False
    angles = []
    for index in range(4):
        previous = points[(index - 1) % 4]
        current = points[index]
        following = points[(index + 1) % 4]
        first = (float(previous[0]) - float(current[0]), float(previous[1]) - float(current[1]))
        second = (float(following[0]) - float(current[0]), float(following[1]) - float(current[1]))
        cosine = (first[0] * second[0] + first[1] * second[1]) / max(
            1e-9, math.hypot(*first) * math.hypot(*second)
        )
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    if min(angles) < 5.0 or max(angles) > 175.0:
        return False
    return True


def transform_points(points: Sequence[Sequence[float]], matrix) -> list[list[float]]:
    import numpy as np

    array = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([array, np.ones(len(array))])
    mapped = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    mapped = mapped[:, :2] / np.maximum(mapped[:, 2:3], 1e-12)
    return mapped.tolist()


def clipped_visibility(points: Sequence[Sequence[float]], width: int, height: int) -> float:
    try:
        from shapely.geometry import Polygon, box
    except ImportError:
        # Conservative AABB fallback used only when shapely is unavailable.
        left, top, right, bottom = aabb(points)
        original = max(0.0, right - left) * max(0.0, bottom - top)
        visible = max(0.0, min(right, width) - max(left, 0.0)) * max(
            0.0, min(bottom, height) - max(top, 0.0)
        )
        return visible / max(original, 1e-9)
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= 0:
        return 0.0
    return float(polygon.intersection(box(0, 0, width, height)).area / polygon.area)
