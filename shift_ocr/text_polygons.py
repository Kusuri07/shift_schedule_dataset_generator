"""Text-layout polygon validation and mask/difference fallbacks."""

from __future__ import annotations

import math
from typing import Sequence


def _convex_hull(points):
    values = sorted(set((float(x), float(y)) for x, y in points))
    if len(values) <= 1:
        return values

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower = []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def minimum_area_quad(points) -> list[list[float]]:
    hull = _convex_hull(points)
    if not hull:
        raise ValueError("text mask has no glyph pixels")
    if len(hull) == 1:
        x, y = hull[0]
        return [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]]
    best = None
    for first, second in zip(hull, hull[1:] + hull[:1]):
        angle = math.atan2(second[1] - first[1], second[0] - first[0])
        cosine, sine = math.cos(angle), math.sin(angle)
        rotated = [(x * cosine + y * sine, -x * sine + y * cosine) for x, y in hull]
        xs = [point[0] for point in rotated]
        ys = [point[1] for point in rotated]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if best is None or area < best[0]:
            corners = [
                (min(xs), min(ys)), (max(xs), min(ys)),
                (max(xs), max(ys)), (min(xs), max(ys)),
            ]
            unrotated = [[x * cosine - y * sine, x * sine + y * cosine] for x, y in corners]
            best = (area, unrotated)
    return best[1]


def polygon_from_pillow_mask(image, bbox: Sequence[float], *, threshold: int = 220) -> list[list[float]]:
    left, top, right, bottom = [int(round(value)) for value in bbox]
    left, top = max(0, left), max(0, top)
    right, bottom = min(image.width, right), min(image.height, bottom)
    crop = image.crop((left, top, right, bottom)).convert("L")
    pixels = crop.load()
    points = [
        (x + left, y + top)
        for y in range(crop.height)
        for x in range(crop.width)
        if pixels[x, y] < threshold
    ]
    quad = minimum_area_quad(points)
    first_height = math.dist(quad[0], quad[3])
    second_height = math.dist(quad[1], quad[2])
    padding = max(2.0, min(first_height, second_height) * 0.05)
    return _expanded_quad(quad, padding)


def _expanded_quad(points, padding: float):
    values = [(float(point[0]), float(point[1])) for point in points]
    center = (
        sum(point[0] for point in values) / len(values),
        sum(point[1] for point in values) / len(values),
    )
    radius = sum(math.dist(point, center) for point in values) / len(values)
    scale = 1.0 + padding / max(radius, 1.0)
    return [
        [center[0] + (point[0] - center[0]) * scale, center[1] + (point[1] - center[1]) * scale]
        for point in values
    ]


def polygon_from_glyph_mask(mask, *, padding_ratio: float = 0.05) -> list[list[float]]:
    """Combine every glyph pixel for one text object into one padded quad."""
    import cv2
    import numpy as np

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("glyph mask is empty")
    points = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(points)
    rectangle = cv2.boxPoints(cv2.minAreaRect(hull))
    height = min(
        float(np.linalg.norm(rectangle[1] - rectangle[0])),
        float(np.linalg.norm(rectangle[2] - rectangle[1])),
    )
    return _expanded_quad(rectangle, max(2.0, height * padding_ratio))


def difference_mask(with_text, without_text):
    import cv2
    import numpy as np

    first = cv2.cvtColor(with_text, cv2.COLOR_BGR2GRAY) if with_text.ndim == 3 else with_text
    second = cv2.cvtColor(without_text, cv2.COLOR_BGR2GRAY) if without_text.ndim == 3 else without_text
    difference = cv2.absdiff(first, second)
    _threshold, mask = cv2.threshold(difference, 8, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def max_corner_distance(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> float:
    # Quads may start at a different corner or wind in the other direction.
    def rotations(points):
        values = list(points)
        for reverse in (False, True):
            current = values[::-1] if reverse else values
            for offset in range(4):
                yield current[offset:] + current[:offset]

    return min(
        max(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])) for a, b in zip(first, candidate))
        for candidate in rotations(list(second))
    )


def choose_text_polygon(
    layout_polygon: Sequence[Sequence[float]] | None,
    *,
    glyph_mask=None,
    with_text=None,
    without_text=None,
    tolerance_px: float = 2.0,
) -> tuple[list[list[float]], str]:
    mask_polygon = polygon_from_glyph_mask(glyph_mask) if glyph_mask is not None else None
    if layout_polygon is not None and (
        mask_polygon is None or max_corner_distance(layout_polygon, mask_polygon) <= tolerance_px
    ):
        return [list(map(float, point)) for point in layout_polygon], "layout_bounds"
    if mask_polygon is not None:
        return mask_polygon, "glyph_mask"
    if with_text is not None and without_text is not None:
        return polygon_from_glyph_mask(difference_mask(with_text, without_text)), "render_difference"
    raise ValueError("no valid text polygon source")
