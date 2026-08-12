"""Markerless schedule-photo registration with explicit acceptance profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .geometry import clipped_visibility, polygon_area, quad_is_plausible, transform_points


WORKING_LONG_SIDE = 2400


@dataclass(frozen=True)
class RegistrationProfile:
    name: str
    min_matches: int
    min_inliers: int
    min_inlier_ratio: float
    min_hull_coverage: float
    min_spatial_regions: int
    median_error_px: float = 4.0
    p90_error_px: float = 8.0


GENERAL = RegistrationProfile("general", 25, 15, 0.45, 0.25, 3)
PARTIAL = RegistrationProfile("partial", 18, 12, 0.50, 0.30, 2)


def normalize_working_image(image):
    import cv2

    height, width = image.shape[:2]
    scale = WORKING_LONG_SIDE / max(width, height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _detect_and_match(reference, photo, method: str):
    import cv2
    import numpy as np

    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=8000)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    else:
        detector = cv2.AKAZE_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    key_ref, desc_ref = detector.detectAndCompute(reference, None)
    key_photo, desc_photo = detector.detectAndCompute(photo, None)
    if desc_ref is None or desc_photo is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), 0
    pairs = matcher.knnMatch(desc_ref, desc_photo, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    source = np.float32([key_ref[match.queryIdx].pt for match in good])
    target = np.float32([key_photo[match.trainIdx].pt for match in good])
    return source, target, len(good)


def _hull_coverage(points, width: int, height: int, visible_area: float | None = None) -> float:
    import cv2
    import numpy as np

    if len(points) < 3:
        return 0.0
    area = float(cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32))))
    return area / max(1.0, visible_area if visible_area is not None else width * height)


def _quadrants(points, width: int, height: int) -> int:
    return len({(int(x >= width / 2), int(y >= height / 2)) for x, y in points})


def _symmetric_errors(source, target, homography):
    import numpy as np

    forward = np.asarray(transform_points(source, homography))
    inverse = np.linalg.inv(homography)
    backward = np.asarray(transform_points(target, inverse))
    return (np.linalg.norm(forward - target, axis=1) + np.linalg.norm(backward - source, axis=1)) / 2


def _refine_ecc(reference, photo, homography):
    import cv2
    import numpy as np

    warp = np.asarray(homography, dtype=np.float32)
    try:
        _score, refined = cv2.findTransformECC(
            photo.astype(np.float32) / 255.0,
            reference.astype(np.float32) / 255.0,
            warp,
            cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6),
            inputMask=None,
            gaussFiltSize=5,
        )
        return refined
    except cv2.error:
        return homography


def _common_visible_areas(homography, ref_width: int, ref_height: int, photo_width: int, photo_height: int):
    import cv2
    import numpy as np

    ref_corners = np.asarray([[0, 0], [ref_width, 0], [ref_width, ref_height], [0, ref_height]], np.float32)
    photo_corners = np.asarray([[0, 0], [photo_width, 0], [photo_width, photo_height], [0, photo_height]], np.float32)
    projected_ref = np.asarray(transform_points(ref_corners, homography), np.float32)
    visible_photo, _polygon_photo = cv2.intersectConvexConvex(projected_ref, photo_corners)
    visible_ref_polygon = np.asarray(transform_points(photo_corners, np.linalg.inv(homography)), np.float32)
    visible_ref, _polygon_ref = cv2.intersectConvexConvex(visible_ref_polygon, ref_corners)
    return max(1.0, float(visible_ref)), max(1.0, float(visible_photo))


def register(reference_image, photo_image, *, partial: bool = False, refine_ecc: bool = True) -> dict[str, Any]:
    import cv2
    import numpy as np

    profile = PARTIAL if partial else GENERAL
    ref, ref_scale = normalize_working_image(reference_image)
    photo, photo_scale = normalize_working_image(photo_image)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) if ref.ndim == 3 else ref
    photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY) if photo.ndim == 3 else photo
    method = "sift"
    source, target, match_count = _detect_and_match(ref_gray, photo_gray, method)
    if match_count < profile.min_matches:
        method = "akaze"
        source, target, match_count = _detect_and_match(ref_gray, photo_gray, method)
    if match_count < 4:
        return {"accepted": False, "reason": "insufficient_matches", "method": method, "match_count": match_count}

    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    if homography is None or mask is None:
        return {"accepted": False, "reason": "homography_failed", "method": method, "match_count": match_count}
    inliers = mask.ravel().astype(bool)
    source_inliers, target_inliers = source[inliers], target[inliers]
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / max(1, match_count)

    reference_corners = [[0, 0], [ref.shape[1], 0], [ref.shape[1], ref.shape[0]], [0, ref.shape[0]]]
    projected = transform_points(reference_corners, homography)
    initial_geometry_valid = quad_is_plausible(projected, photo.shape[1], photo.shape[0], min_area_ratio=0.01)
    if refine_ecc and initial_geometry_valid and inlier_count >= profile.min_inliers:
        homography = _refine_ecc(ref_gray, photo_gray, homography)
        projected = transform_points(reference_corners, homography)

    errors = _symmetric_errors(source_inliers, target_inliers, homography) if inlier_count else np.array([np.inf])
    median_error = float(np.median(errors))
    p90_error = float(np.percentile(errors, 90))
    visible_ref_area = visible_photo_area = None
    if partial:
        visible_ref_area, visible_photo_area = _common_visible_areas(
            homography, ref.shape[1], ref.shape[0], photo.shape[1], photo.shape[0]
        )
    coverage_ref = _hull_coverage(source_inliers, ref.shape[1], ref.shape[0], visible_ref_area)
    coverage_photo = _hull_coverage(target_inliers, photo.shape[1], photo.shape[0], visible_photo_area)
    coverage = min(coverage_ref, coverage_photo)
    spatial_regions = min(
        _quadrants(source_inliers, ref.shape[1], ref.shape[0]),
        _quadrants(target_inliers, photo.shape[1], photo.shape[0]),
    )
    geometry_valid = quad_is_plausible(projected, photo.shape[1], photo.shape[0], min_area_ratio=0.005 if partial else 0.02)
    checks = {
        "matches": match_count >= profile.min_matches,
        "inliers": inlier_count >= profile.min_inliers,
        "inlier_ratio": inlier_ratio >= profile.min_inlier_ratio,
        "coverage": coverage >= profile.min_hull_coverage,
        "spatial_regions": spatial_regions >= profile.min_spatial_regions,
        "median_error": median_error <= profile.median_error_px,
        "p90_error": p90_error <= profile.p90_error_px,
        "geometry": geometry_valid,
    }
    # Map full-resolution reference coordinates directly to full-resolution photo.
    scale_ref_to_work = np.diag([ref_scale, ref_scale, 1.0])
    scale_work_to_photo = np.diag([1.0 / photo_scale, 1.0 / photo_scale, 1.0])
    full_homography = scale_work_to_photo @ homography @ scale_ref_to_work
    return {
        "accepted": all(checks.values()),
        "profile": asdict(profile),
        "method": method,
        "match_count": match_count,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "hull_coverage": coverage,
        "spatial_regions": spatial_regions,
        "symmetric_error_median_px_2400": median_error,
        "symmetric_error_p90_px_2400": p90_error,
        "normalized_error_median": median_error / WORKING_LONG_SIDE,
        "normalized_error_p90": p90_error / WORKING_LONG_SIDE,
        "checks": checks,
        "homography": full_homography.tolist(),
    }


def transfer_objects(objects: Sequence[Mapping[str, Any]], registration: Mapping[str, Any], width: int, height: int):
    if not registration.get("accepted"):
        raise ValueError("registration must be accepted before transferring labels")
    output: list[dict[str, Any]] = []
    matrix = registration["homography"]
    for source in objects:
        item = dict(source)
        for key in ("cell_polygon", "text_polygon"):
            if item.get(key):
                item[key] = transform_points(item[key], matrix)
        if item.get("text_polygon"):
            import numpy as np
            points = np.asarray(item["text_polygon"], dtype=float)
            center = points.mean(axis=0)
            text_height = max(1.0, float(points[:, 1].max() - points[:, 1].min()))
            padding = max(3.0, text_height * 0.08)
            radius = max(1.0, float(np.linalg.norm(points - center, axis=1).mean()))
            item["text_polygon"] = (center + (points - center) * (1.0 + padding / radius)).tolist()
            item["text_polygon_margin_px"] = padding
        visibility = clipped_visibility(item["cell_polygon"], width, height)
        if visibility < 0.20:
            continue
        item["visibility"] = visibility
        item["ignore"] = visibility < 0.60
        item["registration_profile"] = registration["profile"]["name"]
        item["registration_high_confidence"] = bool(
            registration["symmetric_error_median_px_2400"] <= 4
            and registration["symmetric_error_p90_px_2400"] <= 8
        )
        output.append(item)
    return output
