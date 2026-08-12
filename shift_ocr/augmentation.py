"""On-the-fly image/polygon augmentation and recognizer crop simulation."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .geometry import clipped_visibility, transform_points


@dataclass(frozen=True)
class AugmentationRecipe:
    seed: int
    rotation_deg: float
    scale: float
    translate_x: float
    translate_y: float
    perspective: tuple[tuple[float, float], ...]
    blur_sigma: float
    motion_blur: int
    brightness: float
    contrast: float
    gamma: float
    noise_sigma: float
    jpeg_quality: int
    shadow_strength: float
    reflection_strength: float


def sample_recipe(seed: int) -> AugmentationRecipe:
    rng = random.Random(seed)
    return AugmentationRecipe(
        seed=seed,
        rotation_deg=rng.uniform(-15, 15),
        scale=rng.uniform(0.75, 1.25),
        translate_x=rng.uniform(-0.12, 0.12),
        translate_y=rng.uniform(-0.12, 0.12),
        perspective=tuple((rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)) for _ in range(4)),
        blur_sigma=rng.uniform(0, 1.8),
        motion_blur=rng.choice([0, 0, 0, 3, 5, 7]),
        brightness=rng.uniform(0.75, 1.25),
        contrast=rng.uniform(0.75, 1.35),
        gamma=rng.uniform(0.75, 1.35),
        noise_sigma=rng.uniform(0, 10),
        jpeg_quality=rng.randint(40, 95),
        shadow_strength=rng.uniform(0, 0.35),
        reflection_strength=rng.uniform(0, 0.28),
    )


def homography_for_recipe(recipe: AugmentationRecipe, width: int, height: int):
    import cv2
    import numpy as np

    center = np.array([width / 2, height / 2], dtype=np.float32)
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    angle = math.radians(recipe.rotation_deg)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    target = (corners - center) @ rotation.T * recipe.scale + center
    target[:, 0] += recipe.translate_x * width
    target[:, 1] += recipe.translate_y * height
    target += np.asarray(recipe.perspective, dtype=np.float32) * np.array([width, height], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners, target)


def _photometric(image, recipe: AugmentationRecipe):
    import cv2
    import numpy as np

    value = image.astype(np.float32) * recipe.contrast
    value += (recipe.brightness - 1.0) * 127.5
    value = np.clip(value / 255.0, 0, 1) ** (1.0 / recipe.gamma) * 255.0
    rng = np.random.default_rng(recipe.seed)
    height, width = value.shape[:2]
    if recipe.shadow_strength > 0.03:
        shadow = np.zeros((height, width), np.uint8)
        start = int(rng.uniform(-0.2, 0.7) * width)
        band = int(rng.uniform(0.15, 0.45) * width)
        polygon = np.asarray([[start, 0], [start + band, 0], [start + band // 2, height], [start - band // 2, height]], np.int32)
        cv2.fillPoly(shadow, [polygon], 255)
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(3.0, width * 0.015)).astype(np.float32) / 255.0
        value *= 1.0 - shadow[..., None] * recipe.shadow_strength
    if recipe.reflection_strength > 0.03:
        yy, xx = np.mgrid[0:height, 0:width]
        center_x = rng.uniform(0.15, 0.85) * width
        center_y = rng.uniform(0.15, 0.85) * height
        sigma_x = max(1.0, rng.uniform(0.08, 0.22) * width)
        sigma_y = max(1.0, rng.uniform(0.12, 0.30) * height)
        glare = np.exp(-(((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2) / 2.0)
        value += glare[..., None] * 255.0 * recipe.reflection_strength
    if recipe.blur_sigma > 0.15:
        value = cv2.GaussianBlur(value, (0, 0), recipe.blur_sigma)
    if recipe.motion_blur >= 3:
        kernel = np.zeros((recipe.motion_blur, recipe.motion_blur), dtype=np.float32)
        kernel[recipe.motion_blur // 2, :] = 1.0 / recipe.motion_blur
        value = cv2.filter2D(value, -1, kernel)
    if recipe.noise_sigma > 0:
        value += rng.normal(0, recipe.noise_sigma, value.shape)
    value = np.clip(value, 0, 255).astype(np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", value, [cv2.IMWRITE_JPEG_QUALITY, recipe.jpeg_quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded_ok else value


def augment_image_and_objects(image, objects: Sequence[Mapping[str, Any]], recipe: AugmentationRecipe):
    import cv2

    height, width = image.shape[:2]
    matrix = homography_for_recipe(recipe, width, height)
    warped = cv2.warpPerspective(image, matrix, (width, height), borderValue=(245, 245, 245))
    transformed: list[dict[str, Any]] = []
    for source in objects:
        item = dict(source)
        for key in ("cell_polygon", "text_polygon"):
            if item.get(key):
                item[key] = transform_points(item[key], matrix)
        visibility = clipped_visibility(item["cell_polygon"], width, height)
        item["visibility"] = visibility
        # Geometric augmentation may add a visibility-based ignore flag, but it
        # cannot make a source annotation trustworthy again.  In particular,
        # partially registered real-photo labels can remain inside the warped
        # canvas even though their source content was already clipped.
        item["ignore"] = bool(item.get("ignore")) or 0.20 <= visibility < 0.60
        if visibility >= 0.20:
            transformed.append(item)
    return _photometric(warped, recipe), transformed, matrix, asdict(recipe)


def jitter_quad(
    quad: Sequence[Sequence[float]], *, seed: int, jitter_ratio: tuple[float, float] = (0.02, 0.08),
    margin_ratio: tuple[float, float] = (0.0, 0.12), rotation_deg: float = 3.0,
) -> list[list[float]]:
    import cv2
    import numpy as np

    rng = random.Random(seed)
    points = np.asarray(quad, dtype=np.float32)
    width = max(float(np.linalg.norm(points[1] - points[0])), float(np.linalg.norm(points[2] - points[3])))
    height = max(float(np.linalg.norm(points[3] - points[0])), float(np.linalg.norm(points[2] - points[1])))
    jitter = rng.uniform(*jitter_ratio) * min(width, height)
    points += np.asarray([[rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in range(4)])
    center = points.mean(axis=0)
    margin = rng.uniform(*margin_ratio)
    points = center + (points - center) * (1 + 2 * margin)
    matrix = cv2.getRotationMatrix2D(tuple(center), rng.uniform(-rotation_deg, rotation_deg), 1.0)
    homogeneous = np.column_stack([points, np.ones(4, dtype=np.float32)])
    return (homogeneous @ matrix.T).tolist()


def bucket_width(content_width: int) -> int:
    for width in (160, 320, 640):
        if content_width <= width:
            return width
    return 640


def rectify_cell(
    image, quad: Sequence[Sequence[float]], target_height: int = 48,
    target_width: int | None = None,
):
    import cv2
    import numpy as np

    source = np.asarray(quad, dtype=np.float32)
    top = np.linalg.norm(source[1] - source[0])
    bottom = np.linalg.norm(source[2] - source[3])
    left = np.linalg.norm(source[3] - source[0])
    right = np.linalg.norm(source[2] - source[1])
    aspect = max(top, bottom) / max(1.0, max(left, right))
    if target_width is not None and target_width not in {160, 320, 640}:
        raise ValueError("recognizer target_width must be 160, 320 or 640")
    # 640 is the recognizer's largest declared width bucket.  During training,
    # force the source annotation's indexed bucket so jitter/predicted quads
    # cannot make one supposedly homogeneous micro-batch return mixed widths.
    width = target_width or bucket_width(max(8, int(round(target_height * aspect))))
    content_width = min(width, max(8, int(round(target_height * aspect))))
    destination = np.asarray(
        [[0, 0], [content_width - 1, 0], [content_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    crop = cv2.warpPerspective(image, matrix, (content_width, target_height), borderValue=(255, 255, 255))
    padded = np.full((target_height, width, crop.shape[2]), 255, dtype=crop.dtype)
    padded[:, :content_width] = crop
    return padded, width, content_width, matrix
