import math
from typing import Optional, Tuple

DEFAULT_FOV_DEG = 84.0
DEFAULT_CAR_LENGTH_M = 4.5
DEFAULT_CAR_WIDTH_M = 1.8

BAND_1_MAX_M = 200.0
BAND_2_MAX_M = 400.0


def compute_focal_length_px(
    image_width_px: int, fov_deg: float = DEFAULT_FOV_DEG
) -> float:
    """Derive the camera's focal length in pixels from its horizontal FOV.

    Assumes square pixels, so the same focal length applies to both axes.
    """
    if image_width_px <= 0:
        raise ValueError(f"image_width_px must be positive, got {image_width_px}")
    if not 0 < fov_deg < 180:
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")

    fov_rad = math.radians(fov_deg)
    return image_width_px / (2.0 * math.tan(fov_rad / 2.0))


def yolo_bbox_to_pixel_dims(
    width_norm: float, height_norm: float, image_width_px: int, image_height_px: int
) -> Tuple[float, float]:
    """Convert normalized YOLO bbox width/height to pixel dimensions."""
    return width_norm * image_width_px, height_norm * image_height_px


def estimate_distance_m(
    bbox_width_px: float,
    bbox_height_px: float,
    image_width_px: int,
    fov_deg: float = DEFAULT_FOV_DEG,
    car_length_m: float = DEFAULT_CAR_LENGTH_M,
    car_width_m: float = DEFAULT_CAR_WIDTH_M,
) -> float:
    if bbox_width_px <= 0 or bbox_height_px <= 0:
        raise ValueError(
            "bbox_width_px and bbox_height_px must be positive, got "
            f"({bbox_width_px}, {bbox_height_px})"
        )

    focal_length_px = compute_focal_length_px(image_width_px, fov_deg)

    bbox_major = max(bbox_width_px, bbox_height_px)
    bbox_minor = min(bbox_width_px, bbox_height_px)

    z_from_length = (car_length_m * focal_length_px) / bbox_major
    z_from_width = (car_width_m * focal_length_px) / bbox_minor

    return (z_from_length + z_from_width) / 2.0


def classify_distance_band(
    distance_m: float,
    band_1_max_m: float = BAND_1_MAX_M,
    band_2_max_m: float = BAND_2_MAX_M,
) -> Optional[str]:
    if distance_m < 0:
        raise ValueError(f"distance_m must be non-negative, got {distance_m}")

    if distance_m < band_1_max_m:
        return "band_1"
    if distance_m <= band_2_max_m:
        return "band_2"
    return None


def classify_bbox_distance_band(
    bbox_width_px: float,
    bbox_height_px: float,
    image_width_px: int,
    fov_deg: float = DEFAULT_FOV_DEG,
    car_length_m: float = DEFAULT_CAR_LENGTH_M,
    car_width_m: float = DEFAULT_CAR_WIDTH_M,
    band_1_max_m: float = BAND_1_MAX_M,
    band_2_max_m: float = BAND_2_MAX_M,
) -> Tuple[float, Optional[str]]:
    """Estimate a bbox's distance and classify it into a band in one call."""
    distance_m = estimate_distance_m(
        bbox_width_px,
        bbox_height_px,
        image_width_px,
        fov_deg=fov_deg,
        car_length_m=car_length_m,
        car_width_m=car_width_m,
    )
    return distance_m, classify_distance_band(distance_m, band_1_max_m, band_2_max_m)
