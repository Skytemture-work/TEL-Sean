"""Geometry helpers for instance-segmentation masks and ZED depth.

The important detail for this project is that a detected hole is not a solid
object.  The depth at the hole centre is commonly the background behind the
board.  These helpers therefore use the segmentation contour for the hole
geometry and sample an annulus just outside it, where the board surface is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskGeometry:
    """Useful image geometry derived from one instance mask."""

    contour: np.ndarray
    center_u: float
    center_v: float
    left: int
    top: int
    width_px: float
    height_px: float
    area_px: float
    equivalent_diameter_px: float


def contour_geometry(contour: np.ndarray) -> Optional[MaskGeometry]:
    """Return centroid, bounds and equivalent diameter for one contour."""

    points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return None

    contour_i = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    area = float(cv2.contourArea(contour_i))
    if area <= 0.0:
        return None

    moments = cv2.moments(contour_i)
    if moments["m00"] == 0.0:
        return None

    left, top, width, height = cv2.boundingRect(contour_i)
    return MaskGeometry(
        contour=contour_i,
        center_u=float(moments["m10"] / moments["m00"]),
        center_v=float(moments["m01"] / moments["m00"]),
        left=int(left),
        top=int(top),
        width_px=float(width),
        height_px=float(height),
        area_px=area,
        equivalent_diameter_px=float(2.0 * np.sqrt(area / np.pi)),
    )


def polygon_geometry(polygon: Sequence[Sequence[float]]) -> Optional[MaskGeometry]:
    """Convert an Ultralytics ``result.masks.xy`` polygon to geometry."""

    return contour_geometry(np.asarray(polygon, dtype=np.float32))


def binary_mask_geometry(
    mask: np.ndarray, frame_shape: Tuple[int, int]
) -> Optional[MaskGeometry]:
    """Extract the largest contour from a low-resolution binary mask."""

    height, width = frame_shape[:2]
    array = np.asarray(mask)
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        return None

    resized = cv2.resize(
        array.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    binary = (resized > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return contour_geometry(max(contours, key=cv2.contourArea))


def _mask_from_polygon(
    polygon: np.ndarray, frame_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    geometry = contour_geometry(polygon)
    if geometry is None:
        return None
    height, width = frame_shape[:2]
    full_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(full_mask, [geometry.contour], 1)
    return full_mask.astype(bool)


def _mask_from_binary(
    mask: np.ndarray, frame_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    height, width = frame_shape[:2]
    array = np.asarray(mask)
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        return None
    resized = cv2.resize(
        array.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0.5


def sample_board_depth(
    depth_np: np.ndarray,
    *,
    polygon: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    ring_px: int = 6,
    minimum_samples: int = 10,
) -> Optional[float]:
    """Sample board depth just outside a hole mask.

    The hole itself is excluded.  A robust median/MAD filter removes invalid
    values and a small number of edge/background outliers.  Returned depth is
    in the same units as ``depth_np`` (metres for the current ZED setup).
    """

    depth = np.asarray(depth_np)
    if depth.ndim > 2:
        depth = depth[..., 0]
    if depth.ndim != 2:
        return None

    frame_shape = depth.shape[:2]
    if polygon is not None:
        inside = _mask_from_polygon(polygon, frame_shape)
    elif mask is not None:
        inside = _mask_from_binary(mask, frame_shape)
    else:
        return None
    if inside is None or not np.any(inside):
        return None

    ring_px = max(1, int(ring_px))
    kernel_size = ring_px * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dilated = cv2.dilate(inside.astype(np.uint8), kernel) > 0
    ring = dilated & ~inside

    values = depth[ring]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < minimum_samples:
        return None

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(0.03, 3.0 * 1.4826 * mad)
    inliers = values[np.abs(values - median) <= threshold]
    if inliers.size < minimum_samples:
        return None
    return float(np.median(inliers))
