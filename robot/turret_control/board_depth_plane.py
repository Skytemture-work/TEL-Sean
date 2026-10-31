"""Estimate a visible target-board plane from rectified ZED depth pixels."""

from dataclasses import dataclass, replace
import math
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from .board import BoardTarget
from .board_pose import BoardPose, camera_matrix, distortion_coefficients
from .config import TurretConfig
from .official_board import BOARD_HEIGHT_CM, BOARD_WIDTH_CM


@dataclass(frozen=True)
class DepthPlane:
    normal: np.ndarray
    offset_m: float
    inlier_count: int
    sample_count: int
    median_error_m: float

    def depth_at(self, u: float, v: float, config: TurretConfig) -> Optional[float]:
        ray = np.asarray(((u - config.camera_center_x_px) / config.focal_length_x_px,
                          (v - config.camera_center_y_px) / config.focal_length_y_px, 1.0))
        denominator = float(self.normal @ ray)
        if abs(denominator) < 1e-8:
            return None
        depth = -self.offset_m / denominator
        return depth if math.isfinite(depth) and depth > 0 else None


def fit_board_depth_plane(
    depth_data: np.ndarray,
    surface_mask: np.ndarray,
    pose: BoardPose,
    projected: Dict[str, BoardTarget],
    observed: Sequence[BoardTarget],
    config: TurretConfig,
) -> Optional[DepthPlane]:
    """Fit same-frame ZED depth within the visible board and check PnP agreement.

    These conservative thresholds are initial values; field data is required
    to characterize stereo noise and disagreement at competition distances.
    """
    depth = np.asarray(depth_data)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    mask = np.asarray(surface_mask)
    if depth.ndim != 2 or mask.shape != depth.shape:
        return None
    height, width = depth.shape
    if (width, height) != (config.image_width_px, config.image_height_px):
        return None
    # This point projection uses the SDK's rectified pinhole image.
    if any(config.camera_distortion_coeffs):
        return None
    rotation, _ = cv2.Rodrigues(pose.rvec)
    normal_ref = rotation[:, 2]
    origin = np.asarray(pose.tvec, dtype=np.float64).reshape(3)
    corners = np.asarray(((0, 0, 0), (BOARD_WIDTH_CM / 100, 0, 0),
                          (BOARD_WIDTH_CM / 100, BOARD_HEIGHT_CM / 100, 0),
                          (0, BOARD_HEIGHT_CM / 100, 0)), dtype=np.float64)
    if np.any((rotation @ corners.T + origin.reshape(3, 1))[2] <= 0):
        return None
    image_corners, _ = cv2.projectPoints(corners, pose.rvec, pose.tvec,
                                         camera_matrix(config), distortion_coefficients(config))
    quad = image_corners.reshape(-1, 2)
    if not np.isfinite(quad).all() or np.any(np.abs(quad) > 1e6):
        return None
    roi = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(roi, np.round(quad).astype(np.int32), 1)
    roi = (roi > 0) & mask.astype(bool)
    roi = cv2.erode(roi.astype(np.uint8), np.ones((7, 7), np.uint8))
    # Exclude both predicted holes (including occluded holes) and detections.
    for target in list(projected.values()) + list(observed):
        values = (target.u_px, target.v_px, target.width_px, target.height_px)
        if not all(math.isfinite(value) for value in values):
            continue
        if not 0 <= target.u_px < width or not 0 <= target.v_px < height:
            continue
        half_width = max(2, min(width, int(target.width_px * .6 + 4)))
        half_height = max(2, min(height, int(target.height_px * .6 + 4)))
        cv2.rectangle(roi, (int(target.u_px) - half_width, int(target.v_px) - half_height),
                      (int(target.u_px) + half_width, int(target.v_px) + half_height), 0, -1)
    yy, xx = np.mgrid[0:height:6, 0:width:6]
    zz = depth[::6, ::6]
    valid = (roi[::6, ::6] > 0) & np.isfinite(zz) & (zz > 0)
    x, y, z = xx[valid], yy[valid], zz[valid].astype(np.float64)
    if len(z) < 120:
        return None
    if len(z) > 2500:
        selected = np.random.default_rng(0).choice(len(z), 2500, replace=False)
        x, y, z = x[selected], y[selected], z[selected]
    points = np.column_stack(((x - config.camera_center_x_px) * z / config.focal_length_x_px,
                              (y - config.camera_center_y_px) * z / config.focal_length_y_px, z))
    rng = np.random.default_rng(1)
    best = None
    for _ in range(64):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        if abs(float(normal @ normal_ref)) < math.cos(math.radians(20)):
            continue
        residual = np.abs(points @ normal - a @ normal)
        inliers = residual <= .08
        if best is None or np.count_nonzero(inliers) > np.count_nonzero(best):
            best = inliers
    if best is None or np.count_nonzero(best) < max(120, math.ceil(.55 * len(points))):
        return None
    # A narrow strip cannot reliably constrain a whole-board plane.
    if np.ptp(x[best]) < max(24, width * .08) or np.ptp(y[best]) < max(24, height * .08):
        return None
    inlier_points = points[best]
    center = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vh[-1]
    if abs(float(normal @ normal_ref)) < math.cos(math.radians(20)):
        return None
    offset = -float(normal @ center)
    error = float(np.median(np.abs(inlier_points @ normal + offset)))
    if error > .04:
        return None
    plane = DepthPlane(normal, offset, len(inlier_points), len(points), error)
    # Check multiple visible/predicted positions against the independent PnP
    # plane; this rejects coherent foreground/background planes.
    checked = 0
    for target in projected.values():
        if not (0 <= target.u_px < width and 0 <= target.v_px < height):
            continue
        if target.distance_m is None or target.distance_m <= 0:
            continue
        measured = plane.depth_at(target.u_px, target.v_px, config)
        if measured is None or abs(measured - target.distance_m) > max(.75, .2 * target.distance_m):
            return None
        checked += 1
    return plane if checked >= 3 else None


def apply_board_depth_plane(
    projected: Dict[str, BoardTarget], plane: DepthPlane, config: TurretConfig
) -> Dict[str, BoardTarget]:
    result = dict(projected)
    for hole_id, target in projected.items():
        if not (0 <= target.u_px < config.image_width_px and 0 <= target.v_px < config.image_height_px):
            continue
        depth = plane.depth_at(target.u_px, target.v_px, config)
        if depth is not None:
            result[hole_id] = replace(target, distance_m=depth, distance_source="zed_plane")
    return result
