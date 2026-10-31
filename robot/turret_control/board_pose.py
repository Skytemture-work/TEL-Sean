"""Live metric pose tracking for the fixed 2026 target board."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .board import BoardTarget, EngagementPlan
from .config import TurretConfig
from .official_board import BOARD_HEIGHT_CM, BOARD_WIDTH_CM, hole_position


@dataclass(frozen=True)
class BoardPose:
    """Board-to-camera pose and visual-motion state for one frame."""

    rvec: np.ndarray
    tvec: np.ndarray
    timestamp: float
    reprojection_error_px: float
    correspondence_count: int
    translation_speed_m_s: float
    rotation_speed_rad_s: float
    stable: bool
    source: str = "pnp"


def camera_matrix(config: TurretConfig) -> np.ndarray:
    """Build the OpenCV camera matrix from calibrated configuration."""

    return np.asarray(
        [
            [config.focal_length_x_px, 0.0, config.camera_center_x_px],
            [0.0, config.focal_length_y_px, config.camera_center_y_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_coefficients(config: TurretConfig) -> np.ndarray:
    return np.asarray(config.camera_distortion_coeffs, dtype=np.float64).reshape(-1, 1)


def _board_point(hole_id: str) -> np.ndarray:
    x_norm, y_norm = hole_position(hole_id)
    return np.asarray(
        [
            x_norm * BOARD_WIDTH_CM / 100.0,
            y_norm * BOARD_HEIGHT_CM / 100.0,
            0.0,
        ],
        dtype=np.float64,
    )


def _unique_targets_by_id(plan: Optional[EngagementPlan]) -> Dict[str, BoardTarget]:
    targets: Dict[str, BoardTarget] = {}
    if plan is None:
        return targets
    for target in plan.targets:
        if target.hole_id is not None and target.hole_id not in targets:
            targets[target.hole_id] = target
    return targets


def _match_observations(
    observed: Sequence[BoardTarget],
    reference_plan: Optional[EngagementPlan],
) -> Tuple[np.ndarray, np.ndarray]:
    """Match raw YOLO centres to projected official hole identities."""

    references = list(_unique_targets_by_id(reference_plan).values())
    if not observed or not references:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )

    pairs = []
    for observed_index, target in enumerate(observed):
        for reference_index, reference in enumerate(references):
            distance_px = math.hypot(
                target.u_px - reference.u_px,
                target.v_px - reference.v_px,
            )
            # The reference may come from a quad projection or a partial-grid
            # homography.  Keep the gate well below inter-hole spacing while
            # allowing mask-centroid noise and moderate frame-to-frame motion.
            gate_px = max(14.0, min(100.0, reference.size_px * 1.35))
            if distance_px <= gate_px:
                pairs.append(
                    (distance_px, observed_index, reference_index)
                )

    pairs.sort()
    used_observed = set()
    used_references = set()
    object_points = []
    image_points = []
    for _, observed_index, reference_index in pairs:
        if observed_index in used_observed or reference_index in used_references:
            continue
        reference = references[reference_index]
        if reference.hole_id is None:
            continue
        used_observed.add(observed_index)
        used_references.add(reference_index)
        object_points.append(_board_point(reference.hole_id))
        target = observed[observed_index]
        image_points.append((target.u_px, target.v_px))

    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
    )


def _quad_correspondences(board_quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    object_points = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (BOARD_WIDTH_CM / 100.0, 0.0, 0.0),
            (BOARD_WIDTH_CM / 100.0, BOARD_HEIGHT_CM / 100.0, 0.0),
            (0.0, BOARD_HEIGHT_CM / 100.0, 0.0),
        ],
        dtype=np.float64,
    )
    image_points = np.asarray(board_quad, dtype=np.float64).reshape(4, 2)
    return object_points, image_points


def _non_collinear(object_points: np.ndarray) -> bool:
    if len(object_points) < 4:
        return False
    planar = object_points[:, :2]
    return np.linalg.matrix_rank(planar - np.mean(planar, axis=0)) >= 2


def _rotation_delta_rad(first_rvec: np.ndarray, second_rvec: np.ndarray) -> float:
    first_rotation, _ = cv2.Rodrigues(first_rvec)
    second_rotation, _ = cv2.Rodrigues(second_rvec)
    delta = second_rotation @ first_rotation.T
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


class BoardPoseTracker:
    """Estimate, smooth and validate the live board pose every frame."""

    def __init__(self, config: TurretConfig):
        self.config = config
        self._pose: Optional[BoardPose] = None
        self._stable_frames = 0
        self._last_stable_pnp_at: Optional[float] = None

    @property
    def last_pose(self) -> Optional[BoardPose]:
        return self._pose

    def is_fresh(self, now: Optional[float] = None) -> bool:
        if self._pose is None:
            return False
        current = time.monotonic() if now is None else now
        return current - self._pose.timestamp <= self.config.board_pose_stale_timeout_s

    def update(
        self,
        observed_targets: Sequence[BoardTarget],
        reference_plan: Optional[EngagementPlan],
        board_quad: Optional[np.ndarray] = None,
        now: Optional[float] = None,
    ) -> Optional[BoardPose]:
        timestamp = time.monotonic() if now is None else now
        previous_pose = self._pose
        if (
            previous_pose is not None
            and timestamp - previous_pose.timestamp
            > self.config.board_pose_stale_timeout_s
        ):
            # After a long visual gap, reacquire from the new observation
            # instead of smoothing toward a pose from the robot's old place.
            previous_pose = None
            self._stable_frames = 0
        object_points, image_points = _match_observations(
            observed_targets, reference_plan
        )
        if not _non_collinear(object_points):
            if (
                board_quad is None
                or len(object_points)
                < self.config.board_pose_min_quad_hole_support
            ):
                self._stable_frames = 0
                return None
            object_points, image_points = _quad_correspondences(board_quad)

        matrix = camera_matrix(self.config)
        distortion = distortion_coefficients(self.config)
        try:
            if len(object_points) >= 6:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points,
                    image_points,
                    matrix,
                    distortion,
                    iterationsCount=100,
                    reprojectionError=self.config.board_pose_ransac_error_px,
                    confidence=0.995,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                correspondence_count = (
                    int(len(inliers)) if inliers is not None else 0
                )
                if not success or correspondence_count < 4:
                    self._stable_frames = 0
                    return None
                inlier_indices = np.asarray(inliers, dtype=np.int32).reshape(-1)
                object_points = object_points[inlier_indices]
                image_points = image_points[inlier_indices]
            else:
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    matrix,
                    distortion,
                    flags=(
                        cv2.SOLVEPNP_IPPE
                        if hasattr(cv2, "SOLVEPNP_IPPE")
                        else cv2.SOLVEPNP_ITERATIVE
                    ),
                )
                correspondence_count = len(object_points)
                if not success:
                    self._stable_frames = 0
                    return None
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points,
                    image_points,
                    matrix,
                    distortion,
                    rvec,
                    tvec,
                )
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, matrix, distortion
            )
        except (cv2.error, ValueError, TypeError):
            self._stable_frames = 0
            return None

        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            self._stable_frames = 0
            return None
        if float(tvec[2, 0]) <= 0.0:
            self._stable_frames = 0
            return None

        errors = np.linalg.norm(
            projected.reshape(-1, 2) - image_points,
            axis=1,
        )
        reprojection_error = float(np.sqrt(np.mean(errors * errors)))
        if reprojection_error > self.config.board_pose_max_reprojection_error_px:
            self._stable_frames = 0
            return None

        translation_speed = math.inf
        rotation_speed = math.inf
        if previous_pose is not None:
            dt = timestamp - previous_pose.timestamp
            if dt > 1e-4:
                raw_translation_speed = float(
                    np.linalg.norm(tvec - previous_pose.tvec) / dt
                )
                raw_rotation_speed = (
                    _rotation_delta_rad(previous_pose.rvec, rvec) / dt
                )
                if (
                    raw_translation_speed
                    > self.config.board_pose_max_plausible_translation_m_s
                    or raw_rotation_speed
                    > self.config.board_pose_max_plausible_rotation_rad_s
                ):
                    self._stable_frames = 0
                    return None

                alpha = self.config.board_pose_smoothing_alpha
                tvec = (1.0 - alpha) * previous_pose.tvec + alpha * tvec
                rvec = (1.0 - alpha) * previous_pose.rvec + alpha * rvec
                translation_speed = float(
                    np.linalg.norm(tvec - previous_pose.tvec) / dt
                )
                rotation_speed = _rotation_delta_rad(previous_pose.rvec, rvec) / dt

        motion_stable = (
            translation_speed <= self.config.board_pose_fire_translation_m_s
            and rotation_speed <= self.config.board_pose_fire_rotation_rad_s
        )
        self._stable_frames = self._stable_frames + 1 if motion_stable else 0
        stable = self._stable_frames >= self.config.board_pose_stable_frames
        pose = BoardPose(
            rvec=rvec,
            tvec=tvec,
            timestamp=timestamp,
            reprojection_error_px=reprojection_error,
            correspondence_count=correspondence_count,
            translation_speed_m_s=translation_speed,
            rotation_speed_rad_s=rotation_speed,
            stable=stable,
        )
        self._pose = pose
        if stable:
            self._last_stable_pnp_at = timestamp
        else:
            # A newly observed moving pose must not fall back to an old one.
            self._last_stable_pnp_at = None
        return pose

    def update_partial(
        self,
        observed_targets: Sequence[BoardTarget],
        reference_plan: Optional[EngagementPlan],
        now: Optional[float] = None,
    ) -> Optional[BoardPose]:
        """Track a recent stable pose using visible, matched hole centres.

        Sparse observations can only update image-plane translation; depth and
        rotation remain assumptions and the independent PnP age is never reset.
        """
        timestamp = time.monotonic() if now is None else now
        previous = self._pose
        reference_at = self._last_stable_pnp_at
        if (previous is None or reference_plan is None or reference_at is None
                or timestamp < previous.timestamp
                or timestamp - previous.timestamp > self.config.board_pose_stale_timeout_s
                or timestamp - reference_at > self.config.board_pose_partial_max_age_s):
            return None
        projected = list(project_targets_from_pose(reference_plan, previous, self.config).values())
        if len(projected) < 2:
            return None
        positions = np.asarray([(t.u_px, t.v_px) for t in projected])
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        distances += np.eye(len(projected)) * 1e9
        spacing = float(np.min(distances))
        gate = min(20.0, spacing * 0.2)
        pairs = []
        for observed in observed_targets:
            if (not all(math.isfinite(v) for v in
                        (observed.u_px, observed.v_px, observed.size_px, observed.confidence))
                    or observed.confidence < self.config.board_pose_partial_min_confidence):
                continue
            delta = positions - (observed.u_px, observed.v_px)
            distance = np.linalg.norm(delta, axis=1)
            index = int(np.argmin(distance))
            expected = projected[index]
            if (distance[index] <= gate and expected.size_px > 0
                    and 0.75 * expected.size_px <= observed.size_px <= 1.3 * expected.size_px):
                pairs.append((float(distance[index]), index, observed, expected))
        matches = []
        used = set()
        for _, index, observed, expected in sorted(pairs, key=lambda item: item[0]):
            if index not in used:
                matches.append((observed, expected))
                used.add(index)
        if not matches:
            return None
        shifts = np.asarray([(o.u_px - e.u_px, o.v_px - e.v_px) for o, e in matches])
        shift = np.median(shifts, axis=0)
        if np.max(np.linalg.norm(shifts - shift, axis=1)) > max(2.0, spacing * 0.04):
            return None
        # Undistort the paired image points before estimating lateral motion.
        points = np.asarray([(e.u_px, e.v_px) for _, e in matches]
                            + [(o.u_px, o.v_px) for o, _ in matches], dtype=np.float64)
        normalized = cv2.undistortPoints(points.reshape(-1, 1, 2), camera_matrix(self.config),
                                         distortion_coefficients(self.config)).reshape(-1, 2)
        rotation, _ = cv2.Rodrigues(previous.rvec)
        translations = []
        for index, (_, expected) in enumerate(matches):
            depth = float((rotation @ _board_point(expected.hole_id).reshape(3, 1) + previous.tvec)[2, 0])
            translations.append((normalized[index + len(matches)] - normalized[index]) * depth)
        displacement = np.median(np.asarray(translations), axis=0)
        dt = timestamp - previous.timestamp
        if dt <= 1e-4:
            return None
        speed = float(np.linalg.norm(displacement) / dt)
        if speed > self.config.board_pose_fire_translation_m_s:
            self._last_stable_pnp_at = None
            return None
        tvec = previous.tvec.copy()
        tvec[:2, 0] += displacement
        pose = BoardPose(
            rvec=previous.rvec.copy(), tvec=tvec, timestamp=timestamp,
            reprojection_error_px=float(np.max(np.linalg.norm(shifts - shift, axis=1))),
            correspondence_count=len(matches), translation_speed_m_s=speed,
            rotation_speed_rad_s=0.0, stable=True, source="partial",
        )
        self._pose = pose
        return pose


def project_targets_from_pose(
    template_plan: EngagementPlan,
    pose: BoardPose,
    config: TurretConfig,
) -> Dict[str, BoardTarget]:
    """Project every unique official target using the newest metric pose."""

    matrix = camera_matrix(config)
    distortion = distortion_coefficients(config)
    rotation, _ = cv2.Rodrigues(pose.rvec)
    projected: Dict[str, BoardTarget] = {}
    for hole_id, target in _unique_targets_by_id(template_plan).items():
        center = _board_point(hole_id)
        diameter_m = 0.20 if hole_id in {"H10", "H11", "H12"} else 0.40
        radius = diameter_m / 2.0
        sample_points = np.asarray(
            [
                center,
                center + (-radius, 0.0, 0.0),
                center + (radius, 0.0, 0.0),
                center + (0.0, -radius, 0.0),
                center + (0.0, radius, 0.0),
            ],
            dtype=np.float64,
        )
        image_points, _ = cv2.projectPoints(
            sample_points,
            pose.rvec,
            pose.tvec,
            matrix,
            distortion,
        )
        image_points = image_points.reshape(-1, 2)
        camera_point = rotation @ center.reshape(3, 1) + pose.tvec
        depth_m = float(camera_point[2, 0])
        if depth_m <= 0.0:
            continue
        width_px = float(np.linalg.norm(image_points[2] - image_points[1]))
        height_px = float(np.linalg.norm(image_points[4] - image_points[3]))
        projected[hole_id] = replace(
            target,
            u_px=float(image_points[0, 0]),
            v_px=float(image_points[0, 1]),
            width_px=width_px,
            height_px=height_px,
            diameter_px=(width_px + height_px) / 2.0,
            diameter_m=diameter_m,
            distance_m=depth_m,
            distance_source="board_pose",
        )
    return projected


def prefer_observed_depth(
    projected: Dict[str, BoardTarget], observed: Sequence[BoardTarget]
) -> Dict[str, BoardTarget]:
    """Use same-frame stereo depth for uniquely matched visible hole masks."""
    references = list(projected.items())
    if not references:
        return projected
    pairs = []
    for observation_index, target in enumerate(observed):
        if (target.distance_source != "zed_depth" or target.distance_m is None
                or not math.isfinite(target.distance_m) or target.distance_m <= 0):
            continue
        for hole_id, reference in references:
            distance = math.hypot(target.u_px - reference.u_px, target.v_px - reference.v_px)
            gate = min(40.0, max(8.0, reference.size_px * .6))
            if (distance <= gate and reference.size_px > 0
                    and .5 * reference.size_px <= target.size_px <= 1.5 * reference.size_px):
                pairs.append((distance, observation_index, hole_id))
    result = dict(projected)
    used_observations, used_holes = set(), set()
    for _, observation_index, hole_id in sorted(pairs):
        if observation_index in used_observations or hole_id in used_holes:
            continue
        used_observations.add(observation_index)
        used_holes.add(hole_id)
        result[hole_id] = replace(result[hole_id],
                                  distance_m=observed[observation_index].distance_m,
                                  distance_source="zed_depth")
    return result


def update_plan_geometry(
    plan: Optional[EngagementPlan],
    live_targets: Dict[str, BoardTarget],
) -> None:
    """Update pixels/range in-place while preserving plan index and shot state."""

    if plan is None or not live_targets:
        return
    updated = []
    for target in plan.targets:
        live = live_targets.get(target.hole_id or "")
        if live is None:
            updated.append(target)
            continue
        updated.append(
            replace(
                target,
                u_px=live.u_px,
                v_px=live.v_px,
                width_px=live.width_px,
                height_px=live.height_px,
                diameter_px=live.diameter_px,
                distance_m=live.distance_m,
                distance_source=live.distance_source,
            )
        )
    plan.targets[:] = updated
