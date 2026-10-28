"""Black target-board surface detection and candidate filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .board import BoardTarget
from .official_board import BOARD_HEIGHT_CM, BOARD_WIDTH_CM, hole_position_for_cell


@dataclass(frozen=True)
class BoardSurface:
    """Visible part of the dark board surface."""

    contour: np.ndarray
    mask: np.ndarray
    area_px: float
    quad: Optional[np.ndarray] = None


def _ordered_quad(points: np.ndarray) -> Optional[np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered = np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )
    if abs(cv2.contourArea(ordered.reshape(-1, 1, 2))) < 1.0:
        return None
    return ordered


def _quad_from_contour(
    contour: np.ndarray, width: int, height: int
) -> Optional[np.ndarray]:
    """Return a complete, ordered quadrilateral when all corners are visible."""

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approximation) != 4:
        return None
    if any(
        x <= 1 or y <= 1 or x >= width - 2 or y >= height - 2
        for x, y in approximation.reshape(-1, 2)
    ):
        return None
    return _ordered_quad(approximation.reshape(-1, 2))


def _targets_inside_contour(
    contour: np.ndarray, targets: Sequence[BoardTarget]
) -> list[BoardTarget]:
    """Return detections whose centres are inside one surface candidate."""

    return [
        target
        for target in targets
        if cv2.pointPolygonTest(
            contour, (float(target.u_px), float(target.v_px)), False
        ) >= 0
    ]


def _target_geometry_score(
    targets: Sequence[BoardTarget], quad: Optional[np.ndarray]
) -> float:
    """Score how well candidate detections fit the official 3x4 layout."""

    if not targets or quad is None:
        return 0.0
    source = np.asarray(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        dtype=np.float32,
    )
    try:
        inverse = np.linalg.inv(cv2.getPerspectiveTransform(source, quad))
        points = np.asarray(
            [(target.u_px, target.v_px) for target in targets],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        board_points = cv2.perspectiveTransform(points, inverse).reshape(-1, 2)
    except (cv2.error, np.linalg.LinAlgError):
        return 0.0

    official = np.asarray(
        [
            hole_position_for_cell(column, row)
            for row in range(4)
            for column in range(3)
        ],
        dtype=np.float32,
    )
    errors_cm = []
    nearest_cells = []
    for point in board_points:
        delta = (official - point) * np.asarray(
            (BOARD_WIDTH_CM, BOARD_HEIGHT_CM), dtype=np.float32
        )
        distances = np.linalg.norm(delta, axis=1)
        nearest = int(np.argmin(distances))
        error_cm = float(distances[nearest])
        # This is only a ranking signal.  The actual plan still applies the
        # configured +/-2 cm geometry validation later.
        errors_cm.append(max(0.0, 1.0 - error_cm / 20.0))
        nearest_cells.append(nearest)
    if not errors_cm:
        return 0.0
    unique_ratio = len(set(nearest_cells)) / len(nearest_cells)
    return 0.75 * float(np.mean(errors_cm)) + 0.25 * unique_ratio


def detect_dark_board_surface(
    frame: np.ndarray,
    value_threshold: int = 105,
    min_area_fraction: float = 0.04,
    targets: Optional[Sequence[BoardTarget]] = None,
) -> Optional[BoardSurface]:
    """Find the largest plausible dark board region.

    The contour is allowed to touch the image border, so a board whose lower
    edge is outside the camera view can still be used as an ROI.  ``quad`` is
    provided only when a complete four-corner contour is visible.
    """

    if frame is None or frame.ndim != 3:
        return None
    height, width = frame.shape[:2]
    if height < 10 or width < 10:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    otsu_threshold, _ = cv2.threshold(
        value, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    percentile_threshold = float(np.percentile(value, 30.0))
    thresholds = {
        int(np.clip(value_threshold, 20, 220)),
        int(np.clip(otsu_threshold, 20, 220)),
        int(np.clip(percentile_threshold, 20, 220)),
    }

    frame_area = float(height * width)
    kernel_size = max(5, int(round(min(height, width) * 0.015)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    target_candidates = list(targets or [])
    candidates = []
    for threshold in thresholds:
        dark = cv2.inRange(
            value,
            0,
            int(threshold),
        )
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < frame_area * min_area_fraction:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < width * 0.15 or box_height < height * 0.15:
                continue
            box_area = float(box_width * box_height)
            rectangularity = area / box_area if box_area else 0.0
            if rectangularity < 0.35:
                continue
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area else 0.0
            if solidity < 0.45:
                continue

            touches_border = sum(
                (
                    x <= 1,
                    y <= 1,
                    x + box_width >= width - 2,
                    y + box_height >= height - 2,
                )
            )
            # A threshold that swallows the whole image is usually a bad
            # adaptive threshold, not the target board.
            if area > frame_area * 0.85 and touches_border >= 2:
                continue

            quad = _quad_from_contour(contour, width, height)
            inside_targets = _targets_inside_contour(contour, target_candidates)
            target_count = len(inside_targets)
            target_fraction = (
                target_count / len(target_candidates) if target_candidates else 0.0
            )
            target_count_score = min(target_count, 12) / 12.0
            target_support = 0.5 * target_fraction + 0.5 * target_count_score
            geometry_score = _target_geometry_score(inside_targets, quad)

            min_rect_width, min_rect_height = cv2.minAreaRect(contour)[1]
            oriented_aspect = (
                max(min_rect_width, min_rect_height)
                / min(min_rect_width, min_rect_height)
                if min(min_rect_width, min_rect_height) > 0
                else 0.0
            )
            expected_aspect = BOARD_HEIGHT_CM / BOARD_WIDTH_CM
            aspect_score = 1.0 / (1.0 + abs(oriented_aspect - expected_aspect))
            area_score = min(area / frame_area / 0.15, 1.0)
            # Solid, board-like shapes and detections that agree with the
            # fixed 3x4 layout outrank thin black trusses.
            shape_score = (
                0.35 * rectangularity
                + 0.30 * solidity
                + 0.20 * aspect_score
                + 0.15 * area_score
            )
            score = 0.55 * shape_score + 0.30 * target_support + 0.15 * geometry_score
            candidates.append((score, area, contour, quad, target_count))

    if not candidates:
        return None
    _, area, contour, quad, _target_count = max(
        candidates, key=lambda item: item[0]
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED)

    return BoardSurface(
        contour=contour,
        mask=mask.astype(bool),
        area_px=area,
        quad=quad,
    )


def filter_targets_to_surface(
    targets: Sequence[BoardTarget], surface: Optional[BoardSurface]
) -> list[BoardTarget]:
    """Keep only target centres inside the visible board surface."""

    if surface is None:
        return []
    kept = []
    for target in targets:
        u = int(round(target.u_px))
        v = int(round(target.v_px))
        if 0 <= v < surface.mask.shape[0] and 0 <= u < surface.mask.shape[1]:
            if surface.mask[v, u]:
                kept.append(target)
    return kept
