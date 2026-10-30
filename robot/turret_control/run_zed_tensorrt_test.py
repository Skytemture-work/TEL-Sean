"""Preview YOLO segmentation through TensorRT with ZED depth.

This diagnostic intentionally has no motor-control imports or commands.  It
overlays segmentation results, the depth at each detection centre, and timing.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .board import (
    _cluster_1d,
    build_plan,
    build_plan_from_board_quad,
    build_virtual_plan,
    build_virtual_plan_from_partial,
    estimate_board_distance_from_targets,
)
from .config import TurretConfig
from .detection import boxes_to_board_targets
from .run_yolo_demo import _draw_board_plan
from .board_surface import BoardSurface, detect_dark_board_surface, filter_targets_to_surface
from .official_board import BOARD_HEIGHT_CM, BOARD_WIDTH_CM, hole_position, hole_position_for_cell
from .vision_geometry import polygon_geometry, sample_board_depth


DEFAULT_ENGINE = Path.home() / "Desktop" / "best.engine"


def _valid_estimated_quad(quad, frame_shape):
    height, width = frame_shape[:2]
    return (
        np.isfinite(quad).all() and cv2.isContourConvex(quad.astype(np.float32))
        and 0.01 * height * width <= abs(cv2.contourArea(quad)) <= 0.8 * height * width
        and np.all(quad[:, 0] >= -width) and np.all(quad[:, 0] <= 2 * width)
        and np.all(quad[:, 1] >= -height) and np.all(quad[:, 1] <= 2 * height)
    )


def _estimate_quad_from_large_rows(candidates, frame_shape):
    """Fit the three large-hole rows when the entire small-hole row is hidden."""
    if len(candidates) < 7:
        return None
    median_size = float(np.median([target.size_px for target in candidates]))
    candidates = [t for t in candidates if 0.65 * median_size <= t.size_px <= 1.5 * median_size]
    if len(candidates) < 7:
        return None
    columns = _cluster_1d([t.u_px for t in candidates], 3)
    rows = _cluster_1d([t.v_px for t in candidates], 3)
    cells = {}
    for target in sorted(candidates, key=lambda t: t.confidence, reverse=True):
        col = min(range(3), key=lambda i: abs(target.u_px - columns[i]))
        row = min(range(3), key=lambda i: abs(target.v_px - rows[i]))
        cells.setdefault((col, row), target)
    if len(cells) < 7 or len({c for c, _ in cells}) < 3 or len({r for _, r in cells}) < 3:
        return None
    source = np.asarray([hole_position_for_cell(c, r + 1) for c, r in cells], dtype=np.float32)
    observed = np.asarray([(t.u_px, t.v_px) for t in cells.values()], dtype=np.float32)
    matrix, inliers = cv2.findHomography(source, observed, cv2.RANSAC, max(3.0, median_size * 0.15))
    if matrix is None or inliers is None or int(np.count_nonzero(inliers)) < 7:
        return None
    inlier_cells = [cell for cell, valid in zip(cells, inliers.ravel()) if valid]
    if len({c for c, _ in inlier_cells}) < 3 or len({r for _, r in inlier_cells}) < 3:
        return None
    # A large hole's known diameter must agree with the fitted board scale.
    ratios = []
    for (column, row), target in cells.items():
        x, y = hole_position_for_cell(column, row + 1)
        offsets = np.asarray([[(x - 20 / BOARD_WIDTH_CM, y), (x + 20 / BOARD_WIDTH_CM, y),
                               (x, y - 20 / BOARD_HEIGHT_CM), (x, y + 20 / BOARD_HEIGHT_CM)]], dtype=np.float32)
        projected = cv2.perspectiveTransform(offsets, matrix)[0]
        expected_size = (np.linalg.norm(projected[1] - projected[0]) +
                         np.linalg.norm(projected[3] - projected[2])) / 2.0
        if expected_size > 0:
            ratios.append(target.size_px / expected_size)
    if not ratios or not 0.6 <= float(np.median(ratios)) <= 1.4:
        return None
    corners = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, matrix).reshape(4, 2)
    return quad if _valid_estimated_quad(quad, frame_shape) else None


def _estimate_board_quad(targets, frame_shape):
    """Estimate a preview-only board outline from well-supported hole geometry."""
    candidates = [target for target in targets if target.confidence >= 0.5]
    if len(candidates) < 8:
        return _estimate_quad_from_large_rows(candidates, frame_shape)
    plan = build_virtual_plan_from_partial(candidates, 0.2, 0.4)
    if len(plan.targets) != 12:
        return _estimate_quad_from_large_rows(candidates, frame_shape)
    predicted = np.asarray([(t.u_px, t.v_px) for t in plan.targets], dtype=np.float32)
    points = np.asarray([(t.u_px, t.v_px) for t in candidates], dtype=np.float32)
    distances = np.linalg.norm(points[:, None, :] - predicted[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    threshold = max(4.0, 0.2 * float(np.median([t.size_px for t in candidates])))
    matches = {}
    for index, predicted_index in enumerate(nearest):
        distance = float(distances[index, predicted_index])
        if distance <= threshold:
            matches[int(predicted_index)] = min(matches.get(int(predicted_index), float("inf")), distance)
    supported = [plan.targets[index] for index in matches]
    if (len(supported) < 8 or len({t.column for t in supported}) < 3
            or len({t.row for t in supported}) < 4):
        return _estimate_quad_from_large_rows(candidates, frame_shape)
    expected = np.asarray([hole_position(t.hole_id) for t in plan.targets], dtype=np.float32)
    matrix, _ = cv2.findHomography(expected, predicted, 0)
    if matrix is None:
        return None
    corners = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, matrix).reshape(4, 2)
    if not _valid_estimated_quad(quad, frame_shape):
        return _estimate_quad_from_large_rows(candidates, frame_shape)
    return quad


def _surface_from_estimated_quad(quad, frame_shape):
    contour = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 1)
    return BoardSurface(contour, mask.astype(bool), abs(cv2.contourArea(contour)), quad=None)


def _filter_preview_result(result, raw_targets, surface, minimum_confidence,
                           show_unverified=False):
    """Select the same YOLO boxes and masks as the board-surface filter."""
    if result.boxes is None:
        return None
    indices = [
        index for index, box in enumerate(result.boxes)
        if float(box.conf[0].item()) >= minimum_confidence
    ]
    if len(indices) != len(raw_targets):
        raise ValueError("YOLO boxes and converted board targets do not match")
    visible = raw_targets if surface is None and show_unverified else filter_targets_to_surface(raw_targets, surface)
    selected = {id(target) for target in visible}
    kept = [index for index, target in zip(indices, raw_targets) if id(target) in selected]
    return result[kept] if kept else None


def _grid_geometry_valid(plan, frame_shape=None):
    """Reject collapsed, crossed or implausibly distorted 3x4 preview grids."""
    if plan is None or len(plan.targets) != 12:
        return False
    cells = {(t.column, t.row): t for t in plan.targets}
    if len(cells) != 12 or any((c, r) not in cells for c in range(3) for r in range(4)):
        return False
    points = np.asarray([[(cells[c, r].u_px, cells[c, r].v_px)
                          for c in range(3)] for r in range(4)], dtype=np.float64)
    if not np.isfinite(points).all():
        return False
    if frame_shape is not None:
        h, w = frame_shape[:2]
        if (np.any(points[:, :, 0] < -w) or np.any(points[:, :, 0] > 2 * w)
                or np.any(points[:, :, 1] < -h) or np.any(points[:, :, 1] > 2 * h)):
            return False
    horizontal = points[:, 1:, :] - points[:, :-1, :]
    vertical = points[1:, :, :] - points[:-1, :, :]
    h_lengths = np.linalg.norm(horizontal, axis=2)
    v_lengths = np.linalg.norm(vertical, axis=2)
    median_size = float(np.median([t.size_px for t in plan.targets]))
    minimum_spacing = max(3.0, median_size * 0.4)
    for lengths in (h_lengths, v_lengths):
        if np.min(lengths) < minimum_spacing or np.min(lengths) < 0.3 * np.max(lengths):
            return False
    # Adjacent cells must form consistent, non-crossing rows and columns.
    for vectors in (horizontal.reshape(-1, 2), vertical.reshape(-1, 2)):
        direction = np.mean(vectors / np.linalg.norm(vectors, axis=1)[:, None], axis=0)
        length = np.linalg.norm(direction)
        if length < 0.5:
            return False
        cosines = (vectors @ direction) / (np.linalg.norm(vectors, axis=1) * length)
        if np.min(cosines) < 0.65:
            return False
    for row in range(3):
        for col in range(2):
            h, v = horizontal[row, col], vertical[row, col]
            cross = h[0] * v[1] - h[1] * v[0]
            if cross >= -0.25 * np.linalg.norm(h) * np.linalg.norm(v):
                return False
    # Perspective preserves straight lines, even when they are tilted in pixels.
    for line in list(points) + [points[:, col, :] for col in range(3)]:
        direction = line[-1] - line[0]
        distance = np.abs(direction[0] * (line[:, 1] - line[0, 1]) -
                          direction[1] * (line[:, 0] - line[0, 0])) / np.linalg.norm(direction)
        if np.max(distance) > max(3.0, 0.2 * np.linalg.norm(direction) / (len(line) - 1)):
            return False
    return True


class PreviewBoardTracker:
    """Temporarily track a well-supported board with visible matched holes."""

    def __init__(self, max_reference_age_s=10.0):
        self.max_reference_age_s = max_reference_age_s
        self._quad = None
        self._reference_at = None

    def update(self, targets, frame_shape, now):
        fresh = _estimate_board_quad(targets, frame_shape)
        if fresh is not None:
            fresh_plan = build_plan_from_board_quad(fresh, .2, .4)
            if _grid_geometry_valid(fresh_plan, frame_shape):
                self._quad = fresh.copy()
                self._reference_at = now
                return fresh, False
        if self._quad is None or self._reference_at is None or now - self._reference_at > self.max_reference_age_s:
            return None, False
        candidates = [t for t in targets if t.confidence >= .5]
        if not candidates:
            return None, False
        plan = build_plan_from_board_quad(self._quad, .2, .4)
        if not _grid_geometry_valid(plan, frame_shape):
            return None, False
        predicted = np.asarray([(t.u_px, t.v_px) for t in plan.targets], np.float32)
        observed = np.asarray([(t.u_px, t.v_px) for t in candidates], np.float32)
        distances = np.linalg.norm(observed[:, None, :] - predicted[None, :, :], axis=2)
        between = np.linalg.norm(predicted[:, None, :] - predicted[None, :, :], axis=2)
        between += np.eye(len(predicted), dtype=np.float32) * 1e6
        spacing = float(np.min(between))
        pairs = []
        for observed_index, candidate in enumerate(candidates):
            predicted_index = int(np.argmin(distances[observed_index]))
            distance = float(distances[observed_index, predicted_index])
            expected_size = plan.targets[predicted_index].size_px
            if distance <= spacing * .4 and .55 * expected_size <= candidate.size_px <= 1.6 * expected_size:
                pairs.append((distance, observed_index, predicted_index))
        used_observed, used_predicted, matches = set(), set(), []
        for _, observed_index, predicted_index in sorted(pairs):
            if observed_index not in used_observed and predicted_index not in used_predicted:
                used_observed.add(observed_index)
                used_predicted.add(predicted_index)
                matches.append((predicted_index, observed_index))
        if not matches:
            return None, False
        source = np.asarray([predicted[i] for i, _ in matches], np.float32)
        destination = np.asarray([observed[i] for _, i in matches], np.float32)
        non_collinear = (
            len(matches) >= 3
            and max(abs(cv2.contourArea(np.asarray(triangle, np.float32)))
                    for triangle in itertools.combinations(source, 3)) >= spacing * spacing * .2
        )
        if non_collinear:
            matrix, inliers = cv2.estimateAffine2D(
                source, destination, method=cv2.RANSAC,
                ransacReprojThreshold=max(3.0, spacing * .1),
            )
            if matrix is None or inliers is None or int(np.count_nonzero(inliers)) < 3:
                return None, False
        else:
            # One hole (or collinear holes) cannot determine perspective or
            # rotation. Only translate the recent reference and expire it.
            displacements = destination - source
            shift = np.median(displacements, axis=0)
            if np.max(np.linalg.norm(displacements - shift, axis=1)) > max(2.0, spacing * .08):
                return None, False
            matrix = np.asarray(((1.0, 0.0, shift[0]), (0.0, 1.0, shift[1])), np.float64)
        linear = matrix[:, :2]
        singular = np.linalg.svd(linear, compute_uv=False)
        if (np.linalg.det(linear) <= 0 or min(singular) < .85 or max(singular) > 1.15
                or max(singular) / min(singular) > 1.2):
            return None, False
        tracked = cv2.transform(self._quad.reshape(-1, 1, 2), matrix).reshape(4, 2)
        if not _valid_estimated_quad(tracked, frame_shape):
            return None, False
        if not _grid_geometry_valid(build_plan_from_board_quad(tracked, .2, .4), frame_shape):
            return None, False
        self._quad = tracked
        return tracked, True


def _select_grid_plan(board_ready, plan, quad_plan, inferred_plan,
                      observed_fallback_plan, estimated_quad, config, frame_shape=None):
    """Select a complete preview grid and report whether it is inferred."""
    if board_ready and _grid_geometry_valid(plan, frame_shape):
        return plan, False
    if _grid_geometry_valid(quad_plan, frame_shape):
        return quad_plan, True
    if estimated_quad is not None:
        candidate = build_plan_from_board_quad(
            estimated_quad, config.small_hole_diameter_m, config.large_hole_diameter_m,
            large_first=config.engage_large_first,
            left_to_right=config.engage_left_to_right,
            bottom_to_top=config.engage_bottom_to_top,
        )
        if _grid_geometry_valid(candidate, frame_shape):
            return candidate, True
    if _grid_geometry_valid(inferred_plan, frame_shape):
        return inferred_plan, True
    if _grid_geometry_valid(observed_fallback_plan, frame_shape):
        return observed_fallback_plan, True
    return None, False


def _draw_inferred_board(display: np.ndarray, plan, inferred: bool = True) -> None:
    """Draw the magenta board grid, marking geometry-only predictions."""
    color = (255, 0, 255)
    by_column = {}
    by_row = {}
    for target in plan.targets:
        if target.column is None or target.row is None:
            continue
        point = (round(target.u_px), round(target.v_px))
        by_column.setdefault(target.column, []).append((target.row, point))
        by_row.setdefault(target.row, []).append((target.column, point))
        cv2.drawMarker(display, point, color, cv2.MARKER_CROSS, 18, 2)
        cv2.circle(display, point, 12, color, 1, cv2.LINE_AA)
        cv2.putText(
            display,
            f"{'INFER ' if inferred else ''}{target.hole_id or 'H??'}",
            (point[0] + 8, point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    for points in by_column.values():
        points.sort()
        for (_, start), (_, end) in zip(points, points[1:]):
            cv2.line(display, start, end, color, 1, cv2.LINE_AA)
    for points in by_row.values():
        points.sort()
        for (_, start), (_, end) in zip(points, points[1:]):
            cv2.line(display, start, end, color, 1, cv2.LINE_AA)

    cv2.putText(
        display,
        "INFERRED BOARD ONLY - NO FIRE" if inferred else "OBSERVED BOARD GRID",
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely test a YOLO TensorRT segmentation engine on ZED."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--frames", type=int, default=0, help="0 runs until Q/Esc")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--save-frame",
        type=Path,
        help="save the first partial/inferred overlay frame to this path",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"找不到模型：{args.model}")

    # TensorRT 8.5 still asks for the removed NumPy alias on JetPack 5.
    if "bool" not in np.__dict__:
        np.bool = bool  # type: ignore[attr-defined]

    try:
        import pyzed.sl as sl
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"缺少 ZED/Ultralytics 套件：{exc}") from exc

    model = YOLO(str(args.model), task="segment")
    config = TurretConfig()
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.ULTRA

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"無法開啟 ZED：{status}")

    image = sl.Mat()
    depth = sl.Mat()
    runtime = sl.RuntimeParameters()
    window = "ZED YOLO TensorRT segmentation (Q/Esc: quit)"
    frame_count = 0
    started = time.perf_counter()
    saved_frame = False
    board_tracker = PreviewBoardTracker()

    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print("安全模式：只讀取相機並執行推論，不會連線或控制馬達。")
    print(f"模型：{args.model}")
    try:
        while not args.frames or frame_count < args.frames:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
            depth_data = depth.get_data()

            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=0,
                verbose=False,
            )[0]
            raw_detected = boxes_to_board_targets(result, args.conf)
            board_surface = detect_dark_board_surface(
                frame,
                value_threshold=config.board_dark_value_threshold,
                min_area_fraction=config.board_min_area_fraction,
                targets=raw_detected,
            )
            estimated_quad, tracked_estimate = board_tracker.update(
                raw_detected, frame.shape, time.monotonic()
            )
            estimated_surface = False
            if estimated_quad is not None and (board_surface is None or board_surface.quad is None):
                board_surface = _surface_from_estimated_quad(estimated_quad, frame.shape)
                estimated_surface = True
            elif board_surface is not None and board_surface.quad is None:
                # Irregular merged dark backgrounds are not a board boundary.
                board_surface = None
            detected = filter_targets_to_surface(raw_detected, board_surface)
            holes_unverified = board_surface is None
            preview_result = _filter_preview_result(
                result, raw_detected, board_surface, args.conf,
                show_unverified=holes_unverified,
            )
            quad_plan = None
            if board_surface is not None and board_surface.quad is not None:
                quad_distance = estimate_board_distance_from_targets(
                    detected,
                    board_surface.quad,
                    config.focal_length_x_px,
                    config.small_hole_diameter_m,
                    config.large_hole_diameter_m,
                    config.board_position_tolerance_cm,
                )
                quad_plan = build_plan_from_board_quad(
                    board_surface.quad,
                    config.small_hole_diameter_m,
                    config.large_hole_diameter_m,
                    board_distance_m=quad_distance,
                    large_first=config.engage_large_first,
                    left_to_right=config.engage_left_to_right,
                    bottom_to_top=config.engage_bottom_to_top,
                )
            plan = build_plan(
                detected,
                config.focal_length_x_px,
                config.small_hole_diameter_m,
                config.large_hole_diameter_m,
                config.engage_large_first,
                config.engage_left_to_right,
                config.engage_bottom_to_top,
                config.board_position_tolerance_cm,
            )
            board_ready = not estimated_surface and len(plan.targets) == 12 and all(
                target.hole_id is not None and target.diameter_m is not None
                for target in plan.targets
            ) and _grid_geometry_valid(plan, frame.shape)
            inferred_plan = None
            observed_fallback_plan = None
            preview_candidates = (
                [target for target in raw_detected if target.confidence >= 0.5]
                if holes_unverified else detected
            )
            if not board_ready and len(preview_candidates) == 12:
                # Twelve candidates should use their actual positions.  A
                # partial homography here could move a valid point elsewhere
                # when size classification alone was the failed check.
                candidate = build_virtual_plan(
                    preview_candidates,
                    config.small_hole_diameter_m,
                    config.large_hole_diameter_m,
                    config.engage_large_first,
                    config.engage_left_to_right,
                    config.engage_bottom_to_top,
                    config.board_position_tolerance_cm,
                )
                if len(candidate.targets) != 12:
                    candidate = None
                else:
                    observed_fallback_plan = candidate
            elif not board_ready and len(preview_candidates) >= 4:
                candidate = build_virtual_plan_from_partial(
                    preview_candidates,
                    config.small_hole_diameter_m,
                    config.large_hole_diameter_m,
                    config.engage_large_first,
                    config.engage_left_to_right,
                    config.engage_bottom_to_top,
                    focal_px=config.focal_length_x_px,
                    position_tolerance_cm=config.board_position_tolerance_cm,
                )
                if len(candidate.targets) == 12:
                    inferred_plan = candidate
            # Large class/confidence labels obscure small holes on the 3x4
            # target. Keep thin boxes and masks, then add only compact depth
            # text below.
            display = (
                preview_result.plot(labels=False, line_width=1)
                if preview_result is not None else frame.copy()
            )
            if board_surface is not None and not estimated_surface:
                cv2.drawContours(
                    display,
                    [board_surface.contour],
                    -1,
                    (0, 255, 255),
                    2,
                )
            if estimated_surface:
                cv2.polylines(display, [board_surface.contour], True, (255, 255, 0), 2)
                cv2.putText(display, "TRACKED BOARD - INFERRED" if tracked_estimate else "ESTIMATED BOARD", (12, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            elif holes_unverified:
                cv2.putText(display, "HOLES UNVERIFIED - BOARD UNKNOWN", (12, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
            overlay_plan, overlay_inferred = _select_grid_plan(
                board_ready, plan, quad_plan, inferred_plan, observed_fallback_plan,
                estimated_quad, config, frame.shape,
            )
            if overlay_plan is not None:
                _draw_board_plan(display, overlay_plan)
            if overlay_plan is not None:
                _draw_inferred_board(display, overlay_plan, inferred=overlay_inferred)
                if args.save_frame and overlay_inferred and not saved_frame:
                    args.save_frame.parent.mkdir(parents=True, exist_ok=True)
                    if cv2.imwrite(str(args.save_frame), display):
                        print(f"saved inferred frame: {args.save_frame}")
                        saved_frame = True
            elif raw_detected:
                cv2.putText(display, "GRID UNAVAILABLE", (12, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
            frame_count += 1
            elapsed = time.perf_counter() - started
            fps = frame_count / elapsed if elapsed else 0.0
            count = len(detected)
            detection_text = []

            if preview_result is not None and preview_result.boxes is not None:
                masks = preview_result.masks
                for index, box in enumerate(preview_result.boxes):
                    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
                    polygon = None
                    geometry = None
                    if masks is not None and index < len(masks):
                        try:
                            polygon = masks.xy[index]
                            geometry = polygon_geometry(polygon)
                        except (AttributeError, IndexError, TypeError, ValueError):
                            geometry = None
                    if geometry is None:
                        u, v = round((x1 + x2) / 2), round((y1 + y2) / 2)
                        distance = math.nan
                    else:
                        u, v = round(geometry.center_u), round(geometry.center_v)
                        distance = sample_board_depth(
                            depth_data,
                            polygon=polygon,
                            ring_px=6,
                        )
                        if distance is None:
                            distance = math.nan
                    label = "depth unavailable" if math.isnan(distance) else f"{distance:.2f} m"
                    detection_text.append(
                        f"#{index + 1} conf={float(box.conf[0]):.2f} "
                        f"center=({u},{v}) depth={label}"
                    )
                    marker_color = (0, 165, 255) if holes_unverified else (0, 255, 0)
                    cv2.drawMarker(display, (u, v), marker_color, cv2.MARKER_CROSS, 10, 1)
                    cv2.putText(
                        display,
                        label,
                        (u + 8, max(20, v - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        marker_color,
                        1,
                        cv2.LINE_AA,
                    )

            inference_ms = float(result.speed.get("inference", math.nan))
            if holes_unverified:
                board_text = "BOARD UNKNOWN - HOLES UNVERIFIED"
            elif board_ready and overlay_plan is not None and not overlay_inferred:
                board_text = "BOARD READY"
            elif overlay_plan is not None and overlay_inferred:
                board_text = "PARTIAL -> INFERRED 12/12"
            else:
                board_text = "BOARD WAIT"
            status_text = (
                f"{board_text}  in_roi={count}/{len(raw_detected)} raw  "
                f"plan={len(plan.targets)}/12  "
                f"infer={inference_ms:.1f} ms  avg={fps:.1f} FPS"
            )
            cv2.putText(
                display,
                status_text,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if frame_count == 1 or frame_count % 30 == 0:
                print(status_text)
                for text in detection_text:
                    print(f"  {text}")
                if board_ready:
                    print(plan.describe())
                elif overlay_plan is not None and overlay_inferred:
                    print("inferred geometry only (not fireable):")
                    print(overlay_plan.describe())
            if not args.headless:
                cv2.imshow(window, display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
