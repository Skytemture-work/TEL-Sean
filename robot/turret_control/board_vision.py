"""Shared per-frame board detection, geometry inference and visualization.

This module has no motor or launcher dependencies. Its inferred grid is an
image-space estimate; callers must apply their own pose and control gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .board import (build_plan, build_plan_from_board_quad, build_virtual_plan,
                    build_virtual_plan_from_partial, estimate_board_distance_from_targets)
from .board_surface import detect_dark_board_surface, filter_targets_to_surface
from .detection import boxes_to_board_targets
from .run_yolo_demo import DEFAULT_MODEL as FALLBACK_MODEL, _draw_board_plan
from .run_zed_tensorrt_test import (PreviewBoardTracker, _draw_inferred_board,
                                   _filter_preview_result, _grid_geometry_valid,
                                   _surface_from_estimated_quad)

ENGINE_MODEL = Path.home() / "Desktop" / "best.engine"
DEFAULT_MODEL = ENGINE_MODEL if ENGINE_MODEL.is_file() else FALLBACK_MODEL
DEFAULT_CONFIDENCE = 0.25


@dataclass
class BoardVisionResult:
    result: Any
    raw_targets: list
    targets: list
    surface: Any
    quad: Optional[np.ndarray]
    estimated_surface: bool
    tracked: bool
    plan: Any
    inferred: bool
    confidence: float

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw verified detections, board outline and the complete grid."""
        filtered = _filter_preview_result(self.result, self.raw_targets, self.surface, self.confidence)
        display = filtered.plot() if filtered is not None else frame.copy()
        if self.surface is not None:
            color = (255, 255, 0) if self.estimated_surface else (0, 255, 255)
            cv2.drawContours(display, [self.surface.contour], -1, color, 2)
        if _grid_geometry_valid(self.plan, frame.shape):
            _draw_board_plan(display, self.plan)
            _draw_inferred_board(display, self.plan, inferred=self.inferred, show_banner=False)
        return display


class BoardVision:
    """Run a detector and maintain the short-lived board geometry tracker."""

    def __init__(self, model, config, *, confidence=DEFAULT_CONFIDENCE, imgsz=640,
                 device="0", virtual_board=False):
        self.model = model
        self.config = config
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.virtual_board = virtual_board
        self.tracker = PreviewBoardTracker()

    def process(self, frame: np.ndarray, *, now: float, depth_data=None) -> BoardVisionResult:
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.confidence,
                                    device=self.device, verbose=False)[0]
        return self.analyze(frame, result, now=now, depth_data=depth_data)

    def analyze(self, frame: np.ndarray, result, *, now: float, depth_data=None) -> BoardVisionResult:
        config = self.config
        raw = boxes_to_board_targets(result, self.confidence, depth_data)
        surface = detect_dark_board_surface(
            frame, value_threshold=config.board_dark_value_threshold,
            min_area_fraction=config.board_min_area_fraction, targets=raw,
        )
        estimated_quad, tracked = self.tracker.update(raw, frame.shape, now)
        estimated_surface = estimated_quad is not None
        if estimated_surface:
            surface = _surface_from_estimated_quad(estimated_quad, frame.shape)
        elif surface is not None and surface.quad is None:
            surface = None
        targets = filter_targets_to_surface(raw, surface)
        quad = estimated_quad if estimated_quad is not None else (
            surface.quad if surface is not None else None)

        plan = None
        inferred = False
        if quad is not None:
            distance = estimate_board_distance_from_targets(
                targets, quad, config.focal_length_x_px, config.small_hole_diameter_m,
                config.large_hole_diameter_m, config.board_position_tolerance_cm)
            plan = build_plan_from_board_quad(
                quad, config.small_hole_diameter_m, config.large_hole_diameter_m,
                board_distance_m=distance, large_first=config.engage_large_first,
                left_to_right=config.engage_left_to_right, bottom_to_top=config.engage_bottom_to_top)
            inferred = True
        elif len(targets) == 12:
            if not self.virtual_board:
                plan = build_plan(targets, config.focal_length_x_px, config.small_hole_diameter_m,
                                  config.large_hole_diameter_m, config.engage_large_first,
                                  config.engage_left_to_right, config.engage_bottom_to_top,
                                  config.board_position_tolerance_cm)
            if plan is None or len(plan) != 12:
                plan = build_virtual_plan(targets, config.small_hole_diameter_m,
                                          config.large_hole_diameter_m, config.engage_large_first,
                                          config.engage_left_to_right, config.engage_bottom_to_top,
                                          config.board_position_tolerance_cm)
        if (plan is None or len(plan) != 12) and len(targets) >= 4:
            plan = build_virtual_plan_from_partial(
                targets, config.small_hole_diameter_m, config.large_hole_diameter_m,
                config.engage_large_first, config.engage_left_to_right, config.engage_bottom_to_top,
                focal_px=config.focal_length_x_px, position_tolerance_cm=config.board_position_tolerance_cm)
            inferred = True
        return BoardVisionResult(result, raw, targets, surface, quad, estimated_surface,
                                 tracked, plan, inferred, self.confidence)
