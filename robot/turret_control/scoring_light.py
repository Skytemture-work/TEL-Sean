"""Detection and round control for the three red scoring light strips."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from .board import BoardTarget, EngagementPlan
from .official_board import (
    BOARD_HEIGHT_CM,
    BOARD_WIDTH_CM,
    hole_position_for_cell,
)


SCORE_HOLE_IDS: Tuple[str, str, str] = ("H10", "H11", "H12")


@dataclass(frozen=True)
class ScoreLightState:
    """Current special-scoring round state."""

    phase: str
    active_hole_id: Optional[str]
    scores: Dict[str, float]
    started_at: Optional[float]
    expires_at: Optional[float]
    remaining_s: float


def _ring_mask_from_quad(
    shape: Tuple[int, int],
    board_quad: np.ndarray,
    column: int,
    inner_radius_cm: float,
    outer_radius_cm: float,
) -> np.ndarray:
    """Project a physical annulus around one top hole into the image."""

    height, width = shape
    source = np.asarray(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        dtype=np.float32,
    )
    quad = np.asarray(board_quad, dtype=np.float32).reshape(4, 2)
    try:
        matrix = cv2.getPerspectiveTransform(source, quad)
    except cv2.error:
        return np.zeros((height, width), dtype=np.uint8)

    center_x, center_y = hole_position_for_cell(column, 0)
    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)

    def project_radius(radius_cm: float) -> np.ndarray:
        points = np.column_stack(
            (
                center_x + np.cos(angles) * radius_cm / BOARD_WIDTH_CM,
                center_y + np.sin(angles) * radius_cm / BOARD_HEIGHT_CM,
            )
        ).astype(np.float32)
        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2), matrix
        ).reshape(-1, 2)

    try:
        outer = np.round(project_radius(outer_radius_cm)).astype(np.int32)
        inner = np.round(project_radius(inner_radius_cm)).astype(np.int32)
    except cv2.error:
        return np.zeros((height, width), dtype=np.uint8)

    outer_mask = np.zeros((height, width), dtype=np.uint8)
    inner_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(outer_mask, [outer.reshape(-1, 1, 2)], 255)
    cv2.fillPoly(inner_mask, [inner.reshape(-1, 1, 2)], 255)
    return cv2.subtract(outer_mask, inner_mask)


def _ring_mask_from_target(
    shape: Tuple[int, int],
    target: BoardTarget,
    inner_radius_cm: float,
    outer_radius_cm: float,
    tolerance_cm: float,
) -> np.ndarray:
    """Approximate a projected scoring annulus when no board quad is visible."""

    height, width = shape
    diameter_px = target.size_px
    if diameter_px <= 1.0:
        return np.zeros((height, width), dtype=np.uint8)

    # The physical small hole diameter is 20 cm.  A local ellipse is enough
    # for the fallback path; the quad path above handles strong perspective.
    scale_x = diameter_px / 20.0
    scale_y = diameter_px / 20.0
    inner_x = max(0.0, inner_radius_cm - tolerance_cm) * scale_x
    inner_y = max(0.0, inner_radius_cm - tolerance_cm) * scale_y
    outer_x = (outer_radius_cm + tolerance_cm) * scale_x
    outer_y = (outer_radius_cm + tolerance_cm) * scale_y
    outer_mask = np.zeros((height, width), dtype=np.uint8)
    inner_mask = np.zeros((height, width), dtype=np.uint8)
    center = (round(target.u_px), round(target.v_px))
    cv2.ellipse(
        outer_mask,
        center,
        (max(1, round(outer_x)), max(1, round(outer_y))),
        0.0,
        0.0,
        360.0,
        255,
        cv2.FILLED,
    )
    cv2.ellipse(
        inner_mask,
        center,
        (max(1, round(inner_x)), max(1, round(inner_y))),
        0.0,
        0.0,
        360.0,
        255,
        cv2.FILLED,
    )
    return cv2.subtract(outer_mask, inner_mask)


def detect_score_light_scores(
    frame: np.ndarray,
    plan: Optional[EngagementPlan],
    board_quad: Optional[np.ndarray],
    inner_radius_cm: float = 15.0,
    strip_width_cm: float = 1.4,
    search_tolerance_cm: float = 1.5,
    min_saturation: int = 55,
    min_value: int = 40,
) -> Dict[str, float]:
    """Return a red-light score for H10/H11/H12.

    The small scoring hole is Ø20 cm.  The rule places the LED strip's inner
    edge about 5 cm beyond the circular frame edge, so the nominal inner
    radius is 10 + 5 = 15 cm.  The search tolerance absorbs perspective,
    blur, and the stated field tolerance.
    """

    if frame is None or frame.ndim != 3:
        return {}
    if inner_radius_cm <= 0 or strip_width_cm <= 0 or search_tolerance_cm < 0:
        return {}

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(frame)
    hue, saturation, value = cv2.split(hsv)
    red_hue = (hue <= 12) | (hue >= 168)
    red_pixels = red_hue & (saturation >= min_saturation) & (value >= min_value)
    red_excess = np.maximum(
        red.astype(np.int16)
        - np.maximum(blue, green).astype(np.int16),
        0,
    ).astype(np.float32)

    targets_by_id = {}
    if plan is not None:
        targets_by_id = {
            target.hole_id: target
            for target in plan.targets
            if target.hole_id in SCORE_HOLE_IDS
        }

    scores: Dict[str, float] = {}
    outer_radius_cm = inner_radius_cm + strip_width_cm
    for column, hole_id in enumerate(SCORE_HOLE_IDS):
        if board_quad is not None:
            ring = _ring_mask_from_quad(
                frame.shape[:2],
                board_quad,
                column,
                max(0.1, inner_radius_cm - search_tolerance_cm),
                outer_radius_cm + search_tolerance_cm,
            )
        else:
            target = targets_by_id.get(hole_id)
            if target is None:
                scores[hole_id] = 0.0
                continue
            ring = _ring_mask_from_target(
                frame.shape[:2],
                target,
                inner_radius_cm,
                outer_radius_cm,
                search_tolerance_cm,
            )

        ring_pixels = ring > 0
        count = int(np.count_nonzero(ring_pixels))
        if count == 0:
            scores[hole_id] = 0.0
            continue
        red_fraction = float(np.count_nonzero(red_pixels & ring_pixels)) / count
        red_energy = float(np.mean(red_excess[ring_pixels])) / 255.0
        scores[hole_id] = 0.70 * red_fraction + 0.30 * red_energy
    return scores


class ScoreRoundController:
    """Lock one detected scoring hole for the complete 60-second round."""

    def __init__(
        self,
        duration_s: float = 60.0,
        confirm_frames: int = 3,
        min_score: float = 0.12,
        min_margin: float = 0.05,
        inner_radius_cm: float = 15.0,
        strip_width_cm: float = 1.4,
        search_tolerance_cm: float = 1.5,
        min_saturation: int = 55,
        min_value: int = 40,
        clock: Callable[[], float] = time.monotonic,
    ):
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if confirm_frames <= 0:
            raise ValueError("confirm_frames must be positive")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        if not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_margin must be between 0 and 1")
        if inner_radius_cm <= 0 or strip_width_cm <= 0:
            raise ValueError("scoring light radii and width must be positive")
        if search_tolerance_cm < 0:
            raise ValueError("search_tolerance_cm must not be negative")
        self.duration_s = duration_s
        self.confirm_frames = confirm_frames
        self.min_score = min_score
        self.min_margin = min_margin
        self.inner_radius_cm = inner_radius_cm
        self.strip_width_cm = strip_width_cm
        self.search_tolerance_cm = search_tolerance_cm
        self.min_saturation = min_saturation
        self.min_value = min_value
        self._clock = clock
        self._round_id = 0
        self._phase = "idle"
        self._active_hole_id: Optional[str] = None
        self._candidate_hole_id: Optional[str] = None
        self._candidate_frames = 0
        self._started_at: Optional[float] = None
        self._expires_at: Optional[float] = None

    @property
    def round_id(self) -> int:
        return self._round_id

    def start_round(self, now: Optional[float] = None) -> None:
        """Start a round at the whistle; the light chooses the hole."""

        started_at = self._clock() if now is None else now
        self._round_id += 1
        self._phase = "waiting"
        self._active_hole_id = None
        self._candidate_hole_id = None
        self._candidate_frames = 0
        self._started_at = started_at
        self._expires_at = started_at + self.duration_s

    def stop_round(self) -> None:
        """End the round and ignore the light until the next whistle."""

        self._phase = "expired"
        self._active_hole_id = None
        self._candidate_hole_id = None
        self._candidate_frames = 0

    def update(
        self,
        frame: np.ndarray,
        plan: Optional[EngagementPlan] = None,
        board_quad: Optional[np.ndarray] = None,
        now: Optional[float] = None,
    ) -> ScoreLightState:
        current = self._clock() if now is None else now
        scores = detect_score_light_scores(
            frame,
            plan,
            board_quad,
            inner_radius_cm=self.inner_radius_cm,
            strip_width_cm=self.strip_width_cm,
            search_tolerance_cm=self.search_tolerance_cm,
            min_saturation=self.min_saturation,
            min_value=self.min_value,
        )

        if self._phase == "waiting":
            if self._expires_at is not None and current >= self._expires_at:
                self.stop_round()
            else:
                ranked = sorted(
                    scores.items(), key=lambda item: item[1], reverse=True
                )
                best_hole_id, best_score = (
                    ranked[0] if ranked else (None, 0.0)
                )
                second_score = ranked[1][1] if len(ranked) > 1 else 0.0
                if (
                    best_hole_id is not None
                    and best_score >= self.min_score
                    and best_score - second_score >= self.min_margin
                ):
                    if best_hole_id == self._candidate_hole_id:
                        self._candidate_frames += 1
                    else:
                        self._candidate_hole_id = best_hole_id
                        self._candidate_frames = 1
                    if self._candidate_frames >= self.confirm_frames:
                        self._active_hole_id = best_hole_id
                        self._phase = "active"
                else:
                    self._candidate_hole_id = None
                    self._candidate_frames = 0
        elif self._phase == "active":
            if self._expires_at is not None and current >= self._expires_at:
                self.stop_round()

        remaining = 0.0
        if self._expires_at is not None and self._phase in {"waiting", "active"}:
            remaining = max(0.0, self._expires_at - current)
        return ScoreLightState(
            phase=self._phase,
            active_hole_id=self._active_hole_id,
            scores=scores,
            started_at=self._started_at,
            expires_at=self._expires_at,
            remaining_s=remaining,
        )


def target_for_score_hole(
    plan: Optional[EngagementPlan], hole_id: Optional[str]
) -> Optional[BoardTarget]:
    """Find one scoring target in a board plan."""

    if plan is None or hole_id is None:
        return None
    return next(
        (target for target in plan.targets if target.hole_id == hole_id),
        None,
    )
