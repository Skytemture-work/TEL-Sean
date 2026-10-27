"""Target board model: ranging, size classification, and engagement order.

The board is 3 columns x 4 rows of circular holes.  The top row is the small
Ø20 cm holes; the three rows below are Ø40 cm.  Everything is static, so this
is not a tracking problem -- detect the board once, order the holes, then work
through the list.

Ranging comes free because the hole diameters are known:

    optical_depth = focal_length_px * real_diameter_m / apparent_width_px

That is what unblocks the parallax and drop corrections in ``ballistics``.
"""

from __future__ import annotations

import math
import itertools
from typing import Tuple

import cv2
import numpy as np
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

from .aiming import TargetPixel

# Expected ratio between the large and small hole diameters (40 cm / 20 cm).
_EXPECTED_SIZE_RATIO = 2.0
# How far the measured ratio may stray before we refuse to split the clusters.
_RATIO_TOLERANCE = 0.45
_EXPECTED_SMALL_COUNT = 3
_EXPECTED_LARGE_COUNT = 9


@dataclass(frozen=True)
class BoardTarget:
    """One detected hole."""

    u_px: float
    v_px: float
    width_px: float
    height_px: float
    confidence: float = 1.0
    diameter_m: Optional[float] = None
    distance_m: Optional[float] = None
    column: Optional[int] = None
    engaged: bool = False
    # Stable spatial identity from the board diagram (H01..H12).  This is
    # deliberately separate from the engagement order (#1..#12).
    hole_id: Optional[str] = None
    # Row index from bottom to top within a column: 0..3.
    row: Optional[int] = None

    @property
    def size_px(self) -> float:
        """Apparent diameter.  A circle, so average the two box sides."""
        return (self.width_px + self.height_px) / 2.0

    def to_pixel(self) -> TargetPixel:
        return TargetPixel(self.u_px, self.v_px, self.confidence, self.distance_m)


def distance_from_size(size_px: float, diameter_m: float, focal_px: float) -> float:
    """Optical-axis depth to a circle of known diameter from its size."""
    if size_px <= 0:
        raise ValueError("size_px must be positive")
    if diameter_m <= 0:
        raise ValueError("diameter_m must be positive")
    if focal_px <= 0:
        raise ValueError("focal_px must be positive")
    return focal_px * diameter_m / size_px


def classify_by_size(
    targets: Sequence[BoardTarget],
    small_diameter_m: float,
    large_diameter_m: float,
) -> List[BoardTarget]:
    """Split one-class detections into small and large holes.

    At a given range the large holes are exactly twice the small ones, so the
    two clusters separate cleanly.  The split is taken at the widest gap in the
    sorted sizes, but only when the resulting cluster ratio is near the
    expected 2:1 -- otherwise the frame probably contains only one size and
    guessing would assign the wrong diameter (and therefore the wrong range).
    """
    if not targets:
        return []

    ordered = sorted(targets, key=lambda t: t.size_px)
    if len(ordered) == 1:
        return [replace(ordered[0], diameter_m=None)]

    gaps = [
        (ordered[i + 1].size_px - ordered[i].size_px, i)
        for i in range(len(ordered) - 1)
    ]
    _, split_at = max(gaps, key=lambda pair: pair[0])
    small = ordered[: split_at + 1]
    large = ordered[split_at + 1 :]

    small_mean = sum(t.size_px for t in small) / len(small)
    large_mean = sum(t.size_px for t in large) / len(large)
    ratio = large_mean / small_mean if small_mean > 0 else 0.0

    if (
        len(small) != _EXPECTED_SMALL_COUNT
        or len(large) != _EXPECTED_LARGE_COUNT
        or abs(ratio - _EXPECTED_SIZE_RATIO) > _RATIO_TOLERANCE
    ):
        # One size only (or a bad frame).  Leave the diameter unknown so the
        # aiming layer refuses to fire rather than ranging off a wrong guess.
        return [replace(t, diameter_m=None) for t in targets]

    assignment: Dict[int, float] = {}
    for target in small:
        assignment[id(target)] = small_diameter_m
    for target in large:
        assignment[id(target)] = large_diameter_m

    return [replace(t, diameter_m=assignment[id(t)]) for t in ordered]


def select_board_targets(
    targets: Sequence[BoardTarget], require_size_ratio: bool = True
) -> List[BoardTarget]:
    """Keep the best geometrically valid 3x4 board from YOLO candidates.

    YOLO can return background false positives.  The official target gives us
    a strong prior: exactly 12 holes, three vertical columns of four, and a
    3-small/9-large size split with an approximately 2:1 diameter ratio.
    When there are more than 12 candidates, try small subsets of the highest
    confidence boxes and keep the one with the best confidence and geometry.
    """
    if len(targets) <= 12:
        return list(targets)

    candidates = sorted(targets, key=lambda target: target.confidence, reverse=True)[:16]
    best = None
    best_score = None
    for subset in itertools.combinations(candidates, 12):
        identified = assign_hole_ids(subset)
        if len(identified) != 12 or any(target.hole_id is None for target in identified):
            continue

        sizes = sorted(target.size_px for target in subset)
        small_mean = sum(sizes[:3]) / 3.0
        large_mean = sum(sizes[3:]) / 9.0
        if small_mean <= 0:
            continue
        ratio_error = abs((large_mean / small_mean) - _EXPECTED_SIZE_RATIO)
        if require_size_ratio and ratio_error > _RATIO_TOLERANCE:
            continue

        confidence = sum(target.confidence for target in subset) / 12.0
        # Prefer confidence, then a clean 2:1 size split.
        score = confidence - ratio_error * 0.25
        if best_score is None or score > best_score:
            best = list(subset)
            best_score = score

    return best or []


def build_virtual_plan(
    targets: Sequence[BoardTarget],
    small_diameter_m: float,
    large_diameter_m: float,
    large_first: bool = True,
    left_to_right: bool = True,
    bottom_to_top: bool = True,
) -> "EngagementPlan":
    """Build a plan for a screen/virtual board with non-real hole scale.

    The official row layout still determines small versus large; apparent
    pixel size is deliberately ignored because a displayed image is not a
    physical 20/40 cm target at a known range.
    """
    selected = select_board_targets(targets, require_size_ratio=False)
    identified = assign_hole_ids(selected)
    if len(identified) != 12 or any(target.hole_id is None for target in identified):
        return EngagementPlan([])
    classified = [
        replace(
            target,
            diameter_m=small_diameter_m if target.row == 3 else large_diameter_m,
            distance_m=None,
        )
        for target in identified
    ]
    return EngagementPlan(
        order_targets(
            classified,
            large_first=large_first,
            left_to_right=left_to_right,
            bottom_to_top=bottom_to_top,
        )
    )


def _cluster_1d(values: Sequence[float], count: int) -> List[float]:
    """Small dependency-free 1-D k-means used for partial board fitting."""
    ordered = sorted(values)
    centers = [ordered[round(i * (len(ordered) - 1) / (count - 1))] for i in range(count)]
    for _ in range(20):
        groups = [[] for _ in centers]
        for value in values:
            index = min(range(count), key=lambda i: abs(value - centers[i]))
            groups[index].append(value)
        new_centers = [sum(group) / len(group) if group else centers[i] for i, group in enumerate(groups)]
        if max(abs(a - b) for a, b in zip(centers, new_centers)) < 1e-3:
            break
        centers = new_centers
    return sorted(centers)


def build_virtual_plan_from_partial(
    targets: Sequence[BoardTarget],
    small_diameter_m: float,
    large_diameter_m: float,
    large_first: bool = True,
    left_to_right: bool = True,
    bottom_to_top: bool = True,
) -> "EngagementPlan":
    """Infer missing virtual-board holes from a partial 3x4 grid.

    At least six candidates are required.  Candidate points are assigned to
    three x-columns and four y-rows, then a homography maps the official 12
    normalized positions into the camera image.  This is for screen testing:
    it intentionally does not use apparent hole size or range.
    """
    if len(targets) < 6:
        return EngagementPlan([])
    candidates = sorted(targets, key=lambda target: target.confidence, reverse=True)[:20]
    x_centers = _cluster_1d([target.u_px for target in candidates], 3)
    y_centers = _cluster_1d([target.v_px for target in candidates], 4)
    cells = {}
    for target in candidates:
        column = min(range(3), key=lambda i: abs(target.u_px - x_centers[i]))
        row = min(range(4), key=lambda i: abs(target.v_px - y_centers[i]))
        cell = (column, row)
        old = cells.get(cell)
        if old is None or target.confidence > old.confidence:
            cells[cell] = target
    if len(cells) < 6:
        return EngagementPlan([])

    canonical = []
    observed = []
    for (column, row), target in cells.items():
        canonical.append((0.20 + 0.30 * column, 0.125 + 0.25 * row))
        observed.append((target.u_px, target.v_px))
    if len(canonical) < 4:
        return EngagementPlan([])
    matrix, _ = cv2.findHomography(
        np.asarray(canonical, dtype=np.float32),
        np.asarray(observed, dtype=np.float32),
        cv2.RANSAC,
        30.0,
    )
    if matrix is None:
        return EngagementPlan([])

    official_ids = {
        (0, 0): "H10", (1, 0): "H11", (2, 0): "H12",
        (0, 1): "H03", (1, 1): "H06", (2, 1): "H09",
        (0, 2): "H02", (1, 2): "H05", (2, 2): "H08",
        (0, 3): "H01", (1, 3): "H04", (2, 3): "H07",
    }
    points = np.asarray(
        [[0.20 + 0.30 * column, 0.125 + 0.25 * row] for row in range(4) for column in range(3)],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
    inferred = []
    index = 0
    for row in range(4):
        for column in range(3):
            inferred.append(
                BoardTarget(
                    u_px=float(projected[index, 0]),
                    v_px=float(projected[index, 1]),
                    width_px=1.0,
                    height_px=1.0,
                    confidence=1.0,
                    diameter_m=small_diameter_m if row == 0 else large_diameter_m,
                    distance_m=None,
                    column=column,
                    row=3 - row,
                    hole_id=official_ids[(column, row)],
                )
            )
            index += 1
    return EngagementPlan(
        order_targets(
            inferred,
            large_first=large_first,
            left_to_right=left_to_right,
            bottom_to_top=bottom_to_top,
        )
    )


def apply_ranging(targets: Sequence[BoardTarget], focal_px: float) -> List[BoardTarget]:
    """Fill in ``distance_m`` for every target whose diameter is known."""
    out = []
    for target in targets:
        if target.diameter_m is None:
            out.append(replace(target, distance_m=None))
            continue
        out.append(
            replace(
                target,
                distance_m=distance_from_size(target.size_px, target.diameter_m, focal_px),
            )
        )
    return out


def _matches_board_size_layout(
    targets: Sequence[BoardTarget],
    small_diameter_m: float,
    large_diameter_m: float,
) -> bool:
    """Check that the measured size classes are in the official board rows."""
    for target in targets:
        if target.row is None or target.diameter_m is None:
            return False
        expected = small_diameter_m if target.row == 3 else large_diameter_m
        if not math.isclose(target.diameter_m, expected, rel_tol=0.0, abs_tol=1e-9):
            return False
    return True


def assign_columns(
    targets: Sequence[BoardTarget], tolerance_px: Optional[float] = None
) -> List[BoardTarget]:
    """Group targets into vertical columns by their horizontal position.

    Sorting on raw pixel x would interleave columns as soon as the board is
    slightly rotated or the detector jitters, so holes within
    ``tolerance_px`` of each other count as the same column.  The default
    tolerance is half the widest hole, which is far below the column spacing.
    """
    if not targets:
        return []

    if tolerance_px is None:
        tolerance_px = max(t.size_px for t in targets) * 0.75

    ordered = sorted(targets, key=lambda t: t.u_px)
    out: List[BoardTarget] = []
    column_index = 0
    column_anchor = ordered[0].u_px

    for target in ordered:
        if target.u_px - column_anchor > tolerance_px:
            column_index += 1
            column_anchor = target.u_px
        out.append(replace(target, column=column_index))

    return out


def assign_hole_ids(targets: Sequence[BoardTarget]) -> List[BoardTarget]:
    """Assign stable spatial IDs when a complete 3x4 board is visible.

    The diagram numbers holes bottom-to-top in each column:

        column 1: H01 H02 H03 H10   (large, large, large, small)
        column 2: H04 H05 H06 H11
        column 3: H07 H08 H09 H12

    A partial detection is not guessed.  Every target keeps ``hole_id=None``
    until all three columns contain exactly four detections.
    """
    # Keep an already-established column assignment.  This matters for the
    # partial-board homography path, whose inferred points intentionally use a
    # placeholder box size and therefore cannot be reclustered by box width.
    prepared = list(targets)
    if any(target.column is None for target in prepared):
        prepared = assign_columns(prepared)
    if not prepared:
        return []

    by_column: Dict[int, List[BoardTarget]] = {}
    for target in prepared:
        if target.column is None:
            return [replace(item, hole_id=None, row=None) for item in prepared]
        by_column.setdefault(target.column, []).append(target)

    if set(by_column) != {0, 1, 2} or any(len(items) != 4 for items in by_column.values()):
        return [replace(item, hole_id=None, row=None) for item in prepared]

    identified: List[BoardTarget] = []
    for column in range(3):
        # Pixel v grows downwards, so reverse order gives bottom -> top.
        vertical = sorted(by_column[column], key=lambda target: target.v_px, reverse=True)
        for row, target in enumerate(vertical):
            board_number = column * 3 + row + 1 if row < 3 else 10 + column
            identified.append(
                replace(target, hole_id=f"H{board_number:02d}", row=row)
            )
    return identified


def order_targets(
    targets: Sequence[BoardTarget],
    large_first: bool = True,
    left_to_right: bool = True,
    bottom_to_top: bool = True,
) -> List[BoardTarget]:
    """Put the holes in engagement order.

    Default matches the agreed plan: every large hole first, then the small
    ones; columns left to right; within a column bottom to top.  Working a
    column at a time means the horizontal motor only moves once per column.

    Pixel v grows downward, so "bottom first" is descending v.
    """
    # Keep an already-established column assignment.  This matters for the
    # partial-board homography path, whose inferred points intentionally use a
    # placeholder box size and therefore cannot be reclustered by box width.
    prepared = list(targets)
    if any(target.column is None for target in prepared):
        prepared = assign_columns(prepared)

    diameters = sorted({t.diameter_m for t in prepared if t.diameter_m is not None})
    # Rank 0 engages first.
    size_rank: Dict[Optional[float], int] = {None: len(diameters)}
    for rank, diameter in enumerate(reversed(diameters) if large_first else diameters):
        size_rank[diameter] = rank

    def key(target: BoardTarget):
        column = target.column if target.column is not None else 0
        return (
            size_rank.get(target.diameter_m, len(diameters)),
            column if left_to_right else -column,
            -target.v_px if bottom_to_top else target.v_px,
        )

    return sorted(prepared, key=key)


def build_plan(
    targets: Sequence[BoardTarget],
    focal_px: float,
    small_diameter_m: float,
    large_diameter_m: float,
    large_first: bool = True,
    left_to_right: bool = True,
    bottom_to_top: bool = True,
) -> "EngagementPlan":
    """Classify, range and order one detection of the board."""
    selected = select_board_targets(targets, require_size_ratio=True)
    classified = classify_by_size(selected, small_diameter_m, large_diameter_m)
    if len(classified) != 12 or any(target.diameter_m is None for target in classified):
        return EngagementPlan([])
    ranged = apply_ranging(classified, focal_px)
    identified = assign_hole_ids(ranged)
    if len(identified) != 12 or any(target.hole_id is None for target in identified):
        return EngagementPlan([])
    if not _matches_board_size_layout(
        identified, small_diameter_m, large_diameter_m
    ):
        return EngagementPlan([])
    return EngagementPlan(
        order_targets(
            identified,
            large_first=large_first,
            left_to_right=left_to_right,
            bottom_to_top=bottom_to_top,
        )
    )


class EngagementPlan:
    """Walks an ordered target list, one hole at a time.

    The plan never advances by itself.  The caller marks a hole done only after
    the firing gate actually let a shot go, so a hole that could not be reached
    is skipped explicitly rather than silently.
    """

    def __init__(self, targets: Sequence[BoardTarget]):
        self.targets: List[BoardTarget] = list(targets)
        self._index = 0

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def index(self) -> int:
        return self._index

    @property
    def finished(self) -> bool:
        return self._index >= len(self.targets)

    @property
    def remaining(self) -> int:
        return max(0, len(self.targets) - self._index)

    def current(self) -> Optional[BoardTarget]:
        if self.finished:
            return None
        return self.targets[self._index]

    def mark_engaged(self) -> Optional[BoardTarget]:
        """Record a hit on the current hole and move to the next."""
        if self.finished:
            return None
        self.targets[self._index] = replace(self.targets[self._index], engaged=True)
        self._index += 1
        return self.current()

    def skip(self) -> Optional[BoardTarget]:
        """Give up on the current hole without marking it engaged."""
        if self.finished:
            return None
        self._index += 1
        return self.current()

    def describe(self) -> str:
        lines = []
        for position, target in enumerate(self.targets):
            marker = "✔" if target.engaged else ("→" if position == self._index else " ")
            diameter = "?" if target.diameter_m is None else f"{target.diameter_m * 100:.0f}cm"
            distance = "?" if target.distance_m is None else f"{target.distance_m:.2f}m"
            lines.append(
                f" {marker} #{position + 1:2d} {target.hole_id or 'H??':>3} "
                f"col={target.column} row={target.row} "
                f"u={target.u_px:7.1f} v={target.v_px:7.1f} "
                f"Ø{diameter:>5} range={distance:>6}"
            )
        return "\n".join(lines)
