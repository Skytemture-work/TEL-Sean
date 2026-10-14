"""Target board model: ranging, size classification, and engagement order.

The board is 3 columns x 4 rows of circular holes.  The top row is the small
Ø20 cm holes; the three rows below are Ø40 cm.  Everything is static, so this
is not a tracking problem -- detect the board once, order the holes, then work
through the list.

Ranging comes free because the hole diameters are known:

    distance = focal_length_px * real_diameter_m / apparent_width_px

That is what unblocks the parallax and drop corrections in ``ballistics``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

from .aiming import TargetPixel

# Expected ratio between the large and small hole diameters (40 cm / 20 cm).
_EXPECTED_SIZE_RATIO = 2.0
# How far the measured ratio may stray before we refuse to split the clusters.
_RATIO_TOLERANCE = 0.45


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
    """Range to a circle of known diameter from its apparent size."""
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

    if abs(ratio - _EXPECTED_SIZE_RATIO) > _RATIO_TOLERANCE:
        # One size only (or a bad frame).  Leave the diameter unknown so the
        # aiming layer refuses to fire rather than ranging off a wrong guess.
        return [replace(t, diameter_m=None) for t in targets]

    assignment: Dict[int, float] = {}
    for target in small:
        assignment[id(target)] = small_diameter_m
    for target in large:
        assignment[id(target)] = large_diameter_m

    return [replace(t, diameter_m=assignment[id(t)]) for t in ordered]


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
    prepared = assign_columns(targets)
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
    prepared = assign_columns(targets)

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
    classified = classify_by_size(targets, small_diameter_m, large_diameter_m)
    ranged = apply_ranging(classified, focal_px)
    identified = assign_hole_ids(ranged)
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
