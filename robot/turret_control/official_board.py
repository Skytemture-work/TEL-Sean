"""Official 3x4 target-board geometry.

The official layout has three columns and four rows.  The top row contains
the three small holes; the lower three rows contain the large holes.  The
coordinates below are normalized board coordinates and are intentionally
independent of camera resolution.  Field calibration maps them to pixels.
"""

from __future__ import annotations

from typing import Dict, Tuple


# Front-face dimensions from the official drawing, in centimetres.
BOARD_WIDTH_CM = 300.0
BOARD_HEIGHT_CM = 340.0
HOLE_CENTER_X_CM = (60.0, 150.0, 240.0)
# Measured downward from the top edge.  The bottom row is at 220 cm; with a
# 340 cm board this leaves the documented 120 cm to the floor/bottom edge.
HOLE_CENTER_Y_CM = (50.0, 100.0, 160.0, 220.0)

_HOLE_IDS_BY_CELL = {
    (0, 0): "H10", (1, 0): "H11", (2, 0): "H12",
    (0, 1): "H03", (1, 1): "H06", (2, 1): "H09",
    (0, 2): "H02", (1, 2): "H05", (2, 2): "H08",
    (0, 3): "H01", (1, 3): "H04", (2, 3): "H07",
}


def hole_id_for_cell(column: int, row_from_top: int) -> str:
    """Return the official ID for a zero-based column/row-from-top cell."""
    try:
        return _HOLE_IDS_BY_CELL[(column, row_from_top)]
    except KeyError as exc:
        raise ValueError(f"unknown board cell: column={column}, row={row_from_top}") from exc


def hole_position_cm(column: int, row_from_top: int) -> Tuple[float, float]:
    """Return the physical hole centre from the board's top-left corner."""
    if not 0 <= column < len(HOLE_CENTER_X_CM):
        raise ValueError(f"invalid board column: {column}")
    if not 0 <= row_from_top < len(HOLE_CENTER_Y_CM):
        raise ValueError(f"invalid board row: {row_from_top}")
    return HOLE_CENTER_X_CM[column], HOLE_CENTER_Y_CM[row_from_top]


def hole_position_for_cell(column: int, row_from_top: int) -> Tuple[float, float]:
    """Return normalized (x, y) coordinates for one official board cell."""
    x_cm, y_cm = hole_position_cm(column, row_from_top)
    return x_cm / BOARD_WIDTH_CM, y_cm / BOARD_HEIGHT_CM


# Coordinates are (x, y), normalized from the board's top-left corner.
OFFICIAL_HOLE_LAYOUT: Dict[str, Tuple[float, float]] = {
    hole_id_for_cell(column, row): hole_position_for_cell(column, row)
    for row in range(4)
    for column in range(3)
}


def hole_position(hole_id: str) -> Tuple[float, float]:
    """Return the normalized official position for H01..H12."""
    try:
        return OFFICIAL_HOLE_LAYOUT[hole_id]
    except KeyError as exc:
        raise ValueError(f"unknown official hole id: {hole_id}") from exc
