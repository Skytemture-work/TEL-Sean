"""Official 3x4 target-board geometry.

The official layout has three columns and four rows.  The top row contains
the three small holes; the lower three rows contain the large holes.  The
coordinates below are normalized board coordinates and are intentionally
independent of camera resolution.  Field calibration maps them to pixels.
"""

from __future__ import annotations

from typing import Dict, Tuple


# Coordinates are (x, y), normalized from the board's top-left to bottom-right.
# They are a starting geometry template, not final field calibration values.
OFFICIAL_HOLE_LAYOUT: Dict[str, Tuple[float, float]] = {
    "H10": (0.20, 0.125),
    "H11": (0.50, 0.125),
    "H12": (0.80, 0.125),
    "H03": (0.20, 0.375),
    "H06": (0.50, 0.375),
    "H09": (0.80, 0.375),
    "H02": (0.20, 0.625),
    "H05": (0.50, 0.625),
    "H08": (0.80, 0.625),
    "H01": (0.20, 0.875),
    "H04": (0.50, 0.875),
    "H07": (0.80, 0.875),
}


def hole_position(hole_id: str) -> Tuple[float, float]:
    """Return the normalized official position for H01..H12."""
    try:
        return OFFICIAL_HOLE_LAYOUT[hole_id]
    except KeyError as exc:
        raise ValueError(f"unknown official hole id: {hole_id}") from exc
