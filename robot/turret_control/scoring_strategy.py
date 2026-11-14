"""Maximum-score firing plans for the 2026-09-03 competition rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

from .board import BoardTarget, EngagementPlan
from .scoring_light import SCORE_HOLE_IDS


UPPER_PHASE = "upper"
LOWER_PHASE = "lower"
MATCH_PHASES = (UPPER_PHASE, LOWER_PHASE)

# Lower-half 3x3 cells, viewed from the camera:
#
#     H03 H06 H09
#     H02 H05 H08
#     H01 H04 H07
#
# Center first, then the two 8-point diagonals, then the remaining edge
# centres.  This earns both diagonal lines after five successful grid shots
# and all 34 line points after the first nine successful grid shots.
LOWER_LINE_FIRST_IDS: Tuple[str, ...] = (
    "H05",
    "H03",
    "H07",
    "H09",
    "H01",
    "H06",
    "H04",
    "H02",
    "H08",
)

# Team decision (2026-11-14): the lower half skips the small top bonus holes
# and puts two consecutive balls into each grid cell (9 x 2 = 18), walking
# the grid in a serpentine so the muzzle only ever moves to a neighbour:
# bottom-left column up, centre column down, right column up.
LOWER_SERPENTINE_IDS: Tuple[str, ...] = (
    "H01", "H02", "H03",
    "H06", "H05", "H04",
    "H07", "H08", "H09",
)
LOWER_BALLS_PER_GRID_CELL = 2


@dataclass(frozen=True)
class FiringStrategy:
    """One complete legal shot allocation and its theoretical score."""

    phase: str
    plan: EngagementPlan
    total_balls: int
    bonus_balls: int
    grid_balls: int
    base_points: int
    bonus_points: int
    line_points: int
    theoretical_max_points: int

    @property
    def hole_ids(self) -> Tuple[str, ...]:
        return tuple(target.hole_id or "H??" for target in self.plan.targets)


def _targets_by_id(plan: EngagementPlan) -> Dict[str, BoardTarget]:
    result: Dict[str, BoardTarget] = {}
    for target in plan.targets:
        if target.hole_id is not None and target.hole_id not in result:
            result[target.hole_id] = target
    return result


# Team decision (2026-11-14): upper half = 9 balls into the lit bonus hole
# (only within the first 60 s, while the light is on; H11 when the light is not
# confirmed in time), then the same serpentine grid pairs as the lower half.
# The first magazine (18) runs out after H05 #1; the team reloads and the plan
# continues.
UPPER_BONUS_BALLS = 9
UPPER_BONUS_WINDOW_S = 60.0
UPPER_FALLBACK_BONUS_ID = "H11"
UPPER_GRID_IDS: Tuple[str, ...] = tuple(
    hole_id for hole_id in LOWER_SERPENTINE_IDS for _ in range(2))


def build_max_score_strategy(
    board_plan: EngagementPlan,
    active_bonus_hole_id: Optional[str],
    phase: str,
) -> FiringStrategy:
    """Build the firing plan for one upper/lower-half match.

    Upper half (27 balls, reload after 18): ``UPPER_BONUS_BALLS`` into the
    lit bonus hole, then ``UPPER_GRID_IDS``.  ``active_bonus_hole_id=None``
    (no light seen within the bonus window) gives the grid part only.

    Lower half: 18 balls, no bonus shots (team decision).  Two consecutive
    balls per grid cell, cells visited in ``LOWER_SERPENTINE_IDS`` order to
    minimise muzzle travel.  The scoring light is not used.
    The theoretical maximum is 36 + 34 = 70 points.
    """

    if phase not in MATCH_PHASES:
        raise ValueError(f"unknown match phase: {phase}")
    if active_bonus_hole_id is not None and active_bonus_hole_id not in SCORE_HOLE_IDS:
        raise ValueError(f"not a scoring-light hole: {active_bonus_hole_id}")

    by_id = _targets_by_id(board_plan)
    required_ids = {f"H{number:02d}" for number in range(1, 10)}
    if active_bonus_hole_id is not None and phase == UPPER_PHASE:
        required_ids.add(active_bonus_hole_id)
    missing = sorted(required_ids - set(by_id))
    if missing:
        raise ValueError(f"board plan is missing holes: {', '.join(missing)}")

    if phase == UPPER_PHASE:
        bonus = UPPER_BONUS_BALLS if active_bonus_hole_id is not None else 0
        shots = ([by_id[active_bonus_hole_id]] * bonus if bonus else [])
        shots += [by_id[hole_id] for hole_id in UPPER_GRID_IDS]
        grid = len(UPPER_GRID_IDS)
        return FiringStrategy(
            phase=phase,
            plan=EngagementPlan(shots),
            total_balls=bonus + grid,
            bonus_balls=bonus,
            grid_balls=grid,
            base_points=2 * grid,
            bonus_points=4 * bonus,
            line_points=0,
            theoretical_max_points=2 * grid + 4 * bonus,
        )

    shots = [by_id[hole_id] for hole_id in LOWER_SERPENTINE_IDS
             for _ in range(LOWER_BALLS_PER_GRID_CELL)]
    return FiringStrategy(
        phase=phase,
        plan=EngagementPlan(shots),
        total_balls=18,
        bonus_balls=0,
        grid_balls=18,
        base_points=36,
        bonus_points=0,
        line_points=34,
        theoretical_max_points=70,
    )


def append_reallocated_grid_shots(
    plan: EngagementPlan,
    board_plan: EngagementPlan,
    phase: str,
    count: int,
) -> Tuple[str, ...]:
    """Move unused bonus balls to legal 2-point grid cells.

    A bonus shot that was never sent still leaves a physical ball available.
    Once the 60-second light window closes, append that ball to a large-hole
    cell which has not reached the phase-specific scoring cap.
    """

    if phase not in MATCH_PHASES:
        raise ValueError(f"unknown match phase: {phase}")
    if count <= 0:
        return ()
    cap = 3 if phase == UPPER_PHASE else 2
    by_id = _targets_by_id(board_plan)
    large_ids = tuple(f"H{number:02d}" for number in range(1, 10))
    missing = [hole_id for hole_id in large_ids if hole_id not in by_id]
    if missing:
        raise ValueError(f"board plan is missing holes: {', '.join(missing)}")

    allocations = {
        hole_id: sum(target.hole_id == hole_id for target in plan.targets)
        for hole_id in large_ids
    }
    appended: List[str] = []
    for _ in range(count):
        eligible = [
            by_id[hole_id]
            for hole_id in large_ids
            if allocations[hole_id] < cap
        ]
        if not eligible:
            break
        center_u = sum(target.u_px for target in eligible) / len(eligible)
        center_v = sum(target.v_px for target in eligible) / len(eligible)
        target = min(
            eligible,
            key=lambda item: (
                allocations[item.hole_id],
                -item.confidence,
                -item.size_px,
                math.hypot(item.u_px - center_u, item.v_px - center_v),
                item.hole_id,
            ),
        )
        plan.targets.append(target)
        allocations[target.hole_id] += 1
        appended.append(target.hole_id)
    return tuple(appended)
