"""Maximum-score firing plans for the 2026-09-03 competition rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def _nearest_neighbor_order(
    targets: Iterable[BoardTarget], start: Optional[BoardTarget]
) -> List[BoardTarget]:
    """Greedily reduce turret travel without changing shot allocation."""

    remaining = list(targets)
    ordered: List[BoardTarget] = []
    current = start
    while remaining:
        if current is None:
            index = min(
                range(len(remaining)),
                key=lambda i: remaining[i].hole_id or "",
            )
        else:
            index = min(
                range(len(remaining)),
                key=lambda i: (
                    math.hypot(
                        remaining[i].u_px - current.u_px,
                        remaining[i].v_px - current.v_px,
                    ),
                    remaining[i].hole_id or "",
                ),
            )
        current = remaining.pop(index)
        ordered.append(current)
    return ordered


def _most_reliable_targets(
    targets: Sequence[BoardTarget], count: int
) -> List[BoardTarget]:
    """Choose repeat-shot cells with the strongest visual geometry."""

    if count <= 0:
        return []
    center_u = sum(target.u_px for target in targets) / len(targets)
    center_v = sum(target.v_px for target in targets) / len(targets)
    return sorted(
        targets,
        key=lambda target: (
            -target.confidence,
            -target.size_px,
            math.hypot(target.u_px - center_u, target.v_px - center_v),
            target.hole_id or "",
        ),
    )[:count]


def build_max_score_strategy(
    board_plan: EngagementPlan,
    active_bonus_hole_id: str,
    phase: str,
) -> FiringStrategy:
    """Build a legal maximum-score plan for one upper/lower-half match.

    Rule source: ``2026_rule_0903.pdf`` pages 5, 7, 9, 11, 12 and 14.

    Upper half: 27 balls.  Put three into the active 4-point bonus hole, then
    use all 24 remaining balls in the nine 2-point grid holes.  Every grid is
    attempted before repeats because distinct hit cells are the second
    tie-breaker.  The theoretical maximum is 60 points.

    Lower half: 18 balls.  Put two into the active 4-point bonus hole, then
    put at least one into every 3x3 grid cell to obtain all 34 line points.
    Seven legal second shots consume the remaining balls.  The theoretical
    maximum is 74 points.
    """

    if phase not in MATCH_PHASES:
        raise ValueError(f"unknown match phase: {phase}")
    if active_bonus_hole_id not in SCORE_HOLE_IDS:
        raise ValueError(f"not a scoring-light hole: {active_bonus_hole_id}")

    by_id = _targets_by_id(board_plan)
    required_ids = {active_bonus_hole_id} | {
        f"H{number:02d}" for number in range(1, 10)
    }
    missing = sorted(required_ids - set(by_id))
    if missing:
        raise ValueError(f"board plan is missing holes: {', '.join(missing)}")

    bonus_target = by_id[active_bonus_hole_id]
    large_targets = [by_id[f"H{number:02d}"] for number in range(1, 10)]

    if phase == UPPER_PHASE:
        first_pass = _nearest_neighbor_order(large_targets, bonus_target)
        second_pass = _nearest_neighbor_order(large_targets, first_pass[-1])
        six_best = _most_reliable_targets(large_targets, 6)
        third_pass = _nearest_neighbor_order(six_best, second_pass[-1])
        shots = [bonus_target] * 3 + first_pass + second_pass + third_pass
        return FiringStrategy(
            phase=phase,
            plan=EngagementPlan(shots),
            total_balls=27,
            bonus_balls=3,
            grid_balls=24,
            base_points=48,
            bonus_points=12,
            line_points=0,
            theoretical_max_points=60,
        )

    line_first = [by_id[hole_id] for hole_id in LOWER_LINE_FIRST_IDS]
    seven_best = _most_reliable_targets(large_targets, 7)
    repeat_pass = _nearest_neighbor_order(seven_best, line_first[-1])
    shots = [bonus_target] * 2 + line_first + repeat_pass
    return FiringStrategy(
        phase=phase,
        plan=EngagementPlan(shots),
        total_balls=18,
        bonus_balls=2,
        grid_balls=16,
        base_points=32,
        bonus_points=8,
        line_points=34,
        theoretical_max_points=74,
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
