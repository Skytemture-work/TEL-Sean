"""Persistent motor-space safety limits for the two-axis turret.

The limits are deliberately stored in encoder/motor coordinates rather than
only in physical degrees.  That makes the final safety check independent of
which caller produced a command and preserves the exact positions captured
from DAMIAO feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_LIMITS_PATH = Path(__file__).with_name("mechanism_limits.json")
YAW_DESIGN_LIMIT_DEG = 34.5
PITCH_MIN_DEG = 0.0
PITCH_ACTIVITY_TRANSITION_DEG = 30.0
PITCH_MAX_DEG = 35.0
TRIANGLE_BASE_YAW_DEG = 25.0
HOME_CORRIDOR_TRANSITION_DEG = 33.0


def max_activity_pitch_for_yaw_deg(
    yaw_deg: float,
    *,
    activity_max_deg: float = PITCH_MAX_DEG,
    transition_deg: float = PITCH_ACTIVITY_TRANSITION_DEG,
) -> float:
    """Return the maximum normal-activity pitch for a physical yaw angle.

    The normal workspace is rectangular through 30 degrees elevation.  From
    30 to 35 degrees it narrows linearly from +/-25 degrees yaw to the
    centered apex at 35 degrees.
    """

    yaw_abs = abs(float(yaw_deg))
    activity_max_deg = float(activity_max_deg)
    transition_deg = float(transition_deg)
    if not all(math.isfinite(value) for value in (yaw_abs, activity_max_deg, transition_deg)):
        raise ValueError("yaw_deg must be finite")
    transition_deg = min(transition_deg, activity_max_deg)
    if yaw_abs >= TRIANGLE_BASE_YAW_DEG:
        return transition_deg
    fraction = yaw_abs / TRIANGLE_BASE_YAW_DEG
    return activity_max_deg - (activity_max_deg - transition_deg) * fraction


def constrain_pitch_activity_limit(bounds, calibration):
    """Intersect recorded limits with the calibrated 0–35° activity range."""
    design = sorted(
        calibration.motor_from_muzzle(math.radians(angle))
        for angle in (PITCH_MIN_DEG, PITCH_MAX_DEG)
    )
    lower, upper = max(bounds[0], design[0]), min(bounds[1], design[1])
    if lower >= upper:
        raise ValueError("recorded pitch limits do not overlap the 0–35° activity range")
    return lower, upper


def constrain_pitch_travel_limit(bounds, calibration, home_motor_rad=None):
    """Intersect recorded travel with level and the independently recorded home."""
    explicit_home = home_motor_rad is not None
    if home_motor_rad is None:
        home_motor_rad = calibration.motor_from_muzzle(math.radians(45.0))
    if (not math.isfinite(home_motor_rad)
            or calibration.muzzle_from_motor(home_motor_rad) <= 0):
        raise ValueError("recorded home must be above the measured level position")
    if explicit_home and not bounds[0] - 1e-10 <= home_motor_rad <= bounds[1] + 1e-10:
        raise ValueError("recorded home is outside the retained pitch motor limits")
    design = sorted((calibration.motor_from_muzzle(0.0), home_motor_rad))
    lower, upper = max(bounds[0], design[0]), min(bounds[1], design[1])
    if lower >= upper:
        raise ValueError("recorded pitch limits do not overlap the level-to-home travel range")
    return lower, upper


def constrain_yaw_design_limit(bounds, calibration):
    """Intersect encoder limits with the calibrated physical yaw boundary."""
    design = sorted(
        calibration.motor_from_muzzle(math.radians(angle))
        for angle in (-YAW_DESIGN_LIMIT_DEG, YAW_DESIGN_LIMIT_DEG)
    )
    lower, upper = max(bounds[0], design[0]), min(bounds[1], design[1])
    if lower >= upper:
        raise ValueError("recorded yaw limits do not overlap the ±34.5° design limit")
    return lower, upper


@dataclass(frozen=True)
class AxisLimit:
    """One axis' allowed absolute motor-position interval."""

    axis: str
    motor_id: int
    min_rad: float
    max_rad: float

    def __post_init__(self) -> None:
        if self.min_rad >= self.max_rad:
            raise ValueError(f"{self.axis} motor limits must be increasing")
        if not all(math.isfinite(value) for value in (self.min_rad, self.max_rad)):
            raise ValueError(f"{self.axis} motor limits must be finite")


Point = Tuple[float, float]  # (yaw_motor_rad, pitch_motor_rad)


@dataclass(frozen=True)
class HomeCorridor:
    """Narrow vertical-only connection between the low workspace and home."""

    yaw_center_rad: float
    yaw_tolerance_rad: float
    home_pitch_rad: float
    transition_pitch_rad: float

    def __post_init__(self) -> None:
        values = (
            self.yaw_center_rad,
            self.yaw_tolerance_rad,
            self.home_pitch_rad,
            self.transition_pitch_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("home corridor values must be finite")
        if self.yaw_tolerance_rad < 0.0:
            raise ValueError("home corridor yaw tolerance must be non-negative")


@dataclass(frozen=True)
class WorkspaceLimit:
    """A non-rectangular yaw/pitch safety envelope in motor coordinates."""

    yaw_motor_id: int
    pitch_motor_id: int
    vertices: Tuple[Point, ...]
    anchor: Point
    margin_rad: float = 0.0
    symmetry_yaw_center_rad: Optional[float] = None
    home_corridor: Optional[HomeCorridor] = None

    def __post_init__(self) -> None:
        if self.yaw_motor_id == self.pitch_motor_id:
            raise ValueError("workspace yaw and pitch motor IDs must differ")
        if len(self.vertices) < 3:
            raise ValueError("workspace needs at least three vertices")
        values = [value for point in self.vertices for value in point] + list(self.anchor)
        if self.symmetry_yaw_center_rad is not None:
            values.append(self.symmetry_yaw_center_rad)
        if self.home_corridor is not None:
            values.extend(
                (
                    self.home_corridor.yaw_center_rad,
                    self.home_corridor.yaw_tolerance_rad,
                    self.home_corridor.home_pitch_rad,
                    self.home_corridor.transition_pitch_rad,
                )
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("workspace positions must be finite")
        if not math.isfinite(self.margin_rad) or self.margin_rad < 0.0:
            raise ValueError("workspace margin must be finite and non-negative")
        if _polygon_self_intersects(self.vertices):
            raise ValueError("workspace boundary intersects itself; trace one closed outline")


def _distance_sq_to_segment(point: Point, start: Point, end: Point) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-24:
        return (px - ax) ** 2 + (py - ay) ** 2
    fraction = ((px - ax) * dx + (py - ay) * dy) / length_sq
    fraction = max(0.0, min(1.0, fraction))
    closest_x = ax + fraction * dx
    closest_y = ay + fraction * dy
    return (px - closest_x) ** 2 + (py - closest_y) ** 2


def _point_in_polygon(point: Point, vertices: Sequence[Point]) -> bool:
    """Return true for a point inside or directly on a polygon boundary."""

    x, y = point
    inside = False
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        if _distance_sq_to_segment(point, start, end) <= 1e-18:
            return True
        ax, ay = start
        bx, by = end
        if (ay > y) != (by > y):
            crossing_x = (bx - ax) * (y - ay) / (by - ay) + ax
            if x < crossing_x:
                inside = not inside
    return inside


def _orientation(first: Point, second: Point, third: Point) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _segments_intersect(first_start: Point, first_end: Point, second_start: Point, second_end: Point) -> bool:
    one = _orientation(first_start, first_end, second_start)
    two = _orientation(first_start, first_end, second_end)
    three = _orientation(second_start, second_end, first_start)
    four = _orientation(second_start, second_end, first_end)
    epsilon = 1e-12
    return (
        ((one > epsilon and two < -epsilon) or (one < -epsilon and two > epsilon))
        and ((three > epsilon and four < -epsilon) or (three < -epsilon and four > epsilon))
    )


def _polygon_self_intersects(vertices: Sequence[Point]) -> bool:
    count = len(vertices)
    for first in range(count):
        for second in range(first + 1, count):
            if (second + 1) % count == first or (first + 1) % count == second:
                continue
            if _segments_intersect(
                vertices[first],
                vertices[(first + 1) % count],
                vertices[second],
                vertices[(second + 1) % count],
            ):
                return True
    return False


def workspace_base_contains(workspace: WorkspaceLimit, yaw_rad: float, pitch_rad: float) -> bool:
    """Whether a target is inside the traced, mirrored, inset base region."""

    point = (float(yaw_rad), float(pitch_rad))
    if not all(math.isfinite(value) for value in point):
        return False
    points_to_check = [point]
    if workspace.symmetry_yaw_center_rad is not None:
        points_to_check.append(
            (
                2.0 * workspace.symmetry_yaw_center_rad - point[0],
                point[1],
            )
        )
    margin_sq = workspace.margin_rad ** 2
    for candidate in points_to_check:
        if not _point_in_polygon(candidate, workspace.vertices):
            return False
        if workspace.margin_rad > 0.0 and not all(
            _distance_sq_to_segment(
                candidate,
                workspace.vertices[index],
                workspace.vertices[(index + 1) % len(workspace.vertices)],
            ) >= margin_sq
            for index in range(len(workspace.vertices))
        ):
            return False
    return True


def workspace_in_home_corridor(
    workspace: WorkspaceLimit, yaw_rad: float, pitch_rad: float
) -> bool:
    """Whether a target is in the narrow vertical-only route to 45-degree home."""

    corridor = workspace.home_corridor
    if corridor is None or not all(math.isfinite(v) for v in (yaw_rad, pitch_rad)):
        return False
    pitch_min, pitch_max = sorted(
        (corridor.home_pitch_rad, corridor.transition_pitch_rad)
    )
    return (
        abs(float(yaw_rad) - corridor.yaw_center_rad)
        <= corridor.yaw_tolerance_rad + 1e-12
        and pitch_min <= float(pitch_rad) <= pitch_max
    )


def workspace_contains(workspace: WorkspaceLimit, yaw_rad: float, pitch_rad: float) -> bool:
    """Whether a target is in the base region or vertical-only home corridor."""

    return workspace_base_contains(
        workspace, yaw_rad, pitch_rad
    ) or workspace_in_home_corridor(workspace, yaw_rad, pitch_rad)


def workspace_segment_is_safe(workspace: WorkspaceLimit, start: Point, end: Point,
                              max_step_rad: float = math.radians(.1)) -> bool:
    """Check a motor-space segment at a bounded sample spacing.

    NumPy batches the polygon checks so the motor worker does not spend tens
    of milliseconds in Python loops. The scalar fallback has identical rules.
    """
    values = (*start, *end, max_step_rad)
    if not all(math.isfinite(float(value)) for value in values) or max_step_rad <= 0:
        raise ValueError("segment values must be finite and sampling step positive")
    steps = max(1, math.ceil(max(abs(end[0] - start[0]), abs(end[1] - start[1])) / max_step_rad))
    try:
        import numpy as np
    except ImportError:
        return all(workspace_contains(
            workspace, start[0] + (end[0] - start[0]) * index / steps,
            start[1] + (end[1] - start[1]) * index / steps,
        ) for index in range(steps + 1))
    fraction = np.arange(steps + 1, dtype=np.float64) / steps
    xs = start[0] + (end[0] - start[0]) * fraction
    ys = start[1] + (end[1] - start[1]) * fraction

    def inside_base(candidate_x):
        inside = np.zeros(len(xs), dtype=bool)
        on_edge = np.zeros(len(xs), dtype=bool)
        margin_ok = np.ones(len(xs), dtype=bool)
        for index, (ax, ay) in enumerate(workspace.vertices):
            bx, by = workspace.vertices[(index + 1) % len(workspace.vertices)]
            dx, dy = bx - ax, by - ay
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-24:
                distance_sq = (candidate_x - ax) ** 2 + (ys - ay) ** 2
            else:
                projection = np.clip(((candidate_x - ax) * dx + (ys - ay) * dy) / length_sq, 0., 1.)
                distance_sq = (candidate_x - ax - projection * dx) ** 2 + (ys - ay - projection * dy) ** 2
            on_edge |= distance_sq <= 1e-18
            if workspace.margin_rad > 0:
                margin_ok &= distance_sq >= workspace.margin_rad ** 2
            if ay != by:
                crossing = (ay > ys) != (by > ys)
                crossing_x = dx * (ys - ay) / dy + ax
                inside ^= crossing & (candidate_x < crossing_x)
        return (inside | on_edge) & margin_ok

    base = inside_base(xs)
    if workspace.symmetry_yaw_center_rad is not None:
        base &= inside_base(2.0 * workspace.symmetry_yaw_center_rad - xs)
    corridor = workspace.home_corridor
    if corridor is not None:
        lower, upper = sorted((corridor.home_pitch_rad, corridor.transition_pitch_rad))
        base |= ((np.abs(xs - corridor.yaw_center_rad) <= corridor.yaw_tolerance_rad + 1e-12)
                 & (ys >= lower) & (ys <= upper))
    return bool(np.all(base))


def clamp_workspace_target(
    workspace: WorkspaceLimit,
    start_yaw_rad: float,
    start_pitch_rad: float,
    target_yaw_rad: float,
    target_pitch_rad: float,
) -> Tuple[Point, bool]:
    """Clip to the first unsafe boundary while travelling from a safe point."""

    start = (float(start_yaw_rad), float(start_pitch_rad))
    target = (float(target_yaw_rad), float(target_pitch_rad))
    if not all(math.isfinite(value) for value in start + target):
        raise ValueError("workspace target and start must be finite")
    if workspace_contains(workspace, target[0], target[1]):
        return target, False
    if not workspace_contains(workspace, start[0], start[1]):
        start = workspace.anchor
    if not workspace_contains(workspace, start[0], start[1]):
        raise ValueError("workspace anchor is not inside its safety margin")

    safe_fraction = 0.0
    unsafe_fraction = 1.0
    for step in range(1, 25):
        fraction = step / 24.0
        candidate = (
            start[0] + (target[0] - start[0]) * fraction,
            start[1] + (target[1] - start[1]) * fraction,
        )
        if workspace_contains(workspace, candidate[0], candidate[1]):
            safe_fraction = fraction
        else:
            unsafe_fraction = fraction
            break
    for _ in range(24):
        fraction = (safe_fraction + unsafe_fraction) / 2.0
        candidate = (
            start[0] + (target[0] - start[0]) * fraction,
            start[1] + (target[1] - start[1]) * fraction,
        )
        if workspace_contains(workspace, candidate[0], candidate[1]):
            safe_fraction = fraction
        else:
            unsafe_fraction = fraction
    safe_fraction = max(0.0, safe_fraction - 1e-6)
    return (
        start[0] + (target[0] - start[0]) * safe_fraction,
        start[1] + (target[1] - start[1]) * safe_fraction,
    ), True


def _parse_motor_id(value: object) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def load_axis_limits(path: Path = DEFAULT_LIMITS_PATH) -> Dict[str, AxisLimit]:
    """Load and validate the persisted yaw/pitch motor limits."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    axes = data["axes"]
    result: Dict[str, AxisLimit] = {}
    motor_ids = set()
    for axis in ("yaw", "pitch"):
        values = axes[axis]
        limit = AxisLimit(
            axis=axis,
            motor_id=_parse_motor_id(values["motor_id"]),
            min_rad=float(values["min_rad"]),
            max_rad=float(values["max_rad"]),
        )
        if limit.motor_id in motor_ids:
            raise ValueError("yaw and pitch must have different motor IDs")
        motor_ids.add(limit.motor_id)
        result[axis] = limit
    return result


def load_motor_limits(path: Path = DEFAULT_LIMITS_PATH) -> Dict[int, Tuple[float, float]]:
    """Return ``motor_id -> (minimum_rad, maximum_rad)`` for backend checks."""

    return {
        limit.motor_id: (limit.min_rad, limit.max_rad)
        for limit in load_axis_limits(path).values()
    }


def load_hard_motor_limits(path: Path = DEFAULT_LIMITS_PATH) -> Dict[int, Tuple[float, float]]:
    """Load the original manual guard rails retained across workspace retraces."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    retained = (data.get("workspace") or {}).get("previous_hard_limits_rad")
    if not retained:
        return load_motor_limits(path)
    result = {}
    for raw_motor_id, values in retained.items():
        motor_id = _parse_motor_id(raw_motor_id)
        lower = float(values["min_rad"])
        upper = float(values["max_rad"])
        if lower >= upper or not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(f"invalid retained hard limits for motor 0x{motor_id:02X}")
        result[motor_id] = (lower, upper)
    return result


def load_workspace_limit(path: Path = DEFAULT_LIMITS_PATH) -> Optional[WorkspaceLimit]:
    """Load a traced non-rectangular workspace, if this calibration has one."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    raw_workspace = data.get("workspace")
    if raw_workspace is None:
        return None
    if raw_workspace.get("type") != "polygon":
        raise ValueError("unsupported workspace type")
    vertices = tuple(
        (float(vertex["yaw_rad"]), float(vertex["pitch_rad"]))
        for vertex in raw_workspace["vertices_rad"]
    )
    anchor = raw_workspace["anchor_rad"]
    workspace = WorkspaceLimit(
        yaw_motor_id=_parse_motor_id(raw_workspace["yaw_motor_id"]),
        pitch_motor_id=_parse_motor_id(raw_workspace["pitch_motor_id"]),
        vertices=vertices,
        anchor=(float(anchor["yaw_rad"]), float(anchor["pitch_rad"])),
        margin_rad=math.radians(float(raw_workspace.get("margin_motor_deg", 0.0))),
        symmetry_yaw_center_rad=(
            float(raw_workspace["symmetry_yaw_center_rad"])
            if raw_workspace.get("symmetry_yaw_center_rad") is not None
            else None
        ),
        home_corridor=(
            HomeCorridor(
                yaw_center_rad=float(raw_workspace["home_corridor"]["yaw_center_rad"]),
                yaw_tolerance_rad=math.radians(
                    float(raw_workspace["home_corridor"]["yaw_tolerance_motor_deg"])
                ),
                home_pitch_rad=float(raw_workspace["home_corridor"]["home_pitch_rad"]),
                transition_pitch_rad=float(
                    raw_workspace["home_corridor"]["transition_pitch_rad"]
                ),
            )
            if raw_workspace.get("home_corridor") is not None
            else None
        ),
    )
    axis_limits = load_axis_limits(path)
    if (
        workspace.yaw_motor_id != axis_limits["yaw"].motor_id
        or workspace.pitch_motor_id != axis_limits["pitch"].motor_id
    ):
        raise ValueError("workspace motor IDs do not match axis limits")
    if not workspace_contains(workspace, workspace.anchor[0], workspace.anchor[1]):
        raise ValueError("workspace anchor must be inside the inward safety margin")
    return workspace


def clamp_motor_position(
    position_rad: float,
    limits: Tuple[float, float],
) -> Tuple[float, bool]:
    """Clamp one target and report whether the safety limit clipped it."""

    if not math.isfinite(position_rad):
        raise ValueError("motor position must be finite")
    lower, upper = limits
    if lower >= upper:
        raise ValueError("motor limits must be increasing")
    clipped = max(lower, min(upper, position_rad))
    return clipped, clipped != position_rad


def save_manual_limits(
    path: Path,
    yaw_positions: Mapping[str, float],
    pitch_positions: Mapping[str, float],
    yaw_motor_id: int = 0x03,
    pitch_motor_id: int = 0x01,
    margin_motor_deg: float = 0.0,
) -> Path:
    """Save manually captured endpoints with an optional inward margin.

    ``yaw_positions`` must contain ``left_rad`` and ``right_rad``; pitch must
    contain ``low_rad`` and ``high_rad``.  The endpoint names preserve what
    the operator saw while the normalized min/max values are what the backend
    enforces.
    """

    if not math.isfinite(margin_motor_deg) or margin_motor_deg < 0.0:
        raise ValueError("margin_motor_deg must be finite and non-negative")

    def normalized(values: Mapping[str, float]) -> Tuple[float, float]:
        captured = [float(values[key]) for key in values]
        if len(captured) != 2 or not all(math.isfinite(value) for value in captured):
            raise ValueError("each axis must provide exactly two finite endpoints")
        lower = min(captured) + math.radians(margin_motor_deg)
        upper = max(captured) - math.radians(margin_motor_deg)
        if lower >= upper:
            raise ValueError("safety margin leaves no usable motor range")
        return lower, upper

    yaw_min, yaw_max = normalized(yaw_positions)
    pitch_min, pitch_max = normalized(pitch_positions)
    path = Path(path)

    def axis_data(
        axis: str,
        motor_id: int,
        values: Mapping[str, float],
        lower: float,
        upper: float,
    ) -> dict:
        return {
            "axis": axis,
            "motor_id": f"0x{motor_id:02X}",
            "captured_endpoints_rad": {key: float(value) for key, value in values.items()},
            "min_rad": lower,
            "max_rad": upper,
            "min_motor_deg": math.degrees(lower),
            "max_motor_deg": math.degrees(upper),
        }

    data = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "manual safe-range capture from DAMIAO motor feedback",
        "units": {"motor_position": "rad", "display": "deg"},
        "safety_margin_motor_deg": float(margin_motor_deg),
        "axes": {
            "yaw": axis_data("yaw", yaw_motor_id, yaw_positions, yaw_min, yaw_max),
            "pitch": axis_data("pitch", pitch_motor_id, pitch_positions, pitch_min, pitch_max),
        },
    }

    backup_path = path.with_suffix(path.suffix + ".bak")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        shutil.copy2(str(path), str(backup_path))
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return backup_path
def _rdp(points: Sequence[Point], tolerance_rad: float) -> List[Point]:
    """Ramer-Douglas-Peucker simplification for one open polyline."""

    if len(points) <= 2:
        return list(points)
    first = points[0]
    last = points[-1]
    farthest_index = 0
    farthest_distance = -1.0
    for index in range(1, len(points) - 1):
        distance = _distance_sq_to_segment(points[index], first, last)
        if distance > farthest_distance:
            farthest_index = index
            farthest_distance = distance
    if farthest_distance <= tolerance_rad ** 2:
        return [first, last]
    return (
        _rdp(points[: farthest_index + 1], tolerance_rad)[:-1]
        + _rdp(points[farthest_index:], tolerance_rad)
    )


def _simplify_closed_boundary(points: Sequence[Point], tolerance_rad: float) -> Tuple[Point, ...]:
    """Simplify a closed trace without replacing it with a diagonal shortcut."""

    if len(points) < 3:
        raise ValueError("workspace trace needs at least three samples")
    first_index, second_index = 0, 1
    largest_distance = -1.0
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            distance = (points[first][0] - points[second][0]) ** 2 + (
                points[first][1] - points[second][1]
            ) ** 2
            if distance > largest_distance:
                first_index, second_index = first, second
                largest_distance = distance
    forward = list(points[first_index : second_index + 1])
    backward = list(points[second_index:]) + list(points[: first_index + 1])
    simplified = _rdp(forward, tolerance_rad)[:-1] + _rdp(backward, tolerance_rad)[:-1]
    if len(simplified) < 3:
        raise ValueError("workspace trace collapsed to fewer than three vertices")
    return tuple(simplified)


def _repair_closure_seam(
    points: Sequence[Point],
    closure_tolerance_rad: float,
    join_tolerance_rad: float,
) -> Tuple[Tuple[Point, ...], int]:
    """Trim a small overlap where the end rejoins the beginning of a trace.

    Only the first and last halves are considered, and the join must remain
    near both recorded endpoints.  This repairs hand-traced corner overshoot
    without accepting an intersection in the middle of the workspace.
    """

    if len(points) < 6:
        return tuple(points[:-1]), 0
    start = points[0]
    end = points[-1]
    nearby_sq = closure_tolerance_rad ** 2
    join_sq = join_tolerance_rad ** 2
    split = len(points) // 2
    best = None
    for first in range(split):
        segment_start = points[first]
        segment_end = points[first + 1]
        if min(
            (segment_start[0] - start[0]) ** 2 + (segment_start[1] - start[1]) ** 2,
            (segment_end[0] - start[0]) ** 2 + (segment_end[1] - start[1]) ** 2,
        ) > nearby_sq:
            continue
        for second in range(split + 1, len(points) - 1):
            point = points[second]
            if (point[0] - end[0]) ** 2 + (point[1] - end[1]) ** 2 > nearby_sq:
                continue
            distance_sq = _distance_sq_to_segment(point, segment_start, segment_end)
            if distance_sq > join_sq:
                continue
            candidate = (distance_sq, -(second - first), first, second)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return tuple(points[:-1]), 0

    _, _, first, second = best
    segment_start = points[first]
    segment_end = points[first + 1]
    dx = segment_end[0] - segment_start[0]
    dy = segment_end[1] - segment_start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-24:
        join = segment_start
    else:
        point = points[second]
        fraction = (
            (point[0] - segment_start[0]) * dx
            + (point[1] - segment_start[1]) * dy
        ) / length_sq
        fraction = max(0.0, min(1.0, fraction))
        join = (segment_start[0] + fraction * dx, segment_start[1] + fraction * dy)

    repaired = [join] + list(points[first + 1 : second])
    deduplicated = []
    for point in repaired:
        if not deduplicated or point != deduplicated[-1]:
            deduplicated.append(point)
    if len(deduplicated) < 3:
        return tuple(points[:-1]), 0
    # Report only the discarded beginning/end tails; stationary duplicate
    # feedback samples inside the retained loop are not seam trimming.
    trimmed_samples = first + (len(points) - second)
    return tuple(deduplicated), trimmed_samples


def save_workspace_limits(
    path: Path,
    samples: Sequence[Mapping[str, float]],
    anchor_position: Mapping[str, float],
    yaw_motor_id: int = 0x03,
    pitch_motor_id: int = 0x01,
    margin_motor_deg: float = 0.5,
    simplify_motor_deg: float = 0.25,
    closure_motor_deg: float = 2.0,
    hard_motor_limits: Optional[Mapping[int, Tuple[float, float]]] = None,
    yaw_symmetry_center_rad: Optional[float] = None,
    home_position: Optional[Mapping[str, float]] = None,
    home_corridor_yaw_deg: float = 0.25,
) -> Path:
    """Save a manually traced non-rectangular yaw/pitch safety envelope.

    ``samples`` must follow one closed path around the inside of the safe
    mechanical boundary.  The raw samples are retained for audit; a simplified
    polygon is used at runtime for fast final safety checks.
    """

    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (
            margin_motor_deg,
            simplify_motor_deg,
            closure_motor_deg,
            home_corridor_yaw_deg,
        )
    ):
        raise ValueError("workspace margin, simplification, and closure values must be non-negative")
    if yaw_symmetry_center_rad is not None:
        yaw_symmetry_center_rad = float(yaw_symmetry_center_rad)
        if not math.isfinite(yaw_symmetry_center_rad):
            raise ValueError("yaw symmetry center must be finite")

    normalized_hard_limits: Dict[int, Tuple[float, float]] = {}
    for motor_id, bounds in (hard_motor_limits or {}).items():
        lower, upper = float(bounds[0]), float(bounds[1])
        if lower >= upper or not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(f"invalid hard limits for motor 0x{int(motor_id):02X}")
        normalized_hard_limits[int(motor_id)] = (lower, upper)

    raw_points: List[Point] = []
    constrained_points: List[Point] = []
    stored_samples: List[dict] = []
    hard_limit_violation_count = 0
    for sample in samples:
        yaw = float(sample["yaw_rad"])
        pitch = float(sample["pitch_rad"])
        if not math.isfinite(yaw) or not math.isfinite(pitch):
            raise ValueError("workspace samples must be finite")
        raw_points.append((yaw, pitch))
        constrained_yaw = yaw
        constrained_pitch = pitch
        if yaw_motor_id in normalized_hard_limits:
            constrained_yaw, _ = clamp_motor_position(
                yaw, normalized_hard_limits[yaw_motor_id]
            )
        if pitch_motor_id in normalized_hard_limits:
            constrained_pitch, _ = clamp_motor_position(
                pitch, normalized_hard_limits[pitch_motor_id]
            )
        outside_hard_limits = constrained_yaw != yaw or constrained_pitch != pitch
        if outside_hard_limits:
            hard_limit_violation_count += 1
        constrained_points.append((constrained_yaw, constrained_pitch))
        stored = {"yaw_rad": yaw, "pitch_rad": pitch}
        if "elapsed_s" in sample:
            stored["elapsed_s"] = float(sample["elapsed_s"])
        if outside_hard_limits:
            stored.update(
                {
                    "outside_hard_limits": True,
                    "constrained_yaw_rad": constrained_yaw,
                    "constrained_pitch_rad": constrained_pitch,
                }
            )
        stored_samples.append(stored)

    if len(raw_points) < 20:
        raise ValueError("workspace trace needs at least 20 samples")
    closure_gap = math.hypot(
        constrained_points[0][0] - constrained_points[-1][0],
        constrained_points[0][1] - constrained_points[-1][1],
    )
    if closure_gap > math.radians(closure_motor_deg):
        raise ValueError(
            "workspace trace did not return to its start "
            f"({math.degrees(closure_gap):.2f}° gap; allowed {closure_motor_deg:.2f}°)"
        )

    boundary_points = tuple(constrained_points[:-1])
    vertices = _simplify_closed_boundary(
        boundary_points, math.radians(simplify_motor_deg)
    )
    closure_seam_trimmed_samples = 0
    if _polygon_self_intersects(vertices):
        boundary_points, closure_seam_trimmed_samples = _repair_closure_seam(
            constrained_points,
            closure_tolerance_rad=math.radians(closure_motor_deg),
            join_tolerance_rad=max(
                math.radians(0.05),
                min(math.radians(simplify_motor_deg), math.radians(0.25)),
            ),
        )
        vertices = _simplify_closed_boundary(
            boundary_points, math.radians(simplify_motor_deg)
        )
    anchor = (float(anchor_position["yaw_rad"]), float(anchor_position["pitch_rad"]))
    for motor_id, value in ((yaw_motor_id, anchor[0]), (pitch_motor_id, anchor[1])):
        bounds = normalized_hard_limits.get(motor_id)
        if bounds is not None and not bounds[0] <= value <= bounds[1]:
            raise ValueError(
                f"anchor for motor 0x{motor_id:02X} is outside the previous hard limits"
            )
    workspace = WorkspaceLimit(
        yaw_motor_id=yaw_motor_id,
        pitch_motor_id=pitch_motor_id,
        vertices=vertices,
        anchor=anchor,
        margin_rad=math.radians(margin_motor_deg),
        symmetry_yaw_center_rad=yaw_symmetry_center_rad,
    )
    if not workspace_base_contains(workspace, anchor[0], anchor[1]):
        raise ValueError("anchor must be inside the traced outline and away from its safety margin")

    home_corridor = None
    if home_position is not None:
        home_yaw = float(home_position["yaw_rad"])
        home_pitch = float(home_position["pitch_rad"])
        if not all(math.isfinite(value) for value in (home_yaw, home_pitch)):
            raise ValueError("home position must be finite")
        for motor_id, value in ((yaw_motor_id, home_yaw), (pitch_motor_id, home_pitch)):
            bounds = normalized_hard_limits.get(motor_id)
            if bounds is not None and not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"home for motor 0x{motor_id:02X} is outside hard limits")
        if not workspace_base_contains(workspace, home_yaw, home_pitch):
            # Find the first base-safe point while moving vertically from home
            # toward the operator-selected anchor. This guarantees overlap.
            unsafe_fraction = 0.0
            safe_fraction = None
            for step in range(1, 201):
                fraction = step / 200.0
                pitch = home_pitch + (anchor[1] - home_pitch) * fraction
                if workspace_base_contains(workspace, home_yaw, pitch):
                    safe_fraction = fraction
                    break
                unsafe_fraction = fraction
            if safe_fraction is None:
                raise ValueError("home yaw has no vertical connection to the traced workspace")
            for _ in range(30):
                fraction = (unsafe_fraction + safe_fraction) / 2.0
                pitch = home_pitch + (anchor[1] - home_pitch) * fraction
                if workspace_base_contains(workspace, home_yaw, pitch):
                    safe_fraction = fraction
                else:
                    unsafe_fraction = fraction
            transition_pitch = home_pitch + (anchor[1] - home_pitch) * safe_fraction
            home_corridor = HomeCorridor(
                yaw_center_rad=home_yaw,
                yaw_tolerance_rad=math.radians(home_corridor_yaw_deg),
                home_pitch_rad=home_pitch,
                transition_pitch_rad=transition_pitch,
            )
            workspace = WorkspaceLimit(
                yaw_motor_id=workspace.yaw_motor_id,
                pitch_motor_id=workspace.pitch_motor_id,
                vertices=workspace.vertices,
                anchor=workspace.anchor,
                margin_rad=workspace.margin_rad,
                symmetry_yaw_center_rad=workspace.symmetry_yaw_center_rad,
                home_corridor=home_corridor,
            )

    yaw_observed_min = min(point[0] for point in raw_points)
    yaw_observed_max = max(point[0] for point in raw_points)
    pitch_observed_min = min(point[1] for point in raw_points)
    pitch_observed_max = max(point[1] for point in raw_points)
    # Runtime axis bounds should envelope the repaired polygon, not discarded
    # seam overshoot samples (the latter remain preserved in the raw audit).
    yaw_constrained_min = min(point[0] for point in vertices)
    yaw_constrained_max = max(point[0] for point in vertices)
    pitch_constrained_min = min(point[1] for point in vertices)
    pitch_constrained_max = max(point[1] for point in vertices)
    margin_rad = workspace.margin_rad
    yaw_min, yaw_max = yaw_constrained_min + margin_rad, yaw_constrained_max - margin_rad
    pitch_min, pitch_max = pitch_constrained_min + margin_rad, pitch_constrained_max - margin_rad
    if yaw_motor_id in normalized_hard_limits:
        hard_min, hard_max = normalized_hard_limits[yaw_motor_id]
        yaw_min, yaw_max = max(yaw_min, hard_min), min(yaw_max, hard_max)
    if yaw_symmetry_center_rad is not None:
        symmetric_half_span = min(
            yaw_symmetry_center_rad - yaw_min,
            yaw_max - yaw_symmetry_center_rad,
        )
        if symmetric_half_span <= 0.0:
            raise ValueError("yaw symmetry center is outside the usable workspace")
        yaw_min = yaw_symmetry_center_rad - symmetric_half_span
        yaw_max = yaw_symmetry_center_rad + symmetric_half_span
    if pitch_motor_id in normalized_hard_limits:
        hard_min, hard_max = normalized_hard_limits[pitch_motor_id]
        pitch_min, pitch_max = max(pitch_min, hard_min), min(pitch_max, hard_max)
    if home_position is not None:
        home_pitch = float(home_position["pitch_rad"])
        pitch_min = min(pitch_min, home_pitch)
        pitch_max = max(pitch_max, home_pitch)
    if yaw_min >= yaw_max or pitch_min >= pitch_max:
        raise ValueError("safety margin leaves no usable axis range")

    def axis_data(axis: str, motor_id: int, observed_min: float, observed_max: float, lower: float, upper: float) -> dict:
        return {
            "axis": axis,
            "motor_id": f"0x{motor_id:02X}",
            "captured_endpoints_rad": {
                "observed_min_rad": observed_min,
                "observed_max_rad": observed_max,
            },
            "min_rad": lower,
            "max_rad": upper,
            "min_motor_deg": math.degrees(lower),
            "max_motor_deg": math.degrees(upper),
        }

    data = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "manual closed-outline workspace trace from DAMIAO motor feedback",
        "units": {"motor_position": "rad", "display": "deg"},
        "safety_margin_motor_deg": float(margin_motor_deg),
        "axes": {
            "yaw": axis_data("yaw", yaw_motor_id, yaw_observed_min, yaw_observed_max, yaw_min, yaw_max),
            "pitch": axis_data("pitch", pitch_motor_id, pitch_observed_min, pitch_observed_max, pitch_min, pitch_max),
        },
        "workspace": {
            "type": "polygon",
            "yaw_motor_id": f"0x{yaw_motor_id:02X}",
            "pitch_motor_id": f"0x{pitch_motor_id:02X}",
            "margin_motor_deg": float(margin_motor_deg),
            "simplify_motor_deg": float(simplify_motor_deg),
            "closure_gap_motor_deg": math.degrees(closure_gap),
            "closure_seam_trimmed_samples": closure_seam_trimmed_samples,
            "symmetry_yaw_center_rad": yaw_symmetry_center_rad,
            "home_corridor": (
                {
                    "yaw_center_rad": home_corridor.yaw_center_rad,
                    "yaw_tolerance_motor_deg": math.degrees(home_corridor.yaw_tolerance_rad),
                    "home_pitch_rad": home_corridor.home_pitch_rad,
                    "transition_pitch_rad": home_corridor.transition_pitch_rad,
                }
                if home_corridor is not None
                else None
            ),
            "hard_limit_violation_count": hard_limit_violation_count,
            "previous_hard_limits_rad": {
                f"0x{motor_id:02X}": {"min_rad": bounds[0], "max_rad": bounds[1]}
                for motor_id, bounds in normalized_hard_limits.items()
            },
            "anchor_rad": {"yaw_rad": anchor[0], "pitch_rad": anchor[1]},
            "vertices_rad": [
                {"yaw_rad": yaw, "pitch_rad": pitch} for yaw, pitch in vertices
            ],
            "captured_boundary_samples_rad": stored_samples,
        },
    }

    path = Path(path)
    backup_path = path.with_suffix(path.suffix + ".bak")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        shutil.copy2(str(path), str(backup_path))
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return backup_path
