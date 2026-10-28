"""Fixed-camera board layout calibration and replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .board import BoardTarget, EngagementPlan, build_plan
from .config import TurretConfig


def calibrate_layout(
    targets: Sequence[BoardTarget],
    config: TurretConfig,
    output: Path,
) -> EngagementPlan:
    """Validate one complete detection and save its 12 fixed hole positions."""
    plan = build_plan(
        targets,
        config.focal_length_x_px,
        config.small_hole_diameter_m,
        config.large_hole_diameter_m,
        config.engage_large_first,
        config.engage_left_to_right,
        config.engage_bottom_to_top,
        config.board_position_tolerance_cm,
    )
    if len(plan.targets) != 12 or any(t.hole_id is None for t in plan.targets):
        raise ValueError("need exactly 12 detected holes with valid board geometry")

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "note": "Fixed camera/board layout. Recalibrate if either moves.",
        "targets": [
            {
                "u_px": t.u_px,
                "v_px": t.v_px,
                "width_px": t.width_px,
                "height_px": t.height_px,
                "diameter_px": t.diameter_px,
                "confidence": 1.0,
                "diameter_m": t.diameter_m,
                "distance_m": t.distance_m,
                "hole_id": t.hole_id,
                "row": t.row,
                "column": t.column,
            }
            for t in plan.targets
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def load_layout(path: Path) -> EngagementPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = [
        BoardTarget(
            u_px=float(item["u_px"]),
            v_px=float(item["v_px"]),
            width_px=float(item.get("width_px", 1.0)),
            height_px=float(item.get("height_px", 1.0)),
            confidence=float(item.get("confidence", 1.0)),
            diameter_px=(
                None
                if item.get("diameter_px") is None
                else float(item["diameter_px"])
            ),
            diameter_m=item.get("diameter_m"),
            distance_m=item.get("distance_m"),
            column=item.get("column"),
            hole_id=item["hole_id"],
            row=item.get("row"),
        )
        for item in payload["targets"]
    ]
    if len(targets) != 12 or {t.hole_id for t in targets} != {f"H{i:02d}" for i in range(1, 13)}:
        raise ValueError(f"{path} must contain H01..H12 exactly once")
    return EngagementPlan(targets)
