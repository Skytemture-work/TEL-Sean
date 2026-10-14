"""Small adapter for Ultralytics YOLO detection results."""

from typing import Any, Optional

from .aiming import TargetPixel


def select_best_box(result: Any, minimum_confidence: float = 0.40) -> Optional[TargetPixel]:
    """Select the highest-confidence box from one Ultralytics result.

    The current dataset contains one class (``hole``).  Target selection can
    later be replaced with a game-specific rule without changing the aiming or
    motor code.
    """

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_confidence = minimum_confidence
    for box in boxes:
        confidence = float(box.conf[0].item())
        if confidence < best_confidence:
            continue

        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        best = TargetPixel(
            u_px=(x1 + x2) / 2.0,
            v_px=(y1 + y2) / 2.0,
            confidence=confidence,
        )
        best_confidence = confidence

    return best


def boxes_to_board_targets(result, minimum_confidence: float = 0.40):
    """Every box in one Ultralytics result, as board targets.

    Unlike :func:`select_best_box` this keeps *all* of them: the board is
    static and fully in frame, so the plan is built from one good detection
    rather than re-picking a winner every frame.
    """
    from .board import BoardTarget

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    targets = []
    for box in boxes:
        confidence = float(box.conf[0].item())
        if confidence < minimum_confidence:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        targets.append(
            BoardTarget(
                u_px=(x1 + x2) / 2.0,
                v_px=(y1 + y2) / 2.0,
                width_px=abs(x2 - x1),
                height_px=abs(y2 - y1),
                confidence=confidence,
            )
        )
    return targets
