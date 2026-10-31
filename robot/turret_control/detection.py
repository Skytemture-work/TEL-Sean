"""Small adapter for Ultralytics YOLO detection results."""

from typing import Any, Optional

from .aiming import TargetPixel
from .vision_geometry import MaskGeometry, polygon_geometry, sample_board_depth


def _geometry_for_mask(result: Any, index: int) -> Optional[MaskGeometry]:
    """Read one Ultralytics polygon, if this is a segmentation result."""

    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    try:
        polygons = masks.xy
        if index >= len(polygons):
            return None
        return polygon_geometry(polygons[index])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


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
    for index, box in enumerate(boxes):
        confidence = float(box.conf[0].item())
        if confidence < best_confidence:
            continue

        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        geometry = _geometry_for_mask(result, index)
        best = TargetPixel(
            u_px=geometry.center_u if geometry else (x1 + x2) / 2.0,
            v_px=geometry.center_v if geometry else (y1 + y2) / 2.0,
            confidence=confidence,
        )
        best_confidence = confidence

    return best


def boxes_to_board_targets(result, minimum_confidence: float = 0.40, depth_data=None):
    """Convert every detection into a board target.

    Segmentation masks provide the centre and apparent diameter.  Bounding
    boxes remain a compatibility fallback for detection-only models.
    """
    from .board import BoardTarget

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    targets = []
    for index, box in enumerate(boxes):
        confidence = float(box.conf[0].item())
        if confidence < minimum_confidence:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        geometry = _geometry_for_mask(result, index)
        if geometry is not None:
            u_px = geometry.center_u
            v_px = geometry.center_v
            width_px = geometry.width_px
            height_px = geometry.height_px
            diameter_px = geometry.equivalent_diameter_px
            depth_m = (sample_board_depth(depth_data, polygon=geometry.contour)
                       if depth_data is not None else None)
        else:
            u_px = (x1 + x2) / 2.0
            v_px = (y1 + y2) / 2.0
            width_px = abs(x2 - x1)
            height_px = abs(y2 - y1)
            diameter_px = None
            depth_m = None
        targets.append(
            BoardTarget(
                u_px=u_px,
                v_px=v_px,
                width_px=width_px,
                height_px=height_px,
                confidence=confidence,
                diameter_px=diameter_px,
                distance_m=depth_m,
                distance_source="zed_depth" if depth_m is not None else None,
            )
        )
    return targets
