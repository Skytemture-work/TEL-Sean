"""Preview YOLO segmentation through TensorRT with ZED depth.

This diagnostic intentionally has no motor-control imports or commands.  It
overlays segmentation results, the depth at each detection centre, and timing.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .board import (
    build_plan,
    build_plan_from_board_quad,
    build_virtual_plan,
    build_virtual_plan_from_partial,
    estimate_board_distance_from_targets,
)
from .config import TurretConfig
from .detection import boxes_to_board_targets
from .run_yolo_demo import _draw_board_plan
from .board_surface import detect_dark_board_surface, filter_targets_to_surface
from .vision_geometry import polygon_geometry, sample_board_depth


DEFAULT_ENGINE = Path.home() / "Desktop" / "best.engine"


def _draw_inferred_board(display: np.ndarray, plan) -> None:
    """Draw geometry-only hole guesses in magenta for visual inspection.

    These points are deliberately different from the real board plan. They are
    never passed to aiming or firing code; they only show what the known 3x4
    layout predicts when YOLO misses one or more holes.
    """
    color = (255, 0, 255)  # BGR magenta: inferred, not YOLO-observed
    by_column = {}
    by_row = {}
    for target in plan.targets:
        if target.column is None or target.row is None:
            continue
        point = (round(target.u_px), round(target.v_px))
        by_column.setdefault(target.column, []).append((target.row, point))
        by_row.setdefault(target.row, []).append((target.column, point))
        cv2.drawMarker(display, point, color, cv2.MARKER_CROSS, 18, 2)
        cv2.circle(display, point, 12, color, 1, cv2.LINE_AA)
        cv2.putText(
            display,
            f"INFER {target.hole_id or 'H??'}",
            (point[0] + 8, point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    for points in by_column.values():
        points.sort()
        for (_, start), (_, end) in zip(points, points[1:]):
            cv2.line(display, start, end, color, 1, cv2.LINE_AA)
    for points in by_row.values():
        points.sort()
        for (_, start), (_, end) in zip(points, points[1:]):
            cv2.line(display, start, end, color, 1, cv2.LINE_AA)

    cv2.putText(
        display,
        "INFERRED BOARD ONLY - NO FIRE",
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely test a YOLO TensorRT segmentation engine on ZED."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--frames", type=int, default=0, help="0 runs until Q/Esc")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--save-frame",
        type=Path,
        help="save the first partial/inferred overlay frame to this path",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"找不到模型：{args.model}")

    # TensorRT 8.5 still asks for the removed NumPy alias on JetPack 5.
    if "bool" not in np.__dict__:
        np.bool = bool  # type: ignore[attr-defined]

    try:
        import pyzed.sl as sl
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"缺少 ZED/Ultralytics 套件：{exc}") from exc

    model = YOLO(str(args.model), task="segment")
    config = TurretConfig()
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.ULTRA

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"無法開啟 ZED：{status}")

    image = sl.Mat()
    depth = sl.Mat()
    runtime = sl.RuntimeParameters()
    window = "ZED YOLO TensorRT segmentation (Q/Esc: quit)"
    frame_count = 0
    started = time.perf_counter()
    saved_frame = False

    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print("安全模式：只讀取相機並執行推論，不會連線或控制馬達。")
    print(f"模型：{args.model}")
    try:
        while not args.frames or frame_count < args.frames:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
            depth_data = depth.get_data()

            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=0,
                verbose=False,
            )[0]
            raw_detected = boxes_to_board_targets(result, args.conf)
            board_surface = detect_dark_board_surface(
                frame,
                value_threshold=config.board_dark_value_threshold,
                min_area_fraction=config.board_min_area_fraction,
                targets=raw_detected,
            )
            detected = filter_targets_to_surface(raw_detected, board_surface)
            quad_plan = None
            if board_surface is not None and board_surface.quad is not None:
                quad_distance = estimate_board_distance_from_targets(
                    detected,
                    board_surface.quad,
                    config.focal_length_x_px,
                    config.small_hole_diameter_m,
                    config.large_hole_diameter_m,
                    config.board_position_tolerance_cm,
                )
                quad_plan = build_plan_from_board_quad(
                    board_surface.quad,
                    config.small_hole_diameter_m,
                    config.large_hole_diameter_m,
                    board_distance_m=quad_distance,
                    large_first=config.engage_large_first,
                    left_to_right=config.engage_left_to_right,
                    bottom_to_top=config.engage_bottom_to_top,
                )
            plan = build_plan(
                detected,
                config.focal_length_x_px,
                config.small_hole_diameter_m,
                config.large_hole_diameter_m,
                config.engage_large_first,
                config.engage_left_to_right,
                config.engage_bottom_to_top,
                config.board_position_tolerance_cm,
            )
            board_ready = len(plan.targets) == 12 and all(
                target.hole_id is not None and target.diameter_m is not None
                for target in plan.targets
            )
            inferred_plan = None
            observed_fallback_plan = None
            if not board_ready and len(detected) == 12:
                # Twelve candidates should use their actual positions.  A
                # partial homography here could move a valid point elsewhere
                # when size classification alone was the failed check.
                candidate = build_virtual_plan(
                    detected,
                    config.small_hole_diameter_m,
                    config.large_hole_diameter_m,
                    config.engage_large_first,
                    config.engage_left_to_right,
                    config.engage_bottom_to_top,
                    config.board_position_tolerance_cm,
                )
                if len(candidate.targets) != 12:
                    candidate = None
                else:
                    observed_fallback_plan = candidate
            elif not board_ready and len(detected) >= 4:
                candidate = build_virtual_plan_from_partial(
                    detected,
                    config.small_hole_diameter_m,
                    config.large_hole_diameter_m,
                    config.engage_large_first,
                    config.engage_left_to_right,
                    config.engage_bottom_to_top,
                    focal_px=config.focal_length_x_px,
                    position_tolerance_cm=config.board_position_tolerance_cm,
                )
                if len(candidate.targets) == 12:
                    inferred_plan = candidate
            # Large class/confidence labels obscure small holes on the 3x4
            # target. Keep thin boxes and masks, then add only compact depth
            # text below.
            display = result.plot(labels=False, line_width=1)
            if board_surface is not None:
                cv2.drawContours(
                    display,
                    [board_surface.contour],
                    -1,
                    (0, 255, 255),
                    2,
                )
            display_plan = quad_plan or observed_fallback_plan or inferred_plan or plan
            _draw_board_plan(display, display_plan)
            overlay_plan = quad_plan or inferred_plan
            if overlay_plan is not None:
                _draw_inferred_board(display, overlay_plan)
                if args.save_frame and not saved_frame:
                    args.save_frame.parent.mkdir(parents=True, exist_ok=True)
                    if cv2.imwrite(str(args.save_frame), display):
                        print(f"saved inferred frame: {args.save_frame}")
                        saved_frame = True
            frame_count += 1
            elapsed = time.perf_counter() - started
            fps = frame_count / elapsed if elapsed else 0.0
            count = len(detected)
            detection_text = []

            if result.boxes is not None:
                masks = result.masks
                for index, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
                    polygon = None
                    geometry = None
                    if masks is not None and index < len(masks):
                        try:
                            polygon = masks.xy[index]
                            geometry = polygon_geometry(polygon)
                        except (AttributeError, IndexError, TypeError, ValueError):
                            geometry = None
                    if geometry is None:
                        u, v = round((x1 + x2) / 2), round((y1 + y2) / 2)
                        distance = math.nan
                    else:
                        u, v = round(geometry.center_u), round(geometry.center_v)
                        distance = sample_board_depth(
                            depth_data,
                            polygon=polygon,
                            ring_px=6,
                        )
                        if distance is None:
                            distance = math.nan
                    label = "depth unavailable" if math.isnan(distance) else f"{distance:.2f} m"
                    detection_text.append(
                        f"#{index + 1} conf={float(box.conf[0]):.2f} "
                        f"center=({u},{v}) depth={label}"
                    )
                    cv2.drawMarker(display, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
                    cv2.putText(
                        display,
                        label,
                        (u + 8, max(20, v - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            inference_ms = float(result.speed.get("inference", math.nan))
            if board_ready:
                board_text = "BOARD READY"
            elif inferred_plan is not None:
                board_text = "PARTIAL -> INFERRED 12/12"
            else:
                board_text = "BOARD WAIT"
            status_text = (
                f"{board_text}  holes={count}/{len(raw_detected)} raw  "
                f"plan={len(plan.targets)}/12  "
                f"infer={inference_ms:.1f} ms  avg={fps:.1f} FPS"
            )
            cv2.putText(
                display,
                status_text,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if frame_count == 1 or frame_count % 30 == 0:
                print(status_text)
                for text in detection_text:
                    print(f"  {text}")
                if board_ready:
                    print(plan.describe())
                elif inferred_plan is not None:
                    print("inferred geometry only (not fireable):")
                    print(inferred_plan.describe())
            if not args.headless:
                cv2.imshow(window, display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
