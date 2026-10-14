"""Run YOLO either as a single-target aiming demo or a board preview.

Both modes are intended for laptop development.  They do not open CAN and do
not move a real motor.  The board mode detects the whole board in one frame,
then runs the real size-classification, ranging and engagement-order code.

Example:
    python -m turret_control.run_yolo_demo \
        --model jfwiskamckzxc.v2i.yolo26/runs/segment/runs/segment/hole_seg/weights/best.pt

    python -m turret_control.run_yolo_demo --board --device cpu --zed-uvc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .backends import MockMotorBackend
from .board import EngagementPlan, build_plan
from .config import TurretConfig
from .controller import TurretController
from .detection import boxes_to_board_targets, select_best_box


REPO_MODEL = Path(__file__).resolve().parent.parent / "best_v2.pt"
LOCAL_MODEL = (
    Path(__file__).resolve().parent.parent
    / "jfwiskamckzxc.v2i.yolo26/runs/segment/runs/segment/hole_seg/weights/best.pt"
)
DEFAULT_MODEL = REPO_MODEL if REPO_MODEL.is_file() else LOCAL_MODEL


def _inference_frame(frame, zed_uvc: bool):
    """Return the image passed to YOLO for a normal or side-by-side camera."""
    if zed_uvc:
        return frame[:, : frame.shape[1] // 2].copy()
    return frame


def _draw_board_plan(display, plan: EngagementPlan) -> None:
    """Draw the ordered board plan over the YOLO result image."""
    for number, target in enumerate(plan.targets, 1):
        if target.diameter_m == 0.40:
            color = (0, 220, 0)       # large hole
        elif target.diameter_m == 0.20:
            color = (0, 220, 255)     # small hole
        else:
            color = (0, 0, 255)       # classification is not trustworthy

        left = round(target.u_px - target.width_px / 2.0)
        top = round(target.v_px - target.height_px / 2.0)
        right = round(target.u_px + target.width_px / 2.0)
        bottom = round(target.v_px + target.height_px / 2.0)
        cv2.rectangle(display, (left, top), (right, bottom), color, 2)
        cv2.drawMarker(
            display,
            (round(target.u_px), round(target.v_px)),
            color,
            cv2.MARKER_CROSS,
            16,
            2,
        )

        diameter = "?" if target.diameter_m is None else f"{target.diameter_m * 100:.0f}cm"
        distance = "?" if target.distance_m is None else f"{target.distance_m:.2f}m"
        hole_id = target.hole_id or "H??"
        label = f"#{number} {hole_id} Ø{diameter} {distance}"
        cv2.putText(
            display,
            label,
            (left, max(18, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )


def _print_board_plan(plan: EngagementPlan, detected_count: int) -> None:
    known_count = sum(target.diameter_m is not None for target in plan.targets)
    ranged_count = sum(target.distance_m is not None for target in plan.targets)
    print(
        f"靶板偵測 {detected_count} 個，尺寸分類 {known_count}/{detected_count}，"
        f"測距 {ranged_count}/{detected_count}"
    )
    print(plan.describe() if len(plan) else "（沒有偵測到孔）")


def run_board_preview(
    model,
    camera,
    config: TurretConfig,
    device: str,
    zed_uvc: bool,
    imgsz: int,
    conf: float,
) -> None:
    """Preview detection, board classification, ranging and ordering."""
    window = "YOLO board preview (P: print plan, Q/Esc: quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    last_state = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame read failed")
                break

            inference_frame = _inference_frame(frame, zed_uvc)
            result = model.predict(
                inference_frame,
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False,
            )[0]
            detected = boxes_to_board_targets(result, conf)
            plan = build_plan(
                detected,
                config.focal_length_x_px,
                config.small_hole_diameter_m,
                config.large_hole_diameter_m,
                config.engage_large_first,
                config.engage_left_to_right,
                config.engage_bottom_to_top,
            )

            known_count = sum(target.diameter_m is not None for target in plan.targets)
            ready = len(detected) == 12 and known_count == len(detected)
            state = (len(detected), known_count, ready)
            if state != last_state:
                _print_board_plan(plan, len(detected))
                last_state = state

            display = result.plot()
            _draw_board_plan(display, plan)
            if ready:
                status = "BOARD READY: 12 holes classified and ordered"
                status_color = (0, 220, 0)
            elif known_count != len(detected):
                status = f"WAIT: size classification {known_count}/{len(detected)}"
                status_color = (0, 0, 255)
            else:
                status = f"WAIT: detected {len(detected)}/12 holes"
                status_color = (0, 220, 255)

            cv2.putText(
                display,
                status,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                _print_board_plan(plan, len(detected))
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--board",
        action="store_true",
        help="detect the whole 3x4 board and preview classification/order",
    )
    parser.add_argument(
        "--zed-uvc",
        action="store_true",
        help="use the left half of a side-by-side ZED UVC frame",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "This demo requires Ultralytics. Install it with: "
            "python3 -m pip install ultralytics"
        ) from exc

    model = YOLO(str(args.model))
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Cannot open camera /dev/video{args.camera}")

    config = TurretConfig()
    if args.board:
        run_board_preview(
            model,
            camera,
            config,
            args.device,
            args.zed_uvc,
            args.imgsz,
            args.conf,
        )
        return

    turret = TurretController(
        config,
        MockMotorBackend(motor_ids=[config.yaw_motor_id, config.pitch_motor_id]),
    )
    window = "YOLO turret aiming demo (Q/Esc: quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame read failed")
                break

            inference_frame = _inference_frame(frame, args.zed_uvc)

            result = model.predict(
                inference_frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            target = select_best_box(result, args.conf)
            command = turret.update(target)
            display = result.plot()

            if target is None:
                status = "NO TARGET"
            else:
                status = (
                    f"err yaw={command.yaw_error_rad:+.3f} rad "
                    f"pitch={command.pitch_error_rad:+.3f} rad "
                    f"stable={command.stable} fire={command.fire_allowed}"
                )
                cv2.drawMarker(
                    display,
                    (round(target.u_px), round(target.v_px)),
                    (0, 255, 0),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )

            cv2.putText(
                display,
                status,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        turret.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
