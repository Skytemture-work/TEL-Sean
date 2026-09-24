"""Run the current YOLO model with the safe mock turret backend.

This is intended for laptop development.  It does not open CAN and does not
move a real motor.

Example:
    python -m turret_control.run_yolo_demo \
        --model jfwiskamckzxc.v2i.yolo26/runs/segment/runs/segment/hole_seg/weights/best.pt
"""

import argparse
from pathlib import Path

import cv2

from .backends import MockMotorBackend
from .config import TurretConfig
from .controller import TurretController
from .detection import select_best_box


DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "jfwiskamckzxc.v2i.yolo26/runs/segment/runs/segment/hole_seg/weights/best.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640)
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
    turret = TurretController(config, MockMotorBackend())
    window = "YOLO turret aiming demo (Q/Esc: quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame read failed")
                break

            inference_frame = frame
            if args.zed_uvc:
                inference_frame = frame[:, : frame.shape[1] // 2].copy()

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
