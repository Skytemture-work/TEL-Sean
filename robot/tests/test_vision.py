"""Run the robot's VisionTracker without starting the main controller."""

import argparse
import math
import sys
import time
from pathlib import Path


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("expected a positive number of seconds")
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=positive_seconds,
                        help="exit after this many seconds (default: Ctrl+C)")
    parser.add_argument("--interval", type=positive_seconds, default=0.5,
                        help="seconds between status readings (default: 0.5)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if (script_dir.parent / "vision_target.py").is_file():
        if not (script_dir / "vision_target.py").is_file():
            sys.path.insert(0, str(script_dir.parent))

    from vision_target import VisionTracker

    vision = VisionTracker()
    try:
        vision.start()
        vision.set_active(True)
        print("[Vision Test] Waiting for tracker output. Ctrl+C to stop.", flush=True)
        started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            if args.duration is not None and elapsed >= args.duration:
                break

            (vx, vy, wz), (yaw, pitch), detected = vision.get_vision_data()
            target = "DETECTED" if detected else "NO TARGET"
            print(
                f"[Vision Test {elapsed:6.1f}s] {target} | "
                f"vx={vx} vy={vy} wz={wz} | yaw={yaw} pitch={pitch}",
                flush=True,
            )

            delay = args.interval
            if args.duration is not None:
                remaining = args.duration - (time.monotonic() - started)
                delay = min(delay, max(0, remaining))
            time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        vision.set_active(False)

    print("[Vision Test] Finished.", flush=True)


if __name__ == "__main__":
    main()
