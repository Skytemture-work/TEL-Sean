# Turret control

This is the first hardware-independent version of the horizontal/vertical
baseball aiming controller.

## Laptop demo

The default backend is a mock backend; it does not connect to CAN and does not
move a motor:

```bash
python -m turret_control.run_demo --u 720 --v 360
```

The image coordinates assume a 1280x720 image.  Replace the camera focal
lengths in `TurretConfig` with calibrated values before using real images.

To run the current YOLO model with the same safe mock backend:

```bash
python -m turret_control.run_yolo_demo --device cpu
```

For the current side-by-side ZED UVC stream, add `--zed-uvc` so inference is
performed on the left image:

```bash
python -m turret_control.run_yolo_demo --device cpu --zed-uvc
```

Press `q` to exit.  The displayed `yaw` and `pitch` values are angular errors
in radians; they are not yet calibrated to the final mechanical zero.

## AGX CAN backend

The CAN backend is deliberately not enabled by the demo. Before using it on
the robot, verify the exact motor model, CAN IDs, position mode, zero points,
mechanical limits and the team's existing motor enable/feedback code.

Install the optional dependency on AGX:

```bash
python3 -m pip install python-can
```

The documented position-mode frame is sent by:

```python
from turret_control.backends import DamiaoCanBackend

backend = DamiaoCanBackend(channel="can0", bitrate=1_000_000)
backend.set_position(motor_id=1, position_rad=0.0, max_speed_rad_s=1.0)
backend.close()
```

This module does not guess the vendor-specific enable/disable sequence. Add
that sequence after checking the working AGX motor program.
