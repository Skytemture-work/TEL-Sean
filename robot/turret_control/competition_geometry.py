"""CAD-based installation geometry for the real competition robot.

Distances are design values in metres. The ZED left optical centre is assumed
to be at the same height and depth as the CAD camera-centre marker, 60 mm to
its left. The CAD home drawing uses 45 degrees; the physical home motor
position is configured separately and must not be changed by this model.
"""

from dataclasses import replace
import math

from .config import TurretConfig


def with_competition_geometry(config: TurretConfig) -> TurretConfig:
    camera_pitch = math.radians(20.0)
    # From the left optical centre to the yaw axis in level vehicle axes:
    # +x right, +y up, +z forward. Convert y/z to the tilted camera frame.
    axis_y_level = -0.001998
    axis_z_level = -0.101662
    axis_y_camera = math.cos(camera_pitch) * axis_y_level - math.sin(camera_pitch) * axis_z_level
    axis_z_camera = math.sin(camera_pitch) * axis_y_level + math.cos(camera_pitch) * axis_z_level

    # CAD home drawing: exit is 150 mm forward and 96.5 mm up from the
    # pitch axis at the nominal 45 degree design pose. Rotate back to level.
    design_home = math.radians(45.0)
    exit_y_level = math.cos(design_home) * 0.0965 - math.sin(design_home) * 0.150
    exit_z_level = math.sin(design_home) * 0.0965 + math.cos(design_home) * 0.150
    return replace(
        config,
        camera_on_turret=False,
        use_axis_geometry=True,
        camera_to_yaw_yaw_rad=0.0,
        camera_to_yaw_pitch_rad=camera_pitch,
        yaw_axis_offset_x_m=0.060,
        yaw_axis_offset_y_m=axis_y_camera,
        yaw_axis_offset_z_m=axis_z_camera,
        pitch_axis_offset_x_m=0.0,
        pitch_axis_offset_y_m=0.0721,
        pitch_axis_offset_z_m=0.0,
        pitch_exit_offset_y_m=exit_y_level,
        pitch_exit_offset_z_m=exit_z_level,
        compensate_parallax=True,
        compensate_gravity=False,
        require_distance_to_fire=True,
    )
