import json
from html.parser import HTMLParser
import math
import re
from types import SimpleNamespace
import unittest
from urllib.request import urlopen

from turret_control.dashboard import (
    DASHBOARD_HTML_PATH,
    DashboardState,
    build_dashboard_snapshot,
    start_dashboard_server,
)


class _IdParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, _tag, attrs):
        for name, value in attrs:
            if name == "id":
                self.ids.append(value)


def _target(hole_id, engaged=False, distance_m=5.0):
    return SimpleNamespace(
        hole_id=hole_id,
        engaged=engaged,
        distance_m=distance_m,
        confidence=0.93,
        u_px=640.0,
        v_px=320.0,
        diameter_m=0.40,
    )


class DashboardTelemetryTests(unittest.TestCase):
    def test_snapshot_tracks_shots_and_physical_axis_delta(self):
        plan = SimpleNamespace(
            index=2,
            targets=[
                _target("H05", engaged=True),
                _target("H03"),
                _target("H05"),
                _target("H07", distance_m=5.3),
            ],
        )
        board_plan = SimpleNamespace(
            index=0,
            targets=[_target(f"H{number:02d}", distance_m=4.5 + number / 10.0) for number in range(1, 13)],
        )
        command = SimpleNamespace(
            target_yaw_rad=-0.20,
            target_pitch_rad=-0.30,
            yaw_servo_error_rad=-0.05,
            pitch_servo_error_rad=-0.10,
            yaw_axis_distance_m=5.1,
            pitch_axis_distance_m=5.0,
            yaw_error_rad=0.20,
            pitch_error_rad=0.30,
            blocked_reason=None,
            stable=True,
            fire_allowed=True,
            target_reachable=True,
        )
        status = SimpleNamespace(
            command=command,
            target=plan.targets[2],
            fired=False,
            skipped=False,
        )
        config = SimpleNamespace(
            yaw_zero_rad=0.0,
            pitch_zero_rad=0.0,
            yaw_sign=-1,
            pitch_sign=-1,
        )
        pose = SimpleNamespace(
            correspondence_count=9,
            reprojection_error_px=1.2,
            translation_speed_m_s=0.01,
            rotation_speed_rad_s=0.02,
        )

        snapshot = build_dashboard_snapshot(
            plan=plan,
            board_plan=board_plan,
            engagement_status=status,
            config=config,
            pose=pose,
            pose_ready=True,
            virtual_board=False,
            frame_size_ok=True,
            raw_detection_count=13,
            board_detection_count=11,
            board_surface_found=True,
            score_mode=True,
            match_phase="upper",
            score_state=SimpleNamespace(
                phase="active",
                active_hole_id="H11",
                remaining_s=42.5,
                scores={"H10": 0.1, "H11": 0.8, "H12": 0.05},
            ),
            bonus_hole_id="H11",
            operator_fire_pending=False,
            depth_plane=SimpleNamespace(inlier_count=200, sample_count=250, median_error_m=.013),
        )

        self.assertEqual(snapshot["target"]["hole_id"], "H05")
        self.assertEqual(snapshot["vision"]["depth_plane_inliers"], 200)
        self.assertEqual(snapshot["vision"]["depth_plane_samples"], 250)
        self.assertEqual(snapshot["vision"]["depth_plane_median_error_m"], .013)
        self.assertEqual(snapshot["plan"]["fired_shots"], 1)
        self.assertEqual(snapshot["plan"]["skipped_shots"], 1)
        self.assertEqual(snapshot["plan"]["pending_shots"], 2)
        h05 = next(
            item for item in snapshot["plan"]["holes"] if item["hole_id"] == "H05"
        )
        self.assertEqual(h05["state"], "current")
        self.assertEqual(h05["fired_shots"], 1)
        self.assertEqual(h05["pending_shots"], 1)
        self.assertEqual(snapshot["target"]["yaw"]["direction"], "向右")
        self.assertAlmostEqual(
            snapshot["target"]["yaw"]["delta_deg"],
            math.degrees(0.05),
            places=3,
        )
        self.assertEqual(snapshot["target"]["pitch"]["direction"], "抬高")
        self.assertTrue(snapshot["firing"]["allowed"])

    def test_dashboard_script_only_references_existing_static_ids(self):
        page = DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
        parser = _IdParser()
        parser.feed(page)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))

        references = set(
            re.findall(r'(?:text|\$)\("([A-Za-z][A-Za-z0-9]*)"', page)
        )
        self.assertEqual(sorted(references - set(parser.ids)), [])

    def test_http_server_exposes_page_health_and_json(self):
        state = DashboardState()
        state.update_telemetry(
            {"generated_at_ms": 123, "system": {"online": True}}
        )
        server = start_dashboard_server(state, 0, host="127.0.0.1")
        port = server.server_address[1]
        try:
            with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                page = response.read().decode("utf-8")
            with urlopen(
                f"http://127.0.0.1:{port}/api/status", timeout=2
            ) as response:
                status = json.loads(response.read().decode("utf-8"))
            with urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                health = response.read()
        finally:
            state.stop()
            server.shutdown()
            server.server_close()

        self.assertIn("Vision Fire Control", page)
        self.assertTrue(status["system"]["online"])
        self.assertEqual(health, b"ok")


if __name__ == "__main__":
    unittest.main()
