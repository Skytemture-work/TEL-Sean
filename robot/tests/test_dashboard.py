import json
from html.parser import HTMLParser
import math
import re
from types import SimpleNamespace
import unittest
import time
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen

from turret_control.board import BoardTarget, EngagementPlan
from turret_control.dashboard import (
    DASHBOARD_HTML_PATH,
    DashboardState,
    _plan_telemetry,
    build_dashboard_snapshot,
    start_dashboard_server,
)
from turret_control.run_board_real import _apply_manual_aim_action


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
    def test_aim_only_navigation_does_not_record_a_shot_or_repeat_a_stale_request(self):
        plan = EngagementPlan([BoardTarget(100, 100, 20, 20, hole_id="H01"),
                               BoardTarget(150, 100, 20, 20, hole_id="H02")])
        confirmed = []
        plan, applied = _apply_manual_aim_action(plan, ("confirm", 0, "H01"), confirmed)
        self.assertTrue(applied)
        self.assertEqual(plan.current().hole_id, "H02")
        self.assertEqual(confirmed, ["H01"])
        self.assertFalse(plan.targets[0].engaged)
        telemetry = _plan_telemetry(plan, plan, aim_only=True, confirmed_holes=confirmed)
        self.assertEqual(telemetry["confirmed_shots"], 1)
        self.assertEqual(telemetry["fired_shots"], 0)
        self.assertEqual(telemetry["sequence"][0]["state"], "confirmed")
        plan, applied = _apply_manual_aim_action(plan, ("confirm", 0, "H01"), confirmed)
        self.assertFalse(applied)
        self.assertEqual(plan.current().hole_id, "H02")
        plan, applied = _apply_manual_aim_action(plan, ("reset", 1, "H02"), confirmed)
        self.assertTrue(applied)
        self.assertEqual(plan.current().hole_id, "H01")
        self.assertEqual(confirmed, [])

    def test_manual_http_requests_are_mode_gated_and_queued_with_target_identity(self):
        state = DashboardState()
        state.update_telemetry({"generated_at_ms": int(time.time() * 1000),
                                "manual_control": {"enabled": True, "plan_index": 0,
                                                   "current_hole_id": "H01"}})
        server = start_dashboard_server(state, 0, host="127.0.0.1")
        url = f"http://127.0.0.1:{server.server_address[1]}/api/confirm"

        def post(index, hole, origin=None):
            headers = {"Content-Type": "application/json"}
            if origin:
                headers["Origin"] = origin
            request = Request(url, data=json.dumps({"plan_index": index, "hole_id": hole}).encode(),
                              headers=headers, method="POST")
            return urlopen(request, timeout=2)

        try:
            with self.assertRaises(HTTPError) as blocked:
                post(0, "H01")
            self.assertEqual(blocked.exception.code, 409)
            state.enable_manual_aim()
            with self.assertRaises(HTTPError) as cross_origin:
                post(0, "H01", origin="http://other.example")
            self.assertEqual(cross_origin.exception.code, 403)
            with self.assertRaises(HTTPError) as stale:
                post(0, "H02")
            self.assertEqual(stale.exception.code, 409)
            with post(0, "H01") as response:
                self.assertTrue(json.load(response)["ok"])
            with self.assertRaises(HTTPError) as duplicate:
                post(0, "H01")
            self.assertEqual(duplicate.exception.code, 409)
            self.assertEqual(state.take_manual_action(), ("confirm", 0, "H01"))
        finally:
            state.stop()
            server.shutdown()
            server.server_close()

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

        self.assertIn("Vision Detection Preview", page)
        self.assertTrue(status["system"]["online"])
        self.assertEqual(health, b"ok")


if __name__ == "__main__":
    unittest.main()
