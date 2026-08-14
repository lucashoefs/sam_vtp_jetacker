#!/usr/bin/env python3
"""GeNIE BEV path planner as a ROS 2 node.

The planner package (genie_path_planner) is used UNMODIFIED. Two things are
done around it, both identical for every perception model, so the comparison
stays clean:

  1. PATH BANK PRECOMPUTED AT STARTUP. sample_paths_polynomial() takes 85% of
     plan_on_bev()'s runtime (4.5 s of 5.3 s here) and does not depend on the
     BEV at all. It is deterministic given random_seed, so precomputing yields
     a bit-identical bank - a pure caching win, not a behavioural change.

  2. CURVATURE FILTER. The JetAcker is Ackermann with a ~0.30 m minimum turning
     radius, so paths above ~3.3 1/m curvature are physically undrivable.
     Filtering them from the BANK (once, at startup) rather than after costing
     means both arms always choose from the same executable set.

FRAME CONVENTIONS - the planner and ROS disagree, and this is easy to get
silently wrong:

    planner goal/path :  x = lateral RIGHT,  y = FORWARD
    ROS base_link     :  x = FORWARD,        y = LEFT

    planner_x = -ros_y          ros_x =  planner_y
    planner_y =  ros_x          ros_y = -planner_x

camera_pose is T_world_camera for the OPTICAL frame (x right, y down, z fwd)
into a world frame with z up - which is exactly what TF gives for
odom -> d455_color_optical_frame.

ground_z is 0.0 because base_link sits on the ground below the rear axle.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformListener

from genie_path_planner import (
    PlannerConfig,
    blend_modalities,
    depth_to_bev_height_and_traversability,
    plan_on_bev,
    project_score_to_bev,
)
from genie_path_planner.path_sampling import sample_paths_polynomial

import genie_fast


def quat_to_matrix(x, y, z, w) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def path_max_angle_deg(path_px: np.ndarray, start_rc: tuple[int, int],
                       min_range_px: float = 8.0) -> float:
    """Largest bearing from straight-ahead reached by a path, in degrees.

    A forward-facing camera only observes a wedge, so a path that leaves it is
    a path into terrain the robot has no evidence about. That is a structural
    property of the sensor, not a cost trade-off, so it belongs in the bank
    filter next to curvature rather than in unknown_cost - which cannot
    distinguish "unseen because it is behind me" from "unseen because it is
    occluded".

    Points within min_range_px of the origin are ignored: their bearing is
    dominated by pixel quantisation.
    """
    p = np.asarray(path_px, dtype=np.float64)
    dr = float(start_rc[0]) - p[:, 0]        # rows increase downward
    dc = p[:, 1] - float(start_rc[1])
    rng = np.hypot(dr, dc)
    far = rng >= min_range_px
    if not np.any(far):
        return 0.0
    return float(np.max(np.abs(np.degrees(np.arctan2(dc[far], dr[far])))))


def path_curvature(path_px: np.ndarray, res_m: float) -> float:
    """Max |curvature| along a pixel-space path, in 1/m.

    Menger curvature over consecutive triples: 4A / (|ab||bc||ca|).
    """
    if len(path_px) < 3:
        return 0.0
    p = np.asarray(path_px, dtype=np.float64) * res_m
    a, b, c = p[:-2], p[1:-1], p[2:]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    denom = ab * bc * ca
    ok = denom > 1e-9
    if not np.any(ok):
        return 0.0
    return float(np.max(4.0 * area[ok] / denom[ok]))


class GeniePlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("genie_planner")

        self.declare_parameter("config", "")
        self.declare_parameter("score_topic", "/sam_vtp/probability")
        self.declare_parameter("depth_topic",
                               "/camera/d455/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/d455/color/camera_info")
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        # 49 ms/plan measured on x86 after genie_fast; expect slower on the
        # Orin's A78 cores. Measure /genie_planner/plan_ms and lower this if
        # the loop cannot keep up - the controller runs at 20 Hz off the
        # latest path regardless, so replanning need not be fast.
        self.declare_parameter("plan_rate", 5.0)
        self.declare_parameter("fast_costs", True)
        self.declare_parameter("verify_fast_costs", True)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("ground_z", 0.0)
        self.declare_parameter("min_turn_radius_m", 0.30)
        # Half of the camera's horizontal FOV, less a margin. 0 disables.
        # The D455f is 89 deg wide, so 38 keeps paths comfortably inside what
        # the camera can actually see.
        self.declare_parameter("max_path_angle_deg", 38.0)
        self.declare_parameter("goal_distance_m", 3.0)
        # forward   : goal is straight ahead in BASE_LINK, recomputed from the
        #             current heading each plan. There is no memory of the
        #             original bearing, so every obstacle avoidance
        #             permanently redefines "straight" and heading random-
        #             walks without limit.
        # heading   : goal is 3 m along a FIXED bearing held in odom, but
        #             always projected from the robot's current position.
        #             Uses only yaw (drifts ~0.025 deg/s here), never
        #             position (which dead reckoning gets badly wrong). After
        #             avoiding something the robot returns to its original
        #             bearing - a restoring force `forward` does not have.
        # waypoints : absolute goals in odom, needs position accuracy.
        self.declare_parameter("goal_mode", "heading")
        # Clamp so the goal always keeps a forward component; this chassis
        # cannot pivot, so a goal behind it is unreachable.
        self.declare_parameter("max_heading_error_deg", 80.0)
        self.declare_parameter("publish_bev", True)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.world_frame = str(g("world_frame"))
        self.base_frame = str(g("base_frame"))
        self.depth_scale = float(g("depth_scale"))
        self.ground_z = float(g("ground_z"))
        self.goal_distance = float(g("goal_distance_m"))
        self.goal_mode = str(g("goal_mode"))
        self.max_heading_err = math.radians(float(g("max_heading_error_deg")))
        self.publish_bev = bool(g("publish_bev"))

        # Integral-image acceleration for the cost evaluation: ~8x, with
        # outputs verified identical against upstream at startup. An
        # implementation change only - the planner's decisions are unchanged.
        if bool(g("fast_costs")):
            genie_fast.install(check=bool(g("verify_fast_costs")))

        cfg_path = str(g("config"))
        raw = yaml.safe_load(open(cfg_path)) if cfg_path else {}
        proj = raw.get("projection", {})
        dep = raw.get("depth", {})
        plan = raw.get("planning", {})

        self.bev_res = float(proj.get("bev_resolution_m_per_px", 0.03))
        self.bev_fwd = float(proj.get("bev_forward_range_m", 4.0))
        self.bev_side = float(proj.get("bev_side_range_m", 4.0))
        self.max_ray = float(proj.get("max_ray_distance_m", 8.0))
        self.reliable_depth = float(dep.get("reliable_depth_m", 2.4))
        self.min_depth = float(dep.get("min_depth_m", 0.2))
        self.obstacle_h = float(dep.get("obstacle_height_thresh_m", 0.20))
        self.rgb_w = float(plan.get("rgb_weight", 0.2))
        self.depth_w = float(plan.get("depth_weight", 0.8))
        self.mode = str(plan.get("mode", "rgbd"))

        pcfg = raw.get("planner", {})
        self.pcfg = PlannerConfig(**{
            k: v for k, v in pcfg.items()
            if k in PlannerConfig.__dataclass_fields__
        }) if pcfg else PlannerConfig()

        # ---- precompute and curvature-filter the path bank ---------------
        t0 = time.perf_counter()
        bank = sample_paths_polynomial(
            robot=(self.pcfg.grid_size - 1, self.pcfg.grid_size // 2),
            num_goals=self.pcfg.num_goals,
            num_mid_points_per_goal=self.pcfg.num_mid_points_per_goal,
            num_samples=self.pcfg.path_num_samples,
            grid_size=self.pcfg.grid_size,
            include_random_goals=self.pcfg.include_random_goals,
            random_seed=self.pcfg.random_seed,
        )
        start_rc = (self.pcfg.grid_size - 1, self.pcfg.grid_size // 2)
        kappa_max = 1.0 / float(g("min_turn_radius_m"))
        max_angle = float(g("max_path_angle_deg"))

        after_curv = [p for p in bank
                      if path_curvature(p, self.bev_res) <= kappa_max]
        if max_angle > 0:
            self.bank = [p for p in after_curv
                         if path_max_angle_deg(p, start_rc) <= max_angle]
        else:
            self.bank = after_curv

        self.get_logger().info(
            f"path bank: {len(bank)} sampled in {(time.perf_counter()-t0):.1f}s "
            f"-> {len(after_curv)} drivable (R >= {g('min_turn_radius_m')} m) "
            f"-> {len(self.bank)} within +-{max_angle:.0f} deg of the camera FOV"
        )
        if not self.bank:
            raise RuntimeError(
                "bank filters removed every path; loosen min_turn_radius_m "
                "or max_path_angle_deg")
        if len(self.bank) < 20:
            self.get_logger().warn(
                f"only {len(self.bank)} candidate paths survive; the planner "
                f"has little to choose between, which will blunt any "
                f"difference between perception models")

        # ---- state -------------------------------------------------------
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        # See the note in main(): TF lookups block, so this node needs a
        # MultiThreadedExecutor and a reentrant group, not spin_thread.
        self.cb_group = ReentrantCallbackGroup()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.lock = threading.Lock()
        self.latest = None          # (score, depth, header)
        self.camera_k: Optional[np.ndarray] = None
        self.goal_world: Optional[np.ndarray] = None
        self.ref_yaw: Optional[float] = None
        self.n_plans = 0
        self.no_path = 0
        self.last_heading: Optional[float] = None
        self.heading_changes = 0

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CameraInfo, str(g("info_topic")),
                                 self._on_info, 1)
        self.create_subscription(Image, str(g("score_topic")),
                                 self._on_score, sensor_qos)
        self.create_subscription(Image, str(g("depth_topic")),
                                 self._on_depth, sensor_qos)
        self.create_subscription(PoseStamped, "goal_pose", self._on_goal, 1, callback_group=self.cb_group)

        self.pub_path = self.create_publisher(Path, "~/path", 1)
        self.pub_bev = self.create_publisher(Image, "~/bev", 1)
        self.pub_latency = self.create_publisher(Float32, "~/plan_ms", 10)
        self.pub_switch = self.create_publisher(Float32, "~/heading_change_deg", 10)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        # Distinct signal for the bag: planning failures per run is itself a
        # closed-loop metric, and one the operator needs to see live.
        self.pub_no_path = self.create_publisher(Bool, "~/no_path", 1)

        # The overlay costs ~16 ms/plan and is display-only; it is enabled
        # lazily, the first time something subscribes to ~/bev.
        genie_fast.set_visualization(False)
        self._vis_on = False

        self._depth: Optional[np.ndarray] = None
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.running = True
        self.plan_period = 1.0 / float(g("plan_rate"))
        self.worker.start()
        self.create_timer(1.0, self._health, callback_group=self.cb_group)

    # ------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        if self.camera_k is None:
            self.camera_k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"intrinsics received ({msg.width}x{msg.height})")

    def _on_depth(self, msg: Image) -> None:
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self._depth = d.astype(np.float32) * self.depth_scale

    def _on_score(self, msg: Image) -> None:
        s = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        with self.lock:
            self.latest = (s.astype(np.float32) / 255.0, self._depth, msg.header)

    def _on_reset_heading(self, msg: Bool) -> None:
        """Re-anchor the bearing to wherever the robot is now pointing."""
        if msg.data:
            self.ref_yaw = None
            self.get_logger().info("heading reference cleared; will re-capture")

    def _on_goal(self, msg: PoseStamped) -> None:
        goal = np.array([msg.pose.position.x, msg.pose.position.y])
        # The waypoint follower republishes at 5 Hz; only log real changes.
        if self.goal_world is None or not np.allclose(goal, self.goal_world):
            self.get_logger().info(f"goal set: {goal}")
        self.goal_world = goal

    # ------------------------------------------------------------------
    def _camera_pose(self, header) -> Optional[np.ndarray]:
        """T_world_camera (4x4) for the optical frame at the image stamp."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, header.frame_id, header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.0))
        except Exception:
            # Non-blocking, with a fallback to the newest transform. Blocking
            # here would deadlock against the /tf subscription it waits on.
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.world_frame, header.frame_id, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.0))
            except Exception:
                return None
        t, r = tf.transform.translation, tf.transform.rotation
        T = np.eye(4)
        T[:3, :3] = quat_to_matrix(r.x, r.y, r.z, r.w)
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def _base_yaw(self) -> Optional[float]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.base_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.0))
        except Exception:
            return None
        r = tf.transform.rotation
        return math.atan2(2.0 * (r.w * r.z + r.x * r.y),
                          1.0 - 2.0 * (r.y * r.y + r.z * r.z))

    def _goal_in_planner_frame(self) -> tuple[float, float]:
        """Goal as (x_right, y_forward) metres in base_link."""
        d = self.goal_distance

        if self.goal_mode == "waypoints" and self.goal_world is not None:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, self.world_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.0))
            except Exception:
                return 0.0, d
            t, r = tf.transform.translation, tf.transform.rotation
            R = quat_to_matrix(r.x, r.y, r.z, r.w)
            p = R @ np.array([self.goal_world[0], self.goal_world[1], 0.0]) \
                + np.array([t.x, t.y, t.z])
            return float(-p[1]), float(p[0])

        if self.goal_mode == "heading":
            yaw = self._base_yaw()
            if yaw is None:
                return 0.0, d
            if self.ref_yaw is None:
                self.ref_yaw = yaw
                self.get_logger().info(
                    f"heading reference captured: {math.degrees(yaw):+.1f} deg")
            # Bearing error, wrapped to (-pi, pi]. Positive means the robot has
            # turned LEFT of the reference, so the goal must sit to its right.
            err = math.atan2(math.sin(self.ref_yaw - yaw),
                             math.cos(self.ref_yaw - yaw))
            err = max(-self.max_heading_err, min(self.max_heading_err, err))
            return float(-d * math.sin(err)), float(d * math.cos(err))

        # forward: always straight ahead of the current heading
        return 0.0, d

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            t_start = time.perf_counter()
            try:
                self._plan_once()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"planning failed: {exc}")
            sleep = self.plan_period - (time.perf_counter() - t_start)
            if sleep > 0:
                time.sleep(sleep)

    def _plan_once(self) -> None:
        with self.lock:
            if self.latest is None or self.camera_k is None:
                return
            score, depth, header = self.latest

        pose = self._camera_pose(header)
        if pose is None:
            self.get_logger().warn("no camera pose; is odometry running?",
                                   throttle_duration_sec=5.0)
            return

        t0 = time.perf_counter()
        rgb_bev, rgb_obs, _ = project_score_to_bev(
            score, self.camera_k, pose, self.ground_z,
            self.bev_res, self.bev_fwd, self.bev_side, self.max_ray)

        if self.mode == "rgbd" and depth is not None:
            _, d_bev, d_obs, _ = depth_to_bev_height_and_traversability(
                depth, self.camera_k, pose, self.ground_z,
                self.reliable_depth, self.min_depth, self.obstacle_h,
                self.bev_res, self.bev_fwd, self.bev_side)
            bev, obs = blend_modalities(rgb_bev, rgb_obs, d_bev, d_obs,
                                        self.rgb_w, self.depth_w)
        else:
            bev, obs = rgb_bev, rgb_obs

        gx, gy = self._goal_in_planner_frame()
        result = plan_on_bev(bev, obs, gx, gy, self.bev_res, self.pcfg,
                             candidate_path_bank=self.bank)
        elapsed = (time.perf_counter() - t0) * 1000.0

        xy = np.asarray(result.final_path_xy_m)      # (x_right, y_forward)
        if xy.size == 0:
            self.no_path += 1
            self.pub_no_path.publish(Bool(data=True))
            # Every candidate was rejected by the cost filter. On an Ackermann
            # chassis this is terminal without operator help: it cannot turn on
            # the spot to look elsewhere, and the planner has no reverse. Say
            # WHY, so a stuck run is distinguishable in the log from a bug.
            frac_free = float(np.mean(bev > 0.5)) if bev.size else 0.0
            frac_obs = float(np.mean(obs)) if obs.size else 0.0
            behind = " GOAL IS BEHIND THE ROBOT" if gy < 0 else ""
            self.get_logger().warn(
                f"no path ({self.no_path} consecutive): goal=({gx:+.2f} right, "
                f"{gy:+.2f} fwd){behind}; BEV {100*frac_free:.0f}% traversable, "
                f"{100*frac_obs:.0f}% observed; all {len(self.bank)} candidates "
                f"rejected at threshold_cost={self.pcfg.threshold_cost}",
                throttle_duration_sec=3.0)
            return
        self.no_path = 0
        self.pub_no_path.publish(Bool(data=False))

        msg = Path()
        msg.header.stamp = header.stamp     # source frame stamp, for analysis
        msg.header.frame_id = self.base_frame
        for x_r, y_f in xy:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = float(y_f)     # planner y_fwd -> ROS x_fwd
            p.pose.position.y = float(-x_r)    # planner x_right -> ROS y_left
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.pub_path.publish(msg)

        # Path-switching magnitude: heading of the path at ~1 m ahead. This is
        # the mechanistic link between costmap instability and control, and is
        # one of the metrics the closed-loop comparison reports.
        idx = min(len(xy) - 1, max(1, int(1.0 / max(self.bev_res, 1e-6) / 10)))
        heading = math.degrees(math.atan2(-xy[idx][0], xy[idx][1]))
        if self.last_heading is not None:
            delta = abs(heading - self.last_heading)
            self.pub_switch.publish(Float32(data=float(delta)))
            if delta > 10.0:
                self.heading_changes += 1
        self.last_heading = heading

        self.pub_latency.publish(Float32(data=float(elapsed)))
        self.n_plans += 1

        if self.publish_bev and self.pub_bev.get_subscription_count() > 0:
            vis = np.asarray(result.visualization)
            if vis.ndim == 3 and vis.shape[2] == 3:
                m = self.bridge.cv2_to_imgmsg(vis.astype(np.uint8), encoding="rgb8")
                m.header = msg.header
                self.pub_bev.publish(m)

    # ------------------------------------------------------------------
    def _health(self) -> None:
        want_vis = self.publish_bev and self.pub_bev.get_subscription_count() > 0
        if want_vis != self._vis_on:
            genie_fast.set_visualization(want_vis)
            self._vis_on = want_vis

        self.pub_status.publish(String(data=(
            f"plans={self.n_plans} bank={len(self.bank)} "
            f"heading_changes={self.heading_changes} "
            f"no_path={self.no_path} "
            f"mode={self.goal_mode} "
            f"ref_yaw={math.degrees(self.ref_yaw):+.0f}deg"
            if self.ref_yaw is not None else f"mode={self.goal_mode}")))

    def shutdown(self) -> None:
        self.running = False
        self.worker.join(timeout=2.0)


def main() -> None:
    rclpy.init()
    node = GeniePlannerNode()
    # 4 threads: image callback, TF, timers, and headroom. With the default
    # single-threaded executor a blocking TF lookup starves the very /tf
    # subscription it is waiting on.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
