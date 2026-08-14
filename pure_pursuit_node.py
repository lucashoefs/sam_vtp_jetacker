#!/usr/bin/env python3
"""Pure pursuit path tracker for the JetAcker (Ackermann).

Replaces GeNIE's "align heading, then advance 1 m" controller, which assumes
rotate-in-place. This chassis cannot do that: the firmware zeroes all motors
when linear speed is zero, so steering without forward motion achieves
nothing. Pure pursuit is the standard alternative for car-like platforms.

Arguably it is also the better experiment. Stop-turn-go gives perception a
stationary moment to settle, suppressing exactly the temporal instability
under test; continuous motion is both more realistic and more sensitive to it.

TWO THINGS THAT MATTER MORE THAN THE GEOMETRY

1. The path is stored in ODOM, not base_link. The planner publishes in
   base_link at ~2 Hz, and the control loop runs at 20 Hz - so for up to
   500 ms the robot is following a path anchored to where it used to be. At
   0.25 m/s that is 12 cm of pure tracking error. Transforming once into odom
   on receipt, then back into base_link every tick, removes it.

2. Speed falls with curvature. On a fixed-speed Ackermann the steering angle
   saturates on tight turns and the robot simply understeers off the path.
   Slowing on curvature keeps the demanded radius inside what the linkage can
   actually deliver.

Publishes to cmd_vel_raw, NOT cmd_vel: the deadman sits in between.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformListener


def quat_to_yaw(x, y, z, w) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class PurePursuit(Node):
    def __init__(self) -> None:
        super().__init__("pure_pursuit")

        # ---- chassis (must match jetacker_driver) ------------------------
        self.declare_parameter("wheelbase", 0.216)
        self.declare_parameter("max_steer_deg", 34.0)

        # ---- speed -------------------------------------------------------
        self.declare_parameter("max_speed", 0.25)
        self.declare_parameter("min_speed", 0.10)
        # Lookahead grows with speed: too short oscillates, too long cuts
        # corners. L = max(min_lookahead, gain * v).
        self.declare_parameter("min_lookahead", 0.40)
        self.declare_parameter("lookahead_gain", 1.6)

        # ---- safety ------------------------------------------------------
        self.declare_parameter("path_timeout", 1.5)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("goal_tolerance", 0.10)
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.L = float(g("wheelbase"))
        self.max_steer = math.radians(float(g("max_steer_deg")))
        self.v_max = float(g("max_speed"))
        self.v_min = float(g("min_speed"))
        self.ld_min = float(g("min_lookahead"))
        self.ld_gain = float(g("lookahead_gain"))
        self.path_timeout = float(g("path_timeout"))
        self.goal_tol = float(g("goal_tolerance"))
        self.world_frame = str(g("world_frame"))
        self.base_frame = str(g("base_frame"))

        self.min_radius = self.L / math.tan(self.max_steer)

        self.cb_group = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_odom: Optional[np.ndarray] = None   # (N,2) in world frame
        self.path_time = self.get_clock().now()
        self.enabled = False
        self.at_goal = False
        self.route_done = False
        self.blocked_ticks = 0
        self.reason = "waiting for path"

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Path, "path", self._on_path, qos,
                                 callback_group=self.cb_group)
        self.create_subscription(Bool, "pursuit_enable", self._on_enable, 1,
                                 callback_group=self.cb_group)
        # Without this the robot keeps tracking the last path after the route
        # ends, steering at a goal it has already passed - which looks exactly
        # like aimless wandering.
        self.create_subscription(Bool, "route_complete", self._on_route_done, 1,
                                 callback_group=self.cb_group)

        self.pub_cmd = self.create_publisher(Twist, "cmd_vel_raw", 1)
        self.pub_steer = self.create_publisher(Float32, "~/steer_deg", 10)
        self.pub_xtrack = self.create_publisher(Float32, "~/cross_track_m", 10)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        self.pub_goal = self.create_publisher(Bool, "~/at_goal", 1)

        self.create_timer(1.0 / float(g("control_rate")), self._control,
                          callback_group=self.cb_group)
        self.create_timer(1.0, self._health, callback_group=self.cb_group)

        self.get_logger().info(
            f"pure pursuit: L={self.L} max_steer={math.degrees(self.max_steer):.0f}deg "
            f"R_min={self.min_radius:.2f}m v<={self.v_max}")
        self.get_logger().warn("disabled until pursuit_enable is true")

    # ------------------------------------------------------------------
    def _on_enable(self, msg: Bool) -> None:
        if msg.data != self.enabled:
            self.get_logger().info(f"pursuit {'ENABLED' if msg.data else 'disabled'}")
        self.enabled = bool(msg.data)

    def _on_route_done(self, msg: Bool) -> None:
        if msg.data and not self.route_done:
            self.get_logger().info("route complete; holding position")
        self.route_done = bool(msg.data)

    def _on_path(self, msg: Path) -> None:
        """Store the path in the world frame so it stays valid as we move."""
        if not msg.poses:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, msg.header.frame_id or self.base_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.0))
        except Exception:
            return
        t = tf.transform.translation
        r = tf.transform.rotation
        yaw = quat_to_yaw(r.x, r.y, r.z, r.w)
        c, s = math.cos(yaw), math.sin(yaw)

        pts = np.array([[p.pose.position.x, p.pose.position.y]
                        for p in msg.poses], dtype=np.float64)
        self.path_odom = np.stack([
            t.x + c * pts[:, 0] - s * pts[:, 1],
            t.y + s * pts[:, 0] + c * pts[:, 1],
        ], axis=1)
        self.path_time = self.get_clock().now()

    # ------------------------------------------------------------------
    def _robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.base_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.0))
        except Exception:
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        return t.x, t.y, quat_to_yaw(r.x, r.y, r.z, r.w)

    def _stop(self, reason: str) -> None:
        self.reason = reason
        self.pub_cmd.publish(Twist())

    # ------------------------------------------------------------------
    def _control(self) -> None:
        if not self.enabled:
            self._stop("disabled")
            return
        if self.route_done:
            self._stop("route complete")
            return
        if self.path_odom is None:
            self._stop("no path")
            return

        age = (self.get_clock().now() - self.path_time).nanoseconds * 1e-9
        if age > self.path_timeout:
            # Following a stale path is how a robot drives confidently into
            # something the planner stopped seeing.
            self._stop(f"path {age:.1f}s old")
            return

        pose = self._robot_pose()
        if pose is None:
            self._stop("no pose")
            return
        rx, ry, ryaw = pose

        # Path into base_link coordinates.
        d = self.path_odom - np.array([rx, ry])
        c, s = math.cos(-ryaw), math.sin(-ryaw)
        local = np.stack([c * d[:, 0] - s * d[:, 1],
                          s * d[:, 0] + c * d[:, 1]], axis=1)

        dist_to_end = float(np.hypot(*local[-1]))
        if dist_to_end < self.goal_tol:
            # A LOCAL planner's path end is always a couple of metres ahead
            # and moving, so a short path does not mean "arrived" - it means
            # the planner could not find anywhere to go. Stopping is right;
            # calling it at_goal is not, and it hides a blocked robot behind
            # a success-looking status.
            self.blocked_ticks += 1
            self._stop(f"path only {dist_to_end:.2f} m long - planner blocked")
            if self.blocked_ticks % 40 == 1:
                self.get_logger().warn(
                    f"planner path is {dist_to_end:.2f} m, under the "
                    f"{self.goal_tol} m tolerance: nowhere to drive. Check "
                    f"/arc_planner/mean_reach and the obstacle_cost threshold "
                    f"against what the model actually outputs on open ground.")
            return
        self.blocked_ticks = 0

        rng = np.hypot(local[:, 0], local[:, 1])
        nearest = int(np.argmin(rng))
        self.pub_xtrack.publish(Float32(data=float(rng[nearest])))

        # Provisional speed sets the lookahead, then curvature revises the
        # speed; one pass is enough at these speeds.
        ld = max(self.ld_min, self.ld_gain * self.v_max)

        ahead = np.where((rng >= ld) & (np.arange(len(rng)) >= nearest))[0]
        idx = int(ahead[0]) if len(ahead) else len(local) - 1
        gx, gy = local[idx]

        if gx <= 0.05:
            # Target is beside or behind us and this chassis cannot pivot.
            # Creeping straight lets the planner re-anchor rather than
            # commanding a turn the linkage cannot make.
            self.reason = "target behind; creeping"
            cmd = Twist()
            cmd.linear.x = self.v_min
            self.pub_cmd.publish(cmd)
            return

        # Pure pursuit curvature, then the bicycle model for steering.
        ld_actual = math.hypot(gx, gy)
        kappa = 2.0 * gy / (ld_actual * ld_actual)
        steer = math.atan(self.L * kappa)
        steer = max(-self.max_steer, min(self.max_steer, steer))

        # Slow into curvature: a saturated steering angle at full speed just
        # understeers off the path.
        frac = abs(steer) / self.max_steer
        v = self.v_max - (self.v_max - self.v_min) * frac

        # omega is not free on an Ackermann chassis - it follows from speed
        # and steering angle, and the driver derives the same relation.
        omega = v * math.tan(steer) / self.L

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.pub_cmd.publish(cmd)

        self.pub_steer.publish(Float32(data=float(math.degrees(steer))))
        self.reason = f"tracking idx={idx}/{len(local)}"

    # ------------------------------------------------------------------
    def _health(self) -> None:
        age = (self.get_clock().now() - self.path_time).nanoseconds * 1e-9
        self.pub_status.publish(String(data=(
            f"enabled={self.enabled} blocked={self.blocked_ticks} "
            f"path_age={age:.1f}s {self.reason}")))


def main() -> None:
    rclpy.init()
    node = PurePursuit()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_cmd.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
