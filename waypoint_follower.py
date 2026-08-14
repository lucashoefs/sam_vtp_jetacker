#!/usr/bin/env python3
"""Waypoint follower: sequences route goals for the closed-loop trials.

Waypoints are fixed offsets in ODOM, and odom is reset (by restarting
odometry) at the start of every trial, so every run of a route sees exactly
the same goal sequence. Odometry drift then degrades both arms equally and
cancels in the paired comparison, rather than becoming a confound - which it
would be if goals were re-specified per run.

Publishes goal_pose for the planner and reports progress for the trial log.

    python3 waypoint_follower.py --ros-args -p route:=/home/jetson/jetacker/routes/route_a.yaml

Route file:

    name: route_a
    description: general terrain, vegetation and rocks
    waypoints:
      - [2.0, 0.0]      # x forward, y left, metres, in odom
      - [4.0, 1.5]
      - [6.0, 1.5]
"""

from __future__ import annotations

import math
import time
from typing import List

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String
from tf2_ros import Buffer, TransformListener


class WaypointFollower(Node):
    def __init__(self) -> None:
        super().__init__("waypoint_follower")

        self.declare_parameter("route", "")
        self.declare_parameter("waypoints", [2.0, 0.0])   # flat x,y fallback
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tolerance", 0.5)
        self.declare_parameter("rate", 5.0)
        self.declare_parameter("timeout_per_leg", 120.0)
        self.declare_parameter("autostart", False)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.world_frame = str(g("world_frame"))
        self.base_frame = str(g("base_frame"))
        self.tol = float(g("tolerance"))
        self.leg_timeout = float(g("timeout_per_leg"))

        route_path = str(g("route"))
        if route_path:
            raw = yaml.safe_load(open(route_path))
            self.route_name = str(raw.get("name", "unnamed"))
            self.waypoints: List[List[float]] = [
                [float(p[0]), float(p[1])] for p in raw["waypoints"]]
        else:
            flat = list(g("waypoints"))
            self.route_name = "inline"
            self.waypoints = [[float(flat[i]), float(flat[i + 1])]
                              for i in range(0, len(flat) - 1, 2)]

        if not self.waypoints:
            raise RuntimeError("no waypoints")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.idx = 0
        self.active = bool(g("autostart"))
        self.done = False
        self.t_start = time.monotonic()
        self.t_leg = time.monotonic()
        self.path_len = 0.0
        self._last_xy = None

        self.create_subscription(Bool, "route_start", self._on_start, 1)
        self.create_subscription(Bool, "route_reset", self._on_reset, 1)

        self.pub_goal = self.create_publisher(PoseStamped, "goal_pose", 1)
        self.pub_idx = self.create_publisher(Int32, "~/waypoint_index", 1)
        self.pub_dist = self.create_publisher(Float32, "~/distance_to_goal", 10)
        self.pub_travelled = self.create_publisher(Float32, "~/path_length", 10)
        self.pub_done = self.create_publisher(Bool, "~/route_complete", 1)
        self.pub_status = self.create_publisher(String, "~/status", 10)

        self.create_timer(1.0 / float(g("rate")), self._tick)

        total = sum(math.dist(self.waypoints[i], self.waypoints[i + 1])
                    for i in range(len(self.waypoints) - 1))
        total += math.hypot(*self.waypoints[0])
        self.straight_line = total
        self.get_logger().info(
            f"route '{self.route_name}': {len(self.waypoints)} waypoints, "
            f"{total:.1f} m of straight-line legs, tolerance {self.tol} m")
        if not self.active:
            self.get_logger().warn("waiting for route_start")

    # ------------------------------------------------------------------
    def _on_start(self, msg: Bool) -> None:
        if msg.data and not self.active:
            self.active = True
            self.done = False
            self.t_start = self.t_leg = time.monotonic()
            self.get_logger().info(f"route '{self.route_name}' started")

    def _on_reset(self, msg: Bool) -> None:
        if msg.data:
            self.idx = 0
            self.active = False
            self.done = False
            self.path_len = 0.0
            self._last_xy = None
            self.get_logger().info("route reset")

    # ------------------------------------------------------------------
    def _pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.base_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.0))
        except Exception:
            return None
        return tf.transform.translation.x, tf.transform.translation.y

    def _publish_goal(self) -> None:
        wx, wy = self.waypoints[self.idx]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.pose.position.x = wx
        msg.pose.position.y = wy
        msg.pose.orientation.w = 1.0
        self.pub_goal.publish(msg)
        self.pub_idx.publish(Int32(data=int(self.idx)))

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        pose = self._pose()
        if pose is None:
            return
        rx, ry = pose

        # Realised path length, for the path-ratio metric.
        if self._last_xy is not None and self.active and not self.done:
            self.path_len += math.dist((rx, ry), self._last_xy)
        self._last_xy = (rx, ry)
        self.pub_travelled.publish(Float32(data=float(self.path_len)))

        if self.done or not self.active:
            self.pub_done.publish(Bool(data=self.done))
            return

        wx, wy = self.waypoints[self.idx]
        dist = math.dist((rx, ry), (wx, wy))
        self.pub_dist.publish(Float32(data=float(dist)))

        if dist < self.tol:
            self.idx += 1
            self.t_leg = time.monotonic()
            if self.idx >= len(self.waypoints):
                self.done = True
                self.active = False
                elapsed = time.monotonic() - self.t_start
                ratio = self.path_len / max(self.straight_line, 1e-6)
                self.get_logger().info(
                    f"ROUTE COMPLETE in {elapsed:.1f}s, "
                    f"travelled {self.path_len:.1f} m "
                    f"(ratio {ratio:.2f} of {self.straight_line:.1f} m)")
                self.pub_done.publish(Bool(data=True))
                return
            self.get_logger().info(
                f"waypoint {self.idx}/{len(self.waypoints)}: "
                f"{self.waypoints[self.idx]}")

        # A leg that overruns means the robot is stuck or lost. Report it
        # rather than letting a trial sit there; the operator decides whether
        # that counts as an intervention.
        if time.monotonic() - self.t_leg > self.leg_timeout:
            self.get_logger().warn(
                f"leg {self.idx} has taken {time.monotonic()-self.t_leg:.0f}s "
                f"(limit {self.leg_timeout:.0f}s) - stuck?",
                throttle_duration_sec=10.0)

        self._publish_goal()
        self.pub_status.publish(String(data=(
            f"route={self.route_name} wp={self.idx}/{len(self.waypoints)} "
            f"dist={dist:.2f} travelled={self.path_len:.1f} "
            f"elapsed={time.monotonic()-self.t_start:.0f}s")))


def main() -> None:
    rclpy.init()
    node = WaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
