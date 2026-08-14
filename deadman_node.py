#!/usr/bin/env python3
"""Deadman: the only thing between a stalled pipeline and a moving robot.

The chassis firmware has NO command timeout. Whatever motor speed and steering
angle were last sent persist indefinitely - if a node dies mid-turn, the robot
keeps driving in a circle until someone catches it. The driver's own 300 ms
watchdog covers its publisher dying, but not the case that matters more here:
every node alive, still publishing, on data that stopped being true.

Pure pursuit will happily track a path that is 10 s old. The planner will
happily replan on a traversability map that stopped updating. Neither is a
crash, so neither triggers a crash-based safety net. This node checks the
freshness of the WHOLE chain and zeroes the command if any link goes quiet:

    perception -> planner -> pure pursuit -> HERE -> driver

DEFAULTS TO DISABLED. The robot cannot move until something explicitly
publishes enable=true, so a stack that comes up unattended stays still.

    /cmd_vel_raw  (pure pursuit)  ->  /cmd_vel  (driver)
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class Deadman(Node):
    def __init__(self) -> None:
        super().__init__("deadman")

        self.declare_parameter("cmd_timeout", 0.30)
        self.declare_parameter("path_timeout", 2.0)
        self.declare_parameter("perception_timeout", 2.0)
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("max_speed", 0.35)
        self.declare_parameter("require_perception", True)
        # Publish zeros continuously while stopped. The driver's watchdog
        # would stop it anyway after 300 ms, but an explicit stream of zeros
        # stops it now and leaves an unambiguous record in the bag.
        self.declare_parameter("publish_zeros_when_stopped", True)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cmd_timeout = float(g("cmd_timeout"))
        self.path_timeout = float(g("path_timeout"))
        self.perc_timeout = float(g("perception_timeout"))
        self.max_speed = float(g("max_speed"))
        self.require_perception = bool(g("require_perception"))
        self.publish_zeros = bool(g("publish_zeros_when_stopped"))

        now = self.get_clock().now()
        self.enabled = False          # <- robot cannot move until told to
        self.estop = False
        self.last_cmd: Twist | None = None
        self.t_cmd = now
        self.t_path = now
        self.t_perc = now
        self.reason = "not enabled"
        self.blocks = 0

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Twist, "cmd_vel_raw", self._on_cmd, 1)
        self.create_subscription(Path, "path", self._on_path, qos)
        self.create_subscription(Image, "perception", self._on_perception, qos)
        self.create_subscription(Bool, "deadman_enable", self._on_enable, 1)
        self.create_subscription(Bool, "estop", self._on_estop, 1)

        self.pub_cmd = self.create_publisher(Twist, "cmd_vel", 1)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        self.pub_ok = self.create_publisher(Bool, "~/passing", 1)

        self.create_timer(1.0 / float(g("rate")), self._tick)
        self.create_timer(1.0, self._health)

        self.get_logger().warn(
            "deadman active, DISABLED. Publish true on deadman_enable to "
            "allow motion; publish true on estop to latch a stop.")

    # ------------------------------------------------------------------
    def _on_cmd(self, msg: Twist) -> None:
        self.last_cmd = msg
        self.t_cmd = self.get_clock().now()

    def _on_path(self, _msg: Path) -> None:
        self.t_path = self.get_clock().now()

    def _on_perception(self, _msg: Image) -> None:
        self.t_perc = self.get_clock().now()

    def _on_enable(self, msg: Bool) -> None:
        if msg.data and self.estop:
            self.get_logger().warn("enable ignored: e-stop is latched")
            return
        if msg.data != self.enabled:
            self.get_logger().info(
                f"deadman {'ENABLED' if msg.data else 'disabled'}")
        self.enabled = bool(msg.data)

    def _on_estop(self, msg: Bool) -> None:
        """Latching. Clearing it requires an explicit estop=false."""
        if msg.data and not self.estop:
            self.get_logger().error("E-STOP latched")
            self.enabled = False
        elif not msg.data and self.estop:
            self.get_logger().warn("e-stop cleared; still disabled")
        self.estop = bool(msg.data)

    # ------------------------------------------------------------------
    def _age(self, t) -> float:
        return (self.get_clock().now() - t).nanoseconds * 1e-9

    def _blocked_because(self) -> str | None:
        if self.estop:
            return "e-stop latched"
        if not self.enabled:
            return "not enabled"
        if self.last_cmd is None:
            return "no command yet"
        if self._age(self.t_cmd) > self.cmd_timeout:
            return f"command {self._age(self.t_cmd):.1f}s old"
        if self._age(self.t_path) > self.path_timeout:
            return f"path {self._age(self.t_path):.1f}s old"
        if self.require_perception and self._age(self.t_perc) > self.perc_timeout:
            return f"perception {self._age(self.t_perc):.1f}s old"
        return None

    def _tick(self) -> None:
        why = self._blocked_because()
        if why is not None:
            if self.reason != why:
                self.get_logger().warn(f"blocking motion: {why}")
                self.blocks += 1
            self.reason = why
            if self.publish_zeros:
                self.pub_cmd.publish(Twist())
            self.pub_ok.publish(Bool(data=False))
            return

        out = Twist()
        # Independent speed ceiling. Pure pursuit has its own limit; this one
        # catches a bad parameter or a node that has been reconfigured.
        v = float(self.last_cmd.linear.x)
        out.linear.x = max(-self.max_speed, min(self.max_speed, v))
        out.angular.z = float(self.last_cmd.angular.z)
        self.pub_cmd.publish(out)
        self.reason = "passing"
        self.pub_ok.publish(Bool(data=True))

    def _health(self) -> None:
        self.pub_status.publish(String(data=(
            f"enabled={self.enabled} estop={self.estop} blocks={self.blocks} "
            f"cmd={self._age(self.t_cmd):.1f}s path={self._age(self.t_path):.1f}s "
            f"perc={self._age(self.t_perc):.1f}s :: {self.reason}")))


def main() -> None:
    rclpy.init()
    node = Deadman()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Three zeros on the way out: publishing is best-effort and the
        # process is about to vanish.
        for _ in range(3):
            node.pub_cmd.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
