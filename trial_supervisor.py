#!/usr/bin/env python3
"""Ends a trial without needing the network.

The backyard run this was written for had no terminating condition at all.
`heading` mode has no goal, so nothing ends a trial: the robot wedged itself
under a boat trailer at 87 s and then sat there recording at 10 MB/s until
someone carried it back into WiFi range and killed the tmux server. The stop
path ran entirely through the house router, which is the one thing guaranteed
to be missing at the far end of a run.

Four triggers, none of which need a network:

    tilt        picking the robot up. In 967 samples of real driving the tilt
                from vertical never exceeded 11.7 deg; during the pickup it
                sat at 52 deg median. Any threshold in 25-40 deg separated
                them with zero false positives, so 30 deg has ~13 deg of
                margin on the worst driving sample.
    geofence    straight-line distance from where the trial started.
    duration    wall time since the trial started.
    stuck       the deadman blocking continuously, i.e. the robot has already
                given up but nothing has told it to stop trying.

TILT IS THE ONE THAT MATTERS. It turns "walk over and pick it up" -- which is
what you do anyway -- into the stop command, with no hardware, no pairing and
no radio link.

Order on trigger is not negotiable: latch the e-stop, WAIT for the chassis to
confirm it has actually stopped, and only then shut the stack down. Shutting
down first would drop the deadman and pure pursuit while the last command was
still executing, and the chassis firmware has no timeout of its own.

Everything arms on FIRST MOTION, not at startup, so handling the robot while
placing it cannot trigger anything.
"""

from __future__ import annotations

import math
import os
import signal

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Int32, String

ARMED, TRIGGERED, SETTLING, DONE = "armed", "triggered", "settling", "done"


class TrialSupervisor(Node):
    def __init__(self) -> None:
        super().__init__("trial_supervisor")

        # Any limit set to 0.0 is disabled.
        self.declare_parameter("tilt_deg", 30.0)
        self.declare_parameter("tilt_hold_s", 0.4)
        self.declare_parameter("geofence_m", 25.0)
        self.declare_parameter("max_duration_s", 180.0)
        self.declare_parameter("stuck_timeout_s", 15.0)
        # If the robot never moves at all, end anyway rather than record an
        # empty bag until the disk fills.
        self.declare_parameter("no_start_timeout_s", 180.0)
        self.declare_parameter("move_speed_thresh", 0.05)
        # How long to wait for the chassis to confirm zero before shutting
        # down. The driver's own watchdog is 300 ms, so this is generous.
        self.declare_parameter("settle_timeout_s", 3.0)
        self.declare_parameter("shutdown", True)
        self.declare_parameter("beep", True)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.tilt_deg = float(g("tilt_deg"))
        self.tilt_hold = float(g("tilt_hold_s"))
        self.geofence = float(g("geofence_m"))
        self.max_dur = float(g("max_duration_s"))
        self.stuck_to = float(g("stuck_timeout_s"))
        self.no_start = float(g("no_start_timeout_s"))
        self.move_thresh = float(g("move_speed_thresh"))
        self.settle_to = float(g("settle_timeout_s"))
        self.do_shutdown = bool(g("shutdown"))
        self.do_beep = bool(g("beep"))

        now = self.get_clock().now()
        self.state = ARMED
        self.t_node = now
        self.t_start = None            # first motion
        self.t_trigger = None
        self.t_tilt_over = None        # when tilt first exceeded threshold
        self.t_passing = now           # last time the deadman was passing
        self.origin = None             # pose at first motion
        self.reason = ""
        self.zero_ticks = 0
        self.tilt_now = 0.0
        self.dist = 0.0

        sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Odometry, "odom", self._on_odom, sensor)
        self.create_subscription(Imu, "imu", self._on_imu, sensor)
        self.create_subscription(Bool, "deadman_passing", self._on_passing, 1)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd, 1)

        self.pub_estop = self.create_publisher(Bool, "estop", 1)
        self.pub_beep = self.create_publisher(Int32, "beep", 1)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        # Recorded, so the trial log writes itself.
        self.pub_reason = self.create_publisher(String, "~/reason", 1)

        self.create_timer(0.05, self._tick)
        self.create_timer(1.0, self._health)

        self.get_logger().info(
            f"trial supervisor: tilt>{self.tilt_deg:.0f}deg for "
            f"{self.tilt_hold:.1f}s | geofence {self.geofence:.0f}m | "
            f"{self.max_dur:.0f}s | stuck {self.stuck_to:.0f}s "
            f"| shutdown={self.do_shutdown}")
        self.get_logger().warn(
            "PICK THE ROBOT UP TO STOP IT. Limits arm on first motion.")

    # ------------------------------------------------------------------
    def _age(self, t) -> float:
        return (self.get_clock().now() - t).nanoseconds * 1e-9

    def _on_imu(self, m: Imu) -> None:
        a = m.linear_acceleration
        n = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
        if n < 1e-6:
            return
        # Signed, so an inverted robot reads 180 deg rather than 0.
        self.tilt_now = math.degrees(math.acos(max(-1.0, min(1.0, a.z / n))))

    def _on_passing(self, m: Bool) -> None:
        if m.data:
            self.t_passing = self.get_clock().now()

    def _on_cmd(self, m: Twist) -> None:
        if abs(m.linear.x) < 1e-6:
            self.zero_ticks += 1
        else:
            self.zero_ticks = 0

    def _on_odom(self, m: Odometry) -> None:
        p = m.pose.pose.position
        if self.t_start is None:
            if abs(m.twist.twist.linear.x) > self.move_thresh:
                self.t_start = self.get_clock().now()
                self.origin = (p.x, p.y)
                self.get_logger().info("trial started - limits are live")
            return
        self.dist = math.hypot(p.x - self.origin[0], p.y - self.origin[1])

    # ------------------------------------------------------------------
    def _check(self) -> str | None:
        """The reason to end this trial, or None."""
        if self.t_start is None:
            if self.no_start > 0 and self._age(self.t_node) > self.no_start:
                return f"never moved within {self.no_start:.0f}s"
            return None

        if self.tilt_deg > 0:
            if self.tilt_now > self.tilt_deg:
                if self.t_tilt_over is None:
                    self.t_tilt_over = self.get_clock().now()
                elif self._age(self.t_tilt_over) >= self.tilt_hold:
                    return f"picked up (tilt {self.tilt_now:.0f}deg)"
            else:
                self.t_tilt_over = None

        if self.geofence > 0 and self.dist > self.geofence:
            return f"geofence {self.dist:.1f}m > {self.geofence:.0f}m"
        if self.max_dur > 0 and self._age(self.t_start) > self.max_dur:
            return f"duration {self._age(self.t_start):.0f}s"
        if self.stuck_to > 0 and self._age(self.t_passing) > self.stuck_to:
            return f"stuck {self._age(self.t_passing):.0f}s"
        return None

    def _tick(self) -> None:
        if self.state == ARMED:
            why = self._check()
            if why is None:
                return
            self.reason = why
            self.t_trigger = self.get_clock().now()
            self.state = TRIGGERED
            self.get_logger().error(f"ENDING TRIAL: {why}")
            self.pub_reason.publish(String(data=why))
            # Repeated because this is the message that must not be missed.
            for _ in range(5):
                self.pub_estop.publish(Bool(data=True))
            if self.do_beep:
                self.pub_beep.publish(Int32(data=3))
            self.state = SETTLING
            return

        if self.state == SETTLING:
            self.pub_estop.publish(Bool(data=True))
            stopped = self.zero_ticks >= 5
            waited = self._age(self.t_trigger)
            if not stopped and waited < self.settle_to:
                return
            if stopped:
                self.get_logger().info(f"chassis stopped after {waited:.2f}s")
            else:
                self.get_logger().error(
                    f"chassis did NOT confirm zero within {self.settle_to:.1f}s; "
                    "shutting down anyway - the driver watchdog stops it at 300ms")
            self.state = DONE
            self._finish()

    def _finish(self) -> None:
        if self.do_beep:
            self.pub_beep.publish(Int32(data=2))
        if not self.do_shutdown:
            self.get_logger().warn("shutdown disabled; robot is stopped and "
                                   "the e-stop is latched")
            return
        ppid = os.getppid()
        try:
            with open(f"/proc/{ppid}/cmdline", "rb") as fh:
                cmd = fh.read().decode("utf-8", "replace").replace("\0", " ")
        except OSError:
            cmd = ""
        # Refuse to signal something that is not the launch. Run standalone
        # from a shell and the parent is the shell, and SIGINT would take out
        # the terminal instead of the stack.
        if "launch" not in cmd:
            self.get_logger().error(
                f"parent pid {ppid} ({cmd.strip()!r}) is not a launch process; "
                "not signalling it. Robot is stopped, e-stop latched.")
            return
        self.get_logger().warn(f"SIGINT -> launch (pid {ppid}) to close the bag")
        os.kill(ppid, signal.SIGINT)

    def _health(self) -> None:
        el = f"{self._age(self.t_start):.0f}s" if self.t_start else "-"
        self.pub_status.publish(String(data=(
            f"state={self.state} elapsed={el} dist={self.dist:.1f}m "
            f"tilt={self.tilt_now:.0f}deg "
            f"blocked={self._age(self.t_passing):.1f}s "
            f":: {self.reason or 'running'}")))


def main() -> None:
    rclpy.init()
    node = TrialSupervisor()
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
