#!/usr/bin/env python3
"""JetAcker odometry: commanded speed + IMU yaw rate -> /odom and TF.

Why this is enough. SAM-VTP's memory bank holds 7 slots at ~0.2 s spacing, so
pose is only ever used to relate the current frame to memories at most ~1.2 s
old. Absolute drift is irrelevant; only short-horizon *relative* accuracy
matters. That is a much weaker requirement than SLAM, and open-loop speed with
a gyro-derived heading meets it comfortably.

Sources:
  linear velocity  - the commanded value. There are no wheel encoders exposed
                     on this chassis, so this is open loop and wheel slip is
                     unmodelled. Over 1.2 s the error stays small.
  yaw rate         - the board IMU's gyro z. Much better than deriving yaw from
                     steering geometry, which ignores slip entirely.

The gyro bias is estimated whenever the robot is commanded to stop. An
uncorrected MEMS gyro bias of 1 deg/s is 1.2 degrees of heading error across
the memory window, which is enough to visibly smear the warp.

Frames: odom -> base_link. The camera transform is static and belongs in the
launch file, not here.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Quaternion, Twist, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class JetAckerOdometry(Node):
    def __init__(self) -> None:
        super().__init__("jetacker_odometry")

        self.declare_parameter("rate", 50.0)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("cmd_timeout", 0.5)
        # Scale factor on commanded speed. Measure it: drive a known distance
        # at a fixed command and divide. Open-loop odometry is only as good as
        # this number.
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("gyro_bias_alpha", 0.002)
        self.declare_parameter("stationary_speed_eps", 1e-3)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.odom_frame = str(g("odom_frame"))
        self.base_frame = str(g("base_frame"))
        self.do_tf = bool(g("publish_tf"))
        self.cmd_timeout = float(g("cmd_timeout"))
        self.speed_scale = float(g("speed_scale"))
        self.bias_alpha = float(g("gyro_bias_alpha"))
        self.stationary_eps = float(g("stationary_speed_eps"))

        self.x = self.y = self.yaw = 0.0
        self.v_cmd = 0.0
        self.yaw_rate = 0.0
        self.gyro_bias = 0.0
        self.have_imu = False
        self.last_cmd_t = self.get_clock().now()
        self.last_t = self.get_clock().now()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Twist, "cmd_vel", self._on_cmd, 10)
        self.create_subscription(Imu, "imu", self._on_imu, sensor_qos)

        self.pub_odom = self.create_publisher(Odometry, "odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.do_tf else None

        self.create_timer(1.0 / float(g("rate")), self._tick)
        self.get_logger().info(
            f"odometry: {self.odom_frame} -> {self.base_frame}, "
            f"speed_scale={self.speed_scale}"
        )

    # ------------------------------------------------------------------
    def _on_cmd(self, msg: Twist) -> None:
        self.v_cmd = float(msg.linear.x) * self.speed_scale
        self.last_cmd_t = self.get_clock().now()

    def _on_imu(self, msg: Imu) -> None:
        raw = float(msg.angular_velocity.z)
        # Only learn the bias while stopped, or a real turn gets absorbed into
        # the bias estimate and the heading quietly stops tracking.
        if abs(self.v_cmd) < self.stationary_eps:
            self.gyro_bias += self.bias_alpha * (raw - self.gyro_bias)
        self.yaw_rate = raw - self.gyro_bias
        self.have_imu = True

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        now = self.get_clock().now()
        dt = (now - self.last_t).nanoseconds * 1e-9
        self.last_t = now
        if dt <= 0.0 or dt > 0.5:
            return

        # Match the driver's deadman: if commands stopped, the robot did too.
        if (now - self.last_cmd_t).nanoseconds * 1e-9 > self.cmd_timeout:
            self.v_cmd = 0.0

        v, w = self.v_cmd, (self.yaw_rate if self.have_imu else 0.0)

        # Exact integration of a constant-curvature arc over dt. The midpoint
        # heading matters at low rates: straight Euler drifts left or right
        # through a turn.
        if abs(w) < 1e-6:
            self.x += v * math.cos(self.yaw) * dt
            self.y += v * math.sin(self.yaw) * dt
        else:
            yaw_mid = self.yaw + 0.5 * w * dt
            self.x += v * math.cos(yaw_mid) * dt
            self.y += v * math.sin(yaw_mid) * dt
        self.yaw = math.atan2(math.sin(self.yaw + w * dt),
                              math.cos(self.yaw + w * dt))

        stamp = now.to_msg()
        quat = yaw_to_quat(self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = quat
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        # Honest covariances: x/y grow without bound (no encoders), yaw is
        # gyro-good. Anything consuming this should know not to trust position.
        odom.pose.covariance[0] = 0.05
        odom.pose.covariance[7] = 0.05
        odom.pose.covariance[35] = 0.01
        odom.twist.covariance[0] = 0.05
        odom.twist.covariance[35] = 0.005
        self.pub_odom.publish(odom)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.rotation = quat
            self.tf_broadcaster.sendTransform(tf)


def main() -> None:
    rclpy.init()
    node = JetAckerOdometry()
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
