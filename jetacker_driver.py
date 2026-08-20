#!/usr/bin/env python3
"""JetAcker ROS 2 driver: cmd_vel -> RRC serial commands.

Protocol reverse-engineered and confirmed against the Hiwonder mini STM32
controller (USB 1a86:55d4):

    frame    = AA 55 | func | len | data | crc8_maxim(func+len+data)
    motor    = func 3, [0x01, n] + n x (id_u8 zero-based, float32_le rev/s)
    servo    = func 5, [0x01, dur_u16, n] + n x (id_u8, pos_u16)
    imu      = func 7, 24B = 6 x float32_le: ax,ay,az (g), gx,gy,gz

Chassis facts measured on this robot:
    motor 2 = left rear   (positive rev/s drives forward)
    motor 4 = right rear  (NEGATIVE rev/s drives forward)
    steering = bus servo id 9, centre 500, mechanical stops at 350 / 650
    wheelbase 0.216 m, track 0.195 m, wheel diameter 0.097 m
    steering map: servo = 500 + 1000 * degrees / 240   (+-36 deg)

IMPORTANT: this chassis is Ackermann. It CANNOT rotate in place. A Twist with
linear.x == 0 produces no motion whatever angular.z says - the firmware zeroes
all motors when speed is zero, and steering alone does nothing. Minimum turning
radius is about 0.30 m.

SAFETY: the controller has no command timeout of its own. If this node dies
mid-command the motors keep running. The watchdog below is the only thing that
stops that, so do not disable it.
"""

from __future__ import annotations

import math
import threading

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int32, String

from rrc import RRCBoard


G_TO_MS2 = 9.80665
DEG_TO_RAD = math.pi / 180.0


class JetAckerDriver(Node):
    def __init__(self) -> None:
        super().__init__("jetacker_driver")

        # ---- chassis geometry -------------------------------------------
        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("wheelbase", 0.216)
        self.declare_parameter("track_width", 0.195)
        self.declare_parameter("wheel_diameter", 0.097)
        self.declare_parameter("max_steer_deg", 34.0)   # inside the 36 limit
        self.declare_parameter("max_speed", 0.6)

        # ---- servo ------------------------------------------------------
        self.declare_parameter("steer_servo_id", 9)
        self.declare_parameter("steer_centre", 500)
        self.declare_parameter("steer_min", 360)        # stops are 350/650;
        self.declare_parameter("steer_max", 640)        # stay inside them
        self.declare_parameter("steer_duration", 0.05)  # servo slew time, s

        # ---- motors -----------------------------------------------------
        self.declare_parameter("left_motor_id", 2)
        self.declare_parameter("right_motor_id", 4)
        self.declare_parameter("left_sign", 1.0)
        self.declare_parameter("right_sign", -1.0)

        # ---- safety -----------------------------------------------------
        self.declare_parameter("cmd_timeout", 0.3)
        self.declare_parameter("rate", 20.0)
        self.declare_parameter("publish_imu", True)
        self.declare_parameter("imu_frame", "imu_link")

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.L = float(g("wheelbase"))
        self.T = float(g("track_width"))
        self.wheel_d = float(g("wheel_diameter"))
        self.max_steer = float(g("max_steer_deg")) * DEG_TO_RAD
        self.max_speed = float(g("max_speed"))
        self.servo_id = int(g("steer_servo_id"))
        self.centre = int(g("steer_centre"))
        self.smin, self.smax = int(g("steer_min")), int(g("steer_max"))
        self.steer_dur = float(g("steer_duration"))
        self.lid, self.rid = int(g("left_motor_id")), int(g("right_motor_id"))
        self.lsign, self.rsign = float(g("left_sign")), float(g("right_sign"))
        self.timeout = float(g("cmd_timeout"))
        self.publish_imu = bool(g("publish_imu"))
        self.imu_frame = str(g("imu_frame"))

        # Minimum turning radius, purely informational but worth logging:
        # a planner that hands us tighter curvature than this cannot be followed.
        self.min_radius = self.L / math.tan(self.max_steer)

        # ---- hardware ---------------------------------------------------
        self.board = RRCBoard(str(g("port")))
        self.lock = threading.Lock()
        self.last_cmd_time = self.get_clock().now()
        self.target = (0.0, 0.0)      # (linear m/s, angular rad/s)
        self.stopped = True

        with self.lock:
            self.board.stop_motors()
            self.board.set_bus_servo_position(0.3, [(self.servo_id, self.centre)])

        # ---- ROS --------------------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, qos)
        # Audible/visible confirmation. On a robot in a backyard there is no
        # screen, so a stop that cannot be heard cannot be trusted: you end up
        # walking over to check, which is the thing being avoided. The driver
        # owns the serial port, so this has to live here rather than in
        # whichever node wants to beep.
        self.create_subscription(Int32, "beep", self._on_beep, 1)

        self.pub_steer = self.create_publisher(Float32, "~/steer_angle_deg", 10)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        self.pub_imu = (
            self.create_publisher(Imu, "imu", 10) if self.publish_imu else None
        )

        period = 1.0 / float(g("rate"))
        self.create_timer(period, self._control_loop)
        if self.publish_imu:
            self.create_timer(0.02, self._imu_loop)
        self.create_timer(1.0, self._status_loop)

        self.get_logger().info(
            f"jetacker ready: L={self.L} T={self.T} d={self.wheel_d} "
            f"max_steer={math.degrees(self.max_steer):.0f}deg "
            f"min_radius={self.min_radius:.2f}m"
        )
        self.get_logger().warn(
            "Ackermann chassis: linear.x == 0 produces NO motion, "
            "angular.z alone does nothing"
        )

    # ------------------------------------------------------------------
    def _on_beep(self, msg: Int32) -> None:
        n = max(0, min(8, int(msg.data)))
        if not n:
            return
        # Board-timed: both calls just write a packet, so neither blocks the
        # control loop waiting on the beep to finish.
        with self.lock:
            self.board.set_buzzer(freq=2400, on_s=0.12, off_s=0.10, repeat=n)
            self.board.set_led(on_s=0.12, off_s=0.10, repeat=n)

    # ------------------------------------------------------------------
    def _on_cmd_vel(self, msg: Twist) -> None:
        v = max(-self.max_speed, min(self.max_speed, float(msg.linear.x)))
        self.target = (v, float(msg.angular.z))
        self.last_cmd_time = self.get_clock().now()

    # ------------------------------------------------------------------
    def kinematics(self, v: float, omega: float):
        """Twist -> (left rev/s, right rev/s, steering angle rad).

        Ackermann: the steering angle sets the turn radius, and the rear
        wheels must then differ in speed because they travel different arcs.
        """
        if abs(v) < 1e-6:
            # Firmware zeroes all motors at zero speed, and steering without
            # forward motion achieves nothing. Centre and stop.
            return 0.0, 0.0, 0.0

        if abs(omega) < 1e-6:
            steer = 0.0
            vl = vr = v
        else:
            radius = v / omega                       # signed, of the rear axle
            steer = math.atan(self.L / radius)
            # Clamp to what the linkage can do, then recompute the radius so
            # the wheel speeds stay consistent with the angle actually used.
            steer = max(-self.max_steer, min(self.max_steer, steer))
            radius = self.L / math.tan(steer)
            omega_eff = v / radius
            vl = omega_eff * (radius - self.T / 2.0)
            vr = omega_eff * (radius + self.T / 2.0)

        rps_l = vl / (math.pi * self.wheel_d)
        rps_r = vr / (math.pi * self.wheel_d)
        return rps_l, rps_r, steer

    def steer_to_servo(self, steer_rad: float) -> int:
        """servo = 500 + 1000 * degrees / 240, clamped inside the stops."""
        pos = self.centre + 1000.0 * math.degrees(steer_rad) / 240.0
        return int(max(self.smin, min(self.smax, round(pos))))

    # ------------------------------------------------------------------
    def _control_loop(self) -> None:
        age = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9

        if age > self.timeout:
            # DEADMAN. The board has no timeout of its own; without this the
            # last command runs forever if the publisher dies.
            if not self.stopped:
                self.get_logger().warn(
                    f"no cmd_vel for {age:.2f}s - stopping", throttle_duration_sec=5.0
                )
                with self.lock:
                    self.board.stop_motors()
                    self.board.set_bus_servo_position(
                        0.2, [(self.servo_id, self.centre)]
                    )
                self.stopped = True
            return

        v, omega = self.target
        rps_l, rps_r, steer = self.kinematics(v, omega)
        servo_pos = self.steer_to_servo(steer)

        with self.lock:
            self.board.set_bus_servo_position(
                self.steer_dur, [(self.servo_id, servo_pos)]
            )
            self.board.set_motor_speed([
                (self.lid, self.lsign * rps_l),
                (self.rid, self.rsign * rps_r),
            ])
        self.stopped = abs(v) < 1e-6

        self.pub_steer.publish(Float32(data=float(math.degrees(steer))))

    # ------------------------------------------------------------------
    def _imu_loop(self) -> None:
        with self.lock:
            vals = self.board.read_imu(timeout_s=0.01)
        if not vals:
            return
        ax, ay, az, gx, gy, gz = vals

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.imu_frame
        # Board reports accel in g and gyro in deg/s; ROS wants m/s^2 and rad/s.
        msg.linear_acceleration.x = ax * G_TO_MS2
        msg.linear_acceleration.y = ay * G_TO_MS2
        msg.linear_acceleration.z = az * G_TO_MS2
        msg.angular_velocity.x = gx * DEG_TO_RAD
        msg.angular_velocity.y = gy * DEG_TO_RAD
        msg.angular_velocity.z = gz * DEG_TO_RAD
        # No orientation estimate from this driver; -1 marks it unavailable.
        msg.orientation_covariance[0] = -1.0
        self.pub_imu.publish(msg)

    # ------------------------------------------------------------------
    def _status_loop(self) -> None:
        with self.lock:
            pos = self.board.read_servo_position(self.servo_id)
            volts = self.board.read_bus_servo(7, self.servo_id)
        v, omega = self.target
        self.pub_status.publish(String(data=(
            f"v={v:.2f} w={omega:.2f} servo={pos} "
            f"vbat={volts / 1000.0 if volts else '?'}V stopped={self.stopped}"
        )))

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self.get_logger().info("stopping motors and centring steering")
        try:
            with self.lock:
                self.board.stop_motors()
                self.board.set_bus_servo_position(
                    0.3, [(self.servo_id, self.centre)]
                )
            self.board.close()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"clean shutdown failed: {exc}")


def main() -> None:
    rclpy.init()
    node = JetAckerDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
