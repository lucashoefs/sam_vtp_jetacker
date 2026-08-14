#!/usr/bin/env python3
"""Calibrate jetacker_odometry's speed_scale against a tape measure.

There are no wheel encoders on this chassis, so odometry integrates the
COMMANDED velocity. speed_scale is the one number correcting command to
reality, and everything downstream that uses pose - including SAM-VTP's
memory warp - inherits its error.

    speed_scale_new = speed_scale_current * (actual / odom_reported)

Procedure:
    1. Put a mark on the floor at the rear axle.
    2. Run this, press Enter to set the origin.
    3. Drive STRAIGHT forward 2-3 m with teleop, then stop.
    4. Press Enter again, measure the real distance, type it in.

Notes that matter:
    * Straight only. Steering while calibrating mixes heading error into the
      distance and you cannot separate the two afterwards.
    * Longer is better. At 3 m a 2 cm reading error is under 1%; at 0.5 m the
      same error is 4%.
    * Calibrate on the surface you will drive on. Carpet and hard floor differ
      by more than you would expect, because the correction is really absorbing
      wheel slip.
    * Repeat three times and average. Slip is not perfectly repeatable.

    python3 calibrate_speed.py --current-scale 1.0
"""

from __future__ import annotations

import argparse
import math
import threading

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class SpeedCalibrator(Node):
    def __init__(self, current_scale: float):
        super().__init__("speed_calibrator")
        self.current_scale = current_scale
        self.pose = None
        self.origin = None
        self.path_length = 0.0
        self._last = None
        self.create_subscription(Odometry, "odom", self._on_odom, 10)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.pose = (p.x, p.y)
        if self._last is not None:
            dx, dy = p.x - self._last[0], p.y - self._last[1]
            self.path_length += math.hypot(dx, dy)
        self._last = (p.x, p.y)

    def mark_origin(self) -> None:
        self.origin = self.pose
        self.path_length = 0.0

    def displacement(self):
        """Straight-line distance from the origin, and total path travelled.

        They differ if the robot wandered. A large gap means the run was not
        straight enough to calibrate from.
        """
        if self.origin is None or self.pose is None:
            return None, None
        dx = self.pose[0] - self.origin[0]
        dy = self.pose[1] - self.origin[1]
        return math.hypot(dx, dy), self.path_length


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-scale", type=float, default=1.0,
                    help="the speed_scale the odometry node is running with")
    args = ap.parse_args()

    rclpy.init()
    node = SpeedCalibrator(args.current_scale)
    spin = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin.start()

    print("waiting for /odom ...")
    while node.pose is None and rclpy.ok():
        pass
    print(f"odom is publishing. current speed_scale = {args.current_scale}\n")

    print("Mark the floor at the rear axle, then press Enter to set the origin.")
    input()
    node.mark_origin()
    print("origin set. Drive STRAIGHT forward 2-3 m, then stop.")
    print("press Enter when stopped...")

    stop = threading.Event()

    def live():
        while not stop.is_set():
            straight, path = node.displacement()
            if straight is not None:
                print(f"\r  odom: straight-line {straight:6.3f} m   "
                      f"path {path:6.3f} m ", end="", flush=True)
            stop.wait(0.2)

    ticker = threading.Thread(target=live, daemon=True)
    ticker.start()
    input()
    stop.set()
    ticker.join(timeout=1.0)

    straight, path = node.displacement()
    print(f"\n\nodom straight-line: {straight:.3f} m")
    print(f"odom path length:   {path:.3f} m")
    if path > 0 and (path - straight) / max(path, 1e-6) > 0.05:
        print("  ^ path is >5% longer than the straight line: the run curved.")
        print("    Steering error is contaminating this. Redo it straighter.")

    try:
        actual = float(input("\nmeasured distance with the tape, in metres: "))
    except ValueError:
        print("not a number; aborting")
        rclpy.shutdown()
        return

    if straight <= 1e-6:
        print("odom reported no movement; is the odometry node running?")
        rclpy.shutdown()
        return

    ratio = actual / straight
    new_scale = args.current_scale * ratio

    print(f"\n  actual / odom     = {ratio:.4f}")
    print(f"  new speed_scale   = {new_scale:.4f}")
    if actual < 1.5:
        print("\n  Distance under 1.5 m: measurement error dominates.")
        print("  Redo over 2-3 m before trusting this.")
    if abs(ratio - 1.0) > 0.3:
        print("\n  More than 30% off. That is a lot for slip alone - check")
        print("  wheel_diameter in the driver and that nothing is jammed.")

    print(f"\nApply with:")
    print(f"  ros2 launch ~/jetacker/robot_bringup.launch.py "
          f"speed_scale:={new_scale:.4f}")
    print("or set it as the default in robot_bringup.launch.py once you have")
    print("averaged a few runs.")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
