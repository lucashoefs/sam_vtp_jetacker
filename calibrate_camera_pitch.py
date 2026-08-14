#!/usr/bin/env python3
"""Measure camera pitch and height by fitting the ground plane from depth.

WHY THIS MATTERS

The BEV marks a cell as an obstacle when it sits more than
obstacle_height_thresh_m above the assumed ground plane. That plane comes from
the camera's TF, so an unmeasured pitch does not degrade gracefully - it tilts
the whole plane, and distant ground crosses the height threshold and is
labelled impassable. The symptom is a BEV that is green close in and a solid
red band beyond a metre or two, with the model's own traversability perfectly
correct underneath.

It also feeds SAM-VTP's geometry channels, so the same error skews slope and
height there.

METHOD

Take depth in a near-field, bottom-centre window - almost certainly ground -
unproject to 3D in the optical frame, and fit a plane by least squares with
iterative outlier rejection. Then:

    pitch  = angle between the plane normal and the camera's vertical axis
    height = perpendicular distance from the camera centre to the plane

Point the robot at flat, open ground with nothing within about 3 m. Run it,
take the median over a few seconds, and put the result in the launch file.

    python3 calibrate_camera_pitch.py
"""

from __future__ import annotations

import argparse
import math
import statistics
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def fit_plane(points: np.ndarray, iters: int = 5, keep: float = 0.8):
    """Least-squares plane fit with iterative trimming. Returns (normal, d).

    Plane is n . x + d = 0 with |n| = 1. Trimming rather than RANSAC because
    the window is chosen to be mostly ground already; this just sheds the
    few centimetres of grass, cable or skirting board that creep in.
    """
    pts = points
    for _ in range(iters):
        centroid = pts.mean(axis=0)
        u, s, vt = np.linalg.svd(pts - centroid, full_matrices=False)
        normal = vt[-1]
        normal /= np.linalg.norm(normal)
        d = -float(normal @ centroid)
        if len(pts) < 200:
            break
        resid = np.abs(pts @ normal + d)
        cutoff = np.quantile(resid, keep)
        pts = pts[resid <= cutoff]
    return normal, d


class PitchCalibrator(Node):
    def __init__(self, args):
        super().__init__("pitch_calibrator")
        self.args = args
        self.bridge = CvBridge()
        self.K: Optional[np.ndarray] = None
        self.pitches: list[float] = []
        self.heights: list[float] = []
        self.rolls: list[float] = []

        self.create_subscription(CameraInfo, args.info_topic, self._on_info, 1)
        self.create_subscription(Image, args.depth_topic, self._on_depth,
                                 qos_profile_sensor_data)
        print(f"waiting for {args.depth_topic} ...")
        print("point the robot at flat open ground, nothing within ~3 m\n")

    def _on_info(self, msg: CameraInfo) -> None:
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _on_depth(self, msg: Image) -> None:
        if self.K is None:
            return
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = d.astype(np.float32) * self.args.depth_scale
        h, w = depth.shape

        # Bottom-centre window: the ground the robot is standing on.
        r0 = int(h * self.args.row_lo)
        r1 = int(h * self.args.row_hi)
        c0 = int(w * (0.5 - self.args.col_half))
        c1 = int(w * (0.5 + self.args.col_half))
        win = depth[r0:r1, c0:c1]

        rr, cc = np.mgrid[r0:r1, c0:c1]
        valid = (win > self.args.min_depth) & (win < self.args.max_depth) & np.isfinite(win)
        if valid.sum() < 500:
            print(f"  only {valid.sum()} valid depth pixels - is the ground in view?")
            return

        z = win[valid].astype(np.float64)
        u = cc[valid].astype(np.float64)
        v = rr[valid].astype(np.float64)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        # Optical frame: x right, y down, z forward.
        pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)

        normal, d = fit_plane(pts)
        # Point the normal "up", i.e. against the camera's +y (down) axis.
        if normal[1] > 0:
            normal, d = -normal, -d

        # For a level camera the ground normal is (0,-1,0) in optical coords.
        # Pitching the camera down tilts it toward +z.
        pitch = math.degrees(math.atan2(normal[2], -normal[1]))
        roll = math.degrees(math.atan2(normal[0], -normal[1]))
        height = abs(d)      # |n.0 + d| with unit n

        self.pitches.append(pitch)
        self.heights.append(height)
        self.rolls.append(roll)

        n = len(self.pitches)
        if n % 5 == 0:
            print(f"  n={n:<4} pitch {statistics.median(self.pitches):+6.2f} deg   "
                  f"height {statistics.median(self.heights):.3f} m   "
                  f"roll {statistics.median(self.rolls):+5.2f} deg   "
                  f"({valid.sum()} px)")

        if n >= self.args.samples:
            self._report()
            raise SystemExit(0)

    def _report(self) -> None:
        p = statistics.median(self.pitches)
        hgt = statistics.median(self.heights)
        r = statistics.median(self.rolls)
        ps = statistics.pstdev(self.pitches) if len(self.pitches) > 1 else 0.0

        print("\n" + "=" * 62)
        print(f"  pitch   {p:+.2f} deg   (sd {ps:.2f}; positive = nose DOWN)")
        print(f"  height  {hgt:.3f} m    from the camera centre to the ground")
        print(f"  roll    {r:+.2f} deg   (should be near zero if mounted square)")
        print("=" * 62)

        if ps > 1.0:
            print("\n  sd above 1 deg: the window is probably catching something")
            print("  other than flat ground. Move to open floor and rerun.")
        if abs(r) > 3.0:
            print(f"\n  roll of {r:+.1f} deg suggests the camera is not mounted")
            print("  square. That tilts the BEV sideways and is worth fixing")
            print("  mechanically rather than in software.")

        print(f"\nApply with:")
        print(f"  ros2 launch ~/jetacker/robot_bringup.launch.py "
              f"camera_pitch_deg:={p:.2f}")
        print(f"\nand set CAM_Z in robot_bringup.launch.py to {hgt:.3f} "
              f"if it differs from what is there now.")
        print("\nThen check the BEV: the red band beyond a metre or two should")
        print("go, because flat ground will stop crossing the height threshold.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-topic",
                    default="/camera/d455/aligned_depth_to_color/image_raw")
    ap.add_argument("--info-topic", default="/camera/d455/color/camera_info")
    ap.add_argument("--depth-scale", type=float, default=0.001)
    ap.add_argument("--samples", type=int, default=40)
    # Bottom-centre window, as fractions of the image.
    ap.add_argument("--row-lo", type=float, default=0.70)
    ap.add_argument("--row-hi", type=float, default=0.98)
    ap.add_argument("--col-half", type=float, default=0.25)
    ap.add_argument("--min-depth", type=float, default=0.3)
    ap.add_argument("--max-depth", type=float, default=3.0)
    args = ap.parse_args()

    rclpy.init()
    node = PitchCalibrator(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        if node.pitches and len(node.pitches) < args.samples:
            node._report()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
