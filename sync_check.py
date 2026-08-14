#!/usr/bin/env python3
"""Measure the colour/depth timestamp offset and what it costs in pairing.

ApproximateTimeSynchronizer pairs the two streams within `slop` seconds. When
31% of frames fail to pair, the question is whether the two streams carry the
SAME stamps (in which case pairing should be perfect) or systematically
different ones (in which case slop is the wrong size).

Sampling the topics separately with `ros2 topic echo` cannot answer this - it
reads different frames at different moments. This subscribes to both and
compares stamps directly, then reports what pairing rate each candidate slop
value would have achieved on the data just collected.

Uses camera_info rather than the images: same headers, a thousandth of the
bandwidth, so measuring does not perturb what is measured.

    python3 sync_check.py
    python3 sync_check.py --seconds 20
"""

from __future__ import annotations

import argparse
import statistics

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo


def key(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class SyncCheck(Node):
    def __init__(self, args) -> None:
        super().__init__("sync_check")
        self.args = args
        self.colour: list[float] = []
        self.depth: list[float] = []
        self.create_subscription(CameraInfo, args.color_topic,
                                 lambda m: self.colour.append(key(m)),
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, args.depth_topic,
                                 lambda m: self.depth.append(key(m)),
                                 qos_profile_sensor_data)
        self.create_timer(args.seconds, self._report)
        print(f"collecting for {args.seconds:.0f}s ...")

    def _report(self) -> None:
        c, d = sorted(self.colour), sorted(self.depth)
        print(f"\ncolour {len(c)} frames, depth {len(d)} frames")
        if len(c) < 5 or len(d) < 5:
            print("not enough data - are both topics publishing?")
            raise SystemExit(1)

        def rate(t):
            gaps = [b - a for a, b in zip(t, t[1:])]
            return 1.0 / statistics.median(gaps), gaps

        rc, gc = rate(c)
        rd, gd = rate(d)
        print(f"  colour {rc:5.2f} Hz   median gap {statistics.median(gc)*1000:6.1f} ms")
        print(f"  depth  {rd:5.2f} Hz   median gap {statistics.median(gd)*1000:6.1f} ms")

        # Exact stamp equality: enable_sync should give the two streams the
        # SAME stamp, taken from one frameset.
        exact = len(set(c) & set(d))
        print(f"\n  identical stamps: {exact}/{min(len(c), len(d))} "
              f"({100.0*exact/max(min(len(c), len(d)),1):.0f}%)")

        # Nearest-neighbour offsets.
        offsets = []
        di = 0
        for t in c:
            while di + 1 < len(d) and abs(d[di+1] - t) < abs(d[di] - t):
                di += 1
            offsets.append((d[di] - t) * 1000.0)
        offsets.sort()
        n = len(offsets)
        print(f"  nearest-depth offset (ms): "
              f"p10 {offsets[n//10]:+7.1f}  p50 {offsets[n//2]:+7.1f}  "
              f"p90 {offsets[9*n//10]:+7.1f}")

        print("\n  pairing rate by slop:")
        for slop in (0.005, 0.010, 0.020, 0.033, 0.050, 0.100, 0.150):
            ok = sum(1 for o in offsets if abs(o) <= slop * 1000)
            print(f"    slop {slop*1000:6.0f} ms -> {100.0*ok/n:5.1f}%")

        med = abs(offsets[n // 2])
        print()
        if exact > 0.9 * min(len(c), len(d)):
            print("  Stamps are identical, so enable_sync is working and the")
            print("  pairing loss is NOT a stamp mismatch. Look at queue depth")
            print("  and at whether one stream is simply arriving less often.")
        elif med < 5:
            print("  Offset is tiny but stamps differ. A slop of 20-30 ms should")
            print("  pair essentially everything.")
        else:
            print(f"  Systematic offset of {med:.0f} ms. Set sync_slop above it")
            print("  (roughly double), or find out why enable_sync is not giving")
            print("  the two streams a common frameset stamp.")

        if abs(rc - rd) > 0.15 * max(rc, rd):
            print(f"\n  The streams also run at different rates ({rc:.1f} vs")
            print(f"  {rd:.1f} Hz). No slop value recovers frames that were")
            print("  never published; that gap is an upper bound on pairing.")

        raise SystemExit(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--color-topic", default="/camera/d455/color/camera_info")
    ap.add_argument("--depth-topic",
                    default="/camera/d455/aligned_depth_to_color/camera_info")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    rclpy.init()
    node = SyncCheck(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
