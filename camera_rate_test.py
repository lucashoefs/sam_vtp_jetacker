#!/usr/bin/env python3
"""Raw librealsense frame-rate test, with no ROS in the way.

The ROS wrapper reports 7-8 Hz on a stream configured for 15. This measures
what librealsense itself delivers, which separates three quite different
problems:

  * the camera or USB link cannot sustain the rate  -> every config is slow
  * a CPU-side format conversion is the bottleneck  -> RGB8 slow, YUYV fast
  * depth alignment is the bottleneck               -> align config slow
  * none of the above                               -> the ROS wrapper is

The D455 does NOT produce RGB8 natively. Requesting it makes librealsense
convert YUYV -> RGB8 on the CPU, about 6 M pixel conversions per second at
848x480x15, on a machine whose cores are already busy. Comparing the two
formats is the point of this test.

Stop the ROS stack first - only one process can hold the camera.

    python3 camera_rate_test.py
    python3 camera_rate_test.py --seconds 10 --fps 30
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 missing - run inside the venv, or pip install it")


def measure(label: str, configure, seconds: float, use_align: bool = False):
    pipe = rs.pipeline()
    cfg = rs.config()
    configure(cfg)

    try:
        profile = pipe.start(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<38} FAILED: {exc}")
        return None

    align = rs.align(rs.stream.color) if use_align else None

    times = []
    t_end = time.perf_counter() + seconds
    first = True
    try:
        while time.perf_counter() < t_end:
            try:
                frames = pipe.wait_for_frames(timeout_ms=2000)
            except Exception:
                break
            if align is not None:
                frames = align.process(frames)
            now = time.perf_counter()
            if first:            # discard the startup frame
                first = False
                times = [now]
                continue
            times.append(now)
    finally:
        pipe.stop()

    if len(times) < 3:
        print(f"  {label:<38} only {len(times)} frames")
        return None

    gaps = [(b - a) * 1000.0 for a, b in zip(times, times[1:])]
    hz = 1000.0 / statistics.fmean(gaps)
    print(f"  {label:<38} {hz:5.1f} Hz   "
          f"gap min {min(gaps):5.1f}  med {statistics.median(gaps):5.1f}  "
          f"max {max(gaps):6.1f} ms   n={len(gaps)}")
    return hz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    w, h, f = args.width, args.height, args.fps
    print(f"\n{w}x{h} @ {f} fps requested, {args.seconds:.0f}s per config")
    print(f"(perfect delivery would be {f} Hz, {1000.0/f:.1f} ms per frame)\n")

    tests = [
        ("colour YUYV only",
         lambda c: c.enable_stream(rs.stream.color, w, h, rs.format.yuyv, f), False),
        ("colour RGB8 only  (CPU conversion)",
         lambda c: c.enable_stream(rs.stream.color, w, h, rs.format.rgb8, f), False),
        ("depth only",
         lambda c: c.enable_stream(rs.stream.depth, w, h, rs.format.z16, f), False),
    ]

    def both_yuyv(c):
        c.enable_stream(rs.stream.color, w, h, rs.format.yuyv, f)
        c.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)

    def both_rgb(c):
        c.enable_stream(rs.stream.color, w, h, rs.format.rgb8, f)
        c.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)

    tests += [
        ("colour YUYV + depth", both_yuyv, False),
        ("colour RGB8 + depth", both_rgb, False),
        ("colour RGB8 + depth + ALIGN", both_rgb, True),
    ]

    results = {}
    for label, conf, al in tests:
        results[label] = measure(label, conf, args.seconds, al)
        time.sleep(0.6)          # let the device settle between configs

    print("\n--- reading ---")
    yuyv = results.get("colour YUYV only")
    rgb = results.get("colour RGB8 only  (CPU conversion)")
    both = results.get("colour RGB8 + depth")
    aligned = results.get("colour RGB8 + depth + ALIGN")

    if yuyv and rgb and yuyv > rgb * 1.3:
        print(f"  RGB8 is {yuyv/rgb:.1f}x slower than YUYV: the CPU colour")
        print("  conversion inside librealsense is a real bottleneck here.")
        print("  Request YUYV from the camera and convert on the GPU, or in")
        print("  cv_bridge, instead of making librealsense do it per frame.")
    if both and aligned and both > aligned * 1.3:
        print(f"  Alignment costs {both/aligned:.1f}x. It is needed for the")
        print("  geometry model, but consider aligning at lower resolution.")
    if yuyv and yuyv < f * 0.8:
        print(f"  Even bare YUYV only reaches {yuyv:.1f} Hz of {f}. That is the")
        print("  camera, the USB link or CPU starvation - not the ROS wrapper.")
        print("  Check `lsusb -t` shows 5000M, and watch tegrastats while this")
        print("  runs.")
    if yuyv and yuyv > f * 0.9:
        print(f"  librealsense delivers {yuyv:.1f} Hz cleanly, so the camera is")
        print("  fine and the loss is above it - in the wrapper, the format")
        print("  conversion, or CPU contention from everything else running.")


if __name__ == "__main__":
    main()
