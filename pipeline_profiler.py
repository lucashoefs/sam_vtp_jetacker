#!/usr/bin/env python3
"""End-to-end pipeline profiler: where does the latency go, and what spikes?

Every stage republishes the SOURCE frame's timestamp when it finishes, so the
same frame can be followed through the pipeline and each stage's contribution
measured separately:

    capture ---> mask ---> path ---> (control)
            A          B         C

    A = perception   camera driver + transport + cv_bridge + inference
    B = planning     BEV projection + arc search
    C = control      pure pursuit -> deadman -> driver

Reporting means alone hides the problem: a pipeline at 200 ms mean with 900 ms
spikes drives worse than one at 300 ms flat, because the controller cannot
plan around a delay it does not know is coming. So this reports percentiles
and tail, and on every spike captures the system state at that instant - CPU
frequencies, per-core load, temperatures, memory and swap - which is what
distinguishes thermal throttling from contention from memory pressure.

    python3 pipeline_profiler.py
    python3 pipeline_profiler.py --spike-ms 400 --window 300
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import time
from collections import deque
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header


def stamp_key(h: Header) -> int:
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


class SystemProbe:
    """Cheap reads of the things that actually cause spikes on a Jetson."""

    def __init__(self) -> None:
        self._prev = self._cpu_raw()

    @staticmethod
    def _cpu_raw():
        out = {}
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if not line.startswith("cpu"):
                        break
                    parts = line.split()
                    if parts[0] == "cpu":
                        continue
                    v = [int(x) for x in parts[1:8]]
                    out[parts[0]] = (sum(v), v[3])   # (total, idle)
        except Exception:
            pass
        return out

    def cpu_percent(self):
        cur = self._cpu_raw()
        out = []
        for k in sorted(cur, key=lambda s: int(s[3:])):
            if k not in self._prev:
                continue
            dt = cur[k][0] - self._prev[k][0]
            di = cur[k][1] - self._prev[k][1]
            out.append(100.0 * (1 - di / dt) if dt > 0 else 0.0)
        self._prev = cur
        return out

    @staticmethod
    def cpu_freqs_mhz():
        f = []
        for p in sorted(glob.glob(
                "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")):
            try:
                f.append(int(open(p).read().strip()) / 1000.0)
            except Exception:
                pass
        return f

    @staticmethod
    def temps():
        out = {}
        for z in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
            try:
                name = open(z + "/type").read().strip()
                out[name] = int(open(z + "/temp").read().strip()) / 1000.0
            except Exception:
                pass
        return out

    @staticmethod
    def mem_mb():
        info = {}
        try:
            for line in open("/proc/meminfo"):
                k, v = line.split(":")
                info[k] = int(v.split()[0]) / 1024.0
        except Exception:
            return (0, 0, 0, 0)
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", 0)
        swtot = info.get("SwapTotal", 0)
        swfree = info.get("SwapFree", 0)
        return (total - avail, total, swtot - swfree, swtot)

    def snapshot(self) -> str:
        cpu = self.cpu_percent()
        freqs = self.cpu_freqs_mhz()
        t = self.temps()
        used, total, sw_used, sw_total = self.mem_mb()
        hot = max(t.values()) if t else 0.0
        return (f"cpu[{' '.join(f'{c:3.0f}' for c in cpu)}]% "
                f"freq {min(freqs):.0f}-{max(freqs):.0f}MHz "
                f"temp {hot:.1f}C "
                f"mem {used:.0f}/{total:.0f}MB swap {sw_used:.0f}/{sw_total:.0f}MB")


class PipelineProfiler(Node):
    def __init__(self, args) -> None:
        super().__init__("pipeline_profiler")
        self.args = args
        self.sys = SystemProbe()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=20)

        # frame stamp -> arrival monotonic time, per stage
        self.mask_at: dict[int, float] = {}
        self.path_at: dict[int, float] = {}

        self.a = deque(maxlen=args.window)   # capture -> mask
        self.b = deque(maxlen=args.window)   # mask -> path
        self.total = deque(maxlen=args.window)
        self.mask_gaps = deque(maxlen=args.window)
        self.path_gaps = deque(maxlen=args.window)
        self._last_mask: Optional[float] = None
        self._last_path: Optional[float] = None
        self.spikes = 0
        self.n = 0
        self.t0 = time.monotonic()

        self.create_subscription(Header, args.mask_topic, self._on_mask, qos)
        self.create_subscription(Header, args.path_topic, self._on_path, qos)
        self.create_timer(args.report, self._report)

        print(f"profiling {args.mask_topic} and {args.path_topic}")
        print(f"spike threshold {args.spike_ms:.0f} ms, "
              f"window {args.window} frames\n")

    # ------------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_mask(self, msg: Header) -> None:
        k = stamp_key(msg)
        now = self._now()
        self.mask_at[k] = now
        a_ms = (now - k * 1e-9) * 1000.0
        self.a.append(a_ms)
        if self._last_mask is not None:
            self.mask_gaps.append((now - self._last_mask) * 1000.0)
        self._last_mask = now
        # Bound the join tables; frames older than a second are never matched.
        if len(self.mask_at) > 200:
            cut = k - 1_000_000_000
            self.mask_at = {kk: v for kk, v in self.mask_at.items() if kk > cut}

    def _on_path(self, msg: Header) -> None:
        k = stamp_key(msg)
        now = self._now()
        if self._last_path is not None:
            self.path_gaps.append((now - self._last_path) * 1000.0)
        self._last_path = now

        if k not in self.mask_at:
            return
        b_ms = (now - self.mask_at[k]) * 1000.0
        total_ms = (now - k * 1e-9) * 1000.0
        self.b.append(b_ms)
        self.total.append(total_ms)
        self.n += 1

        if total_ms > self.args.spike_ms:
            self.spikes += 1
            a_ms = (self.mask_at[k] - k * 1e-9) * 1000.0
            worst = "perception" if a_ms > b_ms else "planning"
            print(f"  SPIKE {total_ms:6.0f} ms  "
                  f"(perception {a_ms:5.0f} + planning {b_ms:5.0f}) "
                  f"worst={worst}")
            print(f"        {self.sys.snapshot()}")

    # ------------------------------------------------------------------
    @staticmethod
    def _pct(d: deque):
        if not d:
            return None
        s = sorted(d)
        n = len(s)
        return (statistics.fmean(s), s[n // 2], s[int(0.9 * (n - 1))],
                s[int(0.99 * (n - 1))], s[-1])

    def _report(self) -> None:
        if not self.total:
            print("  waiting for frames to traverse the pipeline...")
            return

        print(f"\n=== {time.monotonic() - self.t0:.0f}s, {self.n} frames, "
              f"{self.spikes} spikes ({100.0 * self.spikes / max(self.n,1):.1f}%) ===")
        print(f"  {'stage':<22} {'mean':>7} {'p50':>7} {'p90':>7} "
              f"{'p99':>7} {'max':>7}")
        for name, d in (("A capture->mask", self.a),
                        ("B mask->path", self.b),
                        ("TOTAL capture->path", self.total)):
            p = self._pct(d)
            if p:
                print(f"  {name:<22} {p[0]:7.0f} {p[1]:7.0f} {p[2]:7.0f} "
                      f"{p[3]:7.0f} {p[4]:7.0f}")

        for name, d in (("mask interval", self.mask_gaps),
                        ("path interval", self.path_gaps)):
            p = self._pct(d)
            if p:
                print(f"  {name:<22} {p[0]:7.0f} {p[1]:7.0f} {p[2]:7.0f} "
                      f"{p[3]:7.0f} {p[4]:7.0f}   "
                      f"({1000.0 / max(p[0], 1e-6):.1f} Hz mean)")

        print(f"  {self.sys.snapshot()}")

        # The interpretation is the useful part; a table of numbers is not.
        tp = self._pct(self.total)
        ap, bp = self._pct(self.a), self._pct(self.b)
        if tp and tp[4] > 3 * tp[1]:
            print(f"\n  Tail is {tp[4] / tp[1]:.1f}x the median. Something is")
            print("  intermittently stalling rather than running slowly.")
        if ap and bp and ap[0] > 3 * bp[0]:
            print("\n  Perception dominates. At 5 fps a frame is already 100 ms")
            print("  old on average before anything touches it; raising the")
            print("  camera rate helps more than optimising the planner.")
        gp = self._pct(self.mask_gaps)
        if gp and gp[4] > 2.5 * gp[1]:
            print("\n  Mask intervals are irregular: frames are being dropped")
            print("  or inference is stalling. Check /sam_vtp/status for")
            print("  dropped and no_pose counts.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-topic", default="/sam_vtp/stamp")
    ap.add_argument("--path-topic", default="/arc_planner/stamp")
    ap.add_argument("--spike-ms", type=float, default=400.0)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--report", type=float, default=10.0)
    args = ap.parse_args()

    rclpy.init()
    node = PipelineProfiler(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._report()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
