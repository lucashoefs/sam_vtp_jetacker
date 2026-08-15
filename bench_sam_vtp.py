#!/usr/bin/env python3
"""Benchmark geometry-aware SAM-VTP on Jetson.

Two things make naive timing of this model misleading:

  * It is RECURRENT. Memory attention conditions each frame on the previous
    num_maskmem frames, so a single-frame measurement means nothing.
  * CUDA graph capture only engages once the memory bank is FULL, and for
    geometry checkpoints also once depth and pose are flowing. Averaging
    across warmup badly understates steady-state throughput.

So this runs a real sequence and reports steady state separately from warmup.

No checkpoint required. Timing depends only on architecture and shapes, not on
weight values, so `--random-weights` builds an untrained model from
SAMVTPConfig defaults and measures exactly the same work. Useful before a
trained checkpoint exists.

    # throughput with no checkpoint
    python3 bench_sam_vtp.py --random-weights

    # with a checkpoint, both modes
    python3 bench_sam_vtp.py --checkpoint ~/sam_vtp/sam_vtp.pt

    # where does the time go
    python3 bench_sam_vtp.py --random-weights --breakdown

    # thermal soak
    python3 bench_sam_vtp.py --random-weights --mode fullgraph --frames 20000
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from sam_vtp_core import (
    SAMVTP,
    SAMVTPConfig,
    CameraIntrinsics,
    SAMVTPStreamProcessor,
    build_model,
    pose_from_optical,
)

# Dotted paths from the SAMVTP root. Most of the network lives under
# sam2_model (a SAM2Core); the geometry encoder and refinement head are
# SAM-VTP additions and sit at the top level.
BREAKDOWN_MODULES = [
    "sam2_model.image_encoder",
    "geometry_encoder",
    "sam2_model.memory_attention",
    "sam2_model.sam_mask_decoder",
    "sam2_model.memory_encoder",
    "refine_head",
]


def resolve_module(root, dotted: str):
    """getattr through a dotted path, or None if any step is missing."""
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def read_temps() -> str:
    """Max thermal-zone reading, for spotting throttling during soak runs."""
    try:
        import glob
        peak, name = -1.0, "?"
        for zone in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
            try:
                t = int(open(zone + "/temp").read().strip()) / 1000.0
                if t > peak:
                    peak, name = t, open(zone + "/type").read().strip()
            except Exception:
                continue
        return f"{name}={peak:.1f}C" if peak > 0 else "temp=?"
    except Exception:
        return "temp=?"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM-VTP Jetson benchmark")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--random-weights", action="store_true",
                   help="build an untrained model; timing is unaffected by weights")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--pool", type=int, default=64,
                   help="distinct frames held in RAM; cycled to reach --frames")
    p.add_argument("--report-every", type=int, default=500)
    p.add_argument("--mode", choices=["fullgraph", "eager", "both"], default="both")
    p.add_argument("--breakdown", action="store_true",
                   help="per-submodule timing (forces eager; hooks do not fire in a graph)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--memory-write-interval", type=float, default=0.0,
                   help="seconds between memory writes; 0 disables")
    p.add_argument("--memory-write-every", type=int, default=1,
                   help="write memory every k-th frame. Set this to fps/5 to "
                        "reproduce the ~1.4 s bank span the model trained with; "
                        "graph replay still applies, via a second captured graph")
    # Measured from camera_info on the D455f at 848x480 (89x59 deg FOV).
    # Override for a different camera or resolution; these affect the warp
    # geometry, not the cost.
    p.add_argument("--fx", type=float, default=428.5935)
    p.add_argument("--fy", type=float, default=427.9190)
    p.add_argument("--cx", type=float, default=424.6508)
    p.add_argument("--cy", type=float, default=249.5679)
    p.add_argument("--native-width", type=int, default=848)
    p.add_argument("--native-height", type=int, default=480)
    p.add_argument("--input-fit", choices=["stretch", "crop", "pad"],
                   default="stretch",
                   help="how a non-square frame is fitted to the square input")
    # Synthetic trajectory. Shapes the warp's workload, not its cost.
    p.add_argument("--speed", type=float, default=0.5, help="m/s forward")
    p.add_argument("--yaw-rate", type=float, default=0.2, help="rad/s")
    p.add_argument("--fps", type=float, default=15.0, help="simulated frame rate")
    return p.parse_args()


def build_the_model(args, device):
    """Load a checkpoint, or construct untrained weights of the same shape."""
    cam = CameraIntrinsics(
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        width=args.native_width, height=args.native_height,
    )

    if args.checkpoint:
        ckpt = Path(args.checkpoint).expanduser()
        if not ckpt.exists():
            raise SystemExit(f"checkpoint not found: {ckpt}")
        model = build_model(str(ckpt), device=device,
                            normalization_mode="0_255", camera=cam)
        print(f"weights     : {ckpt.name}")
        return model, cam

    cfg = SAMVTPConfig()
    model = SAMVTP(cfg).to(device).eval()
    model.input_normalization_mode = "0_255"
    model.camera = cam
    print("weights     : RANDOM (untrained) - timing only, outputs are noise")
    return model, cam


def make_pool(args):
    """A pool of (rgb, depth) pairs at NATIVE camera resolution.

    Generating these already at the model's working size would skip the
    848x480 -> 512x512 resize that the real pipeline performs every frame,
    and would make the declared intrinsics inconsistent with the frames.
    """
    n = min(args.pool, args.frames)
    h, w = args.native_height, args.native_width
    print(f"frames      : {args.frames} (pool of {n} synthetic at {w}x{h})")

    base = torch.rand(1, 3, h // 16, w // 16)
    rgbs, depths = [], []
    for i in range(n):
        drift = torch.roll(base, shifts=(i % 8, i % 8), dims=(2, 3))
        rgb = torch.nn.functional.interpolate(
            drift, size=(h, w), mode="bilinear", align_corners=False)
        rgbs.append((rgb * 255.0).clamp(0, 255))

        # A ground plane receding to the horizon, plus a little noise. Roughly
        # the depth structure a forward-facing camera sees, so the warp does a
        # representative amount of valid-pixel work.
        yy = torch.linspace(0.0, 1.0, h).view(1, 1, h, 1).expand(1, 1, h, w)
        d = 1.0 / (yy * 0.25 + 0.05)
        d = d.clamp(0.3, 20.0) + torch.randn(1, 1, h, w) * 0.02
        # ~8% invalid, as a real stereo camera would give on a textureless floor
        holes = (torch.rand(1, 1, h, w) < 0.08)
        depths.append(torch.where(holes, torch.zeros_like(d), d))

    mb = n * (rgbs[0].numel() + depths[0].numel()) * 4 / 1e6
    print(f"pool memory : {mb:.0f} MB  (resized to "
          f"{args.width}x{args.height} inside step())")
    return rgbs, depths


def make_pose(i: int, args) -> torch.Tensor:
    """Pose i of a constant-curvature trajectory, as the model wants it.

    Built through pose_from_optical so the benchmark exercises the same
    conversion the ROS node uses. World frame is arbitrary; only relative
    transforms are used.
    """
    dt = 1.0 / args.fps
    t = i * dt
    yaw = args.yaw_rate * t
    if abs(args.yaw_rate) < 1e-9:
        x, y = args.speed * t, 0.0
    else:
        r = args.speed / args.yaw_rate
        x, y = r * math.sin(yaw), r * (1.0 - math.cos(yaw))

    # Camera looks along the heading. Optical axes: x right, y down, z forward.
    c, s = math.cos(yaw), math.sin(yaw)
    R = torch.tensor([[-s, 0.0, c],
                      [c,  0.0, s],
                      [0.0, -1.0, 0.0]])
    tr = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2]))
    w = tr / 2 if tr > 1e-6 else 1e-6
    q = torch.tensor([[(R[2, 1] - R[1, 2]) / (4 * w),
                       (R[0, 2] - R[2, 0]) / (4 * w),
                       (R[1, 0] - R[0, 1]) / (4 * w), w]])
    return pose_from_optical([x, y, 0.21], q)


def summarise(name: str, samples_ms: list) -> None:
    if not samples_ms:
        print(f"  {name}: no samples")
        return
    s = sorted(samples_ms)
    mean = statistics.fmean(s)
    print(f"  {name:<22} n={len(s):<5} "
          f"mean={mean:7.1f}ms  med={statistics.median(s):7.1f}  "
          f"p90={s[int(0.90*(len(s)-1))]:7.1f}  p99={s[int(0.99*(len(s)-1))]:7.1f}  "
          f"min={s[0]:7.1f}  max={s[-1]:7.1f}   ({1000.0/mean:5.2f} Hz)")


def run_mode(model, pool, args, use_fullgraph: bool) -> dict:
    rgbs, depths = pool
    label = "CUDA graph" if use_fullgraph else "eager"
    print(f"\n=== {label} ===")

    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats(device)
    model.reset_memory()

    interval = args.memory_write_interval
    stream = SAMVTPStreamProcessor(
        model,
        resize_hw=(args.height, args.width),
        use_fullgraph=use_fullgraph,
        threshold=args.threshold,
        input_fit=args.input_fit,
        memory_write_interval_s=interval if interval > 0 else None,
        memory_write_every=max(1, int(args.memory_write_every)),
    )

    inner_ms, outer_ms = [], []
    n_pool = len(rgbs)
    t_start = time.perf_counter()
    captured_at = None

    for i in range(args.frames):
        pose = make_pose(i, args)
        t0 = time.perf_counter()
        result = stream.step(rgbs[i % n_pool], depth=depths[i % n_pool], pose=pose)
        torch.cuda.synchronize(device)
        outer_ms.append((time.perf_counter() - t0) * 1000.0)
        inner_ms.append(result.elapsed_ms)

        if captured_at is None and "replay active" in result.graph_status:
            captured_at = i
            print(f"  frame {i:>6}: graph replay engaged")
        if i < 3 or i in (16, 17, 18):
            print(f"  frame {i:>6}: {result.elapsed_ms:7.1f}ms  "
                  f"fill={result.memory_fill_level:.2f}  {result.graph_status[:60]}")

        if args.report_every and i > 0 and i % args.report_every == 0:
            chunk = outer_ms[-args.report_every:]
            m = statistics.fmean(chunk)
            print(f"  frame {i:>6}: {m:6.1f}ms ({1000.0/m:5.2f} Hz)  "
                  f"max={max(chunk):6.1f}  {read_temps()}  "
                  f"elapsed={time.perf_counter()-t_start:6.0f}s")

    cut = min((captured_at or 16) + 5, args.frames - 1)
    print(f"\n  steady state = frames {cut}..{args.frames - 1}")
    summarise("model only", inner_ms[cut:])
    summarise("end-to-end", outer_ms[cut:])
    print(f"\n  warmup (frames 0..{cut-1}) total: {sum(outer_ms[:cut])/1000.0:.1f}s")
    print(f"  peak GPU mem: {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB")
    print(f"  final status: {stream.graph_status}")

    tail = outer_ms[cut:]
    if len(tail) >= 200:
        tenth = len(tail) // 10
        early, late = statistics.fmean(tail[:tenth]), statistics.fmean(tail[-tenth:])
        print(f"\n  drift: first 10% {early:6.1f}ms -> last 10% {late:6.1f}ms  "
              f"({100.0*(late-early)/early:+.1f}%)")
        if late > early * 1.05:
            print("  ^ slowing over time - likely thermal throttling")

    return {"inner": inner_ms, "outer": outer_ms, "status": stream.graph_status,
            "cut": cut}


def run_breakdown(model, pool, args) -> None:
    """Per-submodule CUDA time. Eager only: hooks do not fire during replay."""
    rgbs, depths = pool
    print("\n=== submodule breakdown (eager) ===")

    totals = {n: 0.0 for n in BREAKDOWN_MODULES}
    counts = {n: 0 for n in BREAKDOWN_MODULES}
    pending, active, handles = {}, {"on": False}, []

    def make_pre(name):
        def hook(_m, _i):
            if active["on"]:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                pending[name] = e
        return hook

    def make_post(name):
        def hook(_m, _i, _o):
            if active["on"] and name in pending:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                torch.cuda.synchronize()
                totals[name] += pending.pop(name).elapsed_time(e)
                counts[name] += 1
        return hook

    for name in BREAKDOWN_MODULES:
        mod = resolve_module(model, name)
        if mod is None:
            print(f"  (no submodule '{name}', skipping)")
            continue
        handles.append(mod.register_forward_pre_hook(make_pre(name)))
        handles.append(mod.register_forward_hook(make_post(name)))

    model.reset_memory()
    stream = SAMVTPStreamProcessor(model, resize_hw=(args.height, args.width),
                                   use_fullgraph=False, threshold=args.threshold,
                                   input_fit=args.input_fit)

    measure_from, wall_ms, n_pool = 16, [], len(rgbs)
    for i in range(args.frames):
        if i == measure_from:
            active["on"] = True
        t0 = time.perf_counter()
        stream.step(rgbs[i % n_pool], depth=depths[i % n_pool], pose=make_pose(i, args))
        if active["on"]:
            wall_ms.append((time.perf_counter() - t0) * 1000.0)

    for h in handles:
        h.remove()

    n = max(1, len(wall_ms))
    wall = sum(wall_ms) / n
    measured = sum(totals.values()) / n

    print(f"\n  measured over {n} steady-state frames")
    print(f"  {'module':<22} {'ms/frame':>10} {'% of wall':>10} {'calls/frame':>12}")
    for name in BREAKDOWN_MODULES:
        if counts[name] == 0:
            continue
        per = totals[name] / n
        print(f"  {name:<22} {per:>10.2f} {100.0*per/wall:>9.1f}% {counts[name]/n:>12.2f}")
    print(f"  {'-'*56}")
    print(f"  {'sum of above':<22} {measured:>10.2f} {100.0*measured/wall:>9.1f}%")
    print(f"  {'wall per frame':<22} {wall:>10.2f}")
    print(f"  {'unattributed':<22} {wall-measured:>10.2f} "
          f"{100.0*(wall-measured)/wall:>9.1f}%")
    print("\n  Unattributed = warp, pre/post-processing, launch overhead and the")
    print("  Python glue between submodules. If it is large, CUDA graphs help")
    print("  more than kernel-level optimisation does.")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available - nothing worth benchmarking")
        return 1
    if not args.checkpoint and not args.random_weights:
        print("pass --checkpoint PATH, or --random-weights to measure throughput "
              "without one")
        return 1

    print(f"torch       : {torch.__version__}")
    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"resolution  : {args.height}x{args.width}")

    device = torch.device("cuda")
    model, cam = build_the_model(args, device)
    print(f"geometry    : {model.use_geometry}   warp: {model.warp_memory}")
    hfov, vfov = cam.fov_deg()
    print(f"intrinsics  : fx={cam.fx:.1f} fy={cam.fy:.1f} @ "
          f"{cam.width}x{cam.height} native  ({hfov:.0f}x{vfov:.0f} deg FOV)")
    if hfov and abs(hfov - 90.0) > 15.0:
        print(f"              ^ training was 90x90 deg. Get real values from "
              f"camera_info;")
        print(f"                the defaults here are a guess and change the "
              f"warp geometry.")

    # Pay lazy cuDNN/workspace init off the clock so it does not land in the
    # first measured frame.
    if hasattr(model, "warmup"):
        t0 = time.perf_counter()
        model.warmup(n=3, hw=(args.height, args.width))
        print(f"warmup      : {time.perf_counter()-t0:.1f}s")

    pool = make_pool(args)

    if args.breakdown:
        run_breakdown(model, pool, args)
        return 0

    results = {}
    if args.mode in ("fullgraph", "both"):
        results["fullgraph"] = run_mode(model, pool, args, use_fullgraph=True)
    if args.mode in ("eager", "both"):
        results["eager"] = run_mode(model, pool, args, use_fullgraph=False)

    if len(results) == 2:
        cut = max(results["fullgraph"]["cut"], results["eager"]["cut"])
        g = statistics.fmean(results["fullgraph"]["outer"][cut:])
        e = statistics.fmean(results["eager"]["outer"][cut:])
        print("\n=== summary ===")
        print(f"  eager      {e:7.1f} ms/frame  ({1000.0/e:5.2f} Hz)")
        print(f"  CUDA graph {g:7.1f} ms/frame  ({1000.0/g:5.2f} Hz)")
        print(f"  speedup    {e/g:7.2f}x")
        if "replay active" not in results["fullgraph"]["status"]:
            print("\n  NOTE: graph capture did NOT engage - the numbers above are")
            print("  eager vs eager. Fix capture before optimising anything else.")
            print(f"  status: {results['fullgraph']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
