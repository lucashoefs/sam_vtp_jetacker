#!/usr/bin/env python3
"""Property tests for the `veto` depth-fusion mode in arc_planner_node.

Runs without ROS: the fusion step is pure numpy over the same
genie_path_planner functions the node calls, so the branch under test is
reproduced here verbatim rather than approximated.

The properties that matter for the closed-loop perception comparison:
  1. a solid object above the height threshold is non-traversable regardless
     of what the model says (the safety backstop survives);
  2. depth NEVER makes a cell more traversable or marks it observed, so it
     cannot manufacture free space the model did not see;
  3. on flat ground the cost surface is the model's, untouched -- this is what
     rgbd destroys and what the comparison depends on;
  4. two models that differ on flat terrain differ by the SAME amount after
     veto fusion, and by rgb_weight x that amount after rgbd fusion.
"""
from __future__ import annotations

import numpy as np

from genie_path_planner import (
    blend_modalities,
    depth_to_bev_height_and_traversability,
    project_score_to_bev,
)

GROUND_Z = 0.0
BEV_RES, BEV_FWD, BEV_SIDE = 0.03, 3.0, 4.0
RELIABLE_D, MIN_D, OBST_H = 2.4, 0.2, 0.20
MAX_RAY = 6.0
RGB_W, DEPTH_W = 0.2, 0.8

H, W = 120, 212
K = np.array([[428.5935 / 4, 0.0, 424.6508 / 4],
              [0.0, 427.9190 / 4, 249.5679 / 4],
              [0.0, 0.0, 1.0]])


def camera_pose(height_m: float = 0.30, tilt_deg: float = 10.0) -> np.ndarray:
    """Camera at `height_m`, tilted `tilt_deg` DOWN. Optical frame -> world.

    Built column-wise from the three optical axes expressed in world
    coordinates, so the convention is explicit rather than inherited from a
    rotation-order guess:
        optical +x (right)   -> world -y
        optical +y (down)    -> mostly world -z
        optical +z (forward) -> world +x, tilted down by `tilt_deg`
    """
    t = np.radians(tilt_deg)
    fwd = np.array([np.cos(t), 0.0, -np.sin(t)])     # optical z
    right = np.array([0.0, -1.0, 0.0])               # optical x
    down = np.cross(fwd, right)                      # optical y, right-handed
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, fwd], axis=1)
    T[:3, 3] = [0.0, 0.0, height_m]
    return T


def synth_depth(box_rows: slice, box_cols: slice, box_dist: float) -> np.ndarray:
    """Flat ground everywhere, with a solid block at a fixed range.

    A camera-frame point at depth Z projects to d_opt * Z with
    d_opt = ((u-cx)/fx, (v-cy)/fy, 1), so the ground hit is
    Z = -height / (R @ d_opt)_z, defined only where the ray points downward.
    """
    pose = camera_pose()
    ys, xs = np.indices((H, W), dtype=np.float64)
    d_opt = np.stack([(xs - K[0, 2]) / K[0, 0],
                      (ys - K[1, 2]) / K[1, 1],
                      np.ones_like(xs)], axis=-1)
    d_world = d_opt @ pose[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -(pose[2, 3] - GROUND_Z) / d_world[..., 2]
    # rays that point up (or level) never meet the ground: unmeasurable, which
    # a real depth camera reports as far/invalid rather than as a ground hit
    z = np.where((d_world[..., 2] < -1e-9) & (z > 0), z, 10.0)
    depth = np.clip(z, 0.0, 10.0)
    depth[box_rows, box_cols] = box_dist              # the obstacle
    return depth.astype(np.float64)


def fuse(mode: str, score: np.ndarray, depth: np.ndarray):
    """Exactly the branch in arc_planner_node._plan_once.

    Returns (bev, observed, rgb_observed, depth_observed) so the tests can
    separate the zone where the two modalities actually meet from the much
    larger area only one of them reaches.
    """
    pose = camera_pose()
    rgb_bev, rgb_obs, _ = project_score_to_bev(
        score, K, pose, GROUND_Z, BEV_RES, BEV_FWD, BEV_SIDE, MAX_RAY)
    rgb_obs = np.asarray(rgb_obs, dtype=bool)
    if mode == "rgb":
        return np.asarray(rgb_bev, np.float32), rgb_obs, rgb_obs, None

    height, d_bev, d_obs, _ = depth_to_bev_height_and_traversability(
        depth, K, pose, GROUND_Z, RELIABLE_D, MIN_D, OBST_H,
        BEV_RES, BEV_FWD, BEV_SIDE)
    d_obs = np.asarray(d_obs, dtype=bool)

    if mode == "rgbd":
        bev, obs = blend_modalities(rgb_bev, rgb_obs, d_bev, d_obs,
                                    RGB_W, DEPTH_W)
        return np.asarray(bev, np.float32), np.asarray(obs, bool), rgb_obs, d_obs

    bev = np.array(rgb_bev, dtype=np.float32, copy=True)
    obs = rgb_obs.copy()
    tall = d_obs & np.isfinite(height) & (height >= OBST_H)
    bev[tall] = 0.0
    obs[tall] = True
    return bev, obs, rgb_obs, d_obs


def arc_corridor_mask(bev_shape) -> np.ndarray:
    """Cells an arc can actually sweep: 0-2.5 m forward, +/-0.5 m lateral.

    The comparison is decided here, not over the whole 3 x 8 m BEV -- most of
    which no arc ever touches.
    """
    bev_h, bev_w = bev_shape
    rows, cols = np.indices((bev_h, bev_w))
    forward_m = (bev_h - 1 - rows) * BEV_RES
    left_m = ((bev_w // 2) - cols) * BEV_RES
    return (forward_m <= 2.5) & (np.abs(left_m) <= 0.5)


def main() -> int:
    ok = True
    depth = synth_depth(slice(20, 60), slice(80, 130), 1.2)

    # 1. safety backstop, against a model that is wrong ------------------
    optimistic = np.full((H, W), 0.95, dtype=np.float32)
    bev_v, obs_v, rgb_obs, _ = fuse("veto", optimistic, depth)
    bev_r, obs_r, _, _ = fuse("rgb", optimistic, depth)
    n_veto = int(np.count_nonzero(obs_v & (bev_v <= 1e-6)))
    n_rgb = int(np.count_nonzero(obs_r & (bev_r <= 1e-6)))
    print("1. solid block at 1.2 m, model says 0.95 (traversable) everywhere")
    print(f"   veto: {n_veto} cells non-traversable | score-only: {n_rgb}")
    if n_veto == 0:
        print("   FAIL: backstop did not fire"); ok = False
    else:
        print("   OK: geometric backstop overrides an optimistic model")

    # 2. depth is strictly add-only --------------------------------------
    # rgb_bev uses -1.0 for "unobserved", so the comparison is only meaningful
    # where RGB actually observed the cell; newly-observed cells are checked
    # separately, and must all be vetoed.
    raised = int(np.count_nonzero((bev_v > bev_r + 1e-6) & rgb_obs))
    newly = obs_v & ~rgb_obs
    newly_all_vetoed = bool(np.all(bev_v[newly] <= 1e-6)) if newly.any() else True
    print("2. depth may add obstacles, never clear them")
    print(f"   traversability raised on RGB-observed cells: {raised} (must be 0)")
    print(f"   cells newly observed via depth: {int(newly.sum())}, "
          f"all vetoed: {newly_all_vetoed}")
    if raised != 0 or not newly_all_vetoed:
        print("   FAIL: depth cleared or silently observed a cell"); ok = False
    else:
        print("   OK: add-only holds")

    # 3. does a difference between two models survive fusion? ------------
    flat = synth_depth(slice(0, 0), slice(0, 0), 1.2)
    a = np.full((H, W), 0.80, dtype=np.float32)
    b = np.full((H, W), 0.30, dtype=np.float32)     # model B sees a mud patch
    a_v, _, _, d_obs = fuse("veto", a, flat)
    b_v, _, rgb_o, _ = fuse("veto", b, flat)
    a_d, _, _, _ = fuse("rgbd", a, flat)
    b_d, _, _, _ = fuse("rgbd", b, flat)

    both = rgb_o & d_obs                    # where the modalities actually meet
    gap_v = float(np.mean(a_v[both] - b_v[both]))
    gap_d = float(np.mean(a_d[both] - b_d[both]))
    print("3. two models differing by 0.50 on flat ground, in the blended zone")
    print(f"   veto preserves {gap_v:.4f} | rgbd preserves {gap_d:.4f} "
          f"(rgb_weight x 0.50 = {RGB_W * 0.50:.4f})")
    if abs(gap_v - 0.50) > 1e-3:
        print("   FAIL: veto altered the model's cost on flat ground"); ok = False
    elif abs(gap_d - RGB_W * 0.50) > 1e-3:
        print("   FAIL: rgbd attenuation is not rgb_weight"); ok = False
    else:
        print(f"   OK: rgbd attenuates {gap_v / max(gap_d, 1e-9):.0f}x, "
              "veto passes it through intact")

    # 4. how much of the region arcs sweep is actually blended? ----------
    corridor = arc_corridor_mask(a_v.shape)
    in_corr = corridor & rgb_o
    blended = corridor & rgb_o & d_obs
    frac = float(blended.sum()) / max(int(in_corr.sum()), 1)
    print("4. share of the arc-swept corridor (0-2.5 m fwd, +/-0.5 m) where "
          "depth is present")
    print(f"   {blended.sum()}/{in_corr.sum()} cells = {frac:.1%} blended under rgbd")
    print(f"   -> under rgbd the model owns {1 - frac + frac * RGB_W:.0%} of the "
          f"corridor cost; under veto it owns 100% outside vetoed cells")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
