#!/usr/bin/env python3
"""Integral-image acceleration for GeNIE's path cost evaluation.

WHY THIS EXISTS

genie_path_planner evaluates path cost by looping over ~100 points per path
and, at each, slicing an 18x18 window and calling .mean(). With a 277-path
bank that is ~28,000 Python-level slice-and-reduce operations per plan, and it
dominates the runtime once the path bank is precomputed.

Both hot functions are BOX FILTERS. Their per-point quantities are:

    path_cost_with_footprint      : mean of exp(alpha * cost) over the window
    has_majority_of_high_cost_points : count of (cost >= threshold) in the window

A summed-area table computed once per plan turns each window into four array
lookups, and the whole path becomes one vectorised gather. Same arithmetic,
same clipping at the borders, same results - just not in Python.

THIS IS NOT AN ALGORITHM CHANGE. The planner's behaviour is untouched: it sees
identical costs and makes identical decisions. verify() below checks that
against the upstream implementations on random inputs, and the ROS node runs
it at startup, so a mismatch is caught immediately rather than silently
changing an experiment.

The same reasoning the paper already applies to the SAM-TP re-implementation
(Sec. III-E1) applies here: this makes the measured planning cost attributable
to the planning computation rather than to Python overhead.

    import genie_fast
    genie_fast.install()        # patches genie_path_planner in place
"""

from __future__ import annotations

import numpy as np


def _integral(a: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero row and column, float64 for accuracy."""
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    np.cumsum(np.cumsum(np.asarray(a, dtype=np.float64), axis=0), axis=1,
              out=out[1:, 1:])
    return out


def _windows(path: np.ndarray, r_half: int, h: int, w: int):
    """Clipped window bounds for every path point, matching upstream exactly.

    Upstream uses int(round(float(x))) - Python's round is banker's rounding,
    so np.round (also half-to-even) matches it. Using floor(x + 0.5) here
    would differ by one pixel on exact .5 values.
    """
    rc = np.asarray(path, dtype=np.float64)
    r = np.round(rc[:, 0]).astype(np.int64)
    c = np.round(rc[:, 1]).astype(np.int64)
    r1 = np.clip(r - r_half, 0, h)
    r2 = np.clip(r + r_half + 1, 0, h)
    c1 = np.clip(c - r_half, 0, w)
    c2 = np.clip(c + r_half + 1, 0, w)
    return r1, r2, c1, c2


def _box_sums(integ: np.ndarray, r1, r2, c1, c2) -> np.ndarray:
    return (integ[r2, c2] - integ[r1, c2] - integ[r2, c1] + integ[r1, c1])


# ---------------------------------------------------------------------------
# Fast replacements
# ---------------------------------------------------------------------------
def fast_compute_paths_costs(candidate_paths, raw_topdown_score, alpha,
                             footprint_px: int = 20):
    """Vectorised equivalent of costs.compute_paths_costs.

    exp(alpha * score) is computed once for the whole grid; upstream computes
    it per window, but exp is elementwise so the values are the same.
    """
    score = np.asarray(raw_topdown_score, dtype=np.float64)
    h, w = score.shape[:2]
    r_half = max(1, int(footprint_px)) // 2
    integ = _integral(np.exp(float(alpha) * score))

    out = []
    for path in candidate_paths:
        if len(path) == 0:
            out.append((0.0, path))
            continue
        r1, r2, c1, c2 = _windows(path, r_half, h, w)
        valid = (r1 < r2) & (c1 < c2)          # upstream skips empty windows
        if not np.any(valid):
            out.append((0.0, path))
            continue
        sums = _box_sums(integ, r1[valid], r2[valid], c1[valid], c2[valid])
        counts = ((r2 - r1) * (c2 - c1))[valid]
        out.append((float(np.sum(sums / counts)), path))
    return out


def fast_has_majority_of_high_cost_points(raw_topdown_score, candidate_path,
                                          num_points, footprint_px: int = 20,
                                          threshold_points_ratio: float = 0.5,
                                          threshold_cost: float = 0.8) -> bool:
    """Vectorised equivalent of path_selection.has_majority_of_high_cost_points.

    Upstream returns True as soon as one point exceeds the ratio, and also on
    the first empty window. Evaluating every point and reducing with any()
    gives the same boolean, since neither condition depends on order.
    """
    score = np.asarray(raw_topdown_score)
    h, w = score.shape[:2]
    r_half = max(1, int(footprint_px)) // 2
    path = np.asarray(candidate_path)[: int(num_points)]
    if len(path) == 0:
        return False

    r1, r2, c1, c2 = _windows(path, r_half, h, w)
    sizes = (r2 - r1) * (c2 - c1)
    if np.any(sizes <= 0):
        return True                     # upstream: empty region -> True

    integ = _integral((score >= float(threshold_cost)).astype(np.float64))
    hits = _box_sums(integ, r1, r2, c1, c2)
    need = np.ceil(float(threshold_points_ratio) * sizes)
    return bool(np.any(hits >= need))


def fast_filter_paths_with_high_costs(candidate_paths, raw_topdown_score,
                                      num_points, footprint_px: int = 20,
                                      threshold_points_ratio: float = 0.05,
                                      threshold_cost: float = 0.8):
    """Vectorised equivalent of path_selection.filter_paths_with_high_costs.

    Patching at the LOOP rather than at the per-path predicate matters: the
    threshold mask and its summed-area table depend only on the score map, so
    building them once for the whole bank instead of once per path removes
    ~99% of the cumsum work (277 integral images per plan, not 1).
    """
    score = np.asarray(raw_topdown_score)
    h, w = score.shape[:2]
    r_half = max(1, int(footprint_px)) // 2
    integ = _integral((score >= float(threshold_cost)).astype(np.float64))

    keep = []
    for path in candidate_paths:
        p = np.asarray(path)[: int(num_points)]
        if len(p) == 0:
            keep.append(path)
            continue
        r1, r2, c1, c2 = _windows(p, r_half, h, w)
        sizes = (r2 - r1) * (c2 - c1)
        if np.any(sizes <= 0):
            continue                    # upstream: empty region -> reject
        hits = _box_sums(integ, r1, r2, c1, c2)
        need = np.ceil(float(threshold_points_ratio) * sizes)
        if not np.any(hits >= need):
            keep.append(path)
    return keep


def set_visualization(enabled: bool) -> None:
    """Enable/disable the BEV overlay render (~16 ms/plan).

    Display only - the overlay is drawn after the path is chosen and has no
    effect on planning. Worth switching off when nothing is subscribed.
    """
    from genie_path_planner import planner as _planner
    if not hasattr(_planner, "_orig_draw_overlay"):
        _planner._orig_draw_overlay = _planner._draw_plan_overlay
    if enabled:
        _planner._draw_plan_overlay = _planner._orig_draw_overlay
    else:
        def _stub(*args, **kwargs):
            return np.zeros((1, 1, 3), dtype=np.uint8)
        _planner._draw_plan_overlay = _stub


# ---------------------------------------------------------------------------
def verify(seed: int = 0, n_paths: int = 40, grid: int = 240,
           tol: float = 1e-6) -> bool:
    """Check the fast paths against upstream on random data. Raises on drift."""
    from genie_path_planner import costs as _costs
    from genie_path_planner import path_selection as _sel

    ref_cost = getattr(_costs, "_orig_compute_paths_costs",
                       _costs.compute_paths_costs)
    ref_major = getattr(_sel, "_orig_has_majority",
                        _sel.has_majority_of_high_cost_points)

    rng = np.random.default_rng(seed)
    score = rng.random((grid, grid)).astype(np.float32)
    paths = []
    for _ in range(n_paths):
        r = np.linspace(grid - 1, rng.integers(0, grid), 100)
        c = np.linspace(grid // 2, rng.integers(0, grid), 100)
        paths.append(np.stack([r, c], axis=1))
    # include paths that run off-grid, where the border clipping matters
    paths.append(np.stack([np.linspace(grid + 40, -40, 100),
                           np.linspace(-30, grid + 30, 100)], axis=1))

    a = [c for c, _ in ref_cost(paths, score, 1.0, 18)]
    b = [c for c, _ in fast_compute_paths_costs(paths, score, 1.0, 18)]
    max_err = max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(a, b))
    if max_err > tol:
        raise RuntimeError(
            f"cost mismatch: relative error {max_err:.2e} > {tol:.0e}. "
            "Refusing to run - this would change planner decisions.")

    for p in paths:
        x = ref_major(score, p, 60, 18, 0.05, 0.5)
        y = fast_has_majority_of_high_cost_points(score, p, 60, 18, 0.05, 0.5)
        if x != y:
            raise RuntimeError("filter mismatch between fast and upstream")

    ref_filter = getattr(_sel, "_orig_filter", _sel.filter_paths_with_high_costs)
    kept_ref = ref_filter(paths, score, 60, 18, 0.05, 0.5)
    kept_fast = fast_filter_paths_with_high_costs(paths, score, 60, 18, 0.05, 0.5)
    if len(kept_ref) != len(kept_fast) or not all(
            np.array_equal(a, b) for a, b in zip(kept_ref, kept_fast)):
        raise RuntimeError("path filter kept a different set of paths")

    print(f"[genie_fast] verified against upstream: "
          f"max relative cost error {max_err:.2e}")
    return True


def install(check: bool = True) -> None:
    """Patch genie_path_planner in place. Idempotent."""
    from genie_path_planner import costs as _costs
    from genie_path_planner import path_selection as _sel
    from genie_path_planner import planner as _planner

    if getattr(_costs, "_genie_fast_installed", False):
        return

    _costs._orig_compute_paths_costs = _costs.compute_paths_costs
    _sel._orig_has_majority = _sel.has_majority_of_high_cost_points
    _sel._orig_filter = _sel.filter_paths_with_high_costs

    if check:
        verify()

    _costs.compute_paths_costs = fast_compute_paths_costs
    _sel.has_majority_of_high_cost_points = fast_has_majority_of_high_cost_points
    _sel.filter_paths_with_high_costs = fast_filter_paths_with_high_costs
    # planner.py bound these by name at import time, so rebind there as well
    for mod, name, fn in (
        (_planner, "compute_paths_costs", fast_compute_paths_costs),
        (_planner, "filter_paths_with_high_costs", fast_filter_paths_with_high_costs),
    ):
        if hasattr(mod, name):
            setattr(mod, name, fn)
    _costs._genie_fast_installed = True
    print("[genie_fast] installed")


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, ".")
    from genie_path_planner import plan_on_bev, PlannerConfig
    from genie_path_planner.path_sampling import sample_paths_polynomial

    cfg = PlannerConfig()
    rng = np.random.default_rng(0)
    bev = np.clip(0.8 + 0.2 * rng.standard_normal((240, 240)), 0, 1).astype(np.float32)
    bev[40:80, 60:100] = 0.0
    obs = np.ones_like(bev, dtype=bool)
    bank = sample_paths_polynomial(
        robot=(239, 120), num_goals=cfg.num_goals,
        num_mid_points_per_goal=cfg.num_mid_points_per_goal,
        num_samples=cfg.path_num_samples, grid_size=240,
        include_random_goals=True, random_seed=42)

    def bench(label):
        plan_on_bev(bev, obs, 0.0, 3.0, 0.03, cfg, candidate_path_bank=bank)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            p = plan_on_bev(bev, obs, 0.0, 3.0, 0.03, cfg, candidate_path_bank=bank)
            ts.append((time.perf_counter() - t0) * 1000)
        print(f"  {label:<24} {np.mean(ts):7.1f} ms  ({1000/np.mean(ts):5.2f} Hz)")
        return p

    before = bench("upstream")
    install()
    after = bench("genie_fast")
    same = np.allclose(before.final_path_xy_m, after.final_path_xy_m)
    print(f"\n  identical final path: {same}")
