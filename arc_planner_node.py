#!/usr/bin/env python3
"""Fast arc planner: constant-curvature local planning on the GeNIE BEV.

WHY REPLACE GENIE'S PATH SEARCH

GeNIE's planner takes ~515 ms on this hardware, so it replans at 2 Hz. At
0.25 m/s that is a decision every 12 cm, and reactions arrive half a second
late - late enough to contact obstacles. A closed-loop comparison run that way
measures planner latency, not perception quality, which is the opposite of
what the experiment is for.

WHAT IS KEPT AND WHAT IS REPLACED

  KEPT      GeNIE's BEV projection: traversability and depth into a top-down
            costmap, and the blend between them. That is the perception ->
            costmap stage, and it is what the models are being compared
            through.
  REPLACED  Path sampling, cost filtering, k-means clustering and the top-k
            blend. In their place: a fan of constant-curvature arcs, each
            scored once, best one wins.

Constant-curvature arcs are also a better fit than GeNIE's polynomials here.
An Ackermann chassis at a fixed steering angle traces exactly an arc, so every
candidate is executable by construction - no curvature filtering needed, and
no gap between what is planned and what the linkage can do.

TWO PROPERTIES THAT MATTER FOR THE EXPERIMENT

  * NO TEMPORAL SMOOTHING. Each frame is scored independently. Filtering
    across frames would suppress exactly the costmap instability the
    experiment sets out to measure.
  * NO BLENDING. GeNIE averages its top 12 paths, weighted by cost, so the
    output is not a path it evaluated; small changes in which paths make the
    top 12 shift the whole blend. Here the winner is a real arc, and it is
    reported with its curvature so switching is directly measurable.

Publishes the same topics as genie_planner_node, so nothing downstream changes.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, Header, String
from tf2_ros import Buffer, TransformListener

import cv2

from genie_path_planner import (
    blend_modalities,
    depth_to_bev_height_and_traversability,
    project_score_to_bev,
)


def clearance_edt(free: np.ndarray, cell_m: float) -> np.ndarray:
    """Metres from every free cell to the nearest blocked one.

    Was `scipy.ndimage.distance_transform_edt`. Dropped because scipy is not a
    dependency this stack can rely on: on the Jetson the venv's `scipy.ndimage`
    imports but has no `distance_transform_edt`, which raised once per planning
    cycle and stopped the planner dead -- including before it published /bev, so
    the failure also took out the one view that would have shown what was wrong.
    OpenCV is already a hard dependency through cv_bridge, so this cannot fail
    without the node failing first.

    Same function, same semantics: distance from each non-zero cell to the
    nearest zero. Measured against scipy on the real 100x267 grid, max|diff| was
    2e-7 m sparse, 5e-8 m dense, 8e-6 m for a single obstacle -- float32 rounding
    against a 3 cm cell -- and it is ~6.5x faster (0.095 ms vs 0.615 ms).

    The exception is an empty `block_mask`. scipy then returns a bounded but
    arbitrary field; cv2 saturates at 65535, which is 1966 m and would leak an
    absurd clearance into the cost. Nothing is blocking, so the honest answer is
    "further than this grid can express": the grid diagonal.
    """
    if not np.any(~free):
        h, w = free.shape
        return np.full(free.shape, float(np.hypot(h, w) * cell_m), dtype=np.float32)
    return cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE) * cell_m


def quat_to_matrix(x, y, z, w) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def integral(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    np.cumsum(np.cumsum(np.asarray(a, dtype=np.float64), axis=0), axis=1,
              out=out[1:, 1:])
    return out


class ArcPlanner(Node):
    def __init__(self) -> None:
        super().__init__("arc_planner")

        self.declare_parameter("config", "")
        self.declare_parameter("score_topic", "/sam_vtp/probability")
        self.declare_parameter("depth_topic",
                               "/camera/d455/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/d455/color/camera_info")
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("plan_rate", 10.0)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("ground_z", 0.0)
        # Subsample the image before projection.
        #
        # project_score_to_bev unprojects EVERY pixel: 848x480 is 407k points
        # binned into a 240x240 grid of 57.6k cells. At 2 m a pixel subtends
        # ~4.7 mm against 30 mm cells, so the input is ~6x oversampled per
        # axis - about 36x more work than the grid can represent. Measured at
        # 200-500 ms on this hardware, which dwarfs the 2 ms arc search.
        #
        # Stride 3 still gives ~14 mm per pixel at 2 m, finer than a cell, so
        # nothing the BEV can express is lost.
        self.declare_parameter("projection_stride", 3)

        # ---- arc fan ----------------------------------------------------
        self.declare_parameter("num_arcs", 31)
        self.declare_parameter("min_turn_radius_m", 0.30)
        self.declare_parameter("horizon_m", 2.5)
        self.declare_parameter("arc_samples", 40)
        # The robot's own width. This is the only obstacle clearance there is.
        # Measured body width (0.25 m) plus 3 cm. This is the robot, not a
        # safety envelope.
        self.declare_parameter("footprint_m", 0.28)
        # Cap on how far around each arc sweeps. Without it, sampling a fixed
        # ARC LENGTH means a tight curvature wraps past a full circle: at
        # r=0.30 m, 2.5 m of arc is 477 degrees, and the candidate curls back
        # behind the robot. Capping the swept angle keeps every arc a forward
        # manoeuvre, and short tight turns are what they physically are.
        # 90 deg: the point at which an arc stops making forward progress
        # (x = r sin(theta) peaks there). Past it the candidate curls back,
        # which is never a useful manoeuvre for this task.
        self.declare_parameter("max_arc_sweep_deg", 90.0)
        # Curvature spacing exponent. LINEAR spacing (1.0) puts most of the
        # fan at tight radii, and tight arcs are physically short - at
        # r = 0.30 m an arc is 0.47 m long. The result is a planner with
        # almost no gentle long-range options: it can go straight for 2.5 m
        # or commit to a hard turn that reaches 0.5 m, and nothing in
        # between. So it drives at an obstacle it saw metres back and then
        # swerves.
        #
        # An exponent of 2 clusters candidates near straight, where the long
        # arcs live, while keeping a few tight ones for genuine emergencies.
        self.declare_parameter("curvature_exponent", 2.0)

        # ---- scoring ----------------------------------------------------
        self.declare_parameter("unknown_cost", 0.45)
        # cost >= this counts as impassable. 0.75 means traversability <=
        # 0.25, i.e. only confidently bad terrain. 0.60 proved too strict on
        # real data: the model scores texture changes (a different strip of
        # carpet, a shadow line) around 0.3-0.4, and close to the robot there
        # is no depth to outvote it - the blind zone and reliable_depth_m
        # leave near-field cells RGB-only, so the 0.8 depth weight does not
        # apply where it would have helped most.
        self.declare_parameter("obstacle_cost", 0.75)
        # Fraction of a footprint that must be impassable before the arc is
        # cut. `any single cell` is far too brittle: a two-pixel strip across
        # an 11x11 window stopped the entire fan. A real obstacle covers a
        # large share of the footprint; noise and thin strips do not.
        self.declare_parameter("block_ratio", 0.25)
        # Unobserved cells STOP an arc, beyond the near-field blind zone.
        #
        # Without this, occlusion shadows read as free space: the region
        # beside and behind an obstacle is unobserved, scores below the
        # obstacle threshold, and an arc runs straight through a gap that
        # only exists because the robot cannot see into it. That is how the
        # robot both drove into the beanbag and tried to thread a gap it does
        # not fit through - the same cause, twice.
        #
        # With it, an arc reaches exactly as far as the robot can see, which
        # is the honest horizon for a forward camera with no BEV memory.
        self.declare_parameter("unknown_blocks", True)
        # Inside this radius the ground is below the camera's field of view
        # and is unobserved by geometry, not by occlusion. Blocking there
        # would stop every arc at the first sample.
        # Was 0.45: the camera only sees ground from ~0.33 m, and the bumper
        # check starts 0.30 m ahead, so there was almost no margin and
        # near-field noise in the observed mask blocked everything.
        self.declare_parameter("blind_radius_m", 0.70)
        # Extra clearance for the body sweeping wider than base_link on a
        # turn. The chassis is ~0.4 m long, so the corners cut outside the
        # rear-axle path; a point footprint along that path underestimates
        # the swept area.
        self.declare_parameter("turn_margin_m", 0.10)
        # Distance from base_link (the REAR AXLE) to the front bumper.
        #
        # The arcs describe where base_link goes, but the body extends ahead
        # of it. Checking the footprint along the rear-axle path means the
        # bumper is already this far into an obstacle by the time the check
        # trips - which is precisely how the robot drives into things that
        # the BEV shows correctly as red.
        #
        # 0.216 m wheelbase + front overhang. Measure from the rear axle to
        # the foremost point of the chassis.
        # Measured: rear axle to the foremost point of the chassis.
        self.declare_parameter("front_offset_m", 0.27)
        # Hard clearance beyond the body half-width. The footprint alone leaves
        # only (footprint/2 - half_width) = ~4 cm, which is less than the ~8 cm
        # the robot travels during the ~320 ms reaction latency. An arc is
        # blocked if anything comes within body half-width + this.
        # Small buffer for BEV discretisation only - the footprint already
        # encodes the robot's actual width, and clearance should come from
        # planning an early gentle arc, not from padding obstacles until the
        # robot cannot fit through gaps it physically fits through.
        self.declare_parameter("safety_margin_m", 0.03)
        # Among arcs that are NOT blocked, prefer the ones that keep their
        # distance. Without this the planner is indifferent between skimming
        # an obstacle and passing it with room, so it skims - every time,
        # because a shorter route scores marginally better.
        # A mild preference, not a hard requirement: enough to break ties
        # toward the roomier of two viable arcs without refusing tight gaps.
        self.declare_parameter("clearance_pref_m", 0.15)
        self.declare_parameter("clearance_weight", 0.4)
        # Reward for PROGRESS along the goal bearing, in place of a penalty
        # on deviating from it.
        #
        # A fixed penalty on bearing error is what makes a reactive planner
        # walk into obstacles and stop: each cycle it nudges off-bearing, the
        # penalty pulls it back, and it never commits to going around. Scoring
        # progress instead - how far the arc gets, projected onto the desired
        # heading - makes the trade-off automatic. An arc deviating 45 degrees
        # but running 2.5 m makes 1.8 m of progress; a straight arc blocked at
        # 0.5 m makes 0.5 m, so the detour wins without a special case.
        self.declare_parameter("progress_weight", 1.5)
        # Small preference for going straight, to break ties on open ground
        # where the cost surface is nearly flat.
        self.declare_parameter("curvature_weight", 0.15)
        self.declare_parameter("goal_distance_m", 3.0)
        self.declare_parameter("goal_mode", "heading")
        self.declare_parameter("max_heading_error_deg", 80.0)
        self.declare_parameter("publish_bev", True)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.world_frame = str(g("world_frame"))
        self.base_frame = str(g("base_frame"))
        self.depth_scale = float(g("depth_scale"))
        self.ground_z = float(g("ground_z"))
        self.goal_distance = float(g("goal_distance_m"))
        self.goal_mode = str(g("goal_mode"))
        self.max_heading_err = math.radians(float(g("max_heading_error_deg")))
        self.unknown_cost = float(g("unknown_cost"))
        self.obstacle_cost = float(g("obstacle_cost"))
        self.w_progress = float(g("progress_weight"))
        self.w_curv = float(g("curvature_weight"))
        self.footprint_m = float(g("footprint_m"))
        self.unknown_blocks = bool(g("unknown_blocks"))
        self.block_ratio = float(g("block_ratio"))
        self.blind_radius = float(g("blind_radius_m"))
        self.turn_margin = float(g("turn_margin_m"))
        self.safety_margin = float(g("safety_margin_m"))
        self.clearance_pref = float(g("clearance_pref_m"))
        self.w_clear = float(g("clearance_weight"))
        self.publish_bev = bool(g("publish_bev"))
        self.stride = max(1, int(g("projection_stride")))

        # projection settings shared with the GeNIE planner
        raw = yaml.safe_load(open(str(g("config")))) if g("config") else {}
        proj, dep, plan = (raw.get("projection", {}), raw.get("depth", {}),
                           raw.get("planning", {}))
        self.bev_res = float(proj.get("bev_resolution_m_per_px", 0.03))
        self.bev_fwd = float(proj.get("bev_forward_range_m", 4.0))
        self.bev_side = float(proj.get("bev_side_range_m", 4.0))
        self.max_ray = float(proj.get("max_ray_distance_m", 6.0))
        self.reliable_depth = float(dep.get("reliable_depth_m", 2.4))
        self.min_depth = float(dep.get("min_depth_m", 0.2))
        self.obstacle_h = float(dep.get("obstacle_height_thresh_m", 0.20))
        self.rgb_w = float(plan.get("rgb_weight", 0.2))
        self.depth_w = float(plan.get("depth_weight", 0.8))
        self.mode = str(plan.get("mode", "rgbd"))

        self._build_arcs(int(g("num_arcs")), float(g("min_turn_radius_m")),
                         float(g("horizon_m")), int(g("arc_samples")),
                         math.radians(float(g("max_arc_sweep_deg"))),
                         float(g("front_offset_m")),
                         float(g("curvature_exponent")))

        # ---- state -------------------------------------------------------
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cb_group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.latest = None
        self.last_planned_key: Optional[int] = None
        self.skipped_stale = 0
        self._depth: Optional[np.ndarray] = None
        self.camera_k: Optional[np.ndarray] = None
        self.goal_world: Optional[np.ndarray] = None
        self.ref_yaw: Optional[float] = None
        self.n_plans = 0
        self.no_path = 0
        self.last_kappa: Optional[float] = None
        self.kappa_switches = 0
        self._last_reach = 0.0
        self._last_clear = 0.0
        self._last_progress = 0.0
        self._last_best = 0

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CameraInfo, str(g("info_topic")),
                                 self._on_info, 1, callback_group=self.cb_group)
        self.create_subscription(Image, str(g("score_topic")),
                                 self._on_score, qos, callback_group=self.cb_group)
        self.create_subscription(Image, str(g("depth_topic")),
                                 self._on_depth, qos, callback_group=self.cb_group)
        self.create_subscription(PoseStamped, "goal_pose", self._on_goal, 1,
                                 callback_group=self.cb_group)
        self.create_subscription(Bool, "reset_heading", self._on_reset, 1,
                                 callback_group=self.cb_group)

        self.pub_path = self.create_publisher(Path, "~/path", 1)
        self.pub_bev = self.create_publisher(Image, "~/bev", 1)
        self.pub_plan_ms = self.create_publisher(Float32, "~/plan_ms", 10)
        self.pub_proj_ms = self.create_publisher(Float32, "~/project_ms", 10)
        self.pub_kappa = self.create_publisher(Float32, "~/curvature", 10)
        self.pub_switch = self.create_publisher(Float32, "~/heading_change_deg", 10)
        self.pub_no_path = self.create_publisher(Bool, "~/no_path", 1)
        self.pub_reach = self.create_publisher(Float32, "~/mean_reach", 10)
        self.pub_clear = self.create_publisher(Float32, "~/min_clearance_m", 10)
        self.pub_progress = self.create_publisher(Float32, "~/progress_m", 10)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        self.pub_stamp = self.create_publisher(Header, "~/stamp", 10)

        self.plan_period = 1.0 / float(g("plan_rate"))
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        self.create_timer(1.0, self._health, callback_group=self.cb_group)

    # ------------------------------------------------------------------
    def _build_arcs(self, n: int, r_min: float, horizon: float,
                    samples: int, sweep_max: float, front_offset: float,
                    exponent: float = 2.0) -> None:
        """Precompute the arc fan in metres, robot-centred, x forward y left.

        Each arc runs for min(horizon, sweep_max / |kappa|) of arc length, so
        no candidate sweeps further round than sweep_max. Straight arcs get
        the full horizon; the tightest get much less, which is correct - a
        0.30 m radius turn only carries you about half a metre before you have
        turned 75 degrees, and anything beyond that is a different manoeuvre,
        not a longer version of the same one.

        Every arc is executable at a fixed steering angle by construction, so
        there is no separate feasibility filter.
        """
        k_max = 1.0 / r_min
        # Non-linear spacing: dense near straight, sparse at full lock.
        t = np.linspace(-1.0, 1.0, n)
        self.kappas = k_max * np.sign(t) * np.abs(t) ** exponent

        self.arcs = np.zeros((n, samples, 2), dtype=np.float64)
        self.check_pts = np.zeros((n, samples, 2), dtype=np.float64)
        self.arc_lengths = np.zeros(n, dtype=np.float64)
        for i, k in enumerate(self.kappas):
            s_max = horizon if abs(k) < 1e-6 else min(horizon, sweep_max / abs(k))
            self.arc_lengths[i] = s_max
            s = np.linspace(0.0, s_max, samples)
            if abs(k) < 1e-6:
                x, y = s, np.zeros_like(s)
            else:
                r = 1.0 / k
                th = s * k
                x = r * np.sin(th)
                y = r * (1.0 - np.cos(th))
            self.arcs[i, :, 0] = x
            self.arcs[i, :, 1] = y
            # Where the FRONT of the robot is when base_link is at each arc
            # point: offset along the local heading, which on an arc is the
            # swept angle so far. This is what gets collision-checked.
            self.check_pts[i, :, 0] = x + front_offset * np.cos(th if abs(k) > 1e-6
                                                               else np.zeros_like(s))
            self.check_pts[i, :, 1] = y + front_offset * np.sin(th if abs(k) > 1e-6
                                                                else np.zeros_like(s))

        self.get_logger().info(
            f"arc fan: {n} arcs, |kappa| <= {k_max:.2f} 1/m (R >= {r_min} m), "
            f"sweep <= {math.degrees(sweep_max):.0f} deg, "
            f"length {self.arc_lengths.min():.2f}-{self.arc_lengths.max():.2f} m, "
            f"collision-checked {front_offset:.2f} m ahead of base_link; "
            f"{int(np.sum(self.arc_lengths > 0.8 * horizon))}/{n} arcs reach "
            f"80% of the horizon")

    # ------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        if self.camera_k is None:
            self.camera_k = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _on_depth(self, msg: Image) -> None:
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self._depth = d.astype(np.float32) * self.depth_scale

    def _on_score(self, msg: Image) -> None:
        s = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        with self.lock:
            self.latest = (s.astype(np.float32) / 255.0, self._depth, msg.header)

    def _on_goal(self, msg: PoseStamped) -> None:
        self.goal_world = np.array([msg.pose.position.x, msg.pose.position.y])

    def _on_reset(self, msg: Bool) -> None:
        if msg.data:
            self.ref_yaw = None
            self.get_logger().info("heading reference cleared")

    # ------------------------------------------------------------------
    def _camera_pose(self, header) -> Optional[np.ndarray]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, header.frame_id, header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.0))
        except Exception:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.world_frame, header.frame_id, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.0))
            except Exception:
                return None
        t, r = tf.transform.translation, tf.transform.rotation
        T = np.eye(4)
        T[:3, :3] = quat_to_matrix(r.x, r.y, r.z, r.w)
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def _base_yaw(self) -> Optional[float]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.base_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.0))
        except Exception:
            return None
        r = tf.transform.rotation
        return math.atan2(2.0 * (r.w * r.z + r.x * r.y),
                          1.0 - 2.0 * (r.y * r.y + r.z * r.z))

    def _goal_bearing(self) -> float:
        """Desired bearing in base_link: +left, radians."""
        if self.goal_mode == "heading":
            yaw = self._base_yaw()
            if yaw is None:
                return 0.0
            if self.ref_yaw is None:
                self.ref_yaw = yaw
                self.get_logger().info(
                    f"heading reference: {math.degrees(yaw):+.1f} deg")
            err = math.atan2(math.sin(self.ref_yaw - yaw),
                             math.cos(self.ref_yaw - yaw))
            return max(-self.max_heading_err, min(self.max_heading_err, err))
        return 0.0

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            t0 = time.perf_counter()
            try:
                self._plan_once()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"planning failed: {exc}")
            rem = self.plan_period - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

    def _plan_once(self) -> None:
        with self.lock:
            if self.latest is None or self.camera_k is None:
                return
            score, depth, header = self.latest

        # Only plan on a mask we have not already planned on.
        #
        # Without this the timer re-plans on whatever is in `latest`, so when
        # perception stalls the planner keeps republishing paths carrying the
        # OLD frame stamp. Downstream that looks like planning taking seconds
        # - the profiler showed perception frozen at 179 ms while "planning"
        # ramped 240 -> 1875 ms on the same frame - and pure pursuit tracks a
        # path built from a view of the world that has since moved.
        key = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
        if key == self.last_planned_key:
            self.skipped_stale += 1
            return
        self.last_planned_key = key

        pose = self._camera_pose(header)
        if pose is None:
            return

        # ---- projection (GeNIE's, subsampled) ---------------------------
        t0 = time.perf_counter()
        if self.stride > 1:
            n = self.stride
            score_p = np.ascontiguousarray(score[::n, ::n])
            depth_p = (np.ascontiguousarray(depth[::n, ::n])
                       if depth is not None else None)
            # Subsampling by n maps old pixel u to new u/n, so focal length
            # and principal point both scale by 1/n. Exact for this indexing.
            K = self.camera_k.copy()
            K[0, 0] /= n; K[1, 1] /= n
            K[0, 2] /= n; K[1, 2] /= n
        else:
            score_p, depth_p, K = score, depth, self.camera_k

        rgb_bev, rgb_obs, _ = project_score_to_bev(
            score_p, K, pose, self.ground_z,
            self.bev_res, self.bev_fwd, self.bev_side, self.max_ray)
        if self.mode == "rgbd" and depth_p is not None:
            _, d_bev, d_obs, _ = depth_to_bev_height_and_traversability(
                depth_p, K, pose, self.ground_z,
                self.reliable_depth, self.min_depth, self.obstacle_h,
                self.bev_res, self.bev_fwd, self.bev_side)
            bev, obs = blend_modalities(rgb_bev, rgb_obs, d_bev, d_obs,
                                        self.rgb_w, self.depth_w)
        else:
            bev, obs = rgb_bev, rgb_obs
        t_proj = (time.perf_counter() - t0) * 1000.0

        # ---- arc scoring -------------------------------------------------
        t1 = time.perf_counter()
        cost = np.where(np.asarray(obs, dtype=bool),
                        1.0 - np.clip(np.nan_to_num(bev, nan=0.0), 0.0, 1.0),
                        self.unknown_cost).astype(np.float64)
        h, w = cost.shape
        # Robot sits bottom-centre; +x forward is up the rows, +y left is
        # toward lower columns.
        r0, c0 = h - 1, w // 2
        # Collision is checked at the bumper, not at the rear axle.
        rows = r0 - self.check_pts[:, :, 0] / self.bev_res
        cols = c0 - self.check_pts[:, :, 1] / self.bev_res
        # The published path still follows base_link - that is what pure
        # pursuit tracks.
        path_rows = r0 - self.arcs[:, :, 0] / self.bev_res
        path_cols = c0 - self.arcs[:, :, 1] / self.bev_res

        # Per-arc footprint: straight arcs need only the body width, curved
        # ones need the extra the corners sweep.
        widths = self.footprint_m + self.turn_margin * (
            np.abs(self.kappas) / max(np.abs(self.kappas).max(), 1e-6))
        halves = np.maximum(1, np.round(widths / self.bev_res / 2).astype(int))
        half = int(halves.max())
        integ = integral(cost)

        obstacle_mask = cost >= self.obstacle_cost
        unknown_mask = np.zeros_like(obstacle_mask)
        block_mask = obstacle_mask.copy()
        if self.unknown_blocks:
            # Distance of every cell from the robot, so the blind zone right
            # in front is exempted - that is unobserved by geometry, not by
            # occlusion, and blocking it would stop every arc immediately.
            rr_g, cc_g = np.mgrid[0:h, 0:w]
            dist = np.hypot((r0 - rr_g), (c0 - cc_g)) * self.bev_res
            unknown_mask = (~np.asarray(obs, dtype=bool)) & (dist > self.blind_radius)
            block_mask |= unknown_mask
        integ_obs = integral(block_mask.astype(np.float64))

        # Metres from every free cell to the nearest blocking cell. One EDT
        # per frame (~0.1 ms at 100x267) replaces a per-arc search.
        clearance_map = clearance_edt(~block_mask, self.bev_res)
        # Second table over obstacles only, so a failure can be attributed to
        # a real obstacle rather than to unobserved space. Guessing which of
        # the two is responsible has cost several iterations.
        integ_hard = integral(obstacle_mask.astype(np.float64))

        rr = np.round(rows).astype(np.int64)
        cc = np.round(cols).astype(np.int64)
        # Hard block uses body half-width PLUS the safety margin; terrain
        # cost still uses the bare footprint, so a wide margin does not
        # inflate the cost of merely passing near something.
        block_half = np.maximum(
            1, np.round((widths / 2 + self.safety_margin) / self.bev_res).astype(int))
        hh = halves[:, None]
        r1 = np.clip(rr - hh, 0, h)
        r2 = np.clip(rr + hh + 1, 0, h)
        c1 = np.clip(cc - hh, 0, w)
        c2 = np.clip(cc + hh + 1, 0, w)
        area = np.maximum((r2 - r1) * (c2 - c1), 1)

        box = (integ[r2, c2] - integ[r1, c2] - integ[r2, c1] + integ[r1, c1])
        mean_cost = box / area
        bh = block_half[:, None]
        br1 = np.clip(rr - bh, 0, h)
        br2 = np.clip(rr + bh + 1, 0, h)
        bc1 = np.clip(cc - bh, 0, w)
        bc2 = np.clip(cc + bh + 1, 0, w)
        hits = (integ_obs[br2, bc2] - integ_obs[br1, bc2]
                - integ_obs[br2, bc1] + integ_obs[br1, bc1])
        hard = (integ_hard[r2, c2] - integ_hard[r1, c2]
                - integ_hard[r2, c1] + integ_hard[r1, c1])

        # An arc is cut at the first sample where a large enough SHARE of its
        # footprint is impassable. Everything past that point is unreachable,
        # so it does not contribute to the score.
        blocked = hits >= (self.block_ratio * area)
        first_block = np.where(blocked.any(axis=1), blocked.argmax(axis=1),
                               self.arcs.shape[1])
        hard_blocked = hard > 0
        first_hard = np.where(hard_blocked.any(axis=1), hard_blocked.argmax(axis=1),
                              self.arcs.shape[1])
        idx = np.arange(self.arcs.shape[1])[None, :]
        valid = idx < first_block[:, None]
        n_valid = np.maximum(valid.sum(axis=1), 1)

        # Mean terrain cost over the part of the arc actually reachable.
        # Distance now enters through the progress term, not as a separate
        # reach penalty, so it is not counted twice.
        terrain = (mean_cost * valid).sum(axis=1) / n_valid
        reached_m = (first_block / float(self.arcs.shape[1])) * self.arc_lengths
        reach = reached_m / max(self.arc_lengths.max(), 1e-6)

        # Per-arc immediate-contact threshold, needed by the failure report
        # below as well as by the scoring.
        step_m = self.arc_lengths / float(self.arcs.shape[1])
        # Cap at a quarter of each arc's samples: a 0.39 m arc cannot be
        # asked to clear 0.3 m, which would demand 31 of its 40 samples and
        # reject every tight turn regardless of the scene.
        min_clear = np.minimum(
            np.maximum(1, np.ceil(0.3 / np.maximum(step_m, 1e-6))),
            max(1, self.arcs.shape[1] // 4))

        # Worst clearance along each arc, up to where it is blocked.
        clr = clearance_map[np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)]
        clr = np.where(valid, clr, np.inf)
        min_clear = np.min(clr, axis=1)
        min_clear = np.where(np.isfinite(min_clear), min_clear, self.clearance_pref)
        # Only shortfall against the preferred clearance is penalised; beyond
        # it, extra room buys nothing and should not distort the route.
        clearance_pen = np.clip(
            (self.clearance_pref - min_clear) / max(self.clearance_pref, 1e-6),
            0.0, 1.0)

        bearing = self._goal_bearing()

        # Bearing of each arc at the point it actually reaches, not at its
        # nominal end: an arc blocked early has not turned as far as its full
        # geometry suggests, and scoring the unreached end would credit it
        # with a deviation it never makes.
        end_idx = np.clip(first_block - 1, 0, self.arcs.shape[1] - 1)
        rows_i = np.arange(len(self.arcs))
        end_xy = self.arcs[rows_i, end_idx]
        arc_bearing = np.arctan2(end_xy[:, 1], end_xy[:, 0])

        # Distance made good along the desired bearing. Negative for arcs
        # that end up more than 90 degrees off, which is what we want.
        progress_m = reached_m * np.cos(arc_bearing - bearing)
        progress = progress_m / max(self.arc_lengths.max(), 1e-6)

        total = (terrain
                 + self.w_clear * clearance_pen
                 - self.w_progress * progress
                 + self.w_curv * np.abs(self.kappas) / max(np.abs(self.kappas).max(), 1e-6))

        # Arcs blocked within the first ~0.3 m are not merely expensive, they
        # are immediate contact. Drop them rather than let a heading bonus
        # rescue one.
        total = np.where(first_block < min_clear, np.inf, total)

        if not np.isfinite(total).any():
            self.no_path += 1
            self.pub_no_path.publish(Bool(data=True))
            # Attribute the failure: an arc stopped by an obstacle and one
            # stopped by unobserved space need opposite fixes.
            by_unknown = int(np.sum(first_hard > first_block))
            n_arcs = len(self.arcs)
            straight = n_arcs // 2
            self.get_logger().warn(
                f"all {n_arcs} arcs blocked within 0.3 m "
                f"({self.no_path} consecutive). "
                f"{by_unknown}/{n_arcs} stopped by UNOBSERVED space rather "
                f"than obstacles. Straight arc: first block at sample "
                f"{int(first_block[straight])}/{self.arcs.shape[1]} "
                f"({first_block[straight] * self.arc_lengths[straight] / self.arcs.shape[1]:.2f} m), "
                f"obstacle at {int(first_hard[straight])}. "
                f"min_clear={int(min_clear[straight])}. "
                f"Raise blind_radius_m (now {self.blind_radius}) if unknown "
                f"dominates, or lower front_offset_m (now checking "
                f"{self.check_pts[straight, 0, 0]:.2f} m ahead).",
                throttle_duration_sec=3.0)
            return

        best = int(np.argmin(total))
        self.no_path = 0
        self.pub_no_path.publish(Bool(data=False))
        t_plan = (time.perf_counter() - t1) * 1000.0

        # ---- publish -----------------------------------------------------
        pts = self.arcs[best][: max(2, int(first_block[best]))]
        msg = Path()
        msg.header.stamp = header.stamp     # source stamp, for offline joins
        msg.header.frame_id = self.base_frame
        for x, y in pts:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.pub_path.publish(msg)
        self.pub_stamp.publish(msg.header)

        k = float(self.kappas[best])
        self.pub_kappa.publish(Float32(data=k))
        if self.last_kappa is not None:
            d_deg = abs(math.degrees(math.atan2(
                self.arcs[best][-1, 1], self.arcs[best][-1, 0])) -
                math.degrees(math.atan2(
                    self.arcs[self._last_best][-1, 1],
                    self.arcs[self._last_best][-1, 0])))
            self.pub_switch.publish(Float32(data=float(d_deg)))
            if d_deg > 10.0:
                self.kappa_switches += 1
        self.last_kappa = k
        self._last_best = best

        self.pub_proj_ms.publish(Float32(data=float(t_proj)))
        self.pub_plan_ms.publish(Float32(data=float(t_proj + t_plan)))
        self.n_plans += 1

        # Mean reach across the fan: how far, as a fraction of the horizon,
        # arcs get before hitting something. Near 1 means open ground; near 0
        # means boxed in. This is the number that says whether the planner is
        # choosing freely or scraping for any survivor.
        self.pub_progress.publish(Float32(data=float(progress_m[best])))
        self._last_progress = float(progress_m[best])
        self._last_clear = float(min_clear[best]) if np.isfinite(min_clear[best]) else -1.0
        self.pub_clear.publish(Float32(data=self._last_clear))
        self._last_reach = float(np.mean(reach))
        self.pub_reach.publish(Float32(data=self._last_reach))

        if self.publish_bev and self.pub_bev.get_subscription_count() > 0:
            self._publish_bev(cost, np.asarray(obs, dtype=bool),
                              path_rows, path_cols, rows, cols,
                              best, first_block, msg.header)

    # ------------------------------------------------------------------
    def _publish_bev(self, cost, observed, rows, cols, chk_rows, chk_cols,
                     best, first_block, header) -> None:
        """Green traversable, red costly, DARK unknown, grey fan, white winner.

        Rendering unknown at its cost value made it look like mid-quality
        terrain, which is exactly the confusion the colour map should prevent:
        "no data" and "mediocre ground" are different things to a reader even
        though the planner scores them alike.
        """
        h, w = cost.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        c = np.clip(cost, 0, 1)
        img[:, :, 0] = (c * 255).astype(np.uint8)
        img[:, :, 1] = ((1 - c) * 255).astype(np.uint8)
        img[~observed] = (25, 25, 40)          # unobserved: near-black blue

        for i in range(len(self.arcs)):
            n = max(2, int(first_block[i]))
            rr = np.clip(np.round(rows[i][:n]).astype(int), 0, h - 1)
            cc = np.clip(np.round(cols[i][:n]).astype(int), 0, w - 1)
            img[rr, cc] = (255, 255, 255) if i == best else (110, 110, 110)
            if i == best:
                # Faint trace of where the bumper actually sweeps, which is
                # what was collision-checked.
                br = np.clip(np.round(chk_rows[i][:n]).astype(int), 0, h - 1)
                bc = np.clip(np.round(chk_cols[i][:n]).astype(int), 0, w - 1)
                img[br, bc] = (0, 200, 255)
            # Mark where each arc was cut off, so a fan that is being
            # truncated early is obvious at a glance.
            if n < self.arcs.shape[1]:
                img[rr[-1], cc[-1]] = (255, 160, 0)

        m = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
        m.header = header
        self.pub_bev.publish(m)

    def _health(self) -> None:
        self.pub_status.publish(String(data=(
            f"plans={self.n_plans} arcs={len(self.arcs)} "
            f"switches={self.kappa_switches} no_path={self.no_path} "
            f"stale_skips={self.skipped_stale} "
            f"reach={self._last_reach:.2f} clear={self._last_clear:.2f}m "
            f"progress={self._last_progress:.2f}m "
            f"unknown_blocks={self.unknown_blocks} stride={self.stride} "
            f"obs_cost={self.obstacle_cost:.2f} block_ratio={self.block_ratio:.2f} "
            f"mode={self.goal_mode}"
            + (f" ref_yaw={math.degrees(self.ref_yaw):+.0f}deg"
               if self.ref_yaw is not None else ""))))

    def shutdown(self) -> None:
        self.running = False


def main() -> None:
    rclpy.init()
    node = ArcPlanner()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
