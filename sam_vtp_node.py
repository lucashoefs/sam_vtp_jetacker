#!/usr/bin/env python3
"""ROS 2 node for geometry-aware SAM-VTP: RGB + aligned depth + camera pose.

Three conversions happen here, and each is the kind that produces plausible
but wrong output if you get it backwards.

1. DEPTH UNITS. RealSense publishes aligned depth as 16UC1 in MILLIMETRES.
   The model wants float32 METRES. Zeros mean "no return" and become 0.0,
   which the model's own min_depth gate excludes - do not fill them.

2. POSE AXES. The model's warp maps optical axes into a NED body frame
   (x forward, y right, z down) via a fixed permutation, then uses only the
   RELATIVE transform between two poses:

       A = P^T @ R_mem^-1 @ R_cur @ P

   Because it is relative, the world frame's absolute orientation cancels:
   ROS `odom` works and true NED is not required. What does matter is that the
   supplied rotation takes NED-BODY camera axes into the world. TF gives us
   the OPTICAL frame, so we post-multiply by the optical->NED-body rotation,
   which as a scalar-last quaternion is (-0.5, -0.5, -0.5, 0.5).

3. INTRINSICS. Taken from camera_info at NATIVE resolution and handed to
   CameraIntrinsics unscaled. The model rescales internally with the correct
   pixel-centre convention, and handles the non-square 848x480 -> 512x512
   resize correctly because fx and fy scale independently.

Pose is looked up at the IMAGE timestamp, not now(). At 0.5 m/s a 60 ms error
is 3 cm of translation, enough to visibly smear the warp.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, Header, String
from tf2_ros import Buffer, TransformListener

from sam_vtp_core import (
    CameraIntrinsics,
    SAMVTPStreamProcessor,
    build_model,
    depth_from_uint16_mm,
    pose_from_optical,
    tune_for_inference,
)

# The axis conversion lives in the library (pose_from_optical). Do NOT also
# set _R_OPT2NED to the identity: applying the rotation twice silently sends
# the ego-motion displacement down the wrong axis, with no error raised.


class SamVtpGeometryNode(Node):
    def __init__(self) -> None:
        super().__init__("sam_vtp")

        self.declare_parameter("checkpoint", "")
        self.declare_parameter("color_topic", "/camera/d455/color/image_raw")
        self.declare_parameter(
            "depth_topic", "/camera/d455/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("info_topic", "/camera/d455/color/camera_info")
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("model_height", 512)
        self.declare_parameter("model_width", 512)
        self.declare_parameter("threshold", 0.5)
        # Memory writes are spaced so the 7-slot bank spans the ~1.4 s it was
        # trained with (7 slots x stride 2 / 10 fps). Writing every frame at
        # 15 fps collapses that to ~0.5 s and fills the bank with near-duplicate
        # views; the prediction is still produced every frame either way, only
        # the write is gated.
        #
        # Set this to round(camera_fps / 5): 3 at 15 fps, 6 at 30 fps.
        #
        # This used to cost CUDA graph replay, because the write was baked into
        # the capture. It no longer does -- the library captures a second graph
        # for the frames that skip the write and branches between them in
        # Python -- so the span and the speed are no longer a trade.
        self.declare_parameter("memory_write_every", 3)
        # stretch | crop | pad. The D455f is 89x59 deg where training was
        # 90x90: 'stretch' exaggerates vertical structure (slope is a geometry
        # channel), 'crop' matches training aspect but loses horizontal FOV,
        # 'pad' keeps the width and feeds bars. Compare on real data.
        self.declare_parameter("input_fit", "stretch")
        # ApproximateTimeSynchronizer pairs colour with aligned depth. With
        # queue_size=2 any jitter between the two evicts a message before its
        # partner arrives, and the pair is silently lost - it never reaches
        # the worker, so `dropped` stays at zero while HALF the frames vanish.
        # The signature is a mask interval at exactly 2x the camera period.
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("sync_queue", 10)
        self.declare_parameter("depth_scale", 0.001)   # 16UC1 mm -> metres
        # Resetting discards the captured CUDA graph and costs a full warmup,
        # after which inference runs ~2x slower until capture re-engages. Only
        # worth doing when the memory really is describing a vanished scene.
        self.declare_parameter("stale_memory_sec", 5.0)
        # How far the nearest available transform may be from the image stamp
        # before the frame is dropped rather than warped with a stale pose.
        self.declare_parameter("max_pose_skew_s", 0.12)
        self.declare_parameter("publish_probability", True)
        self.declare_parameter("log_every_n", 150)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.model_hw = (int(g("model_height")), int(g("model_width")))
        self.threshold = float(g("threshold"))
        # Same comparison the CPU path used, hoisted out of the per-frame path.
        self.threshold_u8 = int(self.threshold * 255)
        self.world_frame = str(g("world_frame"))
        self.depth_scale = float(g("depth_scale"))
        self.stale_sec = float(g("stale_memory_sec"))
        self.log_every_n = int(g("log_every_n"))
        self.input_fit = str(g("input_fit"))
        self.max_pose_skew = float(g("max_pose_skew_s"))

        checkpoint = str(g("checkpoint"))
        if not checkpoint:
            raise RuntimeError("parameter 'checkpoint' is required")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing to run on CPU")

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        # A blocking lookup_transform() inside a callback needs OTHER
        # callbacks - notably /tf - to keep being serviced, or it waits for a
        # transform that can never arrive. Two ways to get that wrong:
        #   * plain rclpy.spin(): single-threaded, deadlocks until timeout.
        #   * TransformListener(spin_thread=True): adds THIS node to a second
        #     executor while main() adds it to another. Two executors owning
        #     one node is undefined; here /tf silently stopped being
        #     delivered and never resumed.
        # A MultiThreadedExecutor plus a reentrant group is the arrangement
        # that actually works. See main().
        self.cb_group = ReentrantCallbackGroup()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Intrinsics must be known before the model is built, so block here.
        self.get_logger().info(f"waiting for {g('info_topic')} ...")
        self.camera_info: Optional[CameraInfo] = None
        info_sub = self.create_subscription(
            CameraInfo, str(g("info_topic")), self._on_info, 1
        )
        while self.camera_info is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.5)
        self.destroy_subscription(info_sub)

        cam = CameraIntrinsics.from_K(
            self.camera_info.k, self.camera_info.width, self.camera_info.height
        )
        hfov, vfov = cam.fov_deg()
        self.get_logger().info(
            f"intrinsics fx={cam.fx:.1f} fy={cam.fy:.1f} "
            f"cx={cam.cx:.1f} cy={cam.cy:.1f} @ {cam.width}x{cam.height} "
            f"({hfov:.0f}x{vfov:.0f} deg)"
        )
        if abs(vfov - 90.0) > 20.0:
            self.get_logger().warn(
                f"vertical FOV {vfov:.0f} deg vs 90 deg in training; "
                f"input_fit={self.input_fit} decides how that gap is handled")

        self.get_logger().info("loading model...")
        self.model = build_model(
            checkpoint, device=torch.device("cuda"), camera=cam
        )
        self.model.input_normalization_mode = "0_255"
        # NHWC for the whole model: measured 50.3 -> 47.4 ms per frame on the
        # Orin, because narrow convolutions are badly served by NCHW. Must come
        # before warmup(), which is what fixes the shapes and pays for any
        # algorithm search off the clock.
        self.get_logger().info(tune_for_inference(self.model))

        write_every = max(1, int(g("memory_write_every")))
        self.stream = SAMVTPStreamProcessor(
            self.model,
            resize_hw=self.model_hw,
            threshold=self.threshold,
            input_fit=self.input_fit,
            use_fullgraph=True,
            memory_write_every=write_every,
            # Keep the timing sync. Dropping it saves one device drain -- about
            # 1% of a 47 ms frame -- and in exchange latency_ms stops measuring
            # the model and starts measuring kernel launches, which read as
            # ~0.5 ms and tell an operator nothing about whether perception is
            # keeping up. On a robot the number you can trust at 3am is worth
            # more than the 1%.
            time_model_only=True,
        )
        # Pay cuDNN workspace allocation and cache filling off the clock, so
        # the robot does not eat ~400 ms on its first live frame.
        self.model.warmup(n=3, hw=self.model_hw)
        self.get_logger().info(f"model ready; {self.stream.graph_status}")

        # ---- frame handoff (single slot, newest wins) --------------------
        self.lock = threading.Condition()
        self.pending = None
        self.pending_seq = 0
        self.consumed_seq = 0
        self.running = True
        self.dropped = 0
        self.processed = 0
        self.no_pose = 0
        self.skewed_poses = 0
        self._graph_active = False
        self.last_infer_end = 0.0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        color_sub = Subscriber(self, Image, str(g("color_topic")), qos_profile=sensor_qos)
        depth_sub = Subscriber(self, Image, str(g("depth_topic")), qos_profile=sensor_qos)
        self.sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=int(g("sync_queue")),
            slop=float(g("sync_slop")),
        )
        self.sync.registerCallback(self._on_pair)

        # Count raw colour frames independently, so lost PAIRS are visible.
        # camera_info shares the image stamp and costs nothing to subscribe.
        self.frames_in = 0
        self.create_subscription(
            CameraInfo, str(g("info_topic")),
            lambda _m: setattr(self, "frames_in", self.frames_in + 1),
            sensor_qos, callback_group=self.cb_group)

        self.pub_binary = self.create_publisher(Image, "~/binary", sensor_qos)
        self.pub_prob = (
            self.create_publisher(Image, "~/probability", sensor_qos)
            if bool(g("publish_probability")) else None
        )
        self.pub_latency = self.create_publisher(Float32, "~/latency_ms", 10)
        self.pub_age = self.create_publisher(Float32, "~/frame_age_ms", 10)
        self.pub_ready = self.create_publisher(Bool, "~/ready", 10)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        # Echo of the SOURCE frame stamp, published the moment this stage
        # finishes. Lets a profiler join stages by frame without subscribing
        # to the image topics, which would itself distort what it measures.
        self.pub_stamp = self.create_publisher(Header, "~/stamp", 10)

        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self.create_timer(1.0, self._health, callback_group=self.cb_group)

    # ------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        if self.camera_info is None:
            self.camera_info = msg

    # ------------------------------------------------------------------
    def _lookup_pose(self, header) -> Optional[np.ndarray]:
        """Camera pose as [x y z qx qy qz qw] with NED-body camera axes.

        NON-BLOCKING by design. A blocking lookup inside a callback needs /tf
        to keep being delivered while it waits, and that is only true if the
        TF subscriptions sit in a reentrant callback group on a multithreaded
        executor - which TransformListener does not arrange for you. Getting
        it wrong does not fail loudly; /tf simply stops being serviced and
        every lookup times out.

        With odometry at 50 Hz and images ~87 ms old, the transform for an
        image stamp is normally already buffered, so timeout=0 costs nothing.

        If the exact stamp is unavailable we fall back to the newest transform
        and accept it only if it is close enough in time. The warp needs
        short-horizon relative pose, so a few tens of milliseconds of skew is
        immaterial; at 0.3 m/s, 120 ms is 3.6 cm.
        """
        zero = rclpy.duration.Duration(seconds=0.0)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, header.frame_id, header.stamp, timeout=zero)
        except Exception as exact_exc:  # noqa: BLE001
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.world_frame, header.frame_id, rclpy.time.Time(),
                    timeout=zero)
            except Exception:  # noqa: BLE001
                if self.no_pose < 3:
                    stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
                    skew = self.get_clock().now().nanoseconds * 1e-9 - stamp
                    self.get_logger().warn(
                        f"TF unavailable ({type(exact_exc).__name__}): "
                        f"{exact_exc}; image stamp {skew*1000:.0f} ms behind "
                        f"node clock")
                return None

            want = header.stamp.sec + header.stamp.nanosec * 1e-9
            got = (tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9)
            gap = abs(want - got)
            if gap > self.max_pose_skew:
                self.get_logger().warn(
                    f"nearest transform is {gap*1000:.0f} ms from the image "
                    f"stamp (limit {self.max_pose_skew*1000:.0f} ms); "
                    f"dropping frame", throttle_duration_sec=5.0)
                return None
            self.skewed_poses += 1

        t = tf.transform.translation
        r = tf.transform.rotation
        # header.frame_id is the colour OPTICAL frame, so this is exactly the
        # case pose_from_optical is for. Translation and rotation must be of
        # the same frame; mixing base_link position with camera rotation adds
        # a lever-arm error that does not cancel when the robot turns.
        return pose_from_optical([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])

    # ------------------------------------------------------------------
    def _on_pair(self, color_msg: Image, depth_msg: Image) -> None:
        pose = self._lookup_pose(color_msg.header)
        if pose is None:
            self.no_pose += 1
            self.get_logger().warn(
                f"no transform {self.world_frame} -> {color_msg.header.frame_id}; "
                "is odometry running?", throttle_duration_sec=5.0)
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
            depth_raw = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cv_bridge failed: {exc}")
            return

        # 16UC1 millimetres -> float32 metres, PLUS an explicit validity mask.
        # A zero means "no return", not "zero distance". Passing the mask
        # through matters: without it the model falls back to `depth >
        # min_depth`, which clamps holes to 0.1 m and reads them as a surface
        # pressed against the lens - spurious near-field obstacles wherever
        # stereo failed. Outdoors, in sun, on low-texture ground, that is
        # most of the frame edge.
        depth_m, depth_valid = depth_from_uint16_mm(
            depth_raw, scale=self.depth_scale)

        with self.lock:
            if self.pending is not None and self.pending_seq > self.consumed_seq:
                self.dropped += 1
            self.pending = (rgb, depth_m, depth_valid, pose, color_msg.header)
            self.pending_seq += 1
            self.lock.notify()

    # ------------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            with self.lock:
                while self.running and self.pending_seq == self.consumed_seq:
                    self.lock.wait(timeout=0.5)
                if not self.running:
                    return
                rgb, depth, depth_valid, pose, header = self.pending
                self.consumed_seq = self.pending_seq
            try:
                self._infer(rgb, depth, depth_valid, pose, header)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"inference failed: {exc}")

    def _infer(self, rgb, depth, depth_valid, pose, header) -> None:
        now = time.monotonic()
        if (self.stale_sec > 0 and self.processed > 0
                and now - self.last_infer_end > self.stale_sec):
            self.get_logger().warn(
                f"{now - self.last_infer_end:.1f}s frame gap; resetting memory "
                f"(CUDA graph will re-capture over the next ~20 frames)")
            self.stream.reset()

        result = self.stream.step(rgb, depth=depth, pose=pose,
                                  depth_valid=depth_valid)
        self.last_infer_end = time.monotonic()
        self.processed += 1

        if not self._graph_active and "replay active" in result.graph_status:
            self._graph_active = True
            self.get_logger().info(
                f"CUDA graph replay engaged at frame {self.processed} "
                f"({result.elapsed_ms:.0f} ms)")

        src_h, src_w = rgb.shape[:2]
        probs = result.probabilities
        if probs.shape[-2:] != (src_h, src_w):
            probs = F.interpolate(probs, size=(src_h, src_w),
                                  mode="bilinear", align_corners=False)

        # Quantise AND threshold on the GPU, then bring both planes back in a
        # single transfer. The threshold still has to be applied after the
        # resize -- thresholding first and interpolating after would blur the
        # decision boundary back into grey -- but neither step needs the CPU.
        #
        # The CPU version of this was `np.where(prob >= t, 255, 0).astype(u8)`,
        # which builds a 407k-element int64 intermediate before narrowing it,
        # and measured ~20x the cost of the comparison itself.
        prob_u8_gpu = (probs[0, 0].clamp(0, 1) * 255).to(torch.uint8)
        binary_u8_gpu = (prob_u8_gpu >= self.threshold_u8).to(torch.uint8) * 255
        both = torch.stack((prob_u8_gpu, binary_u8_gpu)).cpu().numpy()
        prob_u8, binary_u8 = both[0], both[1]

        msg = self.bridge.cv2_to_imgmsg(binary_u8, encoding="mono8")
        msg.header = header
        self.pub_binary.publish(msg)
        if self.pub_prob is not None:
            msg = self.bridge.cv2_to_imgmsg(prob_u8, encoding="mono8")
            msg.header = header
            self.pub_prob.publish(msg)

        self.pub_stamp.publish(header)
        self.pub_latency.publish(Float32(data=float(result.elapsed_ms)))
        stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        age = (self.get_clock().now().nanoseconds * 1e-9 - stamp) * 1000.0
        self.pub_age.publish(Float32(data=float(age)))

        if self.log_every_n and self.processed % self.log_every_n == 0:
            self.get_logger().info(
                f"frames={self.processed} dropped={self.dropped} "
                f"no_pose={self.no_pose} infer={result.elapsed_ms:.1f}ms "
                f"age={age:.0f}ms"
            )

    # ------------------------------------------------------------------
    def _health(self) -> None:
        ready = self.processed > 0
        self.pub_ready.publish(Bool(data=ready))
        self.pub_status.publish(String(data=(
            f"ready={ready} processed={self.processed} dropped={self.dropped} "
            f"in={self.frames_in} synced={self.processed + self.dropped} "
            f"unsynced={max(0, self.frames_in - self.processed - self.dropped)} "
            f"no_pose={self.no_pose} skewed={self.skewed_poses} "
            f"graph={self.stream.graph_status}"
        )))

    def shutdown(self) -> None:
        with self.lock:
            self.running = False
            self.lock.notify_all()
        self.worker.join(timeout=2.0)


def main() -> None:
    rclpy.init()
    node = SamVtpGeometryNode()
    # 4 threads: image callback, TF, timers, and headroom. With the default
    # single-threaded executor a blocking TF lookup starves the very /tf
    # subscription it is waiting on.
    executor = MultiThreadedExecutor(num_threads=4)
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
