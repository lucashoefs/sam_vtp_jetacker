#!/usr/bin/env python3
"""ROS 2 node for geometry-aware SAM-VTP: RGB + aligned depth.

The model takes no pose, odometry or TF of any kind -- the geometry channels are
a function of depth and intrinsics alone -- so this node has no localisation
dependency and never drops a frame waiting for a transform.

Two conversions happen here, and each is the kind that produces plausible but
wrong output if you get it backwards.

1. DEPTH UNITS. RealSense publishes aligned depth as 16UC1 in MILLIMETRES.
   The model wants float32 METRES. Zeros mean "no return" and become 0.0,
   which the model's own min_depth gate excludes - do not fill them.

2. INTRINSICS. Taken from camera_info at NATIVE resolution and handed to
   CameraIntrinsics unscaled. The model rescales internally with the correct
   pixel-centre convention, and handles the non-square 848x480 -> 512x512
   resize correctly because fx and fy scale independently.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

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

from sam_vtp_core import (
    CameraIntrinsics,
    SAMVTPStreamProcessor,
    build_model,
    depth_from_uint16_mm,
    tune_for_inference,
)

# set _R_OPT2NED to the identity: applying the rotation twice silently sends
# the ego-motion displacement down the wrong axis, with no error raised.

_COMPONENT_NAMES = ("memory", "geometry", "refine")
_TRUE = ("true", "1", "yes", "on")
_FALSE = ("false", "0", "no", "off")


def parse_components(spec: str) -> Dict[str, bool]:
    """Parse `"memory=false,geometry=true"` into kwargs for set_components.

    Every malformed input raises. Nothing here is lenient on purpose: a typo
    that was quietly ignored would run the wrong arm for a whole trial
    session, and the resulting bag would look exactly like a correct one.
    """
    out: Dict[str, bool] = {}
    for item in spec.replace(" ", "").split(","):
        if not item:
            continue
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(
                f"components: expected name=bool, got {item!r} "
                f"(names: {', '.join(_COMPONENT_NAMES)})")
        if name not in _COMPONENT_NAMES:
            raise ValueError(
                f"components: unknown component {name!r} "
                f"(names: {', '.join(_COMPONENT_NAMES)})")
        if name in out:
            raise ValueError(f"components: {name!r} given twice")
        low = value.lower()
        if low in _TRUE:
            out[name] = True
        elif low in _FALSE:
            out[name] = False
        else:
            raise ValueError(f"components: {name}={value!r} is not a boolean")
    return out


class SamVtpGeometryNode(Node):
    def __init__(self) -> None:
        super().__init__("sam_vtp")

        self.declare_parameter("checkpoint", "")
        self.declare_parameter("color_topic", "/camera/d455/color/image_raw")
        self.declare_parameter(
            "depth_topic", "/camera/d455/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("info_topic", "/camera/d455/color/camera_info")
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
        # COMPONENT ABLATION -- which arm of the comparison this run is.
        #
        # Empty means "as the checkpoint was trained", which is what a normal
        # deployment run wants. Otherwise a comma-separated list of name=bool,
        # forwarded to SAMVTP.set_components:
        #
        #   memory=false                  frame-wise + geometry (SAM-TP + geo)
        #   memory=false,geometry=false   frame-wise, RGB only  (SAM-TP)
        #   geometry=false                RGB-only ablation
        #
        # These select which of the checkpoint's weights are USED; they cannot
        # add a component the checkpoint lacks, and asking to enable one it
        # lacks raises rather than running a randomly-initialised module.
        #
        # The resolved set is logged at startup and published in ~/status, so
        # the bag itself records which arm produced it. Without that, a
        # directory of interleaved A/B trials is indistinguishable from a
        # directory of identical ones.
        self.declare_parameter("components", "")
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
        self.declare_parameter("publish_probability", True)
        self.declare_parameter("log_every_n", 150)
        # DEGENERATE OUTPUT WATCH.
        #
        # This model is recurrent: each frame is conditioned on a bank of the
        # previous 7. It was trained on 24-frame clips (~5 s) and at 10 Hz a
        # deployment run is thousands of frames, so the state runs orders of
        # magnitude longer than anything it saw in training. If it drifts to a
        # degenerate fixed point the mask goes uniformly black, the planner
        # sees no traversable ground, and the deadman correctly stops the
        # robot - a total failure that looks, from every health counter, like
        # a perfectly healthy node.
        #
        # Detection is on the PROBABILITY map, not the thresholded mask: a max
        # near zero is collapse, whereas a max around 0.4 is a model still
        # discriminating but below threshold, and the two need opposite fixes.
        self.declare_parameter("degenerate_max_prob", 0.05)
        self.declare_parameter("degenerate_frames", 20)
        # Auto-reset is OFF by default and should stay off for trials. How
        # often the state collapses IS a result about the method; recovering
        # from it silently would hide the very thing worth reporting.
        self.declare_parameter("auto_reset_on_degenerate", False)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.model_hw = (int(g("model_height")), int(g("model_width")))
        self.threshold = float(g("threshold"))
        # Same comparison the CPU path used, hoisted out of the per-frame path.
        self.threshold_u8 = int(self.threshold * 255)
        self.depth_scale = float(g("depth_scale"))
        self.stale_sec = float(g("stale_memory_sec"))
        self.log_every_n = int(g("log_every_n"))
        self.input_fit = str(g("input_fit"))
        # Parsed before anything expensive happens: a bad spec should fail in
        # the first second, not after the model has loaded.
        self.components = parse_components(str(g("components")))
        self.degen_thresh = float(g("degenerate_max_prob"))
        self.degen_frames = int(g("degenerate_frames"))
        self.auto_reset = bool(g("auto_reset_on_degenerate"))
        self._degen_run = 0
        self.degen_events = 0

        checkpoint = str(g("checkpoint"))
        if not checkpoint:
            raise RuntimeError("parameter 'checkpoint' is required")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing to run on CPU")

        self.bridge = CvBridge()
        # Kept: the image/depth subscriptions still sit in a reentrant group on
        # a MultiThreadedExecutor so the synchroniser and the worker do not
        # serialise against each other. See main().
        self.cb_group = ReentrantCallbackGroup()

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
            checkpoint, device=torch.device("cuda"), camera=cam,
            components=self.components or None,
        )
        active = self.model.active_components
        # Sorted so the label is stable across runs and greps cleanly.
        self.arm = ",".join(k for k, v in sorted(active.items()) if v) or "none"
        if self.components:
            self.get_logger().warn(
                f"COMPONENT ABLATION: requested {self.components}, "
                f"running with {self.arm} "
                f"(checkpoint trained with {self.model.trained_components})")
        else:
            self.get_logger().info(f"components: {self.arm} (as trained)")
        self.model.input_normalization_mode = "0_255"
        # NHWC for the whole model: measured 50.3 -> 47.4 ms per frame on the
        # Orin, because narrow convolutions are badly served by NCHW. Must come
        # before warmup(), which is what fixes the shapes and pays for any
        # algorithm search off the clock.
        self.get_logger().info(tune_for_inference(self.model))
        # The library removed the ego-motion warp, so nothing reads pose. Assert
        # it rather than assume it: a checkpoint that still wanted one would
        # otherwise be run silently without it.
        if bool(getattr(self.model, "needs_pose", False)):
            raise RuntimeError(
                "this checkpoint reports needs_pose=True, but this node no "
                "longer supplies a pose. Use a checkpoint trained without the "
                "ego-motion warp.")

        write_every = max(1, int(g("memory_write_every")))
        if not active["memory"] and write_every != 1:
            self.get_logger().info(
                f"memory is off; memory_write_every={write_every} has no effect")
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
        self.pub_maxprob = self.create_publisher(Float32, "~/max_prob", 10)
        self.pub_degen = self.create_publisher(Bool, "~/degenerate", 1)
        # Manual recovery without restarting the process, so a collapse can be
        # cleared mid-run and the event still counted.
        self.create_subscription(Bool, "~/reset_memory", self._on_reset, 1,
                                 callback_group=self.cb_group)
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
    def _on_reset(self, msg: Bool) -> None:
        if msg.data:
            self.stream.reset()
            self._degen_run = 0
            self.get_logger().warn(
                "temporal memory reset by request; CUDA graph will re-capture "
                "over the next ~20 frames")

    # ------------------------------------------------------------------
    def _on_pair(self, color_msg: Image, depth_msg: Image) -> None:
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
            self.pending = (rgb, depth_m, depth_valid, color_msg.header)
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
                rgb, depth, depth_valid, header = self.pending
                self.consumed_seq = self.pending_seq
            try:
                self._infer(rgb, depth, depth_valid, header)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"inference failed: {exc}")

    def _infer(self, rgb, depth, depth_valid, header) -> None:
        now = time.monotonic()
        if (self.stale_sec > 0 and self.processed > 0
                and now - self.last_infer_end > self.stale_sec):
            self.get_logger().warn(
                f"{now - self.last_infer_end:.1f}s frame gap; resetting memory "
                f"(CUDA graph will re-capture over the next ~20 frames)")
            self.stream.reset()

        result = self.stream.step(rgb, depth=depth, depth_valid=depth_valid)
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
        # The CPU version of this was `numpy.where(prob >= t, 255, 0).astype(u8)`,
        # which builds a 407k-element int64 intermediate before narrowing it,
        # and measured ~20x the cost of the comparison itself.
        # Peak probability BEFORE thresholding. This is what separates a
        # collapsed recurrent state (max ~0) from a model still discriminating
        # but sitting under the threshold (max ~0.4) - the two look identical
        # in the binary mask and need opposite fixes.
        #
        # Taken from the same GPU tensor and moved across with the images
        # below, so it costs no extra device->host transfer.
        max_prob_gpu = probs.max()

        prob_u8_gpu = (probs[0, 0].clamp(0, 1) * 255).to(torch.uint8)
        binary_u8_gpu = (prob_u8_gpu >= self.threshold_u8).to(torch.uint8) * 255
        both = torch.stack((prob_u8_gpu, binary_u8_gpu)).cpu().numpy()
        prob_u8, binary_u8 = both[0], both[1]

        max_prob = float(max_prob_gpu.item())
        self.pub_maxprob.publish(Float32(data=max_prob))

        if max_prob < self.degen_thresh:
            self._degen_run += 1
            if self._degen_run == self.degen_frames:
                self.degen_events += 1
                self.pub_degen.publish(Bool(data=True))
                self.get_logger().error(
                    f"OUTPUT COLLAPSED: peak probability {max_prob:.3f} for "
                    f"{self.degen_frames} consecutive frames after "
                    f"{self.processed} total. The mask is uniformly black, so "
                    f"the planner sees no traversable ground and the deadman "
                    f"will stop the robot. Recover with: ros2 topic pub --once "
                    f"/sam_vtp/reset_memory std_msgs/msg/Bool '{{data: true}}'")
                if self.auto_reset:
                    self.get_logger().warn("auto_reset_on_degenerate: resetting")
                    self.stream.reset()
                    self._degen_run = 0
        else:
            if self._degen_run >= self.degen_frames:
                self.pub_degen.publish(Bool(data=False))
                self.get_logger().info(
                    f"output recovered: peak probability {max_prob:.3f}")
            self._degen_run = 0

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
                f"infer={result.elapsed_ms:.1f}ms "
                f"age={age:.0f}ms"
            )

    # ------------------------------------------------------------------
    def _health(self) -> None:
        ready = self.processed > 0
        self.pub_ready.publish(Bool(data=ready))
        self.pub_status.publish(String(data=(
            f"arm={self.arm} "
            f"ready={ready} processed={self.processed} dropped={self.dropped} "
            f"in={self.frames_in} synced={self.processed + self.dropped} "
            f"unsynced={max(0, self.frames_in - self.processed - self.dropped)} "
            f"collapses={self.degen_events} "
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
