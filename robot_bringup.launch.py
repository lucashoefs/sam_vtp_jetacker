"""JetAcker bringup: camera + drive + odometry + perception + monitoring.

FRAME CONVENTION
    base_link sits on the GROUND, directly below the rear axle centre.
    Putting it at ground level rather than axle height makes the ground plane
    z = 0, which the BEV projection downstream wants, and the rear axle is the
    right origin for the bicycle model the odometry integrates.

        odom -> base_link            jetacker_odometry
        base_link -> d455_link       static, measured (below)
        d455_link -> *_optical_frame realsense2_camera

CAMERA STREAMS
    Only what the model consumes is enabled. Aligned depth is required by the
    geometry checkpoint; the camera's own IMU is off because odometry uses the
    chassis board's IMU instead, and running both wastes USB bandwidth on a
    link that is already carrying two video streams.

Usage:
    ros2 launch ~/jetacker/robot_bringup.launch.py
    ros2 launch ~/jetacker/robot_bringup.launch.py model:=true
    ros2 launch ~/jetacker/robot_bringup.launch.py drive:=false record:=true
"""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# numpy/scipy link OpenBLAS and OpenMP, which by default spawn one thread per
# core AND SPIN them between parallel regions rather than sleeping
# (OMP_WAIT_POLICY defaults to ACTIVE). For the small arrays here the parallel
# regions last microseconds, so the threads spend their time busy-waiting: the
# planner was measured at 438% CPU - 4.4 cores - for about 25 ms of real work
# per cycle, starving the camera node until colour dropped from 15 Hz to 8.5.
#
# One thread each, and a passive wait if any library ignores that.
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_WAIT_POLICY": "PASSIVE",
}

HOME = os.path.expanduser("~")

# Integration code lives here and is ours. ~/sam_vtp is the model repository,
# which gets wiped and re-cloned on every model update, so nothing of ours can
# live inside it - only the venv and the checkpoint.
JETACKER_DIR = os.path.join(HOME, "jetacker")
SAM_DIR = os.path.join(HOME, "sam_vtp")
VENV_PY = os.path.join(SAM_DIR, ".venv", "bin", "python")
CHECKPOINT = os.path.join(SAM_DIR, "sam_vtp.pt")

CAM_NS = "camera"
CAM_NAME = "d455"

# --- camera mounting, measured on the robot -------------------------------
# From the ground point below the rear axle centre:
#   0.15 m forward, 0.21 m up, centred laterally.
# (Axle centre is one wheel radius up, ~0.05 m, hence 0.16 m above the axle.)
CAM_X, CAM_Y, CAM_Z = 0.15, 0.0, 0.21

# Recorded topics. `-a` would capture raw colour AND raw + aligned depth at
# ~45 MB/s. Aligned depth is included compressed because the BEV work needs
# it; that alone is ~4 MB/s, so watch disk on long runs.
RECORD_TOPICS = [
    f"/{CAM_NS}/{CAM_NAME}/color/image_raw/compressed",
    f"/{CAM_NS}/{CAM_NAME}/color/camera_info",
    f"/{CAM_NS}/{CAM_NAME}/color/metadata",
    f"/{CAM_NS}/{CAM_NAME}/aligned_depth_to_color/image_raw/compressedDepth",
    f"/{CAM_NS}/{CAM_NAME}/aligned_depth_to_color/camera_info",
    f"/{CAM_NS}/{CAM_NAME}/extrinsics/depth_to_color",
    "/sam_vtp/binary",
    "/sam_vtp/latency_ms",
    "/sam_vtp/frame_age_ms",
    "/sam_vtp/status",
    "/cmd_vel",
    "/cmd_vel_raw",
    "/odom",
    "/genie_planner/path",
    "/arc_planner/path",
    "/genie_planner/plan_ms",
    "/arc_planner/plan_ms",
    "/genie_planner/heading_change_deg",
    "/arc_planner/heading_change_deg",
    "/genie_planner/status",
    "/arc_planner/status",
    "/arc_planner/curvature",
    "/arc_planner/no_path",
    "/pure_pursuit/steer_deg",
    "/pure_pursuit/cross_track_m",
    "/pure_pursuit/status",
    "/deadman/status",
    "/deadman/passing",
    "/waypoint_follower/waypoint_index",
    "/waypoint_follower/distance_to_goal",
    "/waypoint_follower/path_length",
    "/waypoint_follower/route_complete",
    "/goal_pose",
    "/imu",
    "/jetacker_driver/steer_angle_deg",
    "/jetacker_driver/status",
    "/tf",
    "/tf_static",
]


def _flag(context, name: str) -> bool:
    return LaunchConfiguration(name).perform(context).lower() in (
        "true", "1", "yes", "on"
    )


def launch_setup(context, *args, **kwargs):
    fps = LaunchConfiguration("fps").perform(context)
    bag_dir = LaunchConfiguration("bag_dir").perform(context)
    pitch = LaunchConfiguration("camera_pitch_deg").perform(context)
    profile = f"848x480x{fps}"
    pitch_rad = float(pitch) * 3.14159265358979 / 180.0

    actions = [
        LogInfo(msg="=== JetAcker bringup ==="),
        LogInfo(msg=f"camera {profile} aligned-depth | "
                    f"drive={_flag(context, 'drive')} "
                    f"odom={_flag(context, 'odometry')} "
                    f"model={_flag(context, 'model')} "
                    f"planner={_flag(context, 'planner')} "
                    f"planner_type={LaunchConfiguration('planner_type').perform(context)} "
                    f"control={_flag(context, 'control')} "
                    f"goal={LaunchConfiguration('goal_mode').perform(context)} "
                    f"record={_flag(context, 'record')}"),
        LogInfo(msg=f"camera mount: x={CAM_X} z={CAM_Z} pitch={pitch} deg "
                    f"(from ground below rear axle)"),
        LogInfo(msg="Ackermann chassis: linear.x must be non-zero to steer"),
    ]
    if _flag(context, "control"):
        actions.append(LogInfo(
            msg="control started but DISABLED - publish true on "
                "/pursuit_enable then /deadman_enable, then /route_start"))

    if _flag(context, "camera"):
        actions.append(GroupAction(
            scoped=True,
            forwarding=False,
            actions=[IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("realsense2_camera"),
                        "launch", "rs_launch.py",
                    )
                ),
                launch_arguments={
                    "camera_namespace": CAM_NS,
                    "camera_name": CAM_NAME,

                    # --- streams the model needs ---
                    "enable_color": "true",
                    "rgb_camera.color_profile": profile,
                    "enable_depth": "true",
                    "depth_module.depth_profile": profile,
                    # Registers depth into the colour frame so one set of
                    # intrinsics describes both. Required by the geometry
                    # checkpoint; without it the depth and RGB pixels do not
                    # correspond.
                    "align_depth.enable": "true",
                    # Groups colour and depth of the same instant into one
                    # frameset, which keeps their stamps together and makes the
                    # node's ApproximateTimeSynchronizer nearly exact.
                    "enable_sync": "true",

                    # --- everything else off ---
                    # Odometry uses the chassis board IMU, so the camera's is
                    # redundant and costs USB bandwidth on a busy link.
                    "enable_motion": "false",
                    "enable_gyro": "false",
                    "enable_accel": "false",
                    "enable_infra": "false",
                    "enable_infra1": "false",
                    "enable_infra2": "false",
                    "pointcloud.enable": "false",
                    "colorizer.enable": "false",
                    "enable_rgbd": "false",
                    "enable_occupancy": "false",
                    "enable_labeled_point_cloud": "false",
                    "enable_safety": "false",

                    "publish_tf": "true",
                }.items(),
            )],
        ))

    # base_link (ground, below rear axle) -> d455_link.
    # Pitch is the one to get right: a few degrees of error tilts every slope
    # estimate and drifts distant flat ground across the height gate.
    actions.append(Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_camera",
        arguments=[
            "--x", str(CAM_X), "--y", str(CAM_Y), "--z", str(CAM_Z),
            "--roll", "0.0", "--pitch", str(pitch_rad), "--yaw", "0.0",
            "--frame-id", "base_link",
            "--child-frame-id", f"{CAM_NAME}_link",
        ],
        output="log",
    ))

    if _flag(context, "drive"):
        # System python: rclpy and pyserial. cwd matters because the driver
        # does `from rrc import RRCBoard`.
        actions.append(ExecuteProcess(
            cmd=["python3", os.path.join(JETACKER_DIR, "jetacker_driver.py")],
            cwd=JETACKER_DIR,
            output="screen",
            respawn=True,        # watchdog stops the robot; respawn recovers
            respawn_delay=2.0,
        ))

    if _flag(context, "odometry"):
        # Must be publishing before the model starts, or every frame is
        # dropped for want of a pose.
        actions.append(ExecuteProcess(
            cmd=[
                "python3", os.path.join(JETACKER_DIR, "jetacker_odometry.py"),
                "--ros-args",
                "-p", f"speed_scale:={LaunchConfiguration('speed_scale').perform(context)}",
            ],
            cwd=JETACKER_DIR,
            output="screen",
            additional_env=SINGLE_THREAD_ENV,
            respawn=True,
            respawn_delay=2.0,
        ))

    if _flag(context, "model"):
        # venv python: torch. Not respawned - a crash here should be noticed.
        actions.append(ExecuteProcess(
            cmd=[
                VENV_PY,
                os.path.join(JETACKER_DIR, "sam_vtp_node.py"),
                "--ros-args",
                "-p", f"checkpoint:={CHECKPOINT}",
                "-p", f"color_topic:=/{CAM_NS}/{CAM_NAME}/color/image_raw",
                "-p", f"depth_topic:=/{CAM_NS}/{CAM_NAME}/aligned_depth_to_color/image_raw",
                "-p", f"info_topic:=/{CAM_NS}/{CAM_NAME}/color/camera_info",
                "-p", "world_frame:=odom",
                "-p", f"input_fit:={LaunchConfiguration('input_fit').perform(context)}",
                # Memory writes 0.2 s apart whatever the camera rate, so the
                # 7-slot bank spans the ~1.4 s the model trained with.
                "-p", f"memory_write_every:={max(1, round(int(fps) / 5))}",
            ],
            cwd=JETACKER_DIR,
            output="screen",
            additional_env=dict(SINGLE_THREAD_ENV, OMP_NUM_THREADS="2"),
        ))

    if _flag(context, "planner"):
        ptype = LaunchConfiguration("planner_type").perform(context)
        script = ("arc_planner_node.py" if ptype == "arc"
                  else "genie_planner_node.py")
        actions.append(ExecuteProcess(
            cmd=[
                VENV_PY,
                os.path.join(JETACKER_DIR, script),
                "--ros-args",
                "-p", f"config:={JETACKER_DIR}/jetacker_planner.yaml",
                "-p", "score_topic:=/sam_vtp/probability",
                "-p", f"depth_topic:=/{CAM_NS}/{CAM_NAME}/aligned_depth_to_color/image_raw",
                "-p", f"info_topic:=/{CAM_NS}/{CAM_NAME}/color/camera_info",
                "-p", ("plan_rate:=10.0" if ptype == "arc" else "plan_rate:=2.0"),
                "-p", f"goal_distance_m:={LaunchConfiguration('goal_distance').perform(context)}",
                "-p", f"goal_mode:={LaunchConfiguration('goal_mode').perform(context)}",
            ],
            cwd=JETACKER_DIR,
            output="screen",
            additional_env=SINGLE_THREAD_ENV,
        ))

    if _flag(context, "control"):
        max_speed = LaunchConfiguration("max_speed").perform(context)
        route = LaunchConfiguration("route").perform(context)

        # planner path -> cmd_vel_raw
        actions.append(ExecuteProcess(
            additional_env=SINGLE_THREAD_ENV,
            cmd=["python3", os.path.join(JETACKER_DIR, "pure_pursuit_node.py"),
                 "--ros-args",
                 "-r", f"path:=/{'arc_planner' if LaunchConfiguration('planner_type').perform(context) == 'arc' else 'genie_planner'}/path",
                 "-r", "route_complete:=/waypoint_follower/route_complete",
                 "-p", f"max_speed:={max_speed}"],
            cwd=JETACKER_DIR, output="screen", respawn=True, respawn_delay=2.0,
        ))

        # cmd_vel_raw -> cmd_vel, gated on the freshness of the whole chain.
        # NOT respawned: if the safety gate is dying repeatedly, the right
        # outcome is a robot that stays stopped and an operator who notices.
        actions.append(ExecuteProcess(
            cmd=["python3", os.path.join(JETACKER_DIR, "deadman_node.py"),
                 "--ros-args",
                 "-r", f"path:=/{'arc_planner' if LaunchConfiguration('planner_type').perform(context) == 'arc' else 'genie_planner'}/path",
                 "-r", "perception:=/sam_vtp/binary",
                 "-p", f"max_speed:={float(max_speed) + 0.1:.2f}"],
            cwd=JETACKER_DIR, output="screen", additional_env=SINGLE_THREAD_ENV,
        ))

        # In forward mode nothing publishes goal_pose, and the planner falls
        # back to a body-relative goal on its own. Distance travelled comes
        # from /odom in the bag rather than from this node.
        if LaunchConfiguration("goal_mode").perform(context) == "waypoints":
            actions.append(ExecuteProcess(
                cmd=["python3", os.path.join(JETACKER_DIR, "waypoint_follower.py"),
                     "--ros-args", "-p", f"route:={route}"],
                cwd=JETACKER_DIR, output="screen", respawn=True,
                respawn_delay=2.0,
            ))

    if _flag(context, "foxglove"):
        # foxglove_bridge ships an XML launch file, not a Python one.
        #
        # BANDWIDTH. Raw pixels do not fit over WiFi. At 848x480 the colour
        # stream alone is 146 Mbit/s, aligned depth another 98, and the two mask
        # topics 40 each -- ~324 Mbit/s with all four open. The link saturates,
        # the bridge logs "outbox for client ... full", and TCP bufferbloat turns
        # the overflow into tens of seconds of lag rather than clean drops.
        #
        # Two mitigations here; the third is yours. use_compression applies
        # websocket permessage-deflate, which costs Orin CPU but is very
        # effective on the mask topics (large uniform regions). send_buffer_limit
        # caps how much can queue before the bridge drops: a smaller buffer
        # trades completeness for freshness, which is the right trade for
        # watching a robot drive. The third is to subscribe to
        # .../image_raw/compressed rather than .../image_raw in Studio -- the
        # RealSense driver already publishes it (see RECORD_TOPICS) and it is
        # worth more than either parameter below.
        actions.append(GroupAction(
            scoped=True,
            forwarding=False,
            actions=[IncludeLaunchDescription(
                AnyLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("foxglove_bridge"),
                        "launch", "foxglove_bridge_launch.xml",
                    )
                ),
                launch_arguments={
                    "port": "8765",
                    "use_compression": LaunchConfiguration("fg_compress").perform(context),
                    "send_buffer_limit": LaunchConfiguration("fg_buffer").perform(context),
                }.items(),
            )],
        ))

    if _flag(context, "record"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(bag_dir, exist_ok=True)
        actions.append(ExecuteProcess(
            cmd=[
                "ros2", "bag", "record",
                "-s", "mcap",
                "-o", os.path.join(bag_dir, f"run_{stamp}"),
                "--max-bag-duration", "300",  # 5 min files; a crash loses one
            ] + RECORD_TOPICS,
            output="screen",
        ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("camera", default_value="true"),
        DeclareLaunchArgument("drive", default_value="true"),
        DeclareLaunchArgument("odometry", default_value="true"),
        DeclareLaunchArgument("model", default_value="false"),
        DeclareLaunchArgument("planner", default_value="false"),
        # genie : GeNIE's published path search. ~515 ms/plan here, so 2 Hz.
        #         Faithful to prior work, too slow to react at driving speed.
        # arc   : constant-curvature arcs on the SAME GeNIE BEV projection.
        #         ~2 ms of search, so the loop runs at the camera rate. Every
        #         candidate is executable by construction on an Ackermann
        #         chassis, and there is no temporal smoothing, so costmap
        #         instability still reaches the controller.
        DeclareLaunchArgument("planner_type", default_value="arc"),
        # Pure pursuit + deadman + waypoint follower. Both the deadman and
        # pure pursuit still come up DISABLED; this only starts the processes.
        DeclareLaunchArgument("control", default_value="false"),
        # forward   : goal is always N metres straight ahead in the ROBOT's
        #             frame, so odometry never enters the control loop. The
        #             environment defines the route; perception decides the
        #             deviations. This is what GeNIE's planner was built for.
        # waypoints : absolute goals in odom. Needs localisation good enough
        #             to trust over the whole run, which open-loop dead
        #             reckoning on grass is not.
        DeclareLaunchArgument("goal_mode", default_value="heading"),
        DeclareLaunchArgument("goal_distance", default_value="3.0"),
        DeclareLaunchArgument("route", default_value=os.path.join(
            HOME, "jetacker", "routes", "route_a.yaml")),
        DeclareLaunchArgument("max_speed", default_value="0.25"),
        DeclareLaunchArgument("foxglove", default_value="true"),
        # See the foxglove_bridge block for why these exist. Compression costs
        # CPU on a board whose CPU is already shared with the perception node,
        # so turn it off if you would rather spend that elsewhere. The buffer is
        # bytes: 10 MB is the bridge's own default and is far too much for a
        # saturated link, since every queued byte is latency you will watch.
        DeclareLaunchArgument("fg_compress", default_value="true"),
        DeclareLaunchArgument("fg_buffer", default_value="1000000"),
        DeclareLaunchArgument("record", default_value="false"),
        # The D455 offers 848x480 at 5/15/30/60 only - there is no 10, and
        # asking for one does not fail loudly: the driver falls back to its
        # default, which for colour is 1280x720x30.
        #
        # 15 rather than 5. Measured end-to-end latency is ~181 ms (perception
        # 111, planning 61), so at 5 fps the 200 ms frame period WAS the
        # bottleneck - the pipeline sat idle waiting for frames. At 15 fps the
        # camera stops being the constraint and perceptual latency falls
        # roughly threefold.
        #
        # This used to cost temporal context: writing memory every frame at
        # 15 fps gives the 7-slot bank a ~0.5 s span against the ~1.4 s it
        # trained with, and the fix (gating the writes) turned off CUDA graph
        # replay, worth 1.44x. Both are available now - the node's
        # memory_write_every defaults to 3, which puts writes 0.2 s apart at
        # 15 fps and keeps replay. If you change fps, change it too:
        # round(fps / 5).
        DeclareLaunchArgument("fps", default_value="15"),
        # MEASURE THIS. Positive pitches the camera down. Unmeasured pitch is
        # the largest single source of error in the geometry channels.
        DeclareLaunchArgument("camera_pitch_deg", default_value="0.0"),
        # 0.93 indoors on hard flooring; 1.01 outdoors on grass, where the
        # tyres bite differently. Measured by comparing odom displacement
        # against a tape over 1.58 m. Recalibrate per surface - this only
        # affects pose fed to the model's memory warp, not control.
        DeclareLaunchArgument("speed_scale", default_value="1.01"),
        # stretch | crop | pad -- how 848x480 is fitted to the square model
        # input. The D455f is 89x59 deg where training was 90x90, so this is
        # a real accuracy decision; evaluate on data once a checkpoint exists.
        DeclareLaunchArgument("input_fit", default_value="stretch"),
        DeclareLaunchArgument("bag_dir", default_value=os.path.join(HOME, "bags")),
        OpaqueFunction(function=launch_setup),
    ])
