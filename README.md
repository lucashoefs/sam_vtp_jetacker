# sam_vtp_jetacker

ROS 2 deployment of [SAM-VTP](https://github.com/lucashoefs/sam_vtp) on a Hiwonder JetAcker: camera, perception, BEV planning, control, safety gate and trial recording. All perception runs onboard.

**Safety:** this software drives a physical robot. Test with the wheels off the ground and keep the E-STOP within reach. The gate stops the robot on stale perception or a lost path; it does not detect collisions.

## Requirements

- Ackermann JetAcker chassis (`/dev/ttyACM0`), Intel RealSense D455f, Jetson Orin Nano 8 GB (JetPack 6.2)
- ROS 2 Humble with `realsense2_camera`, sourced from `~/ros2_ws`; `foxglove_bridge` is optional
- `tmux`

## Setup

Paths are fixed in `start_stack.sh` and `robot_bringup.launch.py`, so clone to these locations:

```bash
git clone https://github.com/lucashoefs/sam_vtp_jetacker.git ~/jetacker
git clone --recurse-submodules https://github.com/lucashoefs/sam_vtp.git ~/sam_vtp

# Path planner, not distributed with this repository
git clone https://github.com/jiaming-ai/GENIE-SAMTP.git ~/GENIE-SAMTP
git -C ~/GENIE-SAMTP checkout 728aee2
ln -s ~/GENIE-SAMTP/genie_path_planner ~/jetacker/genie_path_planner
```

Create `~/sam_vtp/.venv` and install SAM-VTP as described in its README, plus the planner's dependencies (`pyyaml`, `scikit-learn`, `opencv-python`).

The stack expects two checkpoints: `~/sam_vtp/sam_vtp.pt` (SAM-VTP) and `~/sam_vtp/sam_tp.pt` (frame-wise baseline).

Create them with `export_checkpoint.py`, which strips training state, verifies the result by loading it and writes a `.sha256` file:

```bash
./export_checkpoint.py runs/<run>/best_model.pt ~/sam_vtp/sam_vtp.pt
```

## Usage

```bash
MODEL=vtp ./start_stack.sh auto route_a record   # SAM-VTP
MODEL=tp  ./start_stack.sh auto route_a record   # frame-wise baseline
```

The robot does not move on startup. The script opens a tmux session with panes for launch output, operator commands, status and the E-STOP. Wait for `model ready` in the launch output, then press Enter in the operator pane to publish the enables.

Modes are `base` (camera, drive, odometry), `model` (adds perception), `plan` (adds the planner) and `auto` (adds control). Optional environment variables are `FPS`, `COMPONENTS` (for example `memory=false`), `FOXGLOVE` and `EXTRA`, which is passed to `ros2 launch`.

The default goal mode is `heading`, so route files are only used with `EXTRA="goal_mode:=waypoints"`.

### Ending a trial

`trial_supervisor.py` runs on the robot and needs no network. A trial ends on any of the following, which latches the E-STOP and closes the bag:

| Trigger | Default | Launch argument |
|---|---|---|
| Tilt (lifting the robot), held 0.4 s | 30° | `stop_tilt_deg` |
| Distance from start | 25 m | `geofence_m` |
| Duration | 180 s | `max_trial_s` |
| Safety gate blocking continuously | 15 s | |

Raise the limits on steep ground:

```bash
EXTRA="stop_tilt_deg:=40 geofence_m:=40 max_trial_s:=240" \
  MODEL=vtp ./start_stack.sh auto route_a record
```

## Planning

`arc_planner_node.py` uses GeNIE's BEV projection and cost model, and replaces its path search with constant-curvature arcs so that planning runs at camera rate. It reads the graded map on `/sam_vtp/probability`, with cost `1 - P`. Cells with `P <= 0.25` are impassable.

In the default `veto` mode (`jetacker_planner.yaml`), depth marks cells at least 0.20 m high as obstacles but never makes a cell more traversable, so both models share the same geometric backstop.

## Layout

```
start_stack.sh            tmux session for running trials
robot_bringup.launch.py   launch file for the whole stack
jetacker_planner.yaml     planner configuration
sam_vtp_node.py           perception
arc_planner_node.py       BEV planner with arc search
genie_planner_node.py     GeNIE path search, for comparison
genie_fast.py             integral-image acceleration of GeNIE path costs
pure_pursuit_node.py      path to cmd_vel
deadman_node.py           freshness gate over the whole chain
trial_supervisor.py       trial termination
waypoint_follower.py      route goals (routes/)
jetacker_driver.py        chassis driver (rrc.py: board protocol)
jetacker_odometry.py      commanded speed and IMU yaw; no wheel encoders
export_checkpoint.py      checkpoint export
calibrate_*.py            speed and camera pitch calibration
system_check.sh           power mode, clocks, thermals and throttling
test_veto_fusion.py       veto-mode property tests (no ROS required)
```

## Third-party code

The planner package `genie_path_planner` comes from [GENIE-SAMTP](https://github.com/jiaming-ai/GENIE-SAMTP) and is linked at setup, unmodified. `genie_fast.py` accelerates its path cost evaluation at runtime.
