# sam_vtp_jetacker

ROS 2 deployment of [SAM-VTP](https://github.com/lucashoefs/sam_vtp) on a
Hiwonder JetAcker: camera, perception, BEV planner, pure-pursuit controller and
the safety gate used for the closed-loop experiments in the paper.

Hardware: Ackermann JetAcker, Intel RealSense D455f, Jetson Orin Nano 8 GB
(JetPack 6.2). All perception runs onboard.

## Run

```bash
MODEL=vtp ./start_stack.sh auto route_a record   # SAM-VTP
MODEL=tp  ./start_stack.sh auto route_a record   # frame-wise baseline
```

Nothing moves on startup. The script opens a tmux session; wait for the launch
pane to report `model ready`, then press Enter in the ops pane to publish the
enables. The E-STOP is pre-typed in its own pane, one Enter away.

## Stop

Pick the robot up. Tilting past 30 degrees for 0.4 s latches the e-stop, beeps
twice and closes the bag. Trials also end on their own at 180 s, 25 m, or 15 s
stuck. None of it needs a network. See `trial_supervisor.py`.

On rough ground raise the threshold, or a steep slope will trigger it:

```bash
EXTRA="stop_tilt_deg:=40 geofence_m:=40 max_trial_s:=240" \
  MODEL=vtp ./start_stack.sh auto route_a record
```

## Layout

```
start_stack.sh           tmux session laid out for running trials
robot_bringup.launch.py  camera, drive, odometry, perception, planning, control
sam_vtp_node.py          perception
arc_planner_node.py      BEV planner, constant-curvature arcs at camera rate
genie_planner_node.py    GeNIE's published path search, for comparison
pure_pursuit_node.py     path -> cmd_vel
deadman_node.py          freshness gate over the whole chain
trial_supervisor.py      ends a trial with no network
jetacker_driver.py       chassis serial driver (rrc.py is the board protocol)
system_check.sh          power mode, clocks, thermals, throttling
```

The planner is GeNIE's BEV projection and cost model; `arc_planner_node.py`
replaces its path search with constant-curvature arcs so the loop runs at the
camera rate. Neither is a contribution of the paper.
