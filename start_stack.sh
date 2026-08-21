#!/usr/bin/env bash
# Bring up the stack in a tmux session laid out for running trials.
#
#   ./start_stack.sh base     camera + drive + odometry
#   ./start_stack.sh model    + perception
#   ./start_stack.sh plan     + BEV planner
#   ./start_stack.sh auto     + pure pursuit, deadman, waypoints
#   ./start_stack.sh auto route_b record
#
# Environment overrides (all optional):
#
#   FPS=5            camera rate; the node's memory stride follows it
#   MODEL=vtp|tp    which weights to run. vtp (default) = SAM-VTP,
#                   tp = the SAM-TP baseline. This is the closed-loop A/B.
#   COMPONENTS=...   ablate SAM-VTP's own components, e.g. COMPONENTS=memory=false
#   FOXGLOVE=false   no live monitoring; see the warning below
#   EXTRA=...        anything else passed straight to ros2 launch
#
#   MODEL=tp FOXGLOVE=false ./start_stack.sh auto route_a record
#
# Panes:
#   0  launch output (all nodes)          2  status watch
#   1  operator commands, pre-typed       3  E-STOP, pre-typed
#
# Nothing moves on startup. The deadman and pure pursuit both come up
# DISABLED, so a stack left running unattended stays still.
#
# To END a run with no network: PICK THE ROBOT UP. Tilting it past 30 deg for
# 0.4 s latches the e-stop, beeps twice, and closes the bag. It also ends on
# its own at 180 s, 25 m, or 15 s stuck. See trial_supervisor.py.
#
#   Ctrl-b <arrow>  move   Ctrl-b z  zoom   Ctrl-b d  detach
#   tmux attach -t robot   tmux kill-session -t robot

set -u
MODE="${1:-base}"
ROUTE="${2:-route_a}"
RECORD="${3:-}"
# Override with e.g. FPS=5 ./start_stack.sh auto
FPS="${FPS:-15}"
# Which arm of the perception comparison. Empty = the checkpoint as trained.
COMPONENTS="${COMPONENTS:-}"
# Named rather than a raw path so the two arms cannot be mistyped into each
# other; both files are produced by export_checkpoint.py.
MODEL="${MODEL:-vtp}"
case "$MODEL" in
    vtp) CKPT="$HOME/sam_vtp/sam_vtp.pt" ;;
    tp)  CKPT="$HOME/sam_vtp/sam_tp.pt" ;;
    *)   CKPT="$MODEL" ;;   # an explicit path, for anything else
esac
if [ ! -f "$CKPT" ]; then
    echo "MODEL=$MODEL -> $CKPT does not exist on this machine." >&2
    exit 1
fi
if [ ! -f "$CKPT.sha256" ]; then
    echo "WARNING: no $CKPT.sha256 -- bags from this run cannot be traced to a build." >&2
fi
# Live monitoring. Default unchanged, but see the record warning below: on a
# marginal link this is the one component that can steal CPU from perception
# at the exact moment it matters.
FOXGLOVE="${FOXGLOVE:-true}"
# Free-form passthrough to ros2 launch, for anything without a variable here.
EXTRA="${EXTRA:-}"

SESSION=robot
JETACKER=$HOME/jetacker
SAMVTP=$HOME/sam_vtp
SRC="source $SAMVTP/.venv/bin/activate && source /opt/ros/humble/setup.bash \
&& source $HOME/ros2_ws/install/local_setup.bash"

case "$MODE" in
    base)  ARGS="" ;;
    model) ARGS="model:=true" ;;
    plan)  ARGS="model:=true planner:=true" ;;
    auto)  ARGS="model:=true planner:=true control:=true" ;;
    *) echo "usage: $0 {base|model|plan|auto} [route_a|route_b] [record]"; exit 1 ;;
esac

ARGS="$ARGS route:=$JETACKER/routes/${ROUTE}.yaml fps:=${FPS}"
ARGS="$ARGS foxglove:=${FOXGLOVE}"
# Empty is not "the default" downstream -- the launch file omits the -p rather
# than passing `components:=`, which would kill rclpy.init() before the node
# exists. Omit it here too so the two agree.
ARGS="$ARGS checkpoint:=${CKPT}"
[ -n "$COMPONENTS" ] && ARGS="$ARGS components:=${COMPONENTS}"
[ "$RECORD" = "record" ] && ARGS="$ARGS record:=true"
[ -n "$EXTRA" ] && ARGS="$ARGS $EXTRA"

if [ "$MODE" != "base" ] && [ ! -f "$SAMVTP/sam_vtp.pt" ]; then
    echo "checkpoint not found: $SAMVTP/sam_vtp.pt"; exit 1
fi
if [ "$MODE" = "auto" ] && [ ! -f "$JETACKER/routes/${ROUTE}.yaml" ]; then
    echo "route not found: $JETACKER/routes/${ROUTE}.yaml"; exit 1
fi

# WiFi does not fail cleanly outdoors. A silently dropped link gives no TCP
# RST, so foxglove_bridge can hold its ROS subscriptions for minutes and keep
# serialising and deflate-compressing image topics for a client that is gone.
# That is CPU taken from the perception node on a six-core board, during the
# run. The planner has already once starved the camera node from 15 Hz to 8.5
# through the same contention -- see the SINGLE_THREAD_ENV note in the launch
# file. Watch the robot before the trial, not during it.
if [ "$RECORD" = "record" ] && [ "$FOXGLOVE" != "false" ]; then
    echo "WARNING: recording with foxglove:=true."
    echo "         If the link degrades mid-run the bridge competes with"
    echo "         perception for CPU. FOXGLOVE=false for recorded trials."
    echo
fi

if [ -n "$COMPONENTS" ]; then
    echo "COMPONENT ABLATION: $COMPONENTS"
    echo "  (bag directory and /sam_vtp/status both record this)"
    echo
fi

tmux has-session -t $SESSION 2>/dev/null && {
    echo "session '$SESSION' exists: tmux attach -t $SESSION"
    echo "                or kill:  tmux kill-session -t $SESSION"; exit 1; }

# Power mode first, then clocks. jetson_clocks pins to the ceiling of the
# ACTIVE nvpmodel profile and cannot exceed it, so pinning clocks while the
# board sits in a low-power mode quietly runs the whole stack at that mode's
# cap. The Super modes are worth roughly 1.6x GPU clock and 1.5x memory
# bandwidth over the launch defaults, which is more than any code change here.
#
# Not set automatically: mode IDs differ between boards and JetPack versions,
# and switching modes can reboot. Checked loudly instead.
if command -v nvpmodel >/dev/null 2>&1; then
    PWR_MODE="$(sudo nvpmodel -q 2>/dev/null | tr '\n' ' ')"
    case "$PWR_MODE" in
        *SUPER*|*MAXN*) echo "power mode: $PWR_MODE" ;;
        *) echo "WARNING: power mode is '$PWR_MODE', not a MAXN/Super mode."
           echo "         GPU clock and memory bandwidth are capped below what"
           echo "         this board can do. 'sudo nvpmodel -m <id>' to fix;"
           echo "         'sudo nvpmodel -p --verbose' lists the ids." ;;
    esac
fi

# Pin the clocks. Without this the schedutil governor reads the bursty,
# latency-sensitive load as moderate average utilisation and drops to
# 730 MHz - which measured as a ~2x latency penalty across the pipeline.
if command -v jetson_clocks >/dev/null 2>&1; then
    sudo jetson_clocks 2>/dev/null && echo "clocks pinned to maximum"
fi

# Panes are addressed by ID (%0, %3, ...), never by index.
#
# tmux numbers panes by POSITION and renumbers on every split. The E-STOP split
# below inserts a pane at index 1 and pushes the right-hand column from 1,2 to
# 2,3 -- so `-t stack.3`, written for the pane that was just created, actually
# lands on the status-watch pane. That is not hypothetical: it typed the E-STOP
# setup into the pane running `watch`, where it sat buffered until watch exited
# and then ran as a garbled command, while the real E-STOP pane got nothing at
# all. A safety affordance that is only there on paper is worse than none.
#
# Pane IDs are assigned once and never reused, so they survive any resplit.
tmux new-session -d -s $SESSION -n stack -c "$JETACKER"
P_LAUNCH=$(tmux list-panes -t $SESSION:stack -F '#{pane_id}')

# The launch itself
tmux send-keys -t "$P_LAUNCH" "$SRC && ros2 launch $JETACKER/robot_bringup.launch.py $ARGS" C-m

# Operator commands, queued but NOT run
P_OPS=$(tmux split-window -t "$P_LAUNCH" -h -c "$JETACKER" -P -F '#{pane_id}')
tmux send-keys -t "$P_OPS" "$SRC && clear" C-m
if [ "$MODE" = "auto" ]; then
    # `--once` implies `-w 1`: ros2 topic pub blocks until one matching
    # subscription exists, in an unbounded sleep loop. That is what you want for
    # the two enables -- it removes the discovery race, where a message published
    # before the subscriber is matched is simply dropped and the robot silently
    # does not start.
    #
    # It is exactly what you do NOT want for /route_start. Its only subscriber is
    # waypoint_follower.py, which the launch file starts only when
    # goal_mode:=waypoints, and this script never passes goal_mode, so the launch
    # default of "heading" always applies and nothing subscribes. With the
    # implied -w 1 that publish waits forever: the robot moves, because the two
    # enables ahead of it already landed, and the operator pane hangs on a
    # command that cannot complete. -w 0 makes it the no-op it actually is.
    tmux send-keys -t "$P_OPS" \
      "ros2 topic pub --once /pursuit_enable std_msgs/msg/Bool '{data: true}' && ros2 topic pub --once /deadman_enable std_msgs/msg/Bool '{data: true}' && ros2 topic pub --once -w 0 /route_start std_msgs/msg/Bool '{data: true}'"
fi

# Status watch
P_STATUS=$(tmux split-window -t "$P_OPS" -v -c "$JETACKER" -P -F '#{pane_id}')
tmux send-keys -t "$P_STATUS" "$SRC && clear" C-m
if [ "$MODE" = "auto" ]; then
    tmux send-keys -t "$P_STATUS" \
      "watch -n1 'ros2 topic echo /trial_supervisor/status --once 2>/dev/null; ros2 topic echo /deadman/status --once 2>/dev/null; ros2 topic echo /pure_pursuit/status --once 2>/dev/null; ros2 topic echo /waypoint_follower/status --once 2>/dev/null'" C-m
else
    tmux send-keys -t "$P_STATUS" "ros2 topic echo /sam_vtp/status"
fi

# E-STOP, one Enter away at all times.
# -l 25% rather than -p 25: -p was deprecated and then removed, and on tmux 3.4
# it fails with "size missing", so the pane is never created and every later
# reference to it silently does nothing.
P_ESTOP=$(tmux split-window -t "$P_LAUNCH" -v -l 25% -c "$JETACKER" -P -F '#{pane_id}')
tmux send-keys -t "$P_ESTOP" "$SRC && clear && echo '=== E-STOP: press Enter ===' " C-m
tmux send-keys -t "$P_ESTOP" \
  "ros2 topic pub --once /estop std_msgs/msg/Bool '{data: true}'"

# Fail loudly rather than leaving an unarmed E-STOP looking armed.
for p in "$P_LAUNCH" "$P_OPS" "$P_STATUS" "$P_ESTOP"; do
    [ -n "$p" ] || { echo "ERROR: a tmux pane failed to create; E-STOP may not be armed."; break; }
done

tmux select-pane -t "$P_OPS"
echo "started '$SESSION' mode=$MODE route=$ROUTE fps=$FPS \
foxglove=$FOXGLOVE ${COMPONENTS:+components=$COMPONENTS }${RECORD:+record}"
tmux attach -t $SESSION
