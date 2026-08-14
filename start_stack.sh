#!/usr/bin/env bash
# Bring up the stack in a tmux session laid out for running trials.
#
#   ./start_stack.sh base     camera + drive + odometry
#   ./start_stack.sh model    + perception
#   ./start_stack.sh plan     + BEV planner
#   ./start_stack.sh auto     + pure pursuit, deadman, waypoints
#   ./start_stack.sh auto route_b record
#
# Panes:
#   0  launch output (all nodes)          2  status watch
#   1  operator commands, pre-typed       3  E-STOP, pre-typed
#
# Nothing moves on startup. The deadman and pure pursuit both come up
# DISABLED, so a stack left running unattended stays still.
#
#   Ctrl-b <arrow>  move   Ctrl-b z  zoom   Ctrl-b d  detach
#   tmux attach -t robot   tmux kill-session -t robot

set -u
MODE="${1:-base}"
ROUTE="${2:-route_a}"
RECORD="${3:-}"
# Override with e.g. FPS=5 ./start_stack.sh auto
FPS="${FPS:-15}"

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
[ "$RECORD" = "record" ] && ARGS="$ARGS record:=true"

if [ "$MODE" != "base" ] && [ ! -f "$SAMVTP/sam_vtp.pt" ]; then
    echo "checkpoint not found: $SAMVTP/sam_vtp.pt"; exit 1
fi
if [ "$MODE" = "auto" ] && [ ! -f "$JETACKER/routes/${ROUTE}.yaml" ]; then
    echo "route not found: $JETACKER/routes/${ROUTE}.yaml"; exit 1
fi

tmux has-session -t $SESSION 2>/dev/null && {
    echo "session '$SESSION' exists: tmux attach -t $SESSION"
    echo "                or kill:  tmux kill-session -t $SESSION"; exit 1; }

# Pin the clocks. Without this the schedutil governor reads the bursty,
# latency-sensitive load as moderate average utilisation and drops to
# 730 MHz - which measured as a ~2x latency penalty across the pipeline.
if command -v jetson_clocks >/dev/null 2>&1; then
    sudo jetson_clocks 2>/dev/null && echo "clocks pinned to maximum"
fi

tmux new-session -d -s $SESSION -n stack -c "$JETACKER"

# 0: the launch
tmux send-keys -t $SESSION:stack.0 "$SRC && ros2 launch $JETACKER/robot_bringup.launch.py $ARGS" C-m

# 1: operator commands, queued but NOT run
tmux split-window -t $SESSION:stack -h -c "$JETACKER"
tmux send-keys -t $SESSION:stack.1 "$SRC && clear" C-m
if [ "$MODE" = "auto" ]; then
    tmux send-keys -t $SESSION:stack.1 \
      "ros2 topic pub --once /pursuit_enable std_msgs/msg/Bool '{data: true}' && ros2 topic pub --once /deadman_enable std_msgs/msg/Bool '{data: true}' && ros2 topic pub --once /route_start std_msgs/msg/Bool '{data: true}'"
fi

# 2: status watch
tmux split-window -t $SESSION:stack.1 -v -c "$JETACKER"
tmux send-keys -t $SESSION:stack.2 "$SRC && clear" C-m
if [ "$MODE" = "auto" ]; then
    tmux send-keys -t $SESSION:stack.2 \
      "watch -n1 'ros2 topic echo /deadman/status --once 2>/dev/null; ros2 topic echo /pure_pursuit/status --once 2>/dev/null; ros2 topic echo /waypoint_follower/status --once 2>/dev/null'" C-m
else
    tmux send-keys -t $SESSION:stack.2 "ros2 topic echo /sam_vtp/status"
fi

# 3: E-STOP, one Enter away at all times
tmux split-window -t $SESSION:stack.0 -v -p 25 -c "$JETACKER"
tmux send-keys -t $SESSION:stack.3 "$SRC && clear && echo '=== E-STOP: press Enter ===' " C-m
tmux send-keys -t $SESSION:stack.3 \
  "ros2 topic pub --once /estop std_msgs/msg/Bool '{data: true}'"

tmux select-pane -t $SESSION:stack.1
echo "started '$SESSION' mode=$MODE route=$ROUTE fps=$FPS ${RECORD:+record}"
tmux attach -t $SESSION
