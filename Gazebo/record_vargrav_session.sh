#!/usr/bin/env bash
# Records multiple Gazebo sessions, each with a different gravity_z,
# by editing the world file before each launch (real physics, no
# post-hoc perturbation).
#
# For each gravity value:
#   1. Generate a custom world with that gravity_z
#   2. Launch turtlebot3_gazebo (with the custom world)
#   3. Wait for /imu and /odom to appear
#   4. Start recorder and auto_drive
#   5. Run for SECONDS_PER_GRAVITY seconds
#   6. Kill everything cleanly, save the npz with a tagged name
#
# At the end you'll have one train/val/test set per gravity value,
# ready to be concatenated by concat_recordings.py.
#
# Usage:
#   bash record_vargrav_session.sh
#   # or override defaults:
#   GRAVITIES="-9.81 -8.5 -10.5" SECONDS_PER_GRAVITY=120 \
#       bash record_vargrav_session.sh

set -euo pipefail

# ---- config -------------------------------------------------------- #
GRAVITIES="${GRAVITIES:--11.0 -10.0 -9.0 -8.0 -7.0}"     # 5 sessions
SECONDS_PER_GRAVITY="${SECONDS_PER_GRAVITY:-180}"        # 3 min each
WORLD_FILE="$HOME/.ros/vargrav.world"
OUT_DIR="$HOME/.ros/vargrav"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT_DIR"

# ---- helpers ------------------------------------------------------- #
launch_gazebo() {
    local gz="$1"
    bash "$SCRIPT_DIR/make_world.sh" "$gz" > "$WORLD_FILE"
    echo "[record_vargrav] Generated world with gz=$gz at $WORLD_FILE"

    export TURTLEBOT3_MODEL=waffle
    ros2 launch turtlebot3_gazebo empty_world.launch.py \
        world:="$WORLD_FILE" > /tmp/gazebo.log 2>&1 &
    GAZEBO_PID=$!
    echo "[record_vargrav] Gazebo launched (pid $GAZEBO_PID); waiting for /imu …"
}

wait_for_topic() {
    # Wait up to 60 s for both /imu and /odom to appear and publish.
    local deadline=$((SECONDS + 60))
    while [ $SECONDS -lt $deadline ]; do
        if ros2 topic list 2>/dev/null | grep -q "/imu" && \
           ros2 topic list 2>/dev/null | grep -q "/odom"; then
            sleep 5     # extra grace for plugins to settle
            echo "[record_vargrav] /imu and /odom up."
            return 0
        fi
        sleep 2
    done
    echo "[record_vargrav] ERROR: /imu/ /odom did not appear within 60s."
    return 1
}

run_session() {
    local gz="$1"
    local tag="g${gz//[.-]/_}"     # e.g. -8.5 -> g_8_5

    launch_gazebo "$gz"
    wait_for_topic

    # Recorder (saves to ~/.ros/gazebo_*.npz on Ctrl+C / SIGTERM)
    python3 "$SCRIPT_DIR/gazebo_nclt_recorder.py" \
        > /tmp/recorder.log 2>&1 &
    REC_PID=$!

    # Driver
    python3 "$SCRIPT_DIR/auto_drive.py" \
        > /tmp/driver.log 2>&1 &
    DRV_PID=$!

    echo "[record_vargrav] Recording for ${SECONDS_PER_GRAVITY}s at gz=$gz …"
    sleep "$SECONDS_PER_GRAVITY"

    # Stop driver first (zeroes /cmd_vel), then recorder (saves npz).
    kill -INT $DRV_PID 2>/dev/null || true
    sleep 1
    kill -INT $REC_PID 2>/dev/null || true
    sleep 3

    # Move tagged copies aside
    for split in train val test; do
        if [ -f "$HOME/.ros/gazebo_${split}.npz" ]; then
            mv "$HOME/.ros/gazebo_${split}.npz" \
               "$OUT_DIR/${tag}_${split}.npz"
            echo "[record_vargrav]   saved $OUT_DIR/${tag}_${split}.npz"
        fi
    done

    # Tear down Gazebo
    kill -INT $GAZEBO_PID 2>/dev/null || true
    sleep 5
    pkill -f gzserver 2>/dev/null || true
    pkill -f gzclient 2>/dev/null || true
    sleep 2
}

# ---- main loop ----------------------------------------------------- #
for gz in $GRAVITIES; do
    run_session "$gz"
done

echo
echo "[record_vargrav] All sessions complete."
echo "[record_vargrav] Per-gravity files in: $OUT_DIR"
ls -la "$OUT_DIR"
echo
echo "Next: python3 $SCRIPT_DIR/concat_recordings.py --in-dir $OUT_DIR --out-dir ~/.ros"
