# Gazebo Adaptation (branch: `kalman-net-trial`)

This branch adapts the NCLT-trained KalmanNet (`best_knet_nclt.pt`) to a
TurtleBot3 in Gazebo, without changing the model architecture or the
existing NCLT pipeline.

## Contract preserved

| Tensor | Shape         | Meaning                                  |
|--------|---------------|------------------------------------------|
| `x`    | `[N, T, 6]`   | `[px, py, pz, vx, vy, vz]` world frame   |
| `y`    | `[N, T, 3]`   | `[acc_x, acc_y, acc_z]` body frame (IMU) |
| `dt`   | 0.01 s (100 Hz) | matches NCLT — `F` matrix unchanged    |
| `T`    | 200            | matches NCLT `SEQ_LEN`                  |

## What's added

```
Gazebo/
  gazebo_nclt_recorder.py     # ROS2 node: records /odom + /imu, saves
                              #   gazebo_train.npz / val.npz / test.npz
  gazebo_inference_node.py    # ROS2 node: online KalmanNet step,
                              #   subscribes /imu, publishes
                              #   /kalmannet_nclt/odom
  launch/gazebo_kalmannet.launch.py
Model/
  finetune_gazebo.py          # loads best_knet_nclt.pt, fine-tunes
                              #   on Gazebo data, saves
                              #   best_knet_gazebo.pt
```

## Why this design

1. **Same architecture, same I/O contract.** No surgery on
   `Model/train_model.py`. The NCLT weights load directly into the same
   `KalmanNet` class.
2. **Fine-tune, don't retrain.** NCLT cars and TurtleBots have different
   dynamics, but they share the kinematic structure (constant-velocity
   `F`, IMU as observation). Starting from NCLT priors and fine-tuning
   converges faster than training from scratch.
3. **100 Hz resampling.** Gazebo's `/odom` is ~30 Hz and `/imu` is
   ~200 Hz. The recorder time-syncs both onto a uniform 100 Hz grid so
   `F[i, i+3] = 0.01` stays valid without retraining.

## End-to-end workflow

### 1. Record Gazebo data

```bash
# Terminal A — Gazebo
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo empty_world.launch.py

# Terminal B — recorder
python3 Gazebo/gazebo_nclt_recorder.py

# Terminal C — auto-driver (NCLT-style schedule, cycles forever)
python3 Gazebo/auto_drive.py

# Let it run 8-15 minutes for a strong dataset (~1500-3000 sequences),
# then Ctrl+C the recorder (Terminal B) — it auto-saves on exit.
```
Or instead of `auto_drive.py` in Terminal C, drive manually with
`ros2 run turtlebot3_teleop teleop_keyboard`.

Outputs: `~/.ros/gazebo_train.npz`, `gazebo_val.npz`, `gazebo_test.npz`.

### 2. Fine-tune (Colab or local GPU)

```bash
cd Model/
python3 finetune_gazebo.py \
    --data-dir ~/.ros \
    --init-weights best_knet_nclt.pt \
    --out-weights best_knet_gazebo.pt \
    --epochs 30 --lr 1e-4
```
The script prints metrics for the NCLT-only baseline and the fine-tuned
model on the Gazebo test set so you can quantify the gain.

### 3. Run inference in Gazebo

```bash
cp best_knet_gazebo.pt ~/.ros/
ros2 launch kalmannet_ros2 gazebo_kalmannet.launch.py \
     weights:=~/.ros/best_knet_gazebo.pt
```
`/kalmannet_nclt/odom` carries the live state estimate at IMU rate.

### 4. Evaluate (offline, vs. classical baselines)

`Model/eval_compare.py` runs three methods on `gazebo_test.npz` and
prints a side-by-side table with per-state RMSE, MAE, % error,
inlier precision <1 m, and mean inference latency.

```bash
cd Model/
python3 eval_compare.py \
    --data ~/.ros/gazebo_test.npz \
    --weights best_knet_gazebo.pt
```

Methods compared:
- **KalmanNet** — fine-tuned weights
- **Strapdown INS** — bias-corrected double integration of IMU. Pure
  dead-reckoning baseline showing what happens with no filtering.
- **EKF** — augmented-state classical Kalman filter
  (`x = [px,py,pz,vx,vy,vz,ax,ay,az]`, IMU directly observes accel).
  This is the apples-to-apples classical benchmark.

### 5. Evaluate (live, in Gazebo)

`Gazebo/online_evaluator.py` subscribes to `/odom` (ground truth) and
`/kalmannet_nclt/odom` (estimator), aligns them by nearest timestamp,
and prints the same per-state metrics on Ctrl+C. Run alongside the
launch file:

```bash
ros2 run kalmannet_ros2 online_evaluator
# drive the robot in another terminal, then Ctrl+C the evaluator
```

## Frames and units (gotchas)

- **IMU frame.** Both NCLT and Gazebo IMU report `linear_acceleration`
  in the body frame. Do **not** rotate to the world frame — the model
  was trained on raw body-frame accel.
- **Gravity.** NCLT IMU includes gravity in `acc_z`. Gazebo's default
  `/imu` plugin does too. Leaving it in keeps the distribution consistent
  with NCLT priors.
- **World-frame velocity.** The recorder rotates `/odom` body-frame
  twist to the world frame using the quaternion yaw, matching NCLT
  ground-truth velocity (which is finite-diff of world-frame position).
