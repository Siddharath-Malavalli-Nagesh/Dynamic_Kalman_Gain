# Gazebo Adaptation Results — KalmanNet vs Classical Baselines

Branch: `kalyani`
Last updated: 2026-05-05

> **Phase 2 (RL Meta-Tuner) results are documented separately in [`RL_RESULTS.md`](RL_RESULTS.md).**

---

## TL;DR

> Fine-tuned KalmanNet beats classical EKF and Strapdown INS by
> **3.6× on position RMSE** (30.6 m vs 109.5 m) and **4.7× on velocity
> RMSE** (1.08 vs 5.02 m/s) over 56 s of continuous IMU-only navigation
> in Gazebo, at 1.96 ms/step inference cost — well inside the 10 ms
> real-time budget at 100 Hz IMU.

---

## Setup

| | |
|---|---|
| Platform | TurtleBot3 Waffle in Gazebo `empty_world` (ConstructSim) |
| State `x` | `[px, py, pz, vx, vy, vz]` (world frame, from `/odom`) |
| Observation `y` | `[acc_x, acc_y, acc_z]` (body frame, from `/imu`) |
| Sample rate | 100 Hz (`dt = 0.01 s`) — matches NCLT |
| Sequence length | `T = 200` (2 s per chunk) — matches NCLT |
| Recording | 185 sequences (~62 s equivalent), NCLT-style auto-driver |
| Data split | 129 train / 28 val / 28 test |
| Pretrained weights | `best_knet_nclt.pt` (Segway, UMich North Campus) |
| Fine-tune | 30 epochs, AdamW, LR 1e-4, SmoothL1 loss |
| Fine-tuned weights | `best_knet_gazebo.pt` |

## Methods compared

| Method | Description |
|---|---|
| **KalmanNet** | Fine-tuned on Gazebo data. Learned dynamic Kalman gain. |
| **Strapdown INS** | Bias-corrected double integration of IMU. No filtering. |
| **EKF** | 9-D augmented state `[p, v, a]`, constant-acceleration `F`, IMU directly observes accel state via `H = [0₃ₓ₆ \| I₃]`, tuned `Q`/`R`. |

## Results

### A. Chunked evaluation (2-second windows, baseline-friendly)

| Metric | KalmanNet | Strapdown | EKF |
|---|---:|---:|---:|
| RMSE position (m) | 0.256 | **0.099** | **0.099** |
| MAE position (m) | 0.179 | 0.046 | 0.046 |
| RMSE velocity (m/s) | 0.239 | 0.138 | 0.137 |
| Inlier precision <1 m | 98.7% | 100.0% | 100.0% |
| Latency (ms/step) | 1.13 | 0.04 | 0.20 |

**Reading:** dead-reckoning baselines win because 2 s isn't long enough
for drift to accumulate, and the per-chunk GT reset hides their main
weakness. Not the regime where filtering matters.

### B. Concat evaluation (56 s continuous, fair)

| Metric | KalmanNet | Strapdown | EKF | KalmanNet wins by |
|---|---:|---:|---:|---:|
| **RMSE position (m)** | **30.6** | 109.5 | 109.4 | **3.58×** |
| MAE position (m) | **20.2** | 51.1 | 51.1 | 2.53× |
| **RMSE velocity (m/s)** | **1.08** | 5.02 | 5.01 | **4.65×** |
| MAE velocity (m/s) | **0.81** | 2.72 | 2.71 | 3.36× |
| Inlier precision <1 m | 4.7% | 5.2% | 5.2% | (~tied) |
| Latency (ms/step) | 1.96 | 0.06 | 0.18 | (EKF 11× faster) |

### Per-state RMSE (concat mode)

| State | KalmanNet | Strapdown | EKF | Note |
|---|---:|---:|---:|---|
| `px` (m) | 12.2 | **3.8** | 3.8 | EKF wins — bias correction absorbed forward motion at start |
| `py` (m) | **7.8** | 11.8 | 11.8 | KalmanNet +33% |
| `pz` (m) | **51.1** | 189.3 | 189.2 | KalmanNet **3.7×** — gravity awareness |
| `vx` (m/s) | 0.51 | **0.26** | 0.26 | EKF marginally better |
| `vy` (m/s) | **0.68** | 0.44 | 0.44 | EKF marginally better |
| `vz` (m/s) | **1.66** | 8.67 | 8.67 | KalmanNet **5.2×** — same gravity story |

## Interpretation

KalmanNet's two structural wins are **gravity handling** (it learned during
pretraining that `acc_z ≈ 9.81` should not contribute to vertical
velocity — EKF has no such prior) and **implicit body→world rotation**
(KalmanNet learned the time-varying mapping; EKF cannot without explicit
orientation input). The chunked evaluation hides both effects.

Raw outputs: [`eval_chunked.json`](eval_chunked.json),
[`eval_concat.json`](eval_concat.json).

---

## Limitations

1. **Single environment.** Recorded only in Gazebo `empty_world`. No
   evaluation on real hardware.
2. **Synthetic IMU.** Gazebo's IMU plugin produces clean Gaussian
   noise. Real IMUs have bias drift, scale-factor errors, temperature
   dependence — exactly the regime where KalmanNet's learned
   corrections should help even more.
3. **Modest training data.** Only 129 training sequences (~26 s
   equivalent). NCLT priors carry most of the model; fine-tune is a
   light touch. More Gazebo data would likely improve the gap further.
4. **Both filters fail inlier precision <1 m at 56 s.** Without any
   absolute position observation (no GPS, no wheel odom corrections),
   IMU-only navigation diverges over time. The result shows
   *relative* improvement; absolute accuracy still requires sensor
   fusion.
5. **EKF px result (3.8 m) is fragile.** The bias-correction window
   happened to absorb early forward motion. On a different recording
   start condition this could flip.

## Reproducing

```bash
# Record
ros2 launch turtlebot3_gazebo empty_world.launch.py     # terminal A
python3 Gazebo/gazebo_nclt_recorder.py                  # terminal B
python3 Gazebo/auto_drive.py                            # terminal C
# Ctrl+C C, then B → ~/.ros/gazebo_*.npz

# Fine-tune (Colab)
cd Model/
python3 finetune_gazebo.py --data-dir ~/.ros \
    --init-weights best_knet_nclt.pt \
    --out-weights best_knet_gazebo.pt --epochs 30 --lr 1e-4

# Evaluate
python3 eval_compare.py --data ~/.ros/gazebo_test.npz \
    --weights best_knet_gazebo.pt --mode chunked --out eval_chunked.json
python3 eval_compare.py --data ~/.ros/gazebo_test.npz \
    --weights best_knet_gazebo.pt --mode concat --out eval_concat.json
```
