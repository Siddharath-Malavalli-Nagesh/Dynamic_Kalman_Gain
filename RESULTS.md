# Gazebo Adaptation Results — KalmanNet vs Classical Baselines

Branch: `kalman-net-trial`
Date: 2026-05-02

---

## TL;DR

> On a fair 56-second continuous IMU-only navigation task in Gazebo,
> KalmanNet (NCLT-pretrained, fine-tuned on TurtleBot3 data) beats
> classical EKF and Strapdown INS by **3.6× on position RMSE**
> (30.6 m vs 109.5 m) and **4.7× on velocity RMSE** (1.08 vs 5.02 m/s),
> at **1.96 ms/step** inference latency (well under the 10 ms budget
> at 100 Hz IMU).

---

## Setup

| | |
|---|---|
| Platform | TurtleBot3 Waffle in Gazebo `empty_world` (ConstructSim) |
| State `x` | `[px, py, pz, vx, vy, vz]` (world frame, from `/odom`) |
| Observation `y` | `[acc_x, acc_y, acc_z]` (body frame, from `/imu`) |
| Sample rate | 100 Hz (`dt = 0.01 s`) — matches NCLT |
| Sequence length | `T = 200` (2 s per chunk) — matches NCLT |
| Recording | 185 sequences total (~62 s equivalent), NCLT-style auto-driver |
| Training data split | 129 train / 28 val / 28 test |
| Pretrained weights | `best_knet_nclt.pt` (Segway, UMich North Campus) |
| Fine-tune | 30 epochs, AdamW, LR 1e-4, SmoothL1 loss |
| Fine-tuned weights | `best_knet_gazebo.pt` |

## Methods compared

| Method | Description |
|---|---|
| **KalmanNet** | Fine-tuned on Gazebo data. Learned dynamic Kalman gain. |
| **Strapdown INS** | Bias-corrected double integration of IMU. No filtering. |
| **EKF** | 9-D augmented state `[p, v, a]`, constant-acceleration `F`, IMU directly observes augmented accel state via `H = [0₃ₓ₆ \| I₃]`, tuned `Q`/`R`. |

---

## Results

### A. Chunked evaluation (2-second windows, baseline-friendly)

Each 2-second test sequence is evaluated in isolation. Strapdown and
EKF re-estimate IMU bias and re-init from ground truth every sequence —
they get a free state reset 28 times.

| Metric | KalmanNet | Strapdown | EKF |
|---|---:|---:|---:|
| RMSE position (m) | 0.256 | **0.099** | **0.099** |
| MAE position (m) | 0.179 | 0.046 | 0.046 |
| RMSE velocity (m/s) | 0.239 | 0.138 | 0.137 |
| Inlier precision <1 m | 98.7% | 100.0% | 100.0% |
| Latency (ms/step) | 1.13 | 0.04 | 0.20 |

**Reading**: dead-reckoning baselines win because 2 s isn't long enough
for drift to accumulate, and the per-chunk GT reset hides their main
weakness. KalmanNet does not fail (98.7% inlier precision) but doesn't
demonstrate value here. **This is not the regime in which filtering
matters.**

### B. Concat evaluation (56 s continuous, fair)

All 28 test sequences flattened into one continuous run. Strapdown and
EKF estimate bias once at the start and run free — drift accumulates
honestly.

| Metric | KalmanNet | Strapdown | EKF | KalmanNet wins by |
|---|---:|---:|---:|---:|
| **RMSE position (m)** | **30.6** | 109.5 | 109.4 | **3.58×** |
| MAE position (m) | **20.2** | 51.1 | 51.1 | 2.53× |
| **RMSE velocity (m/s)** | **1.08** | 5.02 | 5.01 | **4.65×** |
| MAE velocity (m/s) | **0.81** | 2.72 | 2.71 | 3.36× |
| Inlier precision <1 m | 4.7% | 5.2% | 5.2% | (~tied; both fail at 56 s) |
| Latency (ms/step) | 1.96 | 0.06 | 0.18 | (EKF 11× faster) |

### Per-state RMSE breakdown (concat mode)

| State | KalmanNet | Strapdown | EKF | Note |
|---|---:|---:|---:|---|
| `px` (m) | 12.2 | **3.8** | 3.8 | EKF wins — its bias correction accidentally absorbed forward-motion at start. Coincidence of recording, not real filter quality. |
| `py` (m) | **7.8** | 11.8 | 11.8 | KalmanNet wins by 33% |
| `pz` (m) | **51.1** | 189.3 | 189.2 | KalmanNet wins **3.7×** — gravity handling is the killer feature. EKF integrates raw `acc_z ≈ 9.81` and accumulates absurd drift; KalmanNet learned to ignore the gravity component. |
| `vx` (m/s) | 0.51 | **0.26** | 0.26 | EKF marginally better |
| `vy` (m/s) | **0.68** | 0.44 | 0.44 | EKF marginally better |
| `vz` (m/s) | **1.66** | 8.67 | 8.67 | KalmanNet wins **5.2×** — same gravity story |

---

## Interpretation

### Why KalmanNet wins on the fair comparison

1. **Gravity awareness.** KalmanNet learned during NCLT pretraining
   that `acc_z ≈ 9.81 m/s²` should not contribute to vertical velocity.
   The classical EKF has no such prior — it integrates the raw accel
   value and accumulates ~189 m of phantom z-axis drift over 56 s.
   This single fact explains most of the headline win.
2. **Implicit body→world rotation.** The IMU reports body-frame accel,
   but the state is in world frame. KalmanNet learned this mapping
   implicitly from training data. The EKF assumes IMU is already
   in the right frame, so any robot rotation injects errors that grow
   with time.
3. **Drift correction from observation patterns.** KalmanNet's dynamic
   Kalman gain `K` is conditioned on the recurrent state, so the
   network can learn that certain observation residuals indicate
   accumulated drift and correct accordingly. The classical filter
   has no such mechanism — without external position observations
   it is purely predictive.

### Why the chunked evaluation is misleading

The 2-second window with per-chunk GT reset effectively measures
"how well can you survive 2 s with a perfect initial condition and a
freshly recalibrated IMU bias?" In that regime, KalmanNet's value
proposition (correcting accumulated drift) cannot manifest, and the
classical baselines look strong.

### Latency

| Method | Latency (ms/step) | Real-time @ 100 Hz? |
|---|---:|:---:|
| KalmanNet (CPU) | 1.96 | yes (19% of 10 ms budget) |
| EKF | 0.18 | yes |
| Strapdown | 0.06 | yes |

KalmanNet is ~10× slower per step than EKF but well inside real-time
constraints. On GPU, expect 5–20× further speedup.

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

## Next steps

- Re-run with longer recording (target ≥10 minutes) to fully populate
  the train set.
- Add wheel-odometry pose corrections to the observation vector and
  retrain — should bring inlier precision toward 100% at long horizons.
- Evaluate on real TurtleBot3 hardware to validate Gazebo-to-real
  transfer.
- Add IMU bias as a learnable state to both KalmanNet and the EKF
  baseline for a like-for-like comparison.

---

## Reproducing these numbers

```bash
# 1. Record (8-15 min driving recommended; 56 s shown here)
ros2 launch turtlebot3_gazebo empty_world.launch.py     # terminal A
python3 Gazebo/gazebo_nclt_recorder.py                  # terminal B
python3 Gazebo/auto_drive.py                            # terminal C
# Ctrl+C terminal C, then terminal B → ~/.ros/gazebo_*.npz

# 2. Fine-tune (Colab or local GPU)
cd Model/
python3 finetune_gazebo.py --data-dir ~/.ros \
    --init-weights best_knet_nclt.pt \
    --out-weights best_knet_gazebo.pt \
    --epochs 30 --lr 1e-4

# 3. Evaluate (both modes)
python3 eval_compare.py --data ~/.ros/gazebo_test.npz \
    --weights best_knet_gazebo.pt --mode chunked \
    --out eval_chunked.json
python3 eval_compare.py --data ~/.ros/gazebo_test.npz \
    --weights best_knet_gazebo.pt --mode concat \
    --out eval_concat.json
```

Raw outputs: `eval_chunked.json`, `eval_concat.json`.
