# Gazebo Adaptation Results — KalmanNet vs Classical Baselines (and RL Meta-Tuner)

Branch: `kalyani`
Last updated: 2026-05-02

---

## TL;DR

> **Phase 1 — Static Gazebo:** Fine-tuned KalmanNet beats classical EKF
> and Strapdown INS by **3.6× on position RMSE** over 56 s of continuous
> IMU-only navigation, at 1.96 ms/step.
>
> **Phase 2 — Variable Gravity (RL Meta-Tuner):** A small PPO policy
> sitting on top of the frozen KalmanNet adapts the per-state gain online
> when world gravity changes mid-run. **Numbers below — fill in after
> Colab training and ConstructSim eval.**

---

# Phase 1 — Static Gazebo (NCLT KalmanNet adapted to TurtleBot3)

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

## Phase 1 interpretation

KalmanNet's two structural wins are **gravity handling** (it learned during
pretraining that `acc_z ≈ 9.81` should not contribute to vertical
velocity — EKF has no such prior) and **implicit body→world rotation**
(KalmanNet learned the time-varying mapping; EKF cannot without explicit
orientation input). The chunked evaluation hides both effects.

Raw outputs: [`eval_chunked.json`](eval_chunked.json),
[`eval_concat.json`](eval_concat.json).

---

# Phase 2 — Variable Gravity with RL Meta-Tuner

## Motivation

The fine-tuned KalmanNet is **rigid**: it was trained on Earth gravity
(`gz = -9.81 m/s²`) and has no mechanism to recalibrate if the
environment changes. A real deployment can't assume gravity is constant
(think: robot operating on a sloped surface where the body-frame
"vertical" component shifts, or an aerial vehicle in a different
atmosphere, or — for stress-testing — explicit gravity perturbations).

To make the filter **adaptive**, we add a small PPO meta-tuner that
observes the innovation signal and outputs per-state-dim multiplicative
adjustments to KalmanNet's learned gain `K`. Base KalmanNet weights
are frozen.

## Architecture

```
              IMU obs y_t
                  │
                  ▼
          ┌──────────────────┐
          │   KalmanNet      │  (frozen)
          │   computes K_t   │
          └──────────────────┘
                  │
       ┌──────────┴──────────┐
       │                     │
       ▼                     ▼
   innovation        K_t (learned gain)
       │                     │
       ▼                     │
  ┌───────────┐              │
  │ PPO MLP   │  s_t ∈ ℝ^6  │
  │  16-16    │──────────┐   │
  └───────────┘          │   │
                         ▼   ▼
                    K_final = K_t * s_t
                         │
                         ▼
                     x_post
```

| | |
|---|---|
| Action space | `Box(-1, 1, shape=(6,))` → gain scaler `1.0 + 0.5·a` ∈ [0.5, 1.5] |
| Observation | innovation (3) + ema innov magnitude (1) + posterior |v| (1) + last action (6) = 11 dims |
| Reward | `-mean((x_post - x_gt)²)` per step |
| Episode | one 200-step training sequence |
| Algorithm | PPO (stable-baselines3) |
| Policy | tiny MLP `16 → 16 → 6` (~500 params), suitable for edge |
| Training compute | ~100k env steps, ~10–15 min on Colab CPU |

## Variable-gravity environment

Real Gazebo physics, not synthetic. The `Gazebo/gravity_modulator.py`
ROS2 node calls `/gazebo/set_physics_properties` every N seconds and
samples a new `gz ~ Uniform([-11, -7.5]) m/s²`. The IMU plugin reports
the new gravity in `linear_acceleration`, so recordings contain real
physics-driven baseline shifts.

Recording protocol: **20 minutes** of driving with gravity changing every
**10 seconds** (~120 distinct gravity regimes per recording).

## Methods compared

| Method | Description |
|---|---|
| **EKF** | Same 9-D augmented-state EKF as Phase 1. No gravity awareness. |
| **KalmanNet (frozen)** | Phase 1 best, but never seen variable gravity. |
| **KalmanNet + RL Meta-Tuner** | This work. Frozen KalmanNet + PPO MLP. |

## Results — variable gravity (concat mode, 56 s+)

> **TODO**: fill in after Colab training and ConstructSim eval.
> Run `python3 Model/eval_metatuner.py --data gazebo_test_vargrav.npz
> --weights best_knet_gazebo.pt --policy meta_tuner_ppo.zip` and copy
> the table from `rl_eval.json`.

| Metric | EKF | KalmanNet (frozen) | KalmanNet + RL | RL wins by |
|---|---:|---:|---:|---:|
| RMSE position (m) | TBD | TBD | TBD | TBD |
| MAE position (m) | TBD | TBD | TBD | TBD |
| RMSE velocity (m/s) | TBD | TBD | TBD | TBD |
| Inlier precision <1 m | TBD | TBD | TBD | TBD |
| Latency (ms/step) | TBD | TBD | TBD | TBD |

### Per-state RMSE

| State | EKF | KalmanNet | KalmanNet + RL |
|---|---:|---:|---:|
| `px` (m) | TBD | TBD | TBD |
| `py` (m) | TBD | TBD | TBD |
| `pz` (m) | TBD | TBD | TBD |
| `vx` (m/s) | TBD | TBD | TBD |
| `vy` (m/s) | TBD | TBD | TBD |
| `vz` (m/s) | TBD | TBD | TBD |

## Phase 2 interpretation (template)

**Expected outcome:** Frozen KalmanNet degrades on variable gravity
(particularly on `pz`/`vz` since gravity changes directly affect `acc_z`).
RL Meta-Tuner reduces that degradation by adjusting the gain to reject
the spurious vertical signal during gravity transitions. Latency cost is
small (~0.1 ms/step extra for the 16-16 MLP).

If RL meta-tuner does not improve over frozen KalmanNet, candidate
explanations to investigate:
1. PPO hasn't learned a useful policy → train longer (300k+ steps),
   try larger policy net, increase exploration.
2. Action space is too restrictive (gain ∈ [0.5, 1.5]) → widen.
3. Observation space lacks the necessary signal → add longer history,
   add Q-values or recurrent features.

---

## Reproducing

```bash
# --- Phase 1 ---
ros2 launch turtlebot3_gazebo empty_world.launch.py     # terminal A
python3 Gazebo/gazebo_nclt_recorder.py                  # terminal B
python3 Gazebo/auto_drive.py                            # terminal C
# Ctrl+C C, then B → ~/.ros/gazebo_*.npz

cd Model/
python3 finetune_gazebo.py --data-dir ~/.ros \
    --init-weights best_knet_nclt.pt \
    --out-weights best_knet_gazebo.pt --epochs 30 --lr 1e-4

python3 eval_compare.py --data ~/.ros/gazebo_test.npz \
    --weights best_knet_gazebo.pt --mode chunked --out eval_chunked.json
python3 eval_compare.py --data ~/.ros/gazebo_test.npz \
    --weights best_knet_gazebo.pt --mode concat --out eval_concat.json

# --- Phase 2 ---
# Variable-gravity recording (4 terminals)
ros2 launch turtlebot3_gazebo empty_world.launch.py     # A
python3 Gazebo/gravity_modulator.py --interval 10 \
        --min-g -11 --max-g -7.5                        # B
python3 Gazebo/gazebo_nclt_recorder.py                  # C
python3 Gazebo/auto_drive.py                            # D
# Drive 15-20 min, then Ctrl+C D, then C
mv ~/.ros/gazebo_train.npz ~/.ros/gazebo_train_vargrav.npz
mv ~/.ros/gazebo_test.npz  ~/.ros/gazebo_test_vargrav.npz

# Train PPO meta-tuner (Colab, GPU optional)
python3 Model/rl_metatuner.py train \
    --weights best_knet_gazebo.pt \
    --data    gazebo_train_vargrav.npz \
    --steps   100000 \
    --out     meta_tuner_ppo.zip

# Evaluate
python3 Model/eval_metatuner.py \
    --data    gazebo_test_vargrav.npz \
    --weights best_knet_gazebo.pt \
    --policy  meta_tuner_ppo.zip \
    --mode    concat \
    --out     rl_eval.json
```

## Limitations

(Phase 1 limitations as before, plus Phase 2:)

6. **PPO trained offline on recorded sequences**, not in true closed loop
   with Gazebo. The action affects the state estimate but not the robot
   trajectory, so there's no feedback loop to learn around. Acceptable
   for a meta-tuner (it doesn't change motion), but worth noting.
7. **Gravity range** explored is `[-11, -7.5] m/s²`. Wider ranges or
   different perturbations (bias drift, IMU misalignment) would test
   generalization further.
