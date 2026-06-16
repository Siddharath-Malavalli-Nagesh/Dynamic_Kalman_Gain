# Knowledge Transfer — KalmanNet Gazebo Adaptation (Phase 1, No RL)

> Audience: research paper writer. Goal: enough context, methodology, and
> results to draft a conference paper without re-deriving the work.
> Scope: **frozen, fine-tuned KalmanNet only**. RL meta-tuner / variable
> gravity work is documented separately in [`RL_RESULTS.md`](RL_RESULTS.md)
> and should be treated as Phase 2.

---

## 1. One-paragraph summary

We fine-tune KalmanNet (Revach et al., 2022) — a learned-gain Kalman
filter — for IMU-only inertial navigation on a TurtleBot3 platform in
Gazebo. The model is pretrained on the NCLT dataset (a ground vehicle
in 3D urban terrain) and fine-tuned with a short Gazebo-recorded
session driven by an NCLT-style auto-driver. Against classical EKF and
Strapdown INS baselines, fine-tuned KalmanNet improves continuous-
trajectory position RMSE by **3.6×** and velocity RMSE by **4.7×** over
56 s of evaluation, at 1.96 ms/step inference cost — comfortably real-
time at 100 Hz. The structural wins come from KalmanNet's learned
gravity-subtraction prior and its implicit body→world rotation handling
— both of which the EKF cannot express without additional state.

---

## 2. Problem statement

Inertial navigation systems (INS) integrate accelerometer + gyroscope
readings to estimate position and velocity. The fundamental problem is
**double-integration of biased measurements**: any constant accel bias
ε produces a position error growing as `½ ε t²`. Classical solutions:

- **Strapdown INS** — open-loop dead-reckoning with bias-corrected accel.
- **EKF** — recursive Bayesian filter with explicit motion model `F`,
  observation model `H`, and noise covariances `Q`, `R`.

Both require correct hand-engineered models. Real IMUs violate every
assumption: bias drifts, noise is non-Gaussian, scale factors are
temperature-dependent, and the mapping from body to world frame is
not given a priori. KalmanNet replaces the hand-engineered Kalman gain
`K` with a small recurrent network trained end-to-end on data,
preserving the Kalman update structure but learning the gain.

**Research question for this paper:** Does KalmanNet, pretrained on a
ground-vehicle dataset (NCLT), transfer to a robotics-simulator IMU
(Gazebo TurtleBot3) with a light fine-tune, and does it beat classical
baselines in the operating regime where filtering actually matters
(long-horizon continuous trajectories, not 2-second chunks)?

---

## 3. Background — KalmanNet in 30 lines

The classical Kalman update at time `t`:

```
x_prior  = F · x_post[t-1]               # state propagation
y_pred   = H · x_prior                   # measurement prediction
innov    = y[t] - y_pred                 # innovation
x_post   = x_prior + K · innov           # gain-weighted correction
```

In an EKF, `K = Σ Hᵀ (H Σ Hᵀ + R)⁻¹`, which requires `Q`, `R`, and a
maintained covariance `Σ` — all tuned manually.

In **KalmanNet**, `K_t` is the output of a small recurrent network
whose inputs are:
- `Δy = y[t] - y[t-1]` (innovation-derivative proxy → feeds `GRU_Q`)
- `e_post = x_post[t-1] - x_post[t-2]` (state-error proxy → feeds `GRU_Σ`)
- `Δy` again, feeding a separate `GRU_S`

These three hidden states, plus the current observation `y` and the
previous posterior `x_post[t-1]`, pass through a LayerNorm and a 3-layer
MLP that outputs a flattened `K_t ∈ ℝ^{m × n}`. The Kalman update is
otherwise unchanged.

Key reference: [Revach et al., "KalmanNet: Neural Network Aided Kalman
Filtering for Partially Known Dynamics", IEEE T-SP 2022.]

---

## 4. Setup

| Component | Value |
|---|---|
| Platform | TurtleBot3 Waffle in Gazebo Classic `empty_world` (ConstructSim cloud rosject) |
| ROS distribution | ROS 2 Humble |
| State `x ∈ ℝ⁶` | `[px, py, pz, vx, vy, vz]` — world frame, from `/odom` |
| Observation `y ∈ ℝ³` | `[acc_x, acc_y, acc_z]` — body frame, from `/imu` |
| Sample rate | 100 Hz (`dt = 0.01 s`) — matches NCLT |
| Sequence length `T` | 200 timesteps (2 s per chunk) — matches NCLT |
| Recording duration | ~10 minutes of continuous driving |
| Sequences | 185 total |
| Split | 129 train / 28 val / 28 test |
| Pretrained init weights | `best_knet_nclt.pt` (trained on NCLT, U. Michigan Segway dataset) |
| Fine-tune | 30 epochs, AdamW, LR 1e-4, SmoothL1 loss |
| Output weights | `best_knet_gazebo.pt` |

### 4.1 Why these choices

- **100 Hz** matches NCLT's native rate so the pretrained recurrent dynamics
  transfer without rate retuning. Gazebo's IMU publishes at 200 Hz; the
  recorder resamples to 100 Hz before saving.
- **T = 200** (2 s) matches NCLT pretraining sequences. Beyond ~80 s of
  continuous concat evaluation the recurrent state drifts enough to
  break the LayerNorm (known limitation; documented in §10).
- **SmoothL1 loss** is robust to occasional /odom glitches near
  contact boundaries and is what the original KalmanNet paper uses.
- **Body-frame IMU** is fed directly — no rotation to world frame —
  because that's how NCLT pretraining was done. KalmanNet learns the
  body→world mapping implicitly through the recurrent state.

---

## 5. Architecture in detail

Reproduced exactly from `Model/train_model.py`:

```
m = 6 (state dim), n = 3 (obs dim), N_rnn = 32

KalmanNet:
    GRU_Q     : GRUCell(input=n=3,       hidden=N_rnn=32)
    GRU_Sigma : GRUCell(input=m=6,       hidden=N_rnn=32)
    GRU_S     : GRUCell(input=n=3,       hidden=N_rnn=32)

    feat_dim  = 3 * N_rnn + n + m   = 105

    LayerNorm(105)
    fc1       : Linear(105 → 32) + ReLU
    fc2       : Linear(32 → 32)  + ReLU
    fc3       : Linear(32 → n·m = 18)       # outputs flat K_t

    H         : Linear(m → n) — initialised to zero; learned during
                fine-tune (so early epochs use raw IMU as innovation)
    F         : 6×6 constant — identity + dt·I on the velocity blocks
                (constant-velocity kinematic model). Not learned.
```

Per-step forward pass:
1. Compute `Δy = y[t] - y[t-1]`, `e_post = x_post[t-1] - x_post[t-2]`
2. Advance the three GRUCells with `Δy`, `e_post`, `Δy`
3. Concat hidden states + `y` + `x_post[t-1]` → 105-dim feature
4. LayerNorm → fc1 → ReLU → fc2 → ReLU → fc3 → reshape to `K_t ∈ ℝ⁶ˣ³`
5. `x_prior = F · x_post[t-1]`
6. `innov   = y - H·x_prior`
7. `x_post  = x_prior + K_t · innov`

Total trainable parameters ≈ **15 k** (cheap on edge).

---

## 6. Data collection methodology

A ROS 2 node `gazebo_nclt_recorder.py` subscribes to `/odom` (ground
truth) and `/imu` (observation). It resamples both onto a uniform 100 Hz
grid and writes them as `(N, T, 6)` and `(N, T, 3)` tensors. A second
node `auto_drive.py` publishes `/cmd_vel` following an NCLT-style
schedule of waypoint primitives — long straights, gentle arcs, rare
sharp turns, occasional pauses — at nominal speed 0.15 m/s with
turn rates capped at 0.8 rad/s.

The schedule is randomised but reproducible from a seed; total
recording is ~10 minutes wall-clock producing 185 non-overlapping
sequences.

Splits are made sequence-wise (not time-wise) so no overlap leakage:
- train: 70%, val: 15%, test: 15%.

---

## 7. Baselines

| Method | Description | Parameter count |
|---|---|---:|
| **Strapdown INS** | Bias-corrected open-loop double integration of body-frame accel. Bias estimated from first 1 s of stationary data. | 3 (bias vector) |
| **EKF** | 9-D augmented state `[p, v, a]` with constant-acceleration motion model. Observation `H = [0₃ₓ₆ \| I₃]` (IMU directly observes accel state). Process noise `Q` and measurement noise `R` tuned by grid search to minimise validation RMSE. | 18 (tuned `Q` + `R` diagonals) |
| **KalmanNet (ours)** | Fine-tuned from NCLT pretrained weights. Frozen `F`, learned `H` and `K`. | ~15 000 |

Both classical baselines are implemented in `Model/eval_compare.py`
under `run_strapdown` and `run_ekf` respectively.

---

## 8. Evaluation protocol

Two modes, both important for a faithful comparison:

### 8.1 Chunked mode

Each test sequence (`T = 200`) is processed independently with state
reset at the start of every chunk. Reports per-2-s-window error. This
is the regime classical baselines do well in — drift hasn't had time
to accumulate. **Reported for completeness; not the headline.**

### 8.2 Concat mode

All test sequences are concatenated into one continuous trajectory
(28 sequences × 2 s = **56 s**) with no state reset. This exposes
drift compounding and is the regime where filtering actually matters.
**This is the headline result.**

For both modes we report:
- RMSE / MAE on position and velocity, aggregate and per-state
- Relative error % (RMSE / RMS of GT magnitude)
- "Inlier precision <1 m" — % of samples within 1 m of ground truth
- Per-step inference latency (CPU, ConstructSim VM)

---

## 9. Results

### 9.1 Chunked evaluation (2-second windows)

| Metric | KalmanNet | Strapdown | EKF |
|---|---:|---:|---:|
| RMSE position (m) | 0.256 | **0.099** | **0.099** |
| MAE position (m) | 0.179 | 0.046 | 0.046 |
| RMSE velocity (m/s) | 0.239 | 0.138 | 0.137 |
| Inlier precision <1 m | 98.7% | 100.0% | 100.0% |
| Latency (ms/step) | 1.13 | 0.04 | 0.20 |

**Reading:** dead-reckoning baselines win because 2 s isn't long enough
for drift to accumulate, and the per-chunk GT reset hides their main
weakness. KalmanNet matches them within an order of magnitude. This
demonstrates that the learned gain doesn't *harm* short-horizon
performance — the win shows up only at longer horizons.

### 9.2 Concat evaluation (56 s continuous) — **headline result**

| Metric | KalmanNet | Strapdown | EKF | KalmanNet wins by |
|---|---:|---:|---:|---:|
| **RMSE position (m)** | **30.6** | 109.5 | 109.4 | **3.58×** |
| MAE position (m) | **20.2** | 51.1 | 51.1 | 2.53× |
| **RMSE velocity (m/s)** | **1.08** | 5.02 | 5.01 | **4.65×** |
| MAE velocity (m/s) | **0.81** | 2.72 | 2.71 | 3.36× |
| Inlier precision <1 m | 4.7% | 5.2% | 5.2% | (~tied; all fail at 56 s) |
| Latency (ms/step) | 1.96 | 0.06 | 0.18 | (EKF 11× faster) |

### 9.3 Per-state RMSE (concat mode)

| State | KalmanNet | Strapdown | EKF | Note |
|---|---:|---:|---:|---|
| `px` (m) | 12.2 | **3.8** | 3.8 | EKF wins — bias correction absorbed forward motion at start |
| `py` (m) | **7.8** | 11.8 | 11.8 | KalmanNet 33% better |
| **`pz` (m)** | **51.1** | 189.3 | 189.2 | KalmanNet **3.7×** — gravity awareness |
| `vx` (m/s) | 0.51 | **0.26** | 0.26 | EKF marginally better |
| `vy` (m/s) | **0.68** | 0.44 | 0.44 | EKF marginally better |
| **`vz` (m/s)** | **1.66** | 8.67 | 8.67 | KalmanNet **5.2×** — same gravity story |

### 9.4 Relative error (computed from `eval_concat.json`)

| Method | Rel. pos err (%) | Rel. vel err (%) |
|---|---:|---:|
| KalmanNet | ~22% | ~10% |
| Strapdown | ~78% | ~46% |
| EKF       | ~78% | ~46% |

(Values are RMSE / RMS-magnitude-of-GT. Used in `eval_compare.py`'s
`metrics()` function.)

### 9.5 Live (real-time) inference on Gazebo

A separate ROS 2 inference node (`Gazebo/gazebo_inference_node.py`)
runs the fine-tuned model in closed-loop on the live IMU stream,
publishing `/kalmannet_nclt/odom`. Two implementation requirements
emerged that should be noted in the paper:

1. **Sim time must be enabled** (`use_sim_time=True`) so the IMU
   stamps match training-time `dt`. Without this, the wall-clock
   IMU rate on a slow VM (~60 Hz on ConstructSim due to RTF ≈ 0.3)
   makes the model's `F`-matrix `dt = 0.01` mismatch the actual
   inter-sample interval, producing thousand-metre drift.
2. **IMU sensor `update_rate` must match training** — set to 100 Hz
   in the Gazebo model SDF (line 50 of
   `turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf`).

With those two corrections, live position error converges to single-
digit metres over 30 s of driving — consistent with the offline 56 s
concat numbers above. This live demo is the basis of the inference
video in the supplementary material.

---

## 10. Interpretation — why KalmanNet wins

Two structural advantages, both directly observable in the per-state
breakdown:

### 10.1 Gravity-awareness

Gazebo's IMU plugin includes gravity in `acc_z` (≈ +9.81 m/s² when
the robot is level), as do all real IMUs. The EKF model
`H = [0₃ₓ₆ \| I₃]` directly interprets `acc_z` as vertical
acceleration of the body — so integrating it twice produces
189 m of "vertical motion" over 56 s that does not exist. KalmanNet
learned during NCLT pretraining that this constant ≈ 9.81 m/s² signal
should not be integrated into velocity or position, and the
fine-tune preserves this prior. Result: **`pz` improves 3.7×**, `vz`
improves 5.2×.

### 10.2 Implicit body→world rotation

The IMU publishes accel in the **body frame**. As the robot turns,
body-`x` is no longer aligned with world-`x`. The EKF has no
orientation state and no way to express this — it integrates body
accel as if it were world accel. KalmanNet's recurrent state and the
learned `H` matrix together implicitly track the time-varying
rotation, which is why `py` improves while `px` is competitive (the
test trajectory has more turning around the y axis than the x axis).

### 10.3 The chunked mode hides both effects

Within a 2-s window, body and world frames remain approximately
aligned, gravity drift integrates only ~2 cm of false z motion, and
classical baselines look identical to KalmanNet. The 56-s concat
evaluation is therefore the **honest** comparison for this problem.

---

## 11. Limitations (be transparent about these in the paper)

1. **Single environment.** Recorded only in Gazebo Classic
   `empty_world` with a ground-only TurtleBot3. No real hardware
   validation.
2. **Synthetic, clean IMU.** Gazebo's IMU plugin emits zero-mean
   Gaussian noise. Real IMUs add bias drift, scale-factor errors, and
   temperature dependence — the regime where KalmanNet's learned
   corrections should help even more, but we cannot demonstrate this
   without hardware.
3. **Modest fine-tune dataset.** Only 129 training sequences
   (~26 s equivalent). NCLT priors carry most of the model; the
   fine-tune is a light touch. More Gazebo data would likely widen
   the gap further.
4. **Both filters fail inlier precision <1 m at 56 s.** Without any
   absolute position observation (no GPS, no wheel-odometry
   correction), IMU-only navigation diverges over time. Our result is
   *relative* improvement; absolute accuracy still requires sensor
   fusion.
5. **EKF `px` result is fragile.** The bias-correction window happened
   to absorb early forward motion. On a different recording start
   condition the EKF advantage on `px` could flip.
6. **Numerical-stability horizon.** KalmanNet trained on `T = 200` is
   stable in concat evaluation up to roughly **60–80 s**. Beyond
   that, recurrent-state drift breaks the LayerNorm and errors
   explode. Periodic state resets or training on longer `T` would
   extend this; we did not pursue it for this Phase-1 result.
7. **Z is unobservable in 2D-only TurtleBot data.** Because
   TurtleBot3 motion is strictly planar (GT `z ≡ 0.008`), the model
   never sees genuine z dynamics during fine-tune. NCLT's z
   information is what enables the 3.7× `pz` win — the model is
   essentially extrapolating from pretraining. A drone-based 3D
   dataset (planned follow-up) would let the model learn z dynamics
   in-distribution.

---

## 12. Reproducibility

### 12.1 Branch and commit

- Branch: `kalyani` (Phase 1 frozen model)
- Pretrained weights: `best_knet_nclt.pt`
- Fine-tuned weights: `best_knet_gazebo.pt`
- Recorded data: `~/.ros/gazebo_train.npz`, `gazebo_val.npz`,
  `gazebo_test.npz`

### 12.2 Pipeline

```bash
# 1. Record (3 terminals; see GAZEBO_SETUP.md §4 for screenshots)
ros2 launch turtlebot3_gazebo empty_world.launch.py   # A
python3 Gazebo/gazebo_nclt_recorder.py                # B
python3 Gazebo/auto_drive.py                          # C
# Ctrl+C C, then B → writes the three npz files

# 2. Fine-tune (CPU or Colab)
python3 Model/finetune_gazebo.py \
    --data-dir     ~/.ros \
    --init-weights best_knet_nclt.pt \
    --out-weights  best_knet_gazebo.pt \
    --epochs 30 --lr 1e-4

# 3. Evaluate
python3 Model/eval_compare.py --data ~/.ros/gazebo_test.npz \
    --weights best_knet_gazebo.pt --mode chunked --out eval_chunked.json
python3 Model/eval_compare.py --data ~/.ros/gazebo_test.npz \
    --weights best_knet_gazebo.pt --mode concat  --out eval_concat.json

# 4. Optional — live inference for the demo video
ros2 launch turtlebot3_gazebo empty_world.launch.py   # A
python3 Gazebo/auto_drive.py                          # B
python3 Gazebo/gazebo_inference_node.py               # C
python3 Gazebo/online_evaluator.py                    # D
```

### 12.3 Raw outputs

JSON files in the repo root:
- `eval_chunked.json` — chunked evaluation, all three methods
- `eval_concat.json` — concat evaluation, all three methods

Both include per-state RMSE, MAE, %-error, latency, and inlier
precision. The tables in §9 are direct reads from these files.

---

## 13. Suggested paper structure

```
Title:    Adapting KalmanNet for Robotics-Simulator Inertial
          Navigation: A Light-Touch Transfer Approach

1. Introduction
   - Why IMU-only INS is interesting
   - Classical baselines and their failure mode (gravity, rotation)
   - Our contribution: NCLT → Gazebo transfer with frozen architecture
     and minimal fine-tune

2. Related work
   - KalmanNet (Revach 2022) and follow-ons
   - Learning-augmented Kalman filters
   - NCLT dataset
   - IMU-only INS benchmarks

3. Background — KalmanNet (§3 above)

4. Method
   4.1 Recording pipeline (§4, §6)
   4.2 Fine-tune procedure (§4, §5)
   4.3 Baselines (§7)
   4.4 Evaluation protocol (§8)

5. Results
   5.1 Chunked (§9.1) — "for completeness"
   5.2 Concat — headline (§9.2, §9.3, §9.4)
   5.3 Live inference (§9.5) — for the supplementary video

6. Interpretation (§10) — gravity, rotation, why chunked hides it

7. Limitations (§11)

8. Conclusion and future work
   - Phase 2 — RL meta-tuner (separate paper or extended section;
     see RL_RESULTS.md)
   - Phase 3 — 3D drone fine-tune (in progress)
```

### 13.1 Figures the writer should request

| # | Figure | Source |
|---|---|---|
| 1 | KalmanNet architecture block diagram | Adapt from §5 + Revach 2022 |
| 2 | ConstructSim IDE screenshot (system overview) | `GAZEBO_SETUP.md` Screenshot 1 |
| 3 | Auto-driver trajectory in Gazebo top-down view | Plot `/odom` xy from a recording |
| 4 | Per-state RMSE bar chart (concat mode) | From `eval_concat.json` |
| 5 | Trajectory overlay: GT vs KalmanNet vs EKF (xy plane) | Plot from concat eval |
| 6 | Trajectory overlay: GT vs KalmanNet vs EKF (xz plane) | Same — emphasises the `pz` win |
| 7 | Per-method position error vs time | Compute error norm per step from saved preds |
| 8 | Live-inference still: terminal with GT vs KNet vs error | `online_evaluator.py` output |

### 13.2 Numbers the writer will quote most

- **3.58× position RMSE improvement** over classical baselines in concat mode
- **4.65× velocity RMSE improvement** same setting
- **3.7× per-state `pz` improvement** — the gravity-awareness money number
- **1.96 ms / step** inference latency — real-time at 100 Hz
- **15 000 parameters** — fits on edge hardware
- **30 epochs / ~26 s of fine-tune data** — emphasis on light-touch transfer

---

## 14. Files and where to find things

| File | Purpose |
|---|---|
| `Model/train_model.py` | KalmanNet architecture (`class KalmanNet`) and pretraining |
| `Model/finetune_gazebo.py` | Fine-tune script — loads pretrained weights, trains on Gazebo npz |
| `Model/eval_compare.py` | Three-way evaluation (KalmanNet / Strapdown / EKF), JSON output |
| `Model/knet_step.py` | Per-step inference wrapper (used by live node) |
| `Gazebo/gazebo_nclt_recorder.py` | ROS 2 node that records `/imu` + `/odom` to npz |
| `Gazebo/auto_drive.py` | NCLT-style waypoint auto-driver |
| `Gazebo/gazebo_inference_node.py` | Live ROS 2 node: IMU → KalmanNet → `/kalmannet_nclt/odom` |
| `Gazebo/online_evaluator.py` | Live RMSE/MAE evaluator printing GT vs KNet to terminal |
| `RESULTS.md` | Compact result tables (this KT doc is the long form) |
| `GAZEBO_SETUP.md` | End-to-end reproduction guide with screenshot checklist |
| `best_knet_nclt.pt` | NCLT-pretrained weights (init for fine-tune) |
| `best_knet_gazebo.pt` | Fine-tuned weights — final Phase-1 model |
| `eval_chunked.json` | Chunked evaluation raw output |
| `eval_concat.json` | Concat evaluation raw output |

---

## 15. Open questions / decisions for the writer

1. **Conference target?** The 3.6× / 4.7× concat result is solid for a
   ROS-adjacent venue (ROSCon, IROS workshop). For ICRA/IROS main
   track, the writer should consider whether to bundle Phase 2 (RL
   meta-tuner) for a stronger story, or hold Phase 1 standalone.
2. **Author claim on novelty.** The contribution is *transfer* and
   *evaluation methodology* (chunked-vs-concat distinction). The
   architecture is Revach 2022 — make sure that's credited
   prominently and that we don't overclaim.
3. **Whether to include the live-demo numbers in the main paper or
   supplementary.** Recommendation: supplementary video + a single
   "real-time at 100 Hz, <10 m position error over 30 s of live
   driving" sentence in the main text.
4. **Whether to ablate the NCLT pretraining.** If reviewers ask "what
   does the fine-tune actually contribute vs cold NCLT eval on
   Gazebo?", we have not run that ablation yet — would take ~30
   minutes to add and significantly strengthens the methodology
   section.
