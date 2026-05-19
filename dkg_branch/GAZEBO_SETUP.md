# Gazebo Setup & Reproduction Guide

This document is written so that an external reader can reproduce all
Gazebo experiments in `RESULTS.md` and `RL_RESULTS.md` from a clean
machine — specifically a [TheConstruct ConstructSim](https://app.theconstructsim.com)
rosject (recommended, no local install), or any Ubuntu 22.04 + ROS 2
Humble box.



## 1. Note

If you reproduce locally instead, the only differences are:
- install `ros-humble-turtlebot3*` and `ros-humble-gazebo-*` yourself
- `source /opt/ros/humble/setup.bash` in every terminal
- everything else (paths, scripts, commands) is identical.

---

## 2. ConstructSim rosject — first-time setup

1. Sign in at <https://app.theconstructsim.com>.
2. Create a new **rosject**: ROS 2 Humble, Gazebo Classic, Ubuntu 22.04.
3. Open the **IDE** view (Code editor + Gazebo + Terminal in one tab).
4. In a terminal, clone the project repo into `~/`:
   ```bash
   cd ~
   git clone <your-fork-url> kalmannet_ros2
   cd kalmannet_ros2/dkg_branch
   ```
5. Install Python deps used by the recorder / training / eval scripts:
   ```bash
   pip install --user numpy scipy torch stable-baselines3 gymnasium
   ```
6. Verify the TurtleBot3 stack:
   ```bash
   export TURTLEBOT3_MODEL=waffle
   ros2 launch turtlebot3_gazebo empty_world.launch.py
   ```
   The Gazebo window in the IDE should show the TurtleBot3 Waffle on an
   empty ground plane. **(Screenshot 1)**

`export TURTLEBOT3_MODEL=waffle` must be set in **every** terminal that
talks to Gazebo (the launch file reads it at startup). Add it to
`~/.bashrc` to avoid forgetting.

---

## 3. Repository layout (what each script does)

```
dkg_branch/
├── Gazebo/
│   ├── gazebo_nclt_recorder.py     ROS 2 node — records /odom + /imu,
│   │                               saves gazebo_{train,val,test}.npz
│   ├── auto_drive.py               ROS 2 node — NCLT-style waypoint
│   │                               schedule on /cmd_vel
│   ├── gazebo_inference_node.py    ROS 2 node — live KalmanNet, pub.
│   │                               /kalmannet_nclt/odom
│   ├── online_evaluator.py         ROS 2 node — live RMSE/MAE between
│   │                               /odom (GT) and /kalmannet_nclt/odom
│   ├── make_world.sh               Emits an SDF world with custom gz
│   ├── record_vargrav_session.sh   Loops over gravities, records each
│   ├── concat_recordings.py        Concatenates per-gravity npz with
│   │                               held-out gravity support
│   └── launch/gazebo_kalmannet.launch.py
├── Model/
│   ├── finetune_gazebo.py          Loads best_knet_nclt.pt, fine-tunes
│   ├── eval_compare.py             KalmanNet vs EKF vs Strapdown
│   ├── rl_metatuner_long.py        PPO meta-tuner training
│   └── eval_metatuner.py           RL evaluation (variable gravity)
└── (root)/best_knet_nclt.pt, best_knet_gazebo.pt
```

All `python3 …` commands below assume CWD is `dkg_branch/`.

---

## 4. Phase 1 — Earth-gravity recording, fine-tune, evaluate

### 4.1 Record a dataset (~10 min wall time)

Open three terminals in the ConstructSim IDE.

```bash
# Terminal A — Gazebo
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo empty_world.launch.py
```
**(Screenshot 1 — Gazebo window with TurtleBot3 spawned, sensors active)**

```bash
# Terminal B — recorder
python3 Gazebo/gazebo_nclt_recorder.py
```
Wait until the recorder prints `[recorder] /imu and /odom subscribed,
recording…` **(Screenshot 2 — recorder terminal showing sequence count
incrementing)**

```bash
# Terminal C — auto driver
python3 Gazebo/auto_drive.py
```
**(Screenshot 3 — IDE split view: Gazebo window with the TurtleBot
mid-motion + three terminals running in parallel)**

After ~10 min, `Ctrl-C` Terminal C, then Terminal B. The recorder
auto-saves to:
```
~/.ros/gazebo_train.npz
~/.ros/gazebo_val.npz
~/.ros/gazebo_test.npz
```

### 4.2 Fine-tune (Colab or local GPU recommended)

```bash
cd Model/
python3 finetune_gazebo.py \
    --data-dir   ~/.ros \
    --init-weights best_knet_nclt.pt \
    --out-weights  best_knet_gazebo.pt \
    --epochs 30 --lr 1e-4
```


#### 4.2.a Training in Google Colab

Open a new Colab notebook, set *Runtime → Change runtime type → T4 GPU*.

**Upload these files** (left sidebar → Files icon → Upload):

- `best_knet_nclt.pt`
- `gazebo_train.npz`, `gazebo_val.npz`, `gazebo_test.npz`
- `finetune_gazebo.py`, `eval_compare.py` (from `dkg_branch/Model/`)
- Phase 2 only: `gazebo_train_vargrav.npz`,
  `gazebo_test_vargrav_short.npz`, `rl_metatuner_long.py`,
  `eval_metatuner.py`

**Then run these cells:**

```python
!pip -q install stable-baselines3 gymnasium
```

```python
# Phase 1 — fine-tune
!python3 finetune_gazebo.py \
    --data-dir   . \
    --init-weights best_knet_nclt.pt \
    --out-weights  best_knet_gazebo.pt \
    --epochs 30 --lr 1e-4
```

```python
# Phase 1 — evaluate
!python3 eval_compare.py --data gazebo_test.npz \
    --weights best_knet_gazebo.pt --mode concat --out eval_concat.json
```

```python
# Phase 2 — RL meta-tuner train + eval
!python3 rl_metatuner_long.py train \
    --weights best_knet_gazebo.pt \
    --data    gazebo_train_vargrav.npz \
    --steps   100000 --out meta_tuner_ppo.zip

!python3 eval_metatuner.py \
    --data    gazebo_test_vargrav_short.npz \
    --weights best_knet_gazebo.pt \
    --policy  meta_tuner_ppo.zip \
    --mode    concat --out rl_eval.json
```

Right-click `best_knet_gazebo.pt` and `meta_tuner_ppo.zip` in the Files
panel → **Download** to and upload them back to ConstructSim.



### 4.3 Evaluate vs. classical baselines

```bash
python3 eval_compare.py --data ~/.ros/gazebo_test.npz \
    --weights best_knet_gazebo.pt --mode chunked --out eval_chunked.json
python3 eval_compare.py --data ~/.ros/gazebo_test.npz \
    --weights best_knet_gazebo.pt --mode concat  --out eval_concat.json
```


### 4.4 Live in-simulation inference (optional)

```bash
cp best_knet_gazebo.pt ~/.ros/
ros2 launch kalmannet_ros2 gazebo_kalmannet.launch.py \
     weights:=~/.ros/best_knet_gazebo.pt
# in another terminal:
ros2 run kalmannet_ros2 online_evaluator
```

---

## 5. Phase 2 — Variable-gravity, RL meta-tuner

### 5.1 Custom-gravity world

Gazebo's physics gravity is set in the world SDF, **not** at runtime.
`make_world.sh` generates a minimal world with the gravity you ask for:

```bash
bash Gazebo/make_world.sh -8.5 > ~/.ros/custom.world
ros2 launch turtlebot3_gazebo empty_world.launch.py \
     world:=~/.ros/custom.world
```
**(Screenshot 8 — Gazebo with the TurtleBot visibly lighter / floatier
under gz = −5, e.g. a small bump produces extra airtime. Capture at two
gravities side-by-side if possible: gz = −9.81 vs gz = −5.)**

### 5.2 Multi-gravity batch recording

`record_vargrav_session.sh` regenerates the world, launches Gazebo,
runs the recorder + auto-driver for N seconds, kills everything, and
loops to the next gravity value:

```bash
GRAVITIES="-14 -12 -10 -8 -6 -5 -3" SECONDS_PER_GRAVITY=400 \
    bash Gazebo/record_vargrav_session.sh
```
This produces `~/.ros/vargrav/g{...}_train.npz` etc. for each gravity.
**(Screenshot 9 — terminal showing the loop progress: e.g.
"[record_vargrav] Recording for 400s at gz=-8.0 …" with prior gravities
already saved.)**

### 5.3 Concat with held-out gravities

```bash
python3 Gazebo/concat_recordings.py \
    --in-dir ~/.ros/vargrav --out-dir ~/.ros \
    --holdout-gravities -14 -3
```
Held-out values appear **only** in the test split (see `RL_RESULTS.md`).

### 5.4 Train the PPO meta-tuner

```bash
python3 Model/rl_metatuner_long.py train \
    --weights best_knet_gazebo.pt \
    --data    ~/.ros/gazebo_train_vargrav.npz \
    --steps   100000 \
    --out     meta_tuner_ppo.zip
```
**(Screenshot 10 — PPO training log: episode reward, policy loss,
value loss climbing/falling over ~100k steps. The stable-baselines3
default table is fine.)**

### 5.5 Evaluate

```bash
python3 Model/eval_metatuner.py \
    --data    ~/.ros/gazebo_test_vargrav_short.npz \
    --weights best_knet_gazebo.pt \
    --policy  meta_tuner_ppo.zip \
    --mode    concat \
    --out     rl_eval.json
```
**(Screenshot 11 — the printed comparison table: EKF vs frozen
KalmanNet vs KalmanNet+RL, including per-state RMSE highlighting the
`pz` / `vz` improvement.)**

---

## 6. Frames, units, and gotchas

- **`TURTLEBOT3_MODEL=waffle`** must be exported in every Gazebo-facing
  terminal.
- **IMU is body-frame.** Do not rotate to world frame before feeding
  KalmanNet — the model was trained on raw body-frame accel.
- **Gravity stays in `acc_z`.** NCLT's IMU includes gravity; Gazebo's
  default plugin does too. Leave it in.
- **Sample rate is 100 Hz.** The recorder resamples `/odom` (~30 Hz)
  and `/imu` (~200 Hz) onto a uniform 100 Hz grid so `dt = 0.01 s`
  matches NCLT.
- **`gz < −13` is unstable.** The TurtleBot can fall through the floor
  during spawn. Phase 2 used `gz ∈ [−14, −3]` and `−14` is the edge of
  the stable envelope.
- **Recorder auto-saves on SIGINT.** Always `Ctrl-C` (not kill -9) or
  the train/val/test split is not written.

---

## 7. Screenshot checklist (send these to the paper writer)

Numbered to match the callouts above. Capture from the ConstructSim
browser tab (full window or cropped to the relevant panel).

| # | Subject | What to capture |
|---|---|---|
| 1 | ConstructSim IDE on first launch | Full browser window: code editor + terminal + empty Gazebo panel. Establishes the platform. |
| 2 | TurtleBot3 spawned in `empty_world` | Gazebo viewport with the Waffle on the ground plane. Default camera angle. |
| 3 | Recorder running | Terminal B output: `/imu` + `/odom` subscribed message and the live sequence counter incrementing. |
| 4 | Three-terminal recording session | Split-pane / tiled view of Gazebo + Terminals A/B/C while data is being collected. The "system in motion" hero shot. |
| 5 | Fine-tune training log | Console output of `finetune_gazebo.py` showing per-epoch train/val loss and final test RMSE. |
| 5b | Colab GPU training | Notebook cell with `nvidia-smi` (T4 attached) above the first few epochs of `finetune_gazebo.py` running. Proves the GPU path works end-to-end. |
| 6 | `eval_compare.py` results table | The printed table of KalmanNet vs EKF vs Strapdown — both **chunked** and **concat** modes if room allows (two screenshots). |
| 7 | Live inference overlay | `rqt_plot` (or rviz2 Path display) showing `/odom` vs `/kalmannet_nclt/odom` XY trajectories overlaid. Best taken after ~30 s of driving. |
| 8 | Variable-gravity demo | Two Gazebo screenshots side-by-side: `gz = -9.81` (normal) vs `gz = -5` (floaty). Same robot pose if possible — even a still frame conveys it. |
| 9 | `record_vargrav_session.sh` progress | Terminal showing the loop mid-run: "Recording for 400s at gz=…" with earlier gravities already saved to `vargrav/`. |
| 10 | PPO training log | stable-baselines3 default table (rollout/ep_rew_mean, train/policy_loss, value_loss) over ~100k steps. |
| 11 | `eval_metatuner.py` results | The printed three-column table (EKF / KalmanNet / KalmanNet+RL) and the per-state RMSE block highlighting the `pz` / `vz` improvement. |

Optional / nice-to-have if time permits:

| # | Subject | Why |
|---|---|---|
| 12 | rviz2 with TurtleBot3 robot model + IMU axes | Visual aid showing body-frame IMU vs world-frame odom — relevant to the "frames and units" section of the paper. |
| 13 | `ros2 topic hz /imu` and `ros2 topic hz /odom` | Demonstrates the raw 30 Hz / 200 Hz rates that motivate the 100 Hz resampler. |
| 14 | NPZ inspection | `python3 -c "import numpy as np; d=np.load('~/.ros/gazebo_train.npz'); print({k:d[k].shape for k in d})"` showing the `x [N,200,6]` / `y [N,200,3]` shapes. Proves the NCLT-contract is preserved. |

When sending the bundle, name files `fig01_constructsim.png`,
`fig02_gazebo_spawn.png`, … so the paper writer can drop them in
directly.
