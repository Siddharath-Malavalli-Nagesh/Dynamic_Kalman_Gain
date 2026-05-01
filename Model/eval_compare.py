#!/usr/bin/env python3
"""
Offline benchmark: KalmanNet vs classical baselines on Gazebo test set.

Methods compared
----------------
1. KalmanNet     — Model/best_knet_gazebo.pt (or best_knet_nclt.pt)
2. Strapdown INS — bias-corrected double integration of IMU. No filter.
                   Pure dead reckoning. Establishes "what happens if you
                   don't filter."
3. EKF           — augmented state [px,py,pz,vx,vy,vz,ax,ay,az] with
                   constant-acceleration F, accel as random walk, IMU as
                   direct observation H = [0 | I_3]. The classical method
                   the mentor wants us to beat.

Inputs
------
  --data    gazebo_test.npz  (keys 'x' shape [N,T,6], 'y' shape [N,T,3])
  --weights best_knet_gazebo.pt

Metrics (all reported per method)
---------------------------------
  RMSE / MAE on position (m) and velocity (m/s)
  Per-state RMSE for [px, py, pz, vx, vy, vz]
  Per-state % error (computed only where |GT| > floor — see metrics())
  Inlier precision (% of timesteps with position error < 1 m)
  Mean inference latency (ms / step)

Outputs
-------
  Console table + eval_results.json

Modes
-----
  --mode chunked  (default)  : evaluate each 2-second test sequence in
                                isolation. Strapdown/EKF re-estimate
                                their bias and re-init from GT every
                                sequence — generous to baselines.
  --mode concat              : flatten all test sequences into one long
                                continuous run. Strapdown/EKF estimate
                                bias once at the start and then run
                                free. Drift accumulates honestly.
                                This is the fair comparison.
"""

import argparse
import json
import time
import numpy as np
import torch

from train_model import KalmanNet, sanitize_data

STATE_NAMES = ["px", "py", "pz", "vx", "vy", "vz"]
POS_FLOOR_M  = 0.5     # ignore % error below this magnitude (avoids div-by-tiny)
VEL_FLOOR_MS = 0.05


# --------------------------------------------------------------------- #
# Method 1 — KalmanNet
# --------------------------------------------------------------------- #
def run_kalmannet(model, x_gt, y_obs, device, batch=32):
    model.eval()
    preds, total_t, total_steps = [], 0.0, 0
    with torch.no_grad():
        for i in range(0, len(x_gt), batch):
            xb = torch.tensor(x_gt[i:i+batch]).float().to(device)
            yb = torch.tensor(y_obs[i:i+batch]).float().to(device)
            t0 = time.time()
            p = model(yb, xb[:, 0, :])
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_t += time.time() - t0
            total_steps += xb.shape[0] * xb.shape[1]
            preds.append(p.cpu().numpy())
    return np.concatenate(preds, 0), (total_t / total_steps) * 1000.0


# --------------------------------------------------------------------- #
# Method 2 — Strapdown INS (no filter)
# --------------------------------------------------------------------- #
def run_strapdown(x_gt, y_obs, dt):
    """
    Bias estimated from the first 10 samples of each sequence (assumes
    those samples are roughly stationary, so the mean accel ≈ gravity +
    sensor bias). Then double-integrate.
    """
    N, T, _ = x_gt.shape
    preds = np.zeros_like(x_gt)
    total_t = 0.0
    for s in range(N):
        t0 = time.time()
        bias = y_obs[s, :10].mean(axis=0)
        a = y_obs[s] - bias
        p = np.zeros((T, 3)); v = np.zeros((T, 3))
        p[0] = x_gt[s, 0, :3]
        v[0] = x_gt[s, 0, 3:]
        for t in range(1, T):
            v[t] = v[t-1] + a[t] * dt
            p[t] = p[t-1] + v[t-1] * dt + 0.5 * a[t] * dt * dt
        preds[s, :, :3] = p
        preds[s, :, 3:] = v
        total_t += time.time() - t0
    return preds, (total_t / (N * T)) * 1000.0


# --------------------------------------------------------------------- #
# Method 3 — EKF with augmented state
# --------------------------------------------------------------------- #
def run_ekf(x_gt, y_obs, dt):
    """
    State (9D)  : [px, py, pz, vx, vy, vz, ax, ay, az]
    Process     : constant-acceleration kinematics; accel = accel + w (RW)
    Observation : H = [0_{3x6} | I_3]; IMU directly observes accel state
                  after per-sequence bias subtraction.
    """
    N, T, _ = x_gt.shape

    F = np.eye(9)
    F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
    F[0, 6] = 0.5 * dt * dt; F[1, 7] = 0.5 * dt * dt; F[2, 8] = 0.5 * dt * dt
    F[3, 6] = dt; F[4, 7] = dt; F[5, 8] = dt

    H = np.zeros((3, 9)); H[0, 6] = H[1, 7] = H[2, 8] = 1.0

    # Conservative noise tuning:
    #   Q small for pos/vel (kinematics nearly exact),
    #   larger for accel (it really does jitter / change).
    #   R from typical Gazebo IMU std (~0.05 m/s² on each axis ⇒ var=2.5e-3).
    Q = np.diag([1e-7]*3 + [1e-5]*3 + [1e-2]*3)
    R = np.diag([2.5e-3]*3)

    preds = np.zeros((N, T, 6))
    total_t = 0.0
    I9 = np.eye(9)

    for s in range(N):
        t0 = time.time()
        bias = y_obs[s, :10].mean(axis=0)
        x = np.zeros(9)
        x[:6] = x_gt[s, 0]
        P = np.eye(9) * 0.1

        preds[s, 0] = x[:6]
        for t in range(1, T):
            x = F @ x
            P = F @ P @ F.T + Q
            innov = (y_obs[s, t] - bias) - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ innov
            P = (I9 - K @ H) @ P
            preds[s, t] = x[:6]
        total_t += time.time() - t0
    return preds, (total_t / (N * T)) * 1000.0


# --------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------- #
def metrics(preds, gt):
    err = preds[:, 1:] - gt[:, 1:]    # drop initial GT-seeded step
    pos_err = err[:, :, :3]
    vel_err = err[:, :, 3:]

    per_state_rmse = np.sqrt(np.mean(err ** 2, axis=(0, 1)))    # (6,)
    per_state_mae  = np.mean(np.abs(err), axis=(0, 1))          # (6,)

    pct = []
    for i, name in enumerate(STATE_NAMES):
        floor = VEL_FLOOR_MS if name.startswith("v") else POS_FLOOR_M
        gt_i = gt[:, 1:, i].ravel()
        e_i  = np.abs(err[:, :, i]).ravel()
        valid = np.abs(gt_i) >= floor
        pct.append(float((e_i[valid] / np.abs(gt_i[valid])).mean() * 100)
                   if valid.any() else float("nan"))

    dists = np.sqrt(np.sum(pos_err ** 2, axis=2))
    precision_1m = float((dists < 1.0).mean() * 100)

    return {
        "rmse_pos_m":     float(np.sqrt(np.mean(pos_err ** 2))),
        "mae_pos_m":      float(np.mean(np.abs(pos_err))),
        "rmse_vel_mps":   float(np.sqrt(np.mean(vel_err ** 2))),
        "mae_vel_mps":    float(np.mean(np.abs(vel_err))),
        "precision_1m_pct": precision_1m,
        "per_state_rmse": dict(zip(STATE_NAMES, [float(v) for v in per_state_rmse])),
        "per_state_mae":  dict(zip(STATE_NAMES, [float(v) for v in per_state_mae])),
        "per_state_pct_err": dict(zip(STATE_NAMES, pct)),
    }


# --------------------------------------------------------------------- #
def print_table(results):
    methods = list(results.keys())
    w = 14

    def header(label):
        print(f"\n{label}")
        print(f"{'':<22}" + "".join(f"{m:>{w}}" for m in methods))

    header("Aggregate")
    for key, label in [
        ("rmse_pos_m",       "RMSE position (m)"),
        ("mae_pos_m",        "MAE  position (m)"),
        ("rmse_vel_mps",     "RMSE velocity (m/s)"),
        ("mae_vel_mps",      "MAE  velocity (m/s)"),
        ("precision_1m_pct", "Precision <1 m (%)"),
        ("latency_ms",       "Latency (ms/step)"),
    ]:
        row = f"{label:<22}"
        for m in methods:
            v = results[m].get(key, float("nan"))
            row += f"{v:>{w}.4f}"
        print(row)

    header("Per-state RMSE")
    for s in STATE_NAMES:
        row = f"  {s:<20}"
        for m in methods:
            v = results[m]["per_state_rmse"][s]
            row += f"{v:>{w}.4f}"
        print(row)

    header("Per-state % error (where |GT| > floor)")
    for s in STATE_NAMES:
        row = f"  {s:<20}"
        for m in methods:
            v = results[m]["per_state_pct_err"][s]
            row += f"{v:>{w}.2f}" if not np.isnan(v) else f"{'n/a':>{w}}"
        print(row)


# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    default="gazebo_test.npz")
    ap.add_argument("--weights", default="best_knet_gazebo.pt")
    ap.add_argument("--dt",      type=float, default=0.01)
    ap.add_argument("--out",     default="eval_results.json")
    ap.add_argument("--mode",    choices=["chunked", "concat"],
                    default="chunked",
                    help="chunked: per-chunk reset (favours baselines); "
                         "concat: one long continuous sequence (fair).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Mode: {args.mode}")

    data  = np.load(args.data)
    x_gt  = sanitize_data(data["x"])
    y_obs = sanitize_data(data["y"])
    print(f"Loaded {args.data}: x={x_gt.shape}, y={y_obs.shape}")

    if args.mode == "concat":
        # Flatten N chunks of length T into one long sequence of length N*T,
        # then re-wrap as a single batch of size 1 so the same eval funcs
        # work without modification.
        N, T, m = x_gt.shape
        n = y_obs.shape[2]
        x_gt  = x_gt.reshape(1, N * T, m)
        y_obs = y_obs.reshape(1, N * T, n)
        print(f"  concat mode → single sequence of length {N * T} "
              f"({(N*T)*args.dt:.1f}s of continuous data)")

    results = {}

    print("\n[1/3] KalmanNet …")
    model = KalmanNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    p, lat = run_kalmannet(model, x_gt, y_obs, device)
    results["KalmanNet"] = {**metrics(p, x_gt), "latency_ms": lat}

    print("[2/3] Strapdown INS …")
    p, lat = run_strapdown(x_gt, y_obs, args.dt)
    results["Strapdown"] = {**metrics(p, x_gt), "latency_ms": lat}

    print("[3/3] EKF …")
    p, lat = run_ekf(x_gt, y_obs, args.dt)
    results["EKF"] = {**metrics(p, x_gt), "latency_ms": lat}

    print_table(results)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
