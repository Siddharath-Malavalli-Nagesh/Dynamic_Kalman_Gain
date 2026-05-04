#!/usr/bin/env python3
"""
Three-way comparison on variable-gravity test data:

  1. EKF                          (classical baseline)
  2. KalmanNet (frozen)           (current best, no adaptivity)
  3. KalmanNet + RL Meta-Tuner    (this work)

Reports per-state RMSE, MAE, % error, inlier precision <1m, latency.
Outputs a JSON for inclusion in RESULTS.md.

Usage
-----
  python3 eval_metatuner.py \\
      --data    gazebo_test_vargrav.npz \\
      --weights best_knet_gazebo.pt \\
      --policy  meta_tuner_ppo.zip \\
      --mode    concat
"""

import argparse
import json
import time
import numpy as np
import torch

from train_model import KalmanNet, sanitize_data
from knet_step import KalmanNetStepper
from eval_compare import run_ekf, metrics, print_table, STATE_NAMES


GAIN_CENTER = 1.0
GAIN_RANGE  = 0.5


def run_kalmannet_frozen(model, x_gt, y_obs, device, batch=32):
    """Vanilla KalmanNet with no gain modulation."""
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


def run_kalmannet_rl(model, policy, x_gt, y_obs, device):
    """KalmanNet with RL meta-tuner adjusting per-step gain."""
    N, T, m = x_gt.shape
    n = y_obs.shape[2]
    stepper = KalmanNetStepper(model, device)
    preds = np.zeros_like(x_gt)
    total_t = 0.0
    total_steps = 0

    for s in range(N):
        x0 = torch.tensor(x_gt[s, 0], dtype=torch.float32)
        stepper.reset(x0)
        # warm-up (first sample establishes y_prev)
        _ = stepper.step(torch.tensor(y_obs[s, 0], dtype=torch.float32))
        preds[s, 0] = x_gt[s, 0]

        innov_ema = 0.0
        last_action = np.zeros(m, dtype=np.float32)
        ema_alpha = 0.1

        # Initial obs from a one-step preview to define innovation
        y1 = torch.tensor(y_obs[s, 1], dtype=torch.float32)
        x_post, innov, _ = stepper.step(y1)
        preds[s, 1] = x_post.detach().cpu().numpy().reshape(-1)

        for t in range(2, T):
            # Build observation matching the training env layout
            innov_np = innov.detach().cpu().numpy().reshape(-1)
            vel_mag = float(np.linalg.norm(preds[s, t - 1, 3:]))
            obs = np.concatenate([
                innov_np,
                np.array([innov_ema], dtype=np.float32),
                np.array([vel_mag], dtype=np.float32),
                last_action,
            ]).astype(np.float32)

            action, _ = policy.predict(obs, deterministic=True)
            last_action = np.clip(action, -1.0, 1.0).astype(np.float32)
            gain_scale = torch.tensor(
                GAIN_CENTER + GAIN_RANGE * last_action, dtype=torch.float32)

            y_t = torch.tensor(y_obs[s, t], dtype=torch.float32)
            tic = time.time()
            x_post, innov, _ = stepper.step(y_t, gain_scale=gain_scale)
            total_t += time.time() - tic
            total_steps += 1

            preds[s, t] = x_post.detach().cpu().numpy().reshape(-1)
            innov_mag = float(np.linalg.norm(innov.detach().cpu().numpy()))
            innov_ema = (1 - ema_alpha) * innov_ema + ema_alpha * innov_mag

    return preds, (total_t / max(total_steps, 1)) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--policy",  required=True)
    ap.add_argument("--dt",      type=float, default=0.01)
    ap.add_argument("--mode",    choices=["chunked", "concat"],
                    default="concat")
    ap.add_argument("--out",     default="rl_eval.json")
    args = ap.parse_args()

    from stable_baselines3 import PPO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Mode: {args.mode}")

    data  = np.load(args.data)
    x_gt  = sanitize_data(data["x"]).astype(np.float32)
    y_obs = sanitize_data(data["y"]).astype(np.float32)
    print(f"Loaded {args.data}: x={x_gt.shape}, y={y_obs.shape}")

    if args.mode == "concat":
        N, T, m = x_gt.shape
        n = y_obs.shape[2]
        x_gt  = x_gt.reshape(1, N * T, m)
        y_obs = y_obs.reshape(1, N * T, n)
        print(f"  concat → 1 sequence of length {N * T}")

    model = KalmanNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    policy = PPO.load(args.policy)

    results = {}

    print("\n[1/3] EKF …")
    p, lat = run_ekf(x_gt, y_obs, args.dt)
    results["EKF"] = {**metrics(p, x_gt), "latency_ms": lat}

    print("[2/3] KalmanNet (frozen) …")
    p, lat = run_kalmannet_frozen(model, x_gt, y_obs, device)
    results["KalmanNet"] = {**metrics(p, x_gt), "latency_ms": lat}

    print("[3/3] KalmanNet + RL Meta-Tuner …")
    p, lat = run_kalmannet_rl(model, policy, x_gt, y_obs, device)
    results["KalmanNet+RL"] = {**metrics(p, x_gt), "latency_ms": lat}

    print_table(results)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
