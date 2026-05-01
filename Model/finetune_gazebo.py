#!/usr/bin/env python3
"""
Fine-tune the NCLT-pretrained KalmanNet on Gazebo data.

Inputs:
  - Model/best_knet_nclt.pt      (NCLT pretrained weights, m=6, n=3)
  - gazebo_train.npz / gazebo_val.npz / gazebo_test.npz
    Each with keys 'x' (states, [N, T, 6]) and 'y' (obs, [N, T, 3])
    produced by the gazebo_nclt_recorder ROS2 node.

Output:
  - Model/best_knet_gazebo.pt    (fine-tuned weights)
  - prints a metrics comparison: NCLT-only vs fine-tuned

NOTE on dt: NCLT trained at 100 Hz with constant-velocity F where
F[0,3]=F[1,4]=F[2,5]=dt=0.01. The recorder also resamples Gazebo
data to 100 Hz, so dt matches and the F matrix doesn't need changing.
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reuse the architecture from train_model.py
from train_model import KalmanNet, sanitize_data, compute_metrics


def load_npz(path):
    d = np.load(path)
    return sanitize_data(d["x"]), sanitize_data(d["y"])


def evaluate(model, loader, device):
    model.eval()
    preds, trues, total_t, total_s = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t0 = time.time()
            p = model(y, x[:, 0, :])
            total_t += time.time() - t0
            total_s += x.shape[0] * x.shape[1]
            preds.append(p[:, 1:, :].cpu())
            trues.append(x[:, 1:, :].cpu())
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    ms_per_step = (total_t / total_s) * 1000.0
    return compute_metrics(preds, trues, ms_per_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".",
                    help="Directory with gazebo_{train,val,test}.npz")
    ap.add_argument("--init-weights", default="best_knet_nclt.pt")
    ap.add_argument("--out-weights",  default="best_knet_gazebo.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr",     type=float, default=1e-4,
                    help="Lower than train_model.py default (1e-3) "
                         "to avoid overwriting NCLT priors.")
    ap.add_argument("--batch",  type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Fine-tuning on Device: {device} ---")

    tx, ty = load_npz(os.path.join(args.data_dir, "gazebo_train.npz"))
    vx, vy = load_npz(os.path.join(args.data_dir, "gazebo_val.npz"))
    sx, sy = load_npz(os.path.join(args.data_dir, "gazebo_test.npz"))

    print(f"train: x={tx.shape}  y={ty.shape}")
    print(f"val  : x={vx.shape}  y={vy.shape}")
    print(f"test : x={sx.shape}  y={sy.shape}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(tx).float(), torch.tensor(ty).float()),
        batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(vx).float(), torch.tensor(vy).float()),
        batch_size=args.batch)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(sx).float(), torch.tensor(sy).float()),
        batch_size=args.batch)

    model = KalmanNet().to(device)

    # 1) Baseline: load NCLT weights, evaluate on Gazebo test set as-is.
    if os.path.exists(args.init_weights):
        model.load_state_dict(torch.load(args.init_weights, map_location=device))
        print(f"\nLoaded NCLT pretrained weights from {args.init_weights}")
        print("\n--- Baseline (NCLT weights, no fine-tuning) on Gazebo test ---")
        for k, v in evaluate(model, test_loader, device).items():
            print(f"  {k:<25}: {v}")
    else:
        print(f"WARNING: {args.init_weights} not found — training from scratch.")

    # 2) Fine-tune
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    crit = nn.SmoothL1Loss()
    best = float("inf")

    print(f"\n--- Fine-tuning for {args.epochs} epochs at lr={args.lr} ---")
    for ep in range(1, args.epochs + 1):
        model.train()
        tr = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            p = model(y, x[:, 0, :])
            loss = crit(p[:, 1:, :], x[:, 1:, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * x.size(0)
        tr /= len(train_loader.dataset)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = model(y, x[:, 0, :])
                vl += crit(p[:, 1:, :], x[:, 1:, :]).item() * x.size(0)
        vl /= len(val_loader.dataset)

        flag = ""
        if vl < best:
            best = vl
            torch.save(model.state_dict(), args.out_weights)
            flag = " (saved)"
        if ep == 1 or ep % 5 == 0:
            print(f"ep {ep:03d} | train {tr:.5f} | val {vl:.5f}{flag}")

    # 3) Final eval with best fine-tuned weights
    model.load_state_dict(torch.load(args.out_weights, map_location=device))
    print(f"\n--- Fine-tuned weights ({args.out_weights}) on Gazebo test ---")
    for k, v in evaluate(model, test_loader, device).items():
        print(f"  {k:<25}: {v}")


if __name__ == "__main__":
    main()
