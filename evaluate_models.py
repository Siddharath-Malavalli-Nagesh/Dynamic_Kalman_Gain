"""
evaluate_models.py
------------------
Compares KalmanNet and an Extended Kalman Filter on the NCLT test set.

Usage:
    python evaluate_models.py

Expected files in the working directory (or adjust paths below):
    best_knet_nclt.pt   – saved KalmanNet weights  (state-dict)
    nclt_test.npz       – test split  (keys: 'x' [B,T,6], 'y' [B,T,3])
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ekf import ExtendedKalmanFilter

# ──────────────────────────────────────────────
# Paths  (edit if needed)
# ──────────────────────────────────────────────
KNET_WEIGHTS = "best_knet_nclt.pt"
TEST_DATA    = "Downloads/nclt_test.npz"
BATCH_SIZE   = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════
# Exact KalmanNet Architecture  (matches training script 1-to-1)
# ══════════════════════════════════════════════

class KalmanNet(nn.Module):
    def __init__(self, m=6, n=3, N_rnn=32):
        super().__init__()
        self.m, self.n, self.N_rnn = m, n, N_rnn

        self.GRU_Q     = nn.GRUCell(n, N_rnn)
        self.GRU_Sigma = nn.GRUCell(m, N_rnn)
        self.GRU_S     = nn.GRUCell(n, N_rnn)

        feat_dim  = N_rnn * 3 + n + m
        self.norm = nn.LayerNorm(feat_dim)

        self.fc1 = nn.Linear(feat_dim, N_rnn)
        self.fc2 = nn.Linear(N_rnn,    N_rnn)
        self.fc3 = nn.Linear(N_rnn,    n * m)

        nn.init.uniform_(self.fc3.weight, -0.001, 0.001)
        nn.init.zeros_(self.fc3.bias)

        self.H = nn.Linear(m, n)
        nn.init.zeros_(self.H.weight)
        nn.init.zeros_(self.H.bias)

        dt = 0.01
        F = torch.eye(m)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        self.F = nn.Parameter(F, requires_grad=False)

    def forward(self, Yseq, x0):
        B, T, _ = Yseq.shape
        device  = Yseq.device

        x_post = torch.zeros(B, T, self.m, device=device)
        x_post[:, 0, :] = x0

        h_Q     = torch.zeros(B, self.N_rnn, device=device)
        h_Sigma = torch.zeros(B, self.N_rnn, device=device)
        h_S     = torch.zeros(B, self.N_rnn, device=device)

        for t in range(1, T):
            y       = Yseq[:, t, :]
            y_prev  = Yseq[:, t - 1, :]
            delta_y = y - y_prev

            e_post = (x_post[:, t - 1, :] - x_post[:, t - 2, :]
                      if t > 1 else torch.zeros_like(x_post[:, 0, :]))

            h_Q     = self.GRU_Q(delta_y, h_Q)
            h_Sigma = self.GRU_Sigma(e_post, h_Sigma)
            h_S     = self.GRU_S(delta_y, h_S)

            feat = torch.cat([h_Q, h_Sigma, h_S, y, x_post[:, t - 1, :]], dim=1)
            feat = self.norm(feat)

            feat = torch.relu(self.fc1(feat))
            feat = torch.relu(self.fc2(feat))
            K    = self.fc3(feat).view(B, self.m, self.n)

            x_prior = torch.matmul(self.F, x_post[:, t - 1, :].unsqueeze(2)).squeeze(2)
            y_pred  = self.H(x_prior)
            innov   = y - y_pred

            correction      = torch.bmm(K, innov.unsqueeze(2)).squeeze(2)
            x_post[:, t, :] = x_prior + correction

        return x_post   # (B, T, 6) — index 0 is x0, predictions start at 1


# ══════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════

def sanitize(arr: np.ndarray) -> np.ndarray:
    arr[np.isinf(arr)] = np.nan
    return np.nan_to_num(arr, nan=0.0)


# ══════════════════════════════════════════════
# Metrics  (NumPy — mirrors training script style exactly)
# ══════════════════════════════════════════════

def compute_metrics(pred: np.ndarray, gt: np.ndarray, latency_ms: float) -> dict:
    """
    Args:
        pred / gt  : (N, T, 6)  — t=1…T-1 (t=0 already excluded by caller)
        latency_ms : ms per timestep
    """
    error     = pred - gt                               # (N, T, 6)
    pos_error = error[:, :, :3]                         # (N, T, 3)

    mse      = float(np.mean(error ** 2))
    rmse_pos = float(np.sqrt(np.mean(pos_error ** 2)))
    mae_pos  = float(np.mean(np.abs(pos_error)))
    err_var  = float(np.var(error))

    distances    = np.linalg.norm(pos_error, axis=2)   # (N, T)
    precision_1m = float(np.mean(distances < 1.0) * 100)

    gt_pos_norm = np.mean(np.linalg.norm(gt[:, :, :3], axis=2)) + 1e-8 
    pct_error = float(
        np.mean(np.linalg.norm(pos_error, axis=2)) / gt_pos_norm * 100
    )

    return dict(
        mse          = mse,
        rmse_pos     = rmse_pos,
        mae_pos      = mae_pos,
        err_var      = err_var,
        precision_1m = precision_1m,
        pct_error    = pct_error,
        latency_ms   = latency_ms,
    )


# ══════════════════════════════════════════════
# KalmanNet inference
# ══════════════════════════════════════════════

def run_kalmannet(model: KalmanNet, test_loader: DataLoader):
    model.eval()
    all_preds, all_trues = [], []
    total_time  = 0.0
    total_steps = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            t0   = time.perf_counter()
            pred = model(y_batch, x_batch[:, 0, :])    # (B, T, 6)
            t1   = time.perf_counter()

            total_time  += (t1 - t0)
            total_steps += x_batch.shape[0] * x_batch.shape[1]

            all_preds.append(pred[:, 1:, :].cpu().numpy())    # exclude t=0
            all_trues.append(x_batch[:, 1:, :].cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)   # (N, T-1, 6)
    trues = np.concatenate(all_trues, axis=0)   # (N, T-1, 6)
    lat   = (total_time / total_steps) * 1000.0

    return preds, trues, lat


# ══════════════════════════════════════════════
# EKF inference
# ══════════════════════════════════════════════

def run_ekf(x_gt: np.ndarray, y_meas: np.ndarray,H_matrix=None,sigma_y=None,vel_std=None):
    """
    Args:
        x_gt   : (N, T, 6)
        y_meas : (N, T, 3)

    Returns:
        preds  : (N, T-1, 6)  — predictions for t=1…T-1
        lat    : ms per timestep
    """
    N, T, _ = x_gt.shape
    sigma_y = np.std(y_meas, axis=(0, 1))
    ekf      = ExtendedKalmanFilter(dt=0.01,sigma_y=sigma_y,H_matrix=H_matrix,vel_std=vel_std)
    preds    = np.zeros((N, T - 1, 6), dtype=np.float64)

    t0 = time.perf_counter()

    for i in range(N):
        ekf.reset(x_gt[i, 0, :])
        for t in range(1, T):
            preds[i, t - 1, :] = ekf.step(y_meas[i, t, :])

    t1  = time.perf_counter()
    lat = (t1 - t0) / (N * (T - 1)) * 1000.0

    return preds, lat


# ══════════════════════════════════════════════
# Pretty-print helpers
# ══════════════════════════════════════════════

def print_results(name: str, m: dict):
    print(f"\n=== {name} RESULTS ===")
    print(f"  MSE (State overall)      : {m['mse']:.6f}")
    print(f"  RMSE (Position)          : {m['rmse_pos']:.4f} meters")
    print(f"  MAE  (Position)          : {m['mae_pos']:.4f} meters")
    print(f"  Error Variance           : {m['err_var']:.4f}")
    print(f"  % Pos Error (rel. mean)  : {m['pct_error']:.4f} %")
    print(f"  Inlier Precision (<1m)   : {m['precision_1m']:.2f} %")
    print(f"  Latency                  : {m['latency_ms']:.4f} ms / step")


def print_comparison(ekf_m: dict, knet_m: dict):
    W = 16
    div = "=" * 58
    print(f"\n{div}")
    print("=== COMPARISON")
    print(div)
    print(f"  {'Metric':<24} {'EKF':>{W}} {'KalmanNet':>{W}}")
    print(f"  {'-'*24} {'-'*W} {'-'*W}")

    rows = [
        ("MSE",               ekf_m['mse'],           knet_m['mse']),
        ("RMSE (pos)",        ekf_m['rmse_pos'],       knet_m['rmse_pos']),
        ("MAE (pos)",         ekf_m['mae_pos'],        knet_m['mae_pos']),
        ("% Error",           ekf_m['pct_error'],      knet_m['pct_error']),
        ("Precision (<1m) %", ekf_m['precision_1m'],   knet_m['precision_1m']),
        ("Latency (ms/step)", ekf_m['latency_ms'],     knet_m['latency_ms']),
    ]

    for label, ev, kv in rows:
        print(f"  {label:<24} {ev:>{W}.4f} {kv:>{W}.4f}")

    print(f"{div}\n")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def main():
    # ── Load & sanitize ────────────────────────
    print(f"Loading test data from: {TEST_DATA}")
    raw     = np.load(TEST_DATA)
    x_np    = sanitize(raw["x"].astype(np.float32))   # (N, T, 6)
    y_np    = sanitize(raw["y"].astype(np.float32))   # (N, T, 3)
    N, T, _ = x_np.shape
    print(f"  Test set: {N} sequences × {T} timesteps  |  device: {DEVICE}")

    # ── DataLoader for batched KalmanNet inference ──
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_np), torch.tensor(y_np)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ── Load KalmanNet ─────────────────────────
    print(f"\nLoading KalmanNet weights from: {KNET_WEIGHTS}")
    model = KalmanNet(m=6, n=3, N_rnn=32).to(DEVICE)
    state = torch.load(KNET_WEIGHTS, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    print("  Weights loaded successfully.")

    # ── KalmanNet inference ────────────────────
    print("\nRunning KalmanNet inference…")
    knet_preds, gt_eval, knet_lat = run_kalmannet(model, test_loader)
    # gt_eval  : (N, T-1, 6)  t=1…T-1

    
    # ── EKF inference ──────────────────────────
    x_flat = x_np[:, 1:, :].reshape(-1, 6)   # (N*T, 6)
    y_flat = y_np[:, 1:, :].reshape(-1, 3)   # (N*T, 3)

    for j in range(3):
        corrs = [np.corrcoef(y_flat[:, j], x_flat[:, i])[0,1] for i in range(6)]
        best  = int(np.argmax(np.abs(corrs)))
    
    H_fit, _, _, _ = np.linalg.lstsq(x_flat, y_flat, rcond=None)  # (6,3)
    H_fit = H_fit.T  # (3,6)

    y_pred_flat = (H_fit @ x_flat.T).T         
    residuals   = y_flat - y_pred_flat           
    sigma_r     = np.std(residuals, axis=0)      
    
    vel_std = np.std(np.diff(x_np[:, :, 3:6], axis=1), axis=(0, 1))

    print("Running EKF inference…")
    ekf_preds, ekf_lat = run_ekf(
        x_np.astype(np.float64),
        y_np.astype(np.float64),
        H_matrix=H_fit,
        sigma_y=sigma_r,
        vel_std=vel_std
    )
    gt_ekf = x_np[:, 1:, :].astype(np.float64)  

    # ── Metrics ────────────────────────────────
    knet_metrics = compute_metrics(
        knet_preds.astype(np.float64),
        gt_eval.astype(np.float64),
        knet_lat,
    )
    ekf_metrics = compute_metrics(ekf_preds, gt_ekf, ekf_lat)

    # ── Output ─────────────────────────────────
    print_results("KALMANNET", knet_metrics)
    print_results("EKF",       ekf_metrics)
    print_comparison(ekf_metrics, knet_metrics)


if __name__ == "__main__":
    main()