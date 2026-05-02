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
# Paths  
# ──────────────────────────────────────────────
KNET_WEIGHTS = "best_knet_nclt.pt"
TEST_DATA    = "Downloads/nclt_test.npz"
BATCH_SIZE   = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════
# Exact KalmanNet Architecture  
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
# Metrics  
# ══════════════════════════════════════════════

# REPLACE the entire compute_metrics function with:

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Args:
        pred / gt : (N, T, 6) — t=1…T-1 already excluded by caller
    Returns separate metrics for position, velocity, and full state.
    Latency is computed and added separately by the caller.
    """
    error     = pred - gt                           # (N, T, 6)
    pos_error = error[:, :, :3]                     # (N, T, 3)
    vel_error = error[:, :, 3:]                     # (N, T, 3)

    pos_dist  = np.linalg.norm(pos_error, axis=2)  # (N, T)
    vel_dist  = np.linalg.norm(vel_error, axis=2)  # (N, T)

    # Position
    mse_pos       = float(np.mean(pos_error ** 2))
    rmse_pos      = float(np.sqrt(np.mean(pos_dist ** 2)))
    mae_pos       = float(np.mean(pos_dist))
    var_pos       = float(np.var(pos_dist))
    precision_1m  = float(np.mean(pos_dist < 1.0) * 100)
    rmse_px = float(np.sqrt(np.mean(pos_error[:, :, 0] ** 2)))
    rmse_py = float(np.sqrt(np.mean(pos_error[:, :, 1] ** 2)))
    rmse_pz = float(np.sqrt(np.mean(pos_error[:, :, 2] ** 2)))
    mae_px  = float(np.mean(np.abs(pos_error[:, :, 0])))
    mae_py  = float(np.mean(np.abs(pos_error[:, :, 1])))
    mae_pz  = float(np.mean(np.abs(pos_error[:, :, 2])))

    # Velocity
    mse_vel  = float(np.mean(vel_error ** 2))
    rmse_vel = float(np.sqrt(np.mean(vel_dist ** 2)))
    mae_vel  = float(np.mean(vel_dist))
    var_vel  = float(np.var(vel_dist))
    rmse_vx = float(np.sqrt(np.mean(vel_error[:, :, 0] ** 2)))
    rmse_vy = float(np.sqrt(np.mean(vel_error[:, :, 1] ** 2)))
    rmse_vz = float(np.sqrt(np.mean(vel_error[:, :, 2] ** 2)))
    mae_vx  = float(np.mean(np.abs(vel_error[:, :, 0])))
    mae_vy  = float(np.mean(np.abs(vel_error[:, :, 1])))
    mae_vz  = float(np.mean(np.abs(vel_error[:, :, 2])))
    # Full state
    mse_full = float(np.mean(error ** 2))

    # % position error — stable, relative to mean GT position magnitude
    gt_pos_norm = np.mean(np.linalg.norm(gt[:, :, :3], axis=2)) + 1e-8
    pct_error   = float(mae_pos / gt_pos_norm * 100)

    return dict(
        # position
        mse_pos      = mse_pos,
        rmse_pos     = rmse_pos,
        mae_pos      = mae_pos,
        var_pos      = var_pos,
        precision_1m = precision_1m,
        rmse_px=rmse_px, rmse_py=rmse_py, rmse_pz=rmse_pz,
        mae_px=mae_px,   mae_py=mae_py,   mae_pz=mae_pz,
        # velocity
        mse_vel      = mse_vel,
        rmse_vel     = rmse_vel,
        mae_vel      = mae_vel,
        var_vel      = var_vel,
        rmse_vx=rmse_vx, rmse_vy=rmse_vy, rmse_vz=rmse_vz,
        mae_vx=mae_vx,   mae_vy=mae_vy,   mae_vz=mae_vz,
        # full state
        mse_full     = mse_full,
        pct_error    = pct_error,
    )

# ══════════════════════════════════════════════
# KalmanNet inference
# ══════════════════════════════════════════════:

def run_kalmannet(model: KalmanNet, test_loader: DataLoader, force_cpu: bool = False):
    target_device = torch.device("cpu") if force_cpu else DEVICE
    model = model.to(target_device)
    model.eval()

    all_preds, all_trues = [], []
    total_time  = 0.0
    total_steps = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(target_device)
            y_batch = y_batch.to(target_device)

            # Sync GPU before timing if using CUDA
            if target_device.type == "cuda":
                torch.cuda.synchronize()

            t0   = time.perf_counter()
            pred = model(y_batch, x_batch[:, 0, :])
            if target_device.type == "cuda":
                torch.cuda.synchronize()
            t1   = time.perf_counter()

            total_time  += (t1 - t0)
            total_steps += x_batch.shape[0] * x_batch.shape[1]

            all_preds.append(pred[:, 1:, :].cpu().numpy())
            all_trues.append(x_batch[:, 1:, :].cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    lat   = (total_time / total_steps) * 1000.0

    return preds, trues, lat, ("CPU" if force_cpu else str(target_device).upper())

def run_kalmannet_stepwise(model: KalmanNet, x_np: np.ndarray, y_np: np.ndarray):
    """
    Step-wise KalmanNet inference on CPU for fair latency measurement.
    Processes one sequence, one timestep at a time — no batching.
    Returns latency only (predictions already computed in run_kalmannet).
    """
    model = model.to("cpu")
    model.eval()

    N, T, _ = x_np.shape
    step_times = []

    with torch.no_grad():
        for i in range(N):
            x_seq = torch.tensor(x_np[i], dtype=torch.float32)   # (T, 6)
            y_seq = torch.tensor(y_np[i], dtype=torch.float32)   # (T, 3)

            # Initialise hidden states — shape matches GRUCell (1, N_rnn)
            h_Q     = torch.zeros(1, model.N_rnn)
            h_Sigma = torch.zeros(1, model.N_rnn)
            h_S     = torch.zeros(1, model.N_rnn)

            x_post = torch.zeros(T, model.m)
            x_post[0] = x_seq[0]   # x0 from ground truth

            for t in range(1, T):
                t0 = time.perf_counter()

                y      = y_seq[t].unsqueeze(0)       # (1, 3)
                y_prev = y_seq[t - 1].unsqueeze(0)   # (1, 3)
                delta_y = y - y_prev

                e_post = (x_post[t - 1] - x_post[t - 2]).unsqueeze(0) \
                         if t > 1 else torch.zeros(1, model.m)

                h_Q     = model.GRU_Q(delta_y, h_Q)
                h_Sigma = model.GRU_Sigma(e_post, h_Sigma)
                h_S     = model.GRU_S(delta_y, h_S)

                feat = torch.cat([h_Q, h_Sigma, h_S,
                                  y, x_post[t - 1].unsqueeze(0)], dim=1)
                feat = model.norm(feat)
                feat = torch.relu(model.fc1(feat))
                feat = torch.relu(model.fc2(feat))
                K    = model.fc3(feat).view(1, model.m, model.n)

                x_prior = (model.F @ x_post[t - 1].unsqueeze(1)).squeeze(1)
                y_pred  = model.H(x_prior.unsqueeze(0)).squeeze(0)
                innov   = y.squeeze(0) - y_pred

                correction  = (K @ innov.unsqueeze(1)).squeeze()
                x_post[t]   = x_prior + correction

                t1 = time.perf_counter()
                step_times.append((t1 - t0) * 1000.0)   # ms

    mean_lat = float(np.mean(step_times))
    std_lat  = float(np.std(step_times))
    return mean_lat, std_lat

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

    step_times = []

    for i in range(N):
        ekf.reset(x_gt[i, 0, :])
        for t in range(1, T):
            t0 = time.perf_counter()
            preds[i, t - 1, :] = ekf.step(y_meas[i, t, :])
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)

    lat_mean = float(np.mean(step_times))
    lat_std  = float(np.std(step_times))
    return preds, lat_mean, lat_std, "CPU"


# ══════════════════════════════════════════════
# Pretty-print helpers
# ══════════════════════════════════════════════

def print_results(name: str, m: dict, latency_ms: float, lat_device: str,latency_std: float):
    print(f"\n=== {name} RESULTS ===")
    print(f"  -- Position (px, py, pz) --")
    print(f"  MSE                      : {m['mse_pos']:.6f} m²")
    print(f"  RMSE                     : {m['rmse_pos']:.4f} m")
    print(f"  RMSE_x / y / z           : {m['rmse_px']:.4f} / {m['rmse_py']:.4f} / {m['rmse_pz']:.4f} m")
    print(f"  MAE                      : {m['mae_pos']:.4f} m")
    print(f"  MAE_x  / y / z           : {m['mae_px']:.4f} / {m['mae_py']:.4f} / {m['mae_pz']:.4f} m")
    print(f"  Error Variance           : {m['var_pos']:.4f}")
    print(f"  Inlier Precision (<1m)   : {m['precision_1m']:.2f} %")
    print(f"  % Pos Error (rel. mean)  : {m['pct_error']:.4f} %")
    print(f"  -- Velocity (vx, vy, vz) --")
    print(f"  MSE                      : {m['mse_vel']:.6f} (m/s)²")
    print(f"  RMSE                     : {m['rmse_vel']:.4f} m/s")
    print(f"  RMSE_vx / vy / vz        : {m['rmse_vx']:.4f} / {m['rmse_vy']:.4f} / {m['rmse_vz']:.4f} m/s")
    print(f"  MAE                      : {m['mae_vel']:.4f} m/s")
    print(f"  MAE_vx  / vy / vz        : {m['mae_vx']:.4f} / {m['mae_vy']:.4f} / {m['mae_vz']:.4f} m/s")
    print(f"  Error Variance           : {m['var_vel']:.4f}")
    print(f"  -- Full State --")
    print(f"  MSE                      : {m['mse_full']:.6f}")
    print(f"  -- Latency (step-wise, CPU) --")
    print(f"  Mean                     : {latency_ms:.4f} ms / step")
    print(f"  Std                      : {latency_std:.4f} ms")



def print_comparison(ekf_m, ekf_lat, ekf_std, ekf_dev, knet_m, knet_step_lat, knet_step_std, knet_dev):
    W = 14
    div = "=" * 60
    print(f"\n{div}")
    print("=== COMPARISON")
    print(div)
    print(f"  {'Metric':<28} {'EKF':>{W}} {'KalmanNet':>{W}}")
    print(f"  {'-'*28} {'-'*W} {'-'*W}")

    rows = [
        ("--- Position ---",       None,                    None),
        ("MSE (m²)",               ekf_m['mse_pos'],        knet_m['mse_pos']),
        ("RMSE (m)",               ekf_m['rmse_pos'],       knet_m['rmse_pos']),
        ("RMSE_x (m)",             ekf_m['rmse_px'],        knet_m['rmse_px']),
        ("RMSE_y (m)",             ekf_m['rmse_py'],        knet_m['rmse_py']),
        ("RMSE_z (m)",             ekf_m['rmse_pz'],        knet_m['rmse_pz']),
        ("MAE (m)",                ekf_m['mae_pos'],        knet_m['mae_pos']),
        ("MAE_x (m)",              ekf_m['mae_px'],         knet_m['mae_px']),
        ("MAE_y (m)",              ekf_m['mae_py'],         knet_m['mae_py']),
        ("MAE_z (m)",              ekf_m['mae_pz'],         knet_m['mae_pz']),
        ("Error Variance",         ekf_m['var_pos'],        knet_m['var_pos']),
        ("Precision (<1m) %",      ekf_m['precision_1m'],   knet_m['precision_1m']),
        ("% Pos Error",            ekf_m['pct_error'],      knet_m['pct_error']),
        ("--- Velocity ---",       None,                    None),
        ("MSE (m/s)²",             ekf_m['mse_vel'],        knet_m['mse_vel']),
        ("RMSE (m/s)",             ekf_m['rmse_vel'],       knet_m['rmse_vel']),
        ("RMSE_vx (m/s)",          ekf_m['rmse_vx'],        knet_m['rmse_vx']),
        ("RMSE_vy (m/s)",          ekf_m['rmse_vy'],        knet_m['rmse_vy']),
        ("RMSE_vz (m/s)",          ekf_m['rmse_vz'],        knet_m['rmse_vz']),
        ("MAE (m/s)",              ekf_m['mae_vel'],        knet_m['mae_vel']),
        ("MAE_vx (m/s)",           ekf_m['mae_vx'],         knet_m['mae_vx']),
        ("MAE_vy (m/s)",           ekf_m['mae_vy'],         knet_m['mae_vy']),
        ("MAE_vz (m/s)",           ekf_m['mae_vz'],         knet_m['mae_vz']),
        ("Error Variance",         ekf_m['var_vel'],        knet_m['var_vel']),
        ("--- Full State ---",     None,                    None),
        ("MSE",                    ekf_m['mse_full'],       knet_m['mse_full']),
        ("--- Latency ---",        None,                    None),
        ("ms/step (CPU, mean)",    ekf_lat,   knet_step_lat),
        ("ms/step (CPU, ±std)",    ekf_std,   knet_step_std),
    ]

    for label, ev, kv in rows:
        if ev is None and kv is None and label.startswith("---"):
            print(f"\n  {label}")
            continue
        e_str = f"{ev:>{W}.4f}" if ev is not None else f"{'—':>{W}}"
        k_str = f"{kv:>{W}.4f}" if kv is not None else f"{'—':>{W}}"
        print(f"  {label:<28} {e_str} {k_str}")

    print(f"\n{div}\n")


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

    # ── KalmanNet on native device ─────────────
    print(f"Running KalmanNet on {DEVICE} for accuracy…")
    knet_preds, gt_eval, knet_lat, knet_dev = run_kalmannet(model, test_loader,
                                                              force_cpu=False)
    print("Running KalmanNet step-wise on CPU for fair latency benchmark…")
    knet_step_lat, knet_step_std = run_kalmannet_stepwise(model, x_np, y_np)
    
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
    ekf_preds, ekf_lat, ekf_std, ekf_dev = run_ekf(
        x_np.astype(np.float64),
        y_np.astype(np.float64),
        H_matrix=H_fit,
        sigma_y=sigma_r,
        vel_std=vel_std
    )
    gt_ekf = x_np[:, 1:, :].astype(np.float64)  

    # ── Metrics ────────────────────────────────
    knet_metrics = compute_metrics(knet_preds.astype(np.float64),
                                   gt_eval.astype(np.float64))
    ekf_metrics = compute_metrics(ekf_preds, gt_ekf)

    # ── Output ─────────────────────────────────
    print_results("KALMANNET", knet_metrics, knet_step_lat, "CPU", knet_step_std,)
    print_results("EKF",       ekf_metrics,  ekf_lat, "CPU",      ekf_std,       )
    print_comparison(ekf_metrics,  ekf_lat,       ekf_std,       "CPU",
                     knet_metrics, knet_step_lat, knet_step_std, "CPU")


if __name__ == "__main__":
    main()