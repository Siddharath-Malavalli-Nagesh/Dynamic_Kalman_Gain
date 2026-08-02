"""
evaluate_models.py
------------------
Three-way comparison: KalmanNet (teacher) | KalmanNet Student (shadow) | EKF

Metrics (per model):
    Position (px, py, pz):
        MSE, RMSE, RMSE per axis (x/y/z), MAE, MAE per axis,
        error variance, inlier precision (<1 m), % pos error
    Velocity (vx, vy, vz):
        MSE, RMSE, RMSE per axis (vx/vy/vz), MAE, MAE per axis,
        error variance
    Full state:
        MSE

Latency (step-wise, no batching):
    CPU : all three models
    GPU : KalmanNet + Student only (EKF is always CPU)

Expected files:
    best_knet_nclt.pt      KalmanNet teacher weights
    best_student_nclt.pt   Student weights
    nclt_test.npz          keys: 'x' (N,T,6), 'y' (N,T,3)
    ekf.py                 ExtendedKalmanFilter
    kalmannet_student.py   KalmanNetStudent
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ekf import ExtendedKalmanFilter
from ukf import UnscentedKalmanFilter
from pf  import ParticleFilter
from kalmannet_student import KalmanNetStudent
import torch.nn.functional as F
# ──────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────
KNET_WEIGHTS    = "best_knet_nclt.pt"
STUDENT_WEIGHTS = "best_student_nclt.pt"
TEST_DATA       = "Downloads/nclt_test.npz"
TRAIN_DATA      = "Downloads/nclt_train.npz"
# How many sequences to use for step-wise latency timing.
# Reduce if CPU timing is too slow on large test sets.
LATENCY_N_SEQ = 20

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_CUDA = torch.cuda.is_available()


# ══════════════════════════════════════════════
# KalmanNet teacher — exact architecture match
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
        F  = torch.eye(m)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        self.F = nn.Parameter(F, requires_grad=False)

    def forward(self, Yseq: torch.Tensor, x0: torch.Tensor):
        B, T, _ = Yseq.shape
        device  = Yseq.device

        x0 = x0.view(-1, self.m)[:B]

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
                      if t > 1 else torch.zeros(B, self.m, device=device))

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

            x_post[:, t, :] = x_prior + torch.bmm(K, innov.unsqueeze(2)).squeeze(2)

        # Returns plain tensor (no gains needed at eval time)
        return x_post   # (B, T, 6)


# ══════════════════════════════════════════════
# Data helper
# ══════════════════════════════════════════════

def _sanitize(arr: np.ndarray) -> np.ndarray:
    arr[np.isinf(arr)] = np.nan
    return np.nan_to_num(arr, nan=0.0)


# ══════════════════════════════════════════════
# Metrics — full per-dimension breakdown
# ══════════════════════════════════════════════

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Args:
        pred, gt : (N, T, 6)  t=1..T-1  (t=0 excluded by caller)
    Returns complete metric dict. Latency added separately.
    """
    error     = pred - gt
    pos_error = error[:, :, :3]   # (N, T, 3)
    vel_error = error[:, :, 3:]   # (N, T, 3)

    pos_dist = np.linalg.norm(pos_error, axis=2)   # (N, T)
    vel_dist = np.linalg.norm(vel_error, axis=2)

    # Per-dimension position
    rmse_px = float(np.sqrt(np.mean(pos_error[:, :, 0] ** 2)))
    rmse_py = float(np.sqrt(np.mean(pos_error[:, :, 1] ** 2)))
    rmse_pz = float(np.sqrt(np.mean(pos_error[:, :, 2] ** 2)))
    mae_px  = float(np.mean(np.abs(pos_error[:, :, 0])))
    mae_py  = float(np.mean(np.abs(pos_error[:, :, 1])))
    mae_pz  = float(np.mean(np.abs(pos_error[:, :, 2])))

    # Per-dimension velocity
    rmse_vx = float(np.sqrt(np.mean(vel_error[:, :, 0] ** 2)))
    rmse_vy = float(np.sqrt(np.mean(vel_error[:, :, 1] ** 2)))
    rmse_vz = float(np.sqrt(np.mean(vel_error[:, :, 2] ** 2)))
    mae_vx  = float(np.mean(np.abs(vel_error[:, :, 0])))
    mae_vy  = float(np.mean(np.abs(vel_error[:, :, 1])))
    mae_vz  = float(np.mean(np.abs(vel_error[:, :, 2])))

    # Grouped position
    mse_pos      = float(np.mean(pos_error ** 2))
    rmse_pos     = float(np.sqrt(np.mean(pos_dist ** 2)))
    mae_pos      = float(np.mean(pos_dist))
    var_pos      = float(np.var(pos_dist))
    precision_1m = float(np.mean(pos_dist < 1.0) * 100)
    gt_pos_norm  = np.mean(np.linalg.norm(gt[:, :, :3], axis=2)) + 1e-8
    pct_error    = float(mae_pos / gt_pos_norm * 100)

    # Grouped velocity
    mse_vel  = float(np.mean(vel_error ** 2))
    rmse_vel = float(np.sqrt(np.mean(vel_dist ** 2)))
    mae_vel  = float(np.mean(vel_dist))
    var_vel  = float(np.var(vel_dist))

    # Full state
    mse_full = float(np.mean(error ** 2))

    return dict(
        rmse_px=rmse_px, rmse_py=rmse_py, rmse_pz=rmse_pz,
        mae_px=mae_px,   mae_py=mae_py,   mae_pz=mae_pz,
        rmse_vx=rmse_vx, rmse_vy=rmse_vy, rmse_vz=rmse_vz,
        mae_vx=mae_vx,   mae_vy=mae_vy,   mae_vz=mae_vz,
        mse_pos=mse_pos,   rmse_pos=rmse_pos, mae_pos=mae_pos,
        var_pos=var_pos,   precision_1m=precision_1m, pct_error=pct_error,
        mse_vel=mse_vel,   rmse_vel=rmse_vel,
        mae_vel=mae_vel,   var_vel=var_vel,
        mse_full=mse_full,
    )


# ══════════════════════════════════════════════
# Accuracy inference — batched, for predictions only
# ══════════════════════════════════════════════

def _infer_torch_batched(model: nn.Module, x_np: np.ndarray,
                          y_np: np.ndarray) -> np.ndarray:
    """
    Returns (N, T-1, 6). Handles both KalmanNet (tensor) and
    KalmanNetStudent (tuple) return types.
    """
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(x_np), torch.tensor(y_np)),
        batch_size=32, shuffle=False,
    )
    preds = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            B       = x_batch.shape[0]
            out     = model(y_batch, x_batch[:, 0, :])
            # Unwrap tuple if student returns (x_post, gains)
            pred    = out[0] if isinstance(out, tuple) else out
            preds.append(pred[:B, 1:, :].cpu().numpy())
    return np.concatenate(preds, axis=0)   # (N, T-1, 6)


def _infer_ekf(x_gt: np.ndarray, y_meas: np.ndarray,
               H_matrix: np.ndarray, H_bias: np.ndarray,
               sigma_r: np.ndarray, accel_std: float = 0.5) -> np.ndarray:
    """Returns (N, T-1, 6). Uses fitted H, bias, residual R, kinematic Q."""
    N, T, _ = x_gt.shape
    preds    = np.zeros((N, T - 1, 6), dtype=np.float64)

    for i in range(N):
        ekf       = ExtendedKalmanFilter(dt=0.01, accel_std=accel_std)
        ekf.H     = H_matrix
        ekf.R     = np.diag(sigma_r ** 2)
        ekf.reset(x_gt[i, 0, :])
        for t in range(1, T):
            z = y_meas[i, t, :] - H_bias   # subtract per-channel bias before filter
            preds[i, t - 1, :] = ekf.step(z)
    return preds


# ══════════════════════════════════════════════
# Step-wise latency — one sequence, one step at a time
# ══════════════════════════════════════════════

def _lat_torch(model: nn.Module, x_np: np.ndarray,
               y_np: np.ndarray, device: torch.device,
               n_seq: int) -> tuple:
    """
    Manually unrolls the model one step at a time on `device`.
    Works for both KalmanNet and KalmanNetStudent.
    Returns (mean_ms, std_ms).
    """
    model = model.to(device)
    model.eval()

    is_student = isinstance(model, KalmanNetStudent)
    N          = x_np.shape[0]
    T          = x_np.shape[1]
    n_seq      = min(n_seq, N)
    step_times = []

    with torch.no_grad():
        for i in range(n_seq):
            x_seq = torch.tensor(x_np[i], dtype=torch.float32, device=device)
            y_seq = torch.tensor(y_np[i], dtype=torch.float32, device=device)

            if is_student:
                h_fused = torch.zeros(1, model.N_rnn, device=device)
            else:
                h_Q     = torch.zeros(1, model.N_rnn, device=device)
                h_Sigma = torch.zeros(1, model.N_rnn, device=device)
                h_S     = torch.zeros(1, model.N_rnn, device=device)

            x_post    = torch.zeros(T, model.m, device=device)
            x_post[0] = x_seq[0]

            for t in range(1, T):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                y       = y_seq[t].unsqueeze(0)      # (1, n)
                y_prev  = y_seq[t - 1].unsqueeze(0)
                delta_y = y - y_prev
                e_post  = ((x_post[t - 1] - x_post[t - 2]).unsqueeze(0)
                           if t > 1
                           else torch.zeros(1, model.m, device=device))

                if is_student:
                    gru_in  = torch.cat([delta_y, e_post, y_prev], dim=1)
                    h_fused = model.GRU_fused(gru_in, h_fused)
                    feat    = torch.cat([h_fused, y,
                                         x_post[t - 1].unsqueeze(0)], dim=1)
                    feat    = F.elu(model.fc1(feat))
                    feat    = model.norm(feat)
                    feat    = F.elu(model.fc2(feat))
                    K       = model.fc3(feat).view(1, model.m, model.n)
                    K       = torch.clamp(K, -1.0, 1.0)
                else:
                    h_Q     = model.GRU_Q(delta_y, h_Q)
                    h_Sigma = model.GRU_Sigma(e_post, h_Sigma)
                    h_S     = model.GRU_S(delta_y, h_S)
                    feat    = torch.cat([h_Q, h_Sigma, h_S,
                                         y, x_post[t - 1].unsqueeze(0)], dim=1)
                    feat    = model.norm(feat)
                    feat    = torch.relu(model.fc1(feat))
                    feat    = torch.relu(model.fc2(feat))
                    K       = model.fc3(feat).view(1, model.m, model.n)

                x_prior   = (model.F @ x_post[t - 1].unsqueeze(1)).squeeze(1)
                y_pred    = model.H(x_prior.unsqueeze(0)).squeeze(0)
                innov     = y.squeeze(0) - y_pred
                x_post[t] = x_prior + (K @ innov.unsqueeze(1)).squeeze()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                step_times.append((t1 - t0) * 1000.0)

    return float(np.mean(step_times)), float(np.std(step_times))


def _lat_ekf(x_gt: np.ndarray, y_meas: np.ndarray, n_seq: int,
             H_matrix: np.ndarray, H_bias: np.ndarray,
             sigma_r: np.ndarray, accel_std: float = 0.5) -> tuple:
    """Returns (mean_ms, std_ms) for EKF on CPU."""
    N, T, _    = x_gt.shape
    n_seq      = min(n_seq, N)
    step_times = []

    for i in range(n_seq):
        ekf        = ExtendedKalmanFilter(dt=0.01, accel_std=accel_std)
        ekf.H      = H_matrix
        ekf.R      = np.diag(sigma_r ** 2)
        ekf.reset(x_gt[i, 0, :])
        for t in range(1, T):
            t0 = time.perf_counter()
            ekf.step(y_meas[i, t, :] - H_bias)
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)
    return float(np.mean(step_times)), float(np.std(step_times))

# ══════════════════════════════════════════════
# UKF inference and latency
# ══════════════════════════════════════════════

def _infer_ukf(x_gt: np.ndarray, y_meas: np.ndarray,
               H_matrix: np.ndarray, H_bias: np.ndarray,
               sigma_r: np.ndarray, accel_std: float = 0.1) -> np.ndarray:
    """Returns (N, T-1, 6)."""
    N, T, _ = x_gt.shape
    preds   = np.zeros((N, T - 1, 6), dtype=np.float64)
    for i in range(N):
        ukf       = UnscentedKalmanFilter(dt=0.01, accel_std=accel_std)
        ukf.H     = H_matrix
        ukf.R     = np.diag(sigma_r ** 2)
        ukf.reset(x_gt[i, 0, :])
        for t in range(1, T):
            z = y_meas[i, t, :] - H_bias
            preds[i, t - 1, :] = ukf.step(z)
    return preds


def _lat_ukf(x_gt: np.ndarray, y_meas: np.ndarray, n_seq: int,
             H_matrix: np.ndarray, H_bias: np.ndarray,
             sigma_r: np.ndarray, accel_std: float = 0.1) -> tuple:
    """Returns (mean_ms, std_ms) for UKF on CPU."""
    N, T, _    = x_gt.shape
    n_seq      = min(n_seq, N)
    step_times = []
    for i in range(n_seq):
        ukf       = UnscentedKalmanFilter(dt=0.01, accel_std=accel_std)
        ukf.H     = H_matrix
        ukf.R     = np.diag(sigma_r ** 2)
        ukf.reset(x_gt[i, 0, :])
        for t in range(1, T):
            z  = y_meas[i, t, :] - H_bias
            t0 = time.perf_counter()
            ukf.step(z)
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)
    return float(np.mean(step_times)), float(np.std(step_times))

def _infer_ukf_aug(x_gt, y_meas, H_matrix, sigma_r, accel_std=0.1):
    """UKF with online bias estimation — no H_bias subtraction needed."""
    N, T, _ = x_gt.shape
    preds   = np.zeros((N, T - 1, 6), dtype=np.float64)
    for i in range(N):
        ukf   = UnscentedKalmanFilter(dt=0.01, accel_std=accel_std,
                                       augment_bias=True)
        ukf.R = np.diag(sigma_r ** 2)
        ukf.reset(x_gt[i, 0, :])
        for t in range(1, T):
            full_state = ukf.step(y_meas[i, t, :])
            preds[i, t - 1, :] = full_state[:6]   # return only state, not bias
    return preds
# ══════════════════════════════════════════════
# PF inference and latency
# ══════════════════════════════════════════════

def _infer_pf(x_gt: np.ndarray, y_meas: np.ndarray,
              H_matrix: np.ndarray, H_bias: np.ndarray,
              sigma_r: np.ndarray, accel_std: float = 0.1,
              N_particles: int = 200) -> np.ndarray:
    """Returns (N, T-1, 6)."""
    N, T, _ = x_gt.shape
    preds   = np.zeros((N, T - 1, 6), dtype=np.float64)
    for i in range(N):
        pf       = ParticleFilter(dt=0.01, accel_std=accel_std,
                                   sigma_y=sigma_r, H_matrix=H_matrix,
                                   N_particles=N_particles)
        pf.reset(x_gt[i, 0, :])
        for t in range(1, T):
            z = y_meas[i, t, :] - H_bias
            preds[i, t - 1, :] = pf.step(z)
    return preds


def _lat_pf(x_gt: np.ndarray, y_meas: np.ndarray, n_seq: int,
            H_matrix: np.ndarray, H_bias: np.ndarray,
            sigma_r: np.ndarray, accel_std: float = 0.1,
            N_particles: int = 200) -> tuple:
    """Returns (mean_ms, std_ms) for PF on CPU."""
    N, T, _    = x_gt.shape
    n_seq      = min(n_seq, N)
    step_times = []
    for i in range(n_seq):
        pf = ParticleFilter(dt=0.01, accel_std=accel_std,
                             sigma_y=sigma_r, H_matrix=H_matrix,
                             N_particles=N_particles)
        pf.reset(x_gt[i, 0, :])
        for t in range(1, T):
            z  = y_meas[i, t, :] - H_bias
            t0 = time.perf_counter()
            pf.step(z)
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)
    return float(np.mean(step_times)), float(np.std(step_times))

# ══════════════════════════════════════════════
# Print helpers
# ══════════════════════════════════════════════

def _print_model_results(name: str, m: dict,
                          cpu_mean: float, cpu_std: float,
                          gpu_mean: float, gpu_std: float):
    gpu_str = (f"{gpu_mean:.4f} ± {gpu_std:.4f} ms"
               if gpu_mean >= 0 else "N/A (CPU-only model)")

    print(f"\n{'=' * 62}")
    print(f"  {name}")
    print(f"{'=' * 62}")

    print(f"\n  ── Position (px, py, pz) {'─' * 30}")
    print(f"  MSE                      : {m['mse_pos']:.6f} m²")
    print(f"  RMSE                     : {m['rmse_pos']:.4f} m")
    print(f"  RMSE  x / y / z          : {m['rmse_px']:.4f} / {m['rmse_py']:.4f} / {m['rmse_pz']:.4f} m")
    print(f"  MAE                      : {m['mae_pos']:.4f} m")
    print(f"  MAE   x / y / z          : {m['mae_px']:.4f} / {m['mae_py']:.4f} / {m['mae_pz']:.4f} m")
    print(f"  Error Variance           : {m['var_pos']:.6f}")
    print(f"  Inlier Precision (<1 m)  : {m['precision_1m']:.2f} %")
    print(f"  % Pos Error (rel. mean)  : {m['pct_error']:.4f} %")

    print(f"\n  ── Velocity (vx, vy, vz) {'─' * 30}")
    print(f"  MSE                      : {m['mse_vel']:.6f} (m/s)²")
    print(f"  RMSE                     : {m['rmse_vel']:.4f} m/s")
    print(f"  RMSE  vx / vy / vz       : {m['rmse_vx']:.4f} / {m['rmse_vy']:.4f} / {m['rmse_vz']:.4f} m/s")
    print(f"  MAE                      : {m['mae_vel']:.4f} m/s")
    print(f"  MAE   vx / vy / vz       : {m['mae_vx']:.4f} / {m['mae_vy']:.4f} / {m['mae_vz']:.4f} m/s")
    print(f"  Error Variance           : {m['var_vel']:.6f}")

    print(f"\n  ── Full State {'─' * 42}")
    print(f"  MSE                      : {m['mse_full']:.6f}")

    print(f"\n  ── Latency (step-wise, 1 seq × 1 step) {'─' * 16}")
    print(f"  CPU   mean ± std         : {cpu_mean:.4f} ± {cpu_std:.4f} ms/step")
    print(f"  GPU   mean ± std         : {gpu_str}")


def _print_comparison(names: list, metrics: list,
                       cpu_means: list, cpu_stds: list,
                       gpu_means: list, gpu_stds: list):
    W   = 14
    NC  = len(names)
    div = "=" * (30 + (W + 1) * NC)

    print(f"\n{div}")
    print("  SEVEN-WAY COMPARISON TABLE")
    print(div)

    hdr = f"  {'Metric':<28}"
    for n in names:
        hdr += f" {n:>{W}}"
    print(hdr)
    print(f"  {'-'*28}" + f" {'-'*W}" * NC)

    def row(label, vals):
        line = f"  {label:<28}"
        for v in vals:
            line += f" {v:>{W}.4f}" if isinstance(v, float) and v >= 0 else f" {'N/A':>{W}}"
        print(line)

    def sec(title):
        print(f"\n  ── {title}")

    sec("Position ──────────────────────────────────")
    row("MSE (m²)",             [m['mse_pos']      for m in metrics])
    row("RMSE (m)",             [m['rmse_pos']     for m in metrics])
    row("RMSE_x (m)",           [m['rmse_px']      for m in metrics])
    row("RMSE_y (m)",           [m['rmse_py']      for m in metrics])
    row("RMSE_z (m)",           [m['rmse_pz']      for m in metrics])
    row("MAE (m)",              [m['mae_pos']      for m in metrics])
    row("MAE_x (m)",            [m['mae_px']       for m in metrics])
    row("MAE_y (m)",            [m['mae_py']       for m in metrics])
    row("MAE_z (m)",            [m['mae_pz']       for m in metrics])
    row("Var (pos)",            [m['var_pos']      for m in metrics])
    row("Precision <1m (%)",    [m['precision_1m'] for m in metrics])
    row("% Pos Error",          [m['pct_error']    for m in metrics])

    sec("Velocity ──────────────────────────────────")
    row("MSE (m/s)²",           [m['mse_vel']      for m in metrics])
    row("RMSE (m/s)",           [m['rmse_vel']     for m in metrics])
    row("RMSE_vx (m/s)",        [m['rmse_vx']      for m in metrics])
    row("RMSE_vy (m/s)",        [m['rmse_vy']      for m in metrics])
    row("RMSE_vz (m/s)",        [m['rmse_vz']      for m in metrics])
    row("MAE (m/s)",            [m['mae_vel']      for m in metrics])
    row("MAE_vx (m/s)",         [m['mae_vx']       for m in metrics])
    row("MAE_vy (m/s)",         [m['mae_vy']       for m in metrics])
    row("MAE_vz (m/s)",         [m['mae_vz']       for m in metrics])
    row("Var (vel)",            [m['var_vel']      for m in metrics])

    sec("Full State ─────────────────────────────────")
    row("MSE",                  [m['mse_full']     for m in metrics])

    sec("CPU Latency  step-wise (ms/step) ─────────")
    row("Mean",                 cpu_means)
    row("Std",                  cpu_stds)

    sec("GPU Latency  step-wise (ms/step) ─────────")
    row("Mean",                 gpu_means)
    row("Std",                  gpu_stds)

    print(f"\n{div}\n")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def main():
    import inspect
    from ekf import ExtendedKalmanFilter
    src = inspect.getsource(ExtendedKalmanFilter.update)
    print(src[:300])
    # ── Load data ─────────────────────────────
    print(f"Loading: {TEST_DATA}")
    raw     = np.load(TEST_DATA)
    x_np    = _sanitize(raw["x"].astype(np.float32))   # (N, T, 6)
    y_np    = _sanitize(raw["y"].astype(np.float32))   # (N, T, 3)
    N, T, _ = x_np.shape
    print(f"  {N} sequences × {T} timesteps | device: {DEVICE}")
    print(f"  Latency timing: first {min(LATENCY_N_SEQ, N)} sequences\n")

    x_f64   = x_np.astype(np.float64)
    y_f64   = y_np.astype(np.float64)
    gt_eval = x_np[:, 1:, :].astype(np.float64)   # (N, T-1, 6)

    print("  Loading training data to fit EKF R and accel_std (train set only)...")
    _tr   = np.load(TRAIN_DATA)
    _tr_x = _sanitize(_tr["x"].astype(np.float32)).astype(np.float64)
    _tr_y = _sanitize(_tr["y"].astype(np.float32)).astype(np.float64)

    # Use physically correct H: observe velocities (vx, vy, vz)
    # y[0] ≈ vx, y[1] ≈ vy, y[2] ≈ vz+gravity_bias
    # This matches the NCLT odometry setup in the KalmanNet paper.
    H_fit = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)

    # Fit bias per channel: mean(y - H @ x) over training set
    _x_flat   = _tr_x[:, 1:, :].reshape(-1, 6)
    _y_flat   = _tr_y[:, 1:, :].reshape(-1, 3)
    _y_pred   = (H_fit @ _x_flat.T).T           # (N*T, 3)
    _residuals = _y_flat - _y_pred
    H_bias_fit = np.mean(_residuals, axis=0)    # (3,) per-channel bias
    sigma_r    = np.std(_residuals, axis=0)     # (3,) measurement noise

    # Estimate accel_std from velocity differences
    _vel_diff = np.diff(_tr_x[:, :, 3:6], axis=1)
    accel_std = float(np.clip(np.std(_vel_diff), 0.01, 1.0))

    print(f"  H matrix: velocity observer (fixed, physically correct)")
    print(f"  Fitted H bias: {np.round(H_bias_fit, 4)}")
    print(f"  Residual sigma_r: {np.round(sigma_r, 4)}")
    print(f"  Estimated accel_std: {accel_std:.4f} m/s")
    del _tr, _tr_x, _tr_y, _x_flat, _y_flat, _y_pred, _residuals, _vel_diff
    # ── Load models ───────────────────────────
    print(f"Loading KalmanNet from:  {KNET_WEIGHTS}")
    knet = KalmanNet(m=6, n=3, N_rnn=32).to(DEVICE)
    knet.load_state_dict(torch.load(KNET_WEIGHTS, map_location=DEVICE))
    knet.eval()
    print(f"  Params: {sum(p.numel() for p in knet.parameters()):,}")

    print(f"Loading Student from:    {STUDENT_WEIGHTS}")
    student = KalmanNetStudent(N_rnn=24).to(DEVICE)
    student.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=DEVICE))
    student.eval()
    print(f"  Params: {sum(p.numel() for p in student.parameters()):,}")

    # ── Batched accuracy inference ─────────────
    print("\nRunning accuracy inference (batched)…")
    knet_preds    = _infer_torch_batched(knet,    x_np, y_np)
    student_preds = _infer_torch_batched(student, x_np, y_np)
    ekf_preds   = _infer_ekf(x_f64, y_f64, H_fit, H_bias_fit, sigma_r, accel_std)
    ukf_preds     = _infer_ukf(x_f64, y_f64, H_fit, H_bias_fit, sigma_r, accel_std)
    print("  Running UKF (augmented bias)...")
    ukf_aug_preds = _infer_ukf_aug(x_f64, y_f64, H_fit, sigma_r, accel_std)
    print("  Running PF (200 particles)...")
    pf_preds    = _infer_pf(x_f64, y_f64, H_fit, H_bias_fit, sigma_r, accel_std,
                               N_particles=200)
    print("  Running PF (500 particles)...")
    pf500_preds = _infer_pf(x_f64, y_f64, H_fit, H_bias_fit, sigma_r, accel_std,
                             N_particles=500)
    print("  Running PF (1000 particles)...")
    pf1000_preds = _infer_pf(x_f64, y_f64, H_fit, H_bias_fit, sigma_r, accel_std,
                              N_particles=1000)
    print("  Done.")

    # ── Metrics ───────────────────────────────
    print("Computing metrics…")
    knet_m  = compute_metrics(knet_preds.astype(np.float64),    gt_eval)
    stu_m   = compute_metrics(student_preds.astype(np.float64), gt_eval)
    ekf_m   = compute_metrics(ekf_preds,   gt_eval)
    ukf_m   = compute_metrics(ukf_preds,    gt_eval)
    pf_m    = compute_metrics(pf_preds,     gt_eval)
    pf500_m = compute_metrics(pf500_preds,gt_eval)
    pf1000_m = compute_metrics(pf1000_preds,gt_eval)
    # ── Step-wise CPU latency ──────────────────
    cpu = torch.device("cpu")
    print(f"\nStep-wise CPU latency ({LATENCY_N_SEQ} seqs)…")

    print("  EKF…")
    ekf_cpu_m,   ekf_cpu_s   = _lat_ekf(x_f64, y_f64, LATENCY_N_SEQ, H_fit, H_bias_fit, sigma_r, accel_std)

    print("  KalmanNet…")
    knet_cpu_m, knet_cpu_s = _lat_torch(knet, x_np, y_np, cpu, LATENCY_N_SEQ)

    print("  Student…")
    stu_cpu_m, stu_cpu_s = _lat_torch(student, x_np, y_np, cpu, LATENCY_N_SEQ)

    print("  UKF...")
    ukf_cpu_m, ukf_cpu_s = _lat_ukf(x_f64, y_f64, LATENCY_N_SEQ,
                                      H_fit, H_bias_fit, sigma_r, accel_std)

    print("  PF(200)...")
    pf_cpu_m, pf_cpu_s   = _lat_pf(x_f64, y_f64, LATENCY_N_SEQ,
                                     H_fit, H_bias_fit, sigma_r, accel_std,
                                     N_particles=200)
    print("  PF(500)...")
    pf500_cpu_m, pf500_cpu_s   = _lat_pf(x_f64, y_f64, LATENCY_N_SEQ,
                                         H_fit, H_bias_fit, sigma_r, accel_std,
                                         N_particles=500)
    print("  PF(1000)...")
    pf1000_cpu_m, pf1000_cpu_s   = _lat_pf(x_f64, y_f64, LATENCY_N_SEQ,
                                             H_fit, H_bias_fit, sigma_r, accel_std,
                                             N_particles=1000)

    # Restore to DEVICE after CPU latency pass
    knet.to(DEVICE)
    student.to(DEVICE)

    # ── Step-wise GPU latency ──────────────────
    ekf_gpu_m = ekf_gpu_s = -1.0   # EKF always CPU-only

    if HAS_CUDA:
        print(f"\nStep-wise GPU latency ({LATENCY_N_SEQ} seqs)…")

        print("  KalmanNet…")
        knet_gpu_m, knet_gpu_s = _lat_torch(knet, x_np, y_np, DEVICE, LATENCY_N_SEQ)

        print("  Student…")
        stu_gpu_m, stu_gpu_s = _lat_torch(student, x_np, y_np, DEVICE, LATENCY_N_SEQ)

        knet.to(DEVICE)
        student.to(DEVICE)
    else:
        print("\n  No CUDA — GPU latency skipped.")
        knet_gpu_m = knet_gpu_s = -1.0
        stu_gpu_m  = stu_gpu_s  = -1.0

    # ── Individual result blocks ───────────────
    _print_model_results(
        "KALMANNET — Teacher",
        knet_m, knet_cpu_m, knet_cpu_s, knet_gpu_m, knet_gpu_s)

    _print_model_results(
        "KALMANNET STUDENT — Shadow",
        stu_m, stu_cpu_m, stu_cpu_s, stu_gpu_m, stu_gpu_s)

    _print_model_results(
        "EKF — Baseline",
        ekf_m, ekf_cpu_m, ekf_cpu_s, -1.0, -1.0)
    _print_model_results(
        "UKF — Unscented Kalman Filter",
        ukf_m, ukf_cpu_m, ukf_cpu_s, -1.0, -1.0)

    _print_model_results(
        "PF — Particle Filter (N=200)",
        pf_m, pf_cpu_m, pf_cpu_s, -1.0, -1.0)
    _print_model_results(
            "PF — Particle Filter (N=500)",
            pf500_m, pf500_cpu_m, pf500_cpu_s, -1.0, -1.0)
    _print_model_results(
                "PF — Particle Filter (N=1000)",
                pf1000_m, pf1000_cpu_m, pf1000_cpu_s, -1.0, -1.0)

    # ── Three-way comparison table ─────────────
    _print_comparison(
        names    = ["EKF", "UKF", "PF(200)","PF(500)","PF(1000)", "KalmanNet", "Student"],
        metrics  = [ekf_m, ukf_m, pf_m,pf500_m,pf1000_m, knet_m, stu_m],
        cpu_means= [ekf_cpu_m, ukf_cpu_m, pf_cpu_m,pf500_cpu_m,pf1000_cpu_m, knet_cpu_m, stu_cpu_m],
        cpu_stds = [ekf_cpu_s, ukf_cpu_s, pf_cpu_s,pf500_cpu_s,pf1000_cpu_s, knet_cpu_s, stu_cpu_s],
        gpu_means= [-1.0,       -1.0,      -1.0, -1.0,-1.0,    knet_gpu_m, stu_gpu_m],
        gpu_stds = [-1.0,       -1.0,      -1.0, -1.0,-1.0,    knet_gpu_s, stu_gpu_s],
    )


if __name__ == "__main__":
    main()