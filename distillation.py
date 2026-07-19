"""
distillation_train.py
---------------------
Distillation training: KalmanNetStudent learns to mimic KalmanNet (teacher).

Loss (three signals):
    L = task_weight * SmoothL1(student_pred, GT)         — fit ground truth
      + alpha        * MSE(student_gain, teacher_gain)   — mimic Kalman gain
      + beta         * SmoothL1(student_pred, teacher_pred)  — mimic trajectory

Schedules (all smooth, no sudden jumps):
    alpha : cosine anneal 0.70 → 0.15  (gain distillation weight)
    beta  : linear decay  0.20 → 0.00  (trajectory imitation weight)
    noise : triangular    0.00 → peak → 0.00  (input augmentation)
    LR    : OneCycleLR warm-up + cosine anneal

Expected files:
    best_knet_nclt.pt        teacher weights
    nclt_train/val/test.npz  data splits

Output:
    best_student_nclt.pt     best student checkpoint (by val position RMSE)
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from kalmannet_student import KalmanNetStudent

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
TEACHER_WEIGHTS = "best_knet_nclt.pt"
STUDENT_WEIGHTS = "best_student_nclt_05.pt"

BATCH_SIZE   = 32
EPOCHS       = 120
LR_MAX       = 3e-3
LR_DIV       = 10.0         # initial LR = LR_MAX / LR_DIV
WEIGHT_DECAY = 1e-5
GRAD_CLIP    = 1.0

ALPHA_START  = 0.70       # gain distillation weight, epoch 1
ALPHA_END    = 0.15         # gain distillation weight, epoch EPOCHS
BETA_START   = 0.20         # trajectory imitation weight, epoch 1
BETA_END     = 0.00         # trajectory imitation weight, epoch EPOCHS
POS_WEIGHT   = 2.0          # position loss multiplier vs velocity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════
# Teacher — exact production architecture, unmodified
# ══════════════════════════════════════════════

class KalmanNetTeacher(nn.Module):

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

        gains = []

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

            gains.append(K)

            x_prior = torch.matmul(self.F,
                                   x_post[:, t - 1, :].unsqueeze(2)).squeeze(2)
            y_pred  = self.H(x_prior)
            innov   = y - y_pred

            x_post[:, t, :] = x_prior + torch.bmm(K, innov.unsqueeze(2)).squeeze(2)

        return x_post, torch.stack(gains, dim=1)   # (B,T,m), (B,T-1,m,n)


# ══════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════

def _pos_vel_smoothl1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SmoothL1 with position states weighted POS_WEIGHT× velocity states."""
    loss_pos = nn.SmoothL1Loss()(pred[:, :, :3], target[:, :, :3])
    loss_vel = nn.SmoothL1Loss()(pred[:, :, 3:], target[:, :, 3:])
    return POS_WEIGHT * loss_pos + loss_vel


def distillation_loss(student_pred:  torch.Tensor,
                      teacher_pred:  torch.Tensor,
                      target:        torch.Tensor,
                      student_gain:  torch.Tensor,
                      teacher_gain:  torch.Tensor,
                      alpha:         float,
                      beta:          float) -> torch.Tensor:
    """
    Three-signal distillation loss.
    All leading dimensions must be identical (enforced by caller's [:B] slicing).

    student_pred / teacher_pred / target : (B, T-1, m)
    student_gain / teacher_gain          : (B, T-1, m, n)
    """
    task_weight = max(1.0 - alpha - beta, 0.0)
    task_loss   = _pos_vel_smoothl1(student_pred, target)
    gain_loss   = nn.MSELoss()(student_gain, teacher_gain)
    traj_loss   = _pos_vel_smoothl1(student_pred, teacher_pred)
    return task_weight * task_loss + alpha * gain_loss + beta * traj_loss


# ══════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════

def _sanitize(arr: np.ndarray) -> np.ndarray:
    arr[np.isinf(arr)] = np.nan
    return np.nan_to_num(arr, nan=0.0)


def _make_loader(path: str, shuffle: bool):
    """Returns (DataLoader, y_std: np.ndarray shape (n,))."""
    raw   = np.load(path)
    x     = _sanitize(raw["x"].astype(np.float32))
    y     = _sanitize(raw["y"].astype(np.float32))
    y_std = np.std(y, axis=(0, 1)).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.tensor(x), torch.tensor(y)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
    )
    return loader, y_std


# ══════════════════════════════════════════════
# Validation helper — alpha/beta independent
# ══════════════════════════════════════════════

def _eval_rmse_pos(model: KalmanNetStudent, loader: DataLoader) -> float:
    model.eval()
    errs = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            B       = x_batch.shape[0]
            preds, _ = model(y_batch, x_batch[:, 0, :])
            err = preds[:B, 1:, :3] - x_batch[:B, 1:, :3]
            errs.append(err.cpu())
    return float(torch.sqrt(torch.mean(torch.cat(errs) ** 2)).item())


# ══════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════

def train():
    print(f"Device: {DEVICE}\n")

    # ── Data ──────────────────────────────────
    print("Loading datasets…")
    train_loader, y_std = _make_loader("Downloads/nclt_train.npz", shuffle=True)
    val_loader,   _     = _make_loader("Downloads/nclt_val.npz",   shuffle=False)
    test_loader,  _     = _make_loader("Downloads/nclt_test.npz",  shuffle=False)

    # Noise base: 5% of per-channel measurement std, shape (1,1,n) for broadcast
    noise_base = torch.tensor(y_std * 0.05, dtype=torch.float32
                               ).view(1, 1, -1).to(DEVICE)
    print(f"  Noise base (5% σ_y): {(y_std * 0.05).tolist()}")

    # ── Teacher (frozen) ───────────────────────
    print(f"\nLoading teacher from: {TEACHER_WEIGHTS}")
    teacher = KalmanNetTeacher(m=6, n=3, N_rnn=32).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("  Teacher frozen.")

    # ── Student ───────────────────────────────
    student   = KalmanNetStudent(N_rnn=24).to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(),
                                 lr=LR_MAX / LR_DIV,
                                 weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = LR_MAX,
        total_steps     = EPOCHS * len(train_loader),
        pct_start       = 0.15,
        anneal_strategy = "cos",
        div_factor      = LR_DIV,
        final_div_factor= 100.0,
    )

    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"  Teacher params: {t_params:,}")
    print(f"  Student params: {s_params:,}  ({s_params / t_params * 100:.1f}% of teacher)\n")

    best_val_rmse = float("inf")

    print(f"Training {EPOCHS} epochs | "
          f"α {ALPHA_START:.2f}→{ALPHA_END:.2f} cosine | "
          f"β {BETA_START:.2f}→{BETA_END:.2f} linear | "
          f"noise triangular\n")

    for epoch in range(1, EPOCHS + 1):
        t = (epoch - 1) / max(EPOCHS - 1, 1)

        # Schedules
        alpha       = ALPHA_END + 0.5 * (ALPHA_START - ALPHA_END) * (1 + np.cos(np.pi * t))
        beta        = BETA_START + (BETA_END - BETA_START) * t
        noise_scale = 1.0 - abs(2.0 * t - 1.0)   # triangular: 0→1→0

        # ── Train ─────────────────────────────
        student.train()
        train_loss = 0.0
        skipped    = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            B       = x_batch.shape[0]
            x0      = x_batch[:, 0, :]
            gt      = x_batch[:, 1:, :]   # (B, T-1, 6)

            try:
                y_noised = y_batch + torch.randn_like(y_batch) * noise_base * noise_scale
                y_noised = torch.nan_to_num(y_noised, nan=0.0, posinf=0.0, neginf=0.0)

                with torch.no_grad():
                    t_post, t_gains = teacher(y_noised, x0)

                optimizer.zero_grad()
                s_post, s_gains = student(y_noised, x0)

                # Slice to true B and t=1..T-1
                s_pred = s_post[:B, 1:, :]
                t_pred = t_post[:B, 1:, :]
                s_gain = s_gains[:B]
                t_gain = t_gains[:B]
                target = gt[:B]

                loss = distillation_loss(s_pred, t_pred, target,
                                         s_gain, t_gain, alpha, beta)

                if not torch.isfinite(loss):
                    skipped += 1
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * B

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    skipped += 1
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                raise

        denom      = max(len(train_loader.dataset) - skipped * BATCH_SIZE, 1)
        train_loss /= denom

        # ── Validate ──────────────────────────
        student.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(DEVICE)
                y_val = y_val.to(DEVICE)
                B     = x_val.shape[0]
                x0    = x_val[:, 0, :]
                gt    = x_val[:, 1:, :]

                t_post, t_gains = teacher(y_val, x0)
                s_post, s_gains = student(y_val, x0)

                loss = distillation_loss(
                    s_post[:B, 1:, :], t_post[:B, 1:, :], gt[:B],
                    s_gains[:B], t_gains[:B], alpha, beta)
                val_loss += loss.item() * B

        val_loss /= len(val_loader.dataset)

        val_rmse = _eval_rmse_pos(student, val_loader)
        saved    = ""
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(student.state_dict(), STUDENT_WEIGHTS)
            saved = f"  ← saved (RMSE={val_rmse:.4f} m)"

        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:03d}/{EPOCHS} | "
                  f"α={alpha:.3f} β={beta:.3f} | "
                  f"LR={lr_now:.2e} | "
                  f"Train={train_loss:.6f} Val={val_loss:.6f} | "
                  f"RMSE={val_rmse:.4f} m{saved}")

    # ══════════════════════════════════════════
    # Final test evaluation
    # ══════════════════════════════════════════
    print("\n--- Evaluating best student on test set ---")
    student.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=DEVICE))
    student.eval()

    all_preds, all_trues = [], []
    total_time = 0.0
    total_steps = 0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(DEVICE)
            y_test = y_test.to(DEVICE)
            B      = x_test.shape[0]

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds, _ = student(y_test, x_test[:, 0, :])
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_time  += (t1 - t0)
            total_steps += B * x_test.shape[1]

            all_preds.append(preds[:B, 1:, :].cpu())
            all_trues.append(x_test[:B, 1:, :].cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    pos_err  = all_preds[:, :, :3] - all_trues[:, :, :3]
    rmse_pos = float(torch.sqrt(torch.mean(pos_err ** 2)).item())
    mae_pos  = float(torch.mean(torch.abs(pos_err)).item())
    dists    = torch.norm(pos_err, dim=2)
    prec_1m  = float((dists < 1.0).float().mean().item() * 100)
    lat_ms   = (total_time / total_steps) * 1000.0

    print(f"\n  RMSE (position) : {rmse_pos:.4f} m")
    print(f"  MAE  (position) : {mae_pos:.4f} m")
    print(f"  Precision (<1m) : {prec_1m:.2f} %")
    print(f"  Latency (batch) : {lat_ms:.4f} ms/step")
    print(f"\n  Saved: {STUDENT_WEIGHTS}")


if __name__ == "__main__":
    train()