"""
distillation_train.py
---------------------
Trains KalmanNetStudent to mimic KalmanNet (teacher) via knowledge distillation.

Distillation loss:
    L = (1 - alpha) * SmoothL1(student_pred, gt) + alpha * MSE(student_gain, teacher_gain)

Alpha schedule:
    Gradual linear decay from 0.7 → 0.3 over all epochs (no sudden jump).

Expected files:
    best_knet_nclt.pt    – trained teacher weights
    nclt_train.npz
    nclt_val.npz
    nclt_test.npz        – keys: 'x' (B,T,6), 'y' (B,T,3)

Output:
    best_student_nclt.pt – best student weights (by val position RMSE)
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kalmannet_student import KalmanNetStudent

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
TEACHER_WEIGHTS = "best_knet_nclt.pt"
STUDENT_WEIGHTS = "best_student_nclt.pt"

BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP    = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════
# Teacher model  (exact copy of production architecture)
# Returns (x_post, gains) so gains supervise the student
# ══════════════════════════════════════════════

class KalmanNetTeacher(nn.Module):

    def __init__(self, m=6, n=3, N_rnn=32):
        super().__init__()
        self.m, self.n, self.N_rnn = m, n, N_rnn

        self.GRU_Q     = nn.GRUCell(n, N_rnn)
        self.GRU_Sigma = nn.GRUCell(m, N_rnn)
        self.GRU_S     = nn.GRUCell(n, N_rnn)

        feat_dim   = N_rnn * 3 + n + m
        self.norm  = nn.LayerNorm(feat_dim)

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

        # ── guard: clamp x0 to actual batch size ──
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

            x_prior = torch.matmul(self.F, x_post[:, t - 1, :].unsqueeze(2)).squeeze(2)
            y_pred  = self.H(x_prior)
            innov   = y - y_pred

            x_post[:, t, :] = x_prior + torch.bmm(K, innov.unsqueeze(2)).squeeze(2)

        gains = torch.stack(gains, dim=1)   # (B, T-1, m, n)
        return x_post, gains


# ══════════════════════════════════════════════
# Distillation loss
# ══════════════════════════════════════════════

def distillation_loss(student_pred, target, student_gain, teacher_gain, alpha):
    """
    Args:
        student_pred  : (B, T-1, m)    — student state predictions t=1…T-1
        target        : (B, T-1, m)    — ground truth states       t=1…T-1
        student_gain  : (B, T-1, m, n) — student Kalman gains
        teacher_gain  : (B, T-1, m, n) — teacher Kalman gains
        alpha         : float          — distillation weight (annealed externally)

    All tensors must share the same leading batch dimension B.
    """
    # Both student_pred and target must be (B, T-1, m) with identical B
    task_loss    = nn.SmoothL1Loss()(student_pred, target)
    distill_loss = nn.MSELoss()(student_gain, teacher_gain)
    return (1.0 - alpha) * task_loss + alpha * distill_loss


# ══════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════

def sanitize(arr: np.ndarray) -> np.ndarray:
    arr[np.isinf(arr)] = np.nan
    return np.nan_to_num(arr, nan=0.0)


def make_loader(path: str, shuffle: bool) -> DataLoader:
    raw = np.load(path)
    x   = sanitize(raw["x"].astype(np.float32))
    y   = sanitize(raw["y"].astype(np.float32))
    return DataLoader(
        TensorDataset(torch.tensor(x), torch.tensor(y)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,   # we handle variable last-batch via x0 clamp
    )


# ══════════════════════════════════════════════
# Validation RMSE helper — alpha-independent save criterion
# ══════════════════════════════════════════════

def eval_rmse(model: KalmanNetStudent, loader: DataLoader) -> float:
    model.eval()
    sq_errs = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            B       = x_batch.shape[0]

            x0 = x_batch[:, 0, :]
            preds, _ = model(y_batch, x0)

            # preds shape is (B, T, m) — slice t=1…T-1, position only
            err = preds[:B, 1:, :3] - x_batch[:B, 1:, :3]
            sq_errs.append(err.cpu())

    all_err = torch.cat(sq_errs, dim=0)
    return float(torch.sqrt(torch.mean(all_err ** 2)).item())


# ══════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════

def train():
    print(f"Device: {DEVICE}")

    # ── Data ──────────────────────────────────
    print("Loading datasets…")
    train_loader = make_loader("Downloads/nclt_train.npz", shuffle=True)
    val_loader   = make_loader("Downloads/nclt_val.npz",   shuffle=False)
    test_loader  = make_loader("Downloads/nclt_test.npz",  shuffle=False)

    # ── Teacher (frozen) ───────────────────────
    print(f"Loading teacher weights from: {TEACHER_WEIGHTS}")
    teacher = KalmanNetTeacher(m=6, n=3, N_rnn=32).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("  Teacher frozen.")

    # ── Student ───────────────────────────────
    student   = KalmanNetStudent(N_rnn=24).to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(),
                                 lr=LR, weight_decay=WEIGHT_DECAY)

    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"  Teacher params : {t_params:,}")
    print(f"  Student params : {s_params:,}  ({s_params / t_params * 100:.1f}% of teacher)\n")

    best_val_rmse = float("inf")

    print(f"Starting distillation for {EPOCHS} epochs  "
          f"(alpha 0.70 → 0.30 linear decay)\n")

    for epoch in range(1, EPOCHS + 1):

        # ── Gradual alpha decay (no sudden jump) ──
        alpha = 0.7 - 0.4 * ((epoch - 1) / max(EPOCHS - 1, 1))

        # ── Train ─────────────────────────────
        student.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            B       = x_batch.shape[0]          # true batch size (may be < BATCH_SIZE)

            x0 = x_batch[:, 0, :]              # (B, 6)
            gt = x_batch[:, 1:, :]             # (B, T-1, 6)

            # Scheduled rollout noise — increases gradually with epoch
            noise_scale  = 0.01 * (epoch / EPOCHS)
            y_batch_noised = y_batch + torch.randn_like(y_batch) * noise_scale

            # Teacher: frozen, no grad
            with torch.no_grad():
                _, teacher_gains = teacher(y_batch_noised, x0)
                # teacher_gains: (B, T-1, m, n)

            # Student forward
            optimizer.zero_grad()
            student_preds, student_gains = student(y_batch_noised, x0)
            # student_preds : (B, T, m)
            # student_gains : (B, T-1, m, n)

            # Slice student predictions to t=1…T-1 and clamp to true B
            s_pred = student_preds[:B, 1:, :]   # (B, T-1, m)
            s_gain = student_gains[:B]           # (B, T-1, m, n)
            t_gain = teacher_gains[:B]           # (B, T-1, m, n)
            target = gt[:B]                      # (B, T-1, m)

            loss = distillation_loss(s_pred, target, s_gain, t_gain, alpha)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item() * B

        train_loss /= len(train_loader.dataset)

        # ── Validate ──────────────────────────
        student.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(DEVICE)
                y_val = y_val.to(DEVICE)
                B     = x_val.shape[0]

                x0     = x_val[:, 0, :]
                gt     = x_val[:, 1:, :]

                _, teacher_gains         = teacher(y_val, x0)
                student_preds, student_gains = student(y_val, x0)

                s_pred = student_preds[:B, 1:, :]
                s_gain = student_gains[:B]
                t_gain = teacher_gains[:B]
                target = gt[:B]

                loss = distillation_loss(s_pred, target, s_gain, t_gain, alpha)
                val_loss += loss.item() * B

        val_loss /= len(val_loader.dataset)

        # Save on position RMSE — not val_loss (which changes with alpha)
        val_rmse = eval_rmse(student, val_loader)
        saved    = ""
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(student.state_dict(), STUDENT_WEIGHTS)
            saved = f"  ← saved (RMSE={val_rmse:.4f} m)"

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{EPOCHS} | alpha={alpha:.3f} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Val RMSE: {val_rmse:.4f} m{saved}")

    # ══════════════════════════════════════════
    # Test evaluation
    # ══════════════════════════════════════════
    print("\n--- Training complete. Evaluating on test set ---")
    student.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=DEVICE))
    student.eval()

    all_preds, all_trues = [], []
    total_time  = 0.0
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
    print(f"  Latency         : {lat_ms:.4f} ms / step")
    print(f"\n  Student weights saved to: {STUDENT_WEIGHTS}")


if __name__ == "__main__":
    train()