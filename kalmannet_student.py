"""
kalmannet_student.py
--------------------
Lite KalmanNet student model for knowledge distillation.

Reductions vs teacher (N_rnn=32):
  - GRU hidden size: 32 → 24
  - fc1 output: feat_dim → N_rnn*2, then fc2: N_rnn*2 → N_rnn  (kept but smaller)
  - fc3: N_rnn → n*m  (unchanged role)

forward() returns (x_post, gains):
    x_post : (B, T, m)       — predicted states (index 0 = x0)
    gains  : (B, T-1, m, n) — Kalman gain per timestep for distillation
"""

import torch
import torch.nn as nn


class KalmanNetStudent(nn.Module):

    m = 6
    n = 3

    def __init__(self, N_rnn: int = 24):
        super().__init__()
        self.N_rnn = N_rnn

        self.GRU_Q     = nn.GRUCell(self.n, N_rnn)
        self.GRU_Sigma = nn.GRUCell(self.m, N_rnn)
        self.GRU_S     = nn.GRUCell(self.n, N_rnn)

        feat_dim   = N_rnn * 3 + self.n + self.m
        self.norm  = nn.LayerNorm(feat_dim)

        self.fc1 = nn.Linear(feat_dim,    N_rnn * 2)
        self.fc2 = nn.Linear(N_rnn * 2,  N_rnn)
        self.fc3 = nn.Linear(N_rnn,       self.n * self.m)

        nn.init.uniform_(self.fc3.weight, -0.001, 0.001)
        nn.init.zeros_(self.fc3.bias)

        self.H = nn.Linear(self.m, self.n)
        nn.init.zeros_(self.H.weight)
        nn.init.zeros_(self.H.bias)

        dt = 0.01
        F  = torch.eye(self.m)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        self.F = nn.Parameter(F, requires_grad=False)

    def forward(self, Yseq: torch.Tensor, x0: torch.Tensor):
        """
        Args:
            Yseq : (B, T, n)
            x0   : (B, m)

        Returns:
            x_post : (B, T, m)
            gains  : (B, T-1, m, n)
        """
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