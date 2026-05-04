"""
kalmannet_student.py
--------------------
Lite KalmanNet student model for knowledge distillation.

Architecture vs teacher (N_rnn=32):
  - Single fused GRUCell replaces three separate GRUCells (lower per-step cost)
  - GRU hidden size: 32 → 24
  - Bottleneck MLP: feat → N_rnn → N_rnn//2 → n*m
  - ELU activations (smoother gradients)
  - H initialised as velocity observer (physically motivated prior)
  - Gain clamped to [-1, 1] (prevents rollout explosion at test time)

forward() always returns (x_post, gains):
    x_post : (B, T, m)       — state predictions; index 0 = x0
    gains  : (B, T-1, m, n) — per-step Kalman gain for distillation supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class KalmanNetStudent(nn.Module):

    m = 6   # state dim
    n = 3   # measurement dim

    def __init__(self, N_rnn: int = 24):
        super().__init__()
        self.N_rnn = N_rnn

        # Fused GRU input: [delta_y(3) | e_post(6) | y_prev(3)] = 12
        fused_input_dim = self.n + self.m + self.n
        self.GRU_fused  = nn.GRUCell(fused_input_dim, N_rnn)

        # Gain MLP input: [h_fused(N_rnn) | y(3) | x_post_prev(6)]
        feat_dim   = N_rnn + self.n + self.m
        hidden_dim = N_rnn
        bottle_dim = N_rnn // 2

        self.fc1  = nn.Linear(feat_dim,   hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, bottle_dim)
        self.fc3  = nn.Linear(bottle_dim, self.n * self.m)

        nn.init.uniform_(self.fc3.weight, -0.001, 0.001)
        nn.init.zeros_(self.fc3.bias)

        # H: initialised to observe velocities (vx, vy, vz)
        self.H = nn.Linear(self.m, self.n, bias=True)
        with torch.no_grad():
            self.H.weight.zero_()
            self.H.weight[0, 3] = 1.0
            self.H.weight[1, 4] = 1.0
            self.H.weight[2, 5] = 1.0
            self.H.bias.zero_()

        # Constant-velocity F (non-trainable)
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
            x_post : (B, T, m)        — states; x_post[:,0,:] = x0
            gains  : (B, T-1, m, n)  — Kalman gains for t=1..T-1
        """
        B, T, _ = Yseq.shape
        device  = Yseq.device

        x0 = x0.view(-1, self.m)[:B]   # last-batch guard

        x_post  = torch.zeros(B, T, self.m, device=device)
        x_post[:, 0, :] = x0
        h_fused = torch.zeros(B, self.N_rnn, device=device)

        gains = []

        for t in range(1, T):
            y       = Yseq[:, t, :]
            y_prev  = Yseq[:, t - 1, :]
            delta_y = y - y_prev

            e_post = (x_post[:, t - 1, :] - x_post[:, t - 2, :]
                      if t > 1 else torch.zeros(B, self.m, device=device))

            gru_in  = torch.cat([delta_y, e_post, y_prev], dim=1)
            h_fused = self.GRU_fused(gru_in, h_fused)

            feat = torch.cat([h_fused, y, x_post[:, t - 1, :]], dim=1)
            feat = F.elu(self.fc1(feat))
            feat = self.norm(feat)
            feat = F.elu(self.fc2(feat))
            K    = self.fc3(feat).view(B, self.m, self.n)
            K    = torch.clamp(K, -1.0, 1.0)

            gains.append(K)

            x_prior = torch.matmul(self.F,
                                   x_post[:, t - 1, :].unsqueeze(2)).squeeze(2)
            y_pred  = self.H(x_prior)
            innov   = y - y_pred

            x_post[:, t, :] = x_prior + torch.bmm(K, innov.unsqueeze(2)).squeeze(2)

        return x_post, torch.stack(gains, dim=1)   # (B,T,m), (B,T-1,m,n)