"""
Per-step KalmanNet inference, factored out so both the RL meta-tuner
env and the ROS2 inference node can use it without code duplication.

A KalmanNetStepper wraps a trained KalmanNet model and exposes a
single-step .step(y_t) call. The optional gain_scale argument
multiplies the learned gain K element-wise (broadcast over the
observation dimension), which is how the RL meta-tuner injects its
action.
"""

import torch


class KalmanNetStepper:

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self.reset()

    def reset(self, x0=None):
        m, N = self.model.m, self.model.N_rnn
        self.h_Q     = torch.zeros(1, N, device=self.device)
        self.h_Sigma = torch.zeros(1, N, device=self.device)
        self.h_S     = torch.zeros(1, N, device=self.device)
        self.x_post   = (x0.view(1, m).to(self.device) if x0 is not None
                         else torch.zeros(1, m, device=self.device))
        self.x_post_1 = self.x_post.clone()
        self.y_prev   = None
        self.last_innov = torch.zeros(1, self.model.n, device=self.device)

    @torch.no_grad()
    def step(self, y_t, gain_scale=None):
        """
        y_t        : (n,) or (1, n) torch tensor on self.device
        gain_scale : (m,) or (1, m) tensor in approx [0.5, 1.5] or None

        Returns x_post (1, m), innov (1, n), K (1, m, n).
        """
        y = y_t.view(1, self.model.n).to(self.device)

        if self.y_prev is None:
            self.y_prev = y
            return self.x_post.clone(), self.last_innov, \
                   torch.zeros(1, self.model.m, self.model.n, device=self.device)

        delta_y = y - self.y_prev
        e_post  = self.x_post - self.x_post_1

        self.h_Q     = self.model.GRU_Q(delta_y, self.h_Q)
        self.h_Sigma = self.model.GRU_Sigma(e_post, self.h_Sigma)
        self.h_S     = self.model.GRU_S(delta_y, self.h_S)

        feat = torch.cat([self.h_Q, self.h_Sigma, self.h_S,
                          y, self.x_post], dim=1)
        feat = self.model.norm(feat)
        feat = torch.relu(self.model.fc1(feat))
        feat = torch.relu(self.model.fc2(feat))
        K    = self.model.fc3(feat).view(1, self.model.m, self.model.n)

        if gain_scale is not None:
            s = gain_scale.view(1, self.model.m, 1).to(self.device)
            K = K * s

        x_prior = torch.matmul(self.model.F,
                               self.x_post.unsqueeze(2)).squeeze(2)
        y_pred  = self.model.H(x_prior)
        innov   = y - y_pred
        x_new   = x_prior + torch.bmm(K, innov.unsqueeze(2)).squeeze(2)

        self.x_post_1 = self.x_post
        self.x_post   = x_new
        self.y_prev   = y
        self.last_innov = innov
        return x_new.clone(), innov, K
