#!/usr/bin/env python3
"""
RL meta-tuner that sits on top of a frozen KalmanNet.

The agent observes a small feature vector each step (innovation, its
running magnitude, posterior velocity magnitude, last action) and
outputs a 6-D vector that scales the per-state-dim Kalman gain
multiplicatively. The base KalmanNet weights are NOT modified.

Action space  : Box(-1, 1, shape=(6,))
                Mapped to gain scaler in [0.5, 1.5] via
                s = 1.0 + 0.5 * action.

Observation   : Box(-inf, inf, shape=(11,))
                [ innov(3), innov_ema_mag(1), |vel|(1), last_action(6) ]

Reward        : -mean((x_post - x_gt) ** 2)  (per step)

Episode       : one training sequence (200 steps).

Algorithm     : PPO (stable-baselines3), tiny MLP policy 16-16.

Usage
-----
  python3 rl_metatuner.py train \\
      --weights best_knet_gazebo.pt \\
      --data    gazebo_train_vargrav.npz \\
      --steps   100000 \\
      --out     meta_tuner_ppo.zip
"""

import argparse
import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from train_model import KalmanNet, sanitize_data
from knet_step import KalmanNetStepper


GAIN_CENTER = 1.0
GAIN_RANGE  = 0.5     # action 1.0 → scaler 1.5; -1.0 → 0.5


class MetaTunerEnv(gym.Env):
    """One sequence per episode. action scales the gain each step."""

    metadata = {"render_modes": []}

    def __init__(self, x_data, y_data, knet_weights, device="cpu",
                 ema_alpha=0.1):
        super().__init__()
        self.x_data = x_data        # (N, T, 6)
        self.y_data = y_data        # (N, T, 3)
        self.N, self.T, self.m = x_data.shape
        self.n = y_data.shape[2]
        self.device = torch.device(device)
        self.ema_alpha = ema_alpha

        self.model = KalmanNet().to(self.device)
        self.model.load_state_dict(
            torch.load(knet_weights, map_location=self.device))
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.stepper = KalmanNetStepper(self.model, self.device)

        # 6-D action: per-state-dim gain multiplier
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.m,), dtype=np.float32)
        # 11-D observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n + 1 + 1 + self.m,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._reset_internal()

    def _reset_internal(self):
        self.t = 0
        self.seq_idx = int(self._rng.integers(0, self.N))
        self.innov_ema_mag = 0.0
        self.last_action = np.zeros(self.m, dtype=np.float32)

    def _build_obs(self, innov, x_post):
        innov_np = innov.detach().cpu().numpy().reshape(-1)
        vel_mag = float(np.linalg.norm(
            x_post.detach().cpu().numpy().reshape(-1)[3:]))
        return np.concatenate([
            innov_np,
            np.array([self.innov_ema_mag], dtype=np.float32),
            np.array([vel_mag], dtype=np.float32),
            self.last_action,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_internal()
        x0 = torch.tensor(self.x_data[self.seq_idx, 0],
                          dtype=torch.float32)
        self.stepper.reset(x0)
        # Take one warm-up step so innov is defined
        y0 = torch.tensor(self.y_data[self.seq_idx, 0], dtype=torch.float32)
        _ = self.stepper.step(y0)
        self.t = 1
        # Build initial obs from a hypothetical zero-action step preview
        y_t = torch.tensor(self.y_data[self.seq_idx, self.t], dtype=torch.float32)
        x_post, innov, _ = self.stepper.step(y_t)   # actually advances
        obs = self._build_obs(innov, x_post)
        # NB: we just advanced state by one; reward will be skipped this turn
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.last_action = action
        gain_scale = torch.tensor(
            GAIN_CENTER + GAIN_RANGE * action,
            dtype=torch.float32)

        self.t += 1
        y_t = torch.tensor(self.y_data[self.seq_idx, self.t],
                           dtype=torch.float32)
        x_gt = self.x_data[self.seq_idx, self.t]
        x_post, innov, _ = self.stepper.step(y_t, gain_scale=gain_scale)

        x_post_np = x_post.detach().cpu().numpy().reshape(-1)
        err = x_post_np - x_gt
        reward = float(-np.mean(err ** 2))

        innov_mag = float(np.linalg.norm(innov.detach().cpu().numpy()))
        self.innov_ema_mag = (
            (1 - self.ema_alpha) * self.innov_ema_mag
            + self.ema_alpha * innov_mag)

        obs = self._build_obs(innov, x_post)
        terminated = self.t >= self.T - 1
        truncated  = False
        info = {"err_mse": -reward}
        return obs, reward, terminated, truncated, info


# --------------------------------------------------------------------- #
def make_env(npz_path, weights_path, device="cpu"):
    data = np.load(npz_path)
    x = sanitize_data(data["x"]).astype(np.float32)
    y = sanitize_data(data["y"]).astype(np.float32)
    return MetaTunerEnv(x, y, weights_path, device=device)


def cmd_train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_fn = lambda: make_env(args.data, args.weights, device=args.device)
    venv = DummyVecEnv([env_fn for _ in range(args.n_envs)])

    policy_kwargs = dict(net_arch=dict(pi=[16, 16], vf=[32, 32]))
    model = PPO("MlpPolicy", venv,
                learning_rate=3e-4,
                n_steps=2048 // args.n_envs,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                policy_kwargs=policy_kwargs,
                verbose=1)

    print(f"Training PPO meta-tuner for {args.steps} env steps …")
    model.learn(total_timesteps=args.steps)
    model.save(args.out)
    print(f"Saved {args.out}")


def cmd_rollout(args):
    """Run one episode with the trained policy and print mean reward."""
    from stable_baselines3 import PPO
    env = make_env(args.data, args.weights, device=args.device)
    pol = PPO.load(args.policy)
    obs, _ = env.reset()
    rs = []
    while True:
        action, _ = pol.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        rs.append(r)
        if term or trunc:
            break
    print(f"Episode reward sum = {sum(rs):.4f}  "
          f"mean per-step = {np.mean(rs):.6f}  steps={len(rs)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--weights", required=True)
    t.add_argument("--data",    required=True)
    t.add_argument("--steps",   type=int, default=100_000)
    t.add_argument("--n-envs",  type=int, default=4)
    t.add_argument("--device",  default="cpu")
    t.add_argument("--out",     default="meta_tuner_ppo.zip")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("rollout")
    r.add_argument("--weights", required=True)
    r.add_argument("--data",    required=True)
    r.add_argument("--policy",  required=True)
    r.add_argument("--device",  default="cpu")
    r.set_defaults(func=cmd_rollout)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
