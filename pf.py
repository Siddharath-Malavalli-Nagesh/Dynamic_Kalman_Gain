"""
pf.py
-----
Bootstrap Particle Filter for constant-velocity state estimation.

State:   x = [px, py, pz, vx, vy, vz]  (dim = 6)
Measure: z in R^3                        (dim = 3)

Algorithm:
  1. Initialise N particles around x0
  2. Each step:
     a. Propagate: add process noise to each particle via F
     b. Weight:    Gaussian likelihood of z under each particle
     c. Resample:  systematic resampling when ESS drops below threshold
     d. Estimate:  weighted mean of particles

N_particles trades accuracy vs latency — 200 gives reasonable accuracy
while keeping per-step cost comparable to UKF.
"""

import numpy as np


class ParticleFilter:

    def __init__(self,
                 dt:          float      = 0.01,
                 accel_std:   float      = 0.1,
                 sigma_y:     np.ndarray = None,
                 H_matrix:    np.ndarray = None,
                 N_particles: int        = 200,
                 resample_threshold: float = 0.5):
        """
        Args:
            dt                 : timestep (s)
            accel_std          : process noise std for velocity (m/s per step)
            sigma_y            : measurement noise std per channel (3,)
            H_matrix           : observation matrix (3, 6)
            N_particles        : number of particles (200 balances speed/accuracy)
            resample_threshold : resample when ESS < threshold * N_particles
        """
        self.dt          = dt
        self.N           = N_particles
        self.resamp_thr  = resample_threshold
        self.n_state     = 6
        self.n_meas      = 3

        # ── State transition F ─────────────────────────────────────────
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt; self.F[1, 4] = dt; self.F[2, 5] = dt

        # ── Process noise: velocity perturbation std ──────────────────
        # Wiener model: position noise propagates from velocity noise
        # pos_noise_std = 0.5 * accel_std * dt^2  (second-order effect)
        # vel_noise_std = accel_std * dt
        self.q_pos = 0.5 * accel_std * dt**2
        self.q_vel = accel_std * dt

        # ── Measurement matrix H ──────────────────────────────────────
        if H_matrix is not None:
            self.H = H_matrix.astype(np.float64)
        else:
            self.H = np.array([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ], dtype=np.float64)

        # ── Measurement noise R ───────────────────────────────────────
        if sigma_y is None:
            sigma_y = np.array([0.1, 0.1, 0.1])
        self.sigma_y = np.asarray(sigma_y, dtype=np.float64)   # (3,)
        self.R_diag  = self.sigma_y ** 2                        # (3,) for fast likelihood

        # ── Particles and weights (initialised in reset) ──────────────
        self.particles = np.zeros((N_particles, 6), dtype=np.float64)
        self.weights   = np.ones(N_particles, dtype=np.float64) / N_particles

    # ------------------------------------------------------------------
    def reset(self, x0: np.ndarray):
        """Initialise particles around x0 with small perturbation."""
        x0 = np.array(x0, dtype=np.float64).flatten()
        assert x0.shape == (6,)

        # Spread particles: tight around known position, looser for velocity
        spread = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        self.particles = x0 + np.random.randn(self.N, 6) * spread
        self.weights   = np.ones(self.N) / self.N

    # ------------------------------------------------------------------
    def _propagate(self):
        """Apply F + process noise to all particles."""
        # F @ particles.T — linear propagation
        self.particles = (self.F @ self.particles.T).T   # (N, 6)

        # Add independent process noise per particle
        noise = np.random.randn(self.N, 6)
        noise[:, :3] *= self.q_pos   # position noise
        noise[:, 3:] *= self.q_vel   # velocity noise
        self.particles += noise

    # ------------------------------------------------------------------
    def _log_likelihood(self, z: np.ndarray) -> np.ndarray:
        """
        Compute log p(z | particle_i) for all particles.
        Gaussian likelihood: log p = -0.5 * sum((z - H@x)^2 / sigma_y^2)
        Returns (N,) log weights.
        """
        z_pred = (self.H @ self.particles.T).T   # (N, 3)
        diff   = z - z_pred                       # (N, 3)
        # Sum of squared standardised residuals
        log_w  = -0.5 * np.sum(diff**2 / self.R_diag, axis=1)
        return log_w

    # ------------------------------------------------------------------
    def _systematic_resample(self):
        """
        Systematic resampling — O(N), lower variance than multinomial.
        Returns resampled particle indices.
        """
        N       = self.N
        cumsum  = np.cumsum(self.weights)
        cumsum[-1] = 1.0   # numerical safety
        step    = 1.0 / N
        start   = np.random.uniform(0, step)
        points  = start + step * np.arange(N)
        indices = np.searchsorted(cumsum, points)
        return indices

    # ------------------------------------------------------------------
    def _update_weights(self, z: np.ndarray):
        """Update weights using measurement likelihood, then normalise."""
        log_w           = self._log_likelihood(z)
        log_w          -= np.max(log_w)           # numerical stability
        self.weights    = np.exp(log_w)
        self.weights   /= self.weights.sum()

    # ------------------------------------------------------------------
    def _maybe_resample(self):
        """Resample if effective sample size (ESS) is too low."""
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.resamp_thr * self.N:
            indices        = self._systematic_resample()
            self.particles = self.particles[indices]
            self.weights   = np.ones(self.N) / self.N

    # ------------------------------------------------------------------
    def estimate(self) -> np.ndarray:
        """Return weighted mean of particles as state estimate."""
        return np.average(self.particles, weights=self.weights, axis=0)

    # ------------------------------------------------------------------
    def step(self, z: np.ndarray) -> np.ndarray:
        """Full filter step: propagate → weight → resample → estimate."""
        z = np.asarray(z, dtype=np.float64).flatten()
        assert z.shape == (3,)

        self._propagate()
        self._update_weights(z)
        self._maybe_resample()
        return self.estimate()