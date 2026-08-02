"""
ukf.py
------
Unscented Kalman Filter for constant-velocity state estimation.

State:   x = [px, py, pz, vx, vy, vz]  (dim = 6)
Measure: z in R^3                        (dim = 3)

The UKF approximates the posterior distribution by propagating a set of
deterministically chosen sigma points through the nonlinear dynamics.
For linear F and H (as here) it is equivalent to the KF, but the sigma
point framework generalises cleanly to nonlinear extensions.

H and R are overridden externally from training-set fitting in
evaluate_models.py, same pattern as ekf.py.
"""

import numpy as np


class UnscentedKalmanFilter:

    def __init__(self,
                 dt:        float      = 0.01,
                 accel_std: float      = 0.1,
                 sigma_y:   np.ndarray = None,
                 H_matrix:  np.ndarray = None,
                 alpha:     float      = 1e-3,
                 beta:      float      = 2.0,
                 kappa:     float      = 0.0,
                 augment_bias: bool = False):
        """
        Args:
            dt        : timestep (s)
            accel_std : process noise — std of velocity change per step (m/s)
            sigma_y   : measurement noise std per channel (3,)
            H_matrix  : observation matrix (3, 6); overrides default
            alpha     : sigma point spread (1e-4 to 1 — smaller = tighter spread)
            beta      : prior knowledge of distribution (2 = Gaussian)
            kappa     : secondary scaling (0 recommended for state estimation)
        """
        self.augment_bias = augment_bias
        
        self.dt  = dt
        self.n   = 6   # state dim
        self.m   = 3   # measurement dim

        # ── UKF scaling parameters ────────────────────────────────────
        self.alpha = alpha
        self.beta  = beta
        self.kappa = kappa
        lam        = alpha**2 * (self.n + kappa) - self.n
        self.lam   = lam

        # Weights for mean and covariance
        n = self.n
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wc = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

        # ── State transition F ─────────────────────────────────────────
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt; self.F[1, 4] = dt; self.F[2, 5] = dt

        # ── Measurement matrix H ──────────────────────────────────────
        if H_matrix is not None:
            self.H = H_matrix.astype(np.float64)
        else:
            self.H = np.array([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ], dtype=np.float64)

        # ── Process noise Q — Wiener kinematic model ──────────────────
        a2   = accel_std ** 2
        q_pp = (dt ** 3) / 3.0 * a2
        q_pv = (dt ** 2) / 2.0 * a2
        q_vv = dt * a2
        self.Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            self.Q[i,     i]     = q_pp
            self.Q[i,     i + 3] = q_pv
            self.Q[i + 3, i]     = q_pv
            self.Q[i + 3, i + 3] = q_vv

        # ── Measurement noise R ───────────────────────────────────────
        if sigma_y is None:
            sigma_y = np.array([0.1, 0.1, 0.1])
        sigma_y = np.asarray(sigma_y, dtype=np.float64)
        self.R  = np.diag(sigma_y ** 2)

        # ── State and covariance ──────────────────────────────────────
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]).astype(np.float64)

        if augment_bias:
                    self.n = 9   # augment state with [bx, by, bz]
                    # Recalculate sigma point weights for n=9
                    n   = self.n
                    lam = alpha**2 * (n + kappa) - n
                    self.lam = lam
                    self.Wm  = np.full(2*n+1, 1/(2*(n+lam)))
                    self.Wc  = np.full(2*n+1, 1/(2*(n+lam)))
                    self.Wm[0] = lam/(n+lam)
                    self.Wc[0] = lam/(n+lam) + (1 - alpha**2 + beta)
        
                    # Extended F: bias states are random walk (identity block)
                    F_aug = np.eye(9)
                    F_aug[:6, :6] = self.F
                    self.F = F_aug
        
                    # Extended H: z = H_vel @ x[:6] + x[6:9]
                    H_aug = np.zeros((3, 9))
                    H_aug[:, :6] = self.H
                    H_aug[:, 6:] = np.eye(3)
                    self.H = H_aug
        
                    # Extended Q: small bias drift
                    Q_aug = np.zeros((9, 9))
                    Q_aug[:6, :6] = self.Q
                    Q_aug[6:, 6:] = np.eye(3) * (0.001 ** 2)   # bias diffuses slowly
                    self.Q = Q_aug
        
                    # Extended P
                    self.P = np.diag([1,1,1,10,10,10,0.01,0.01,0.01]).astype(np.float64)

    # ------------------------------------------------------------------
    def _sigma_points(self):
        """Generate 2n+1 sigma points from current x and P."""
        n   = self.n
        lam = self.lam
        try:
            S = np.linalg.cholesky((n + lam) * self.P)
        except np.linalg.LinAlgError:
            # P not positive definite — regularise
            self.P = (self.P + self.P.T) / 2 + np.eye(n) * 1e-6
            S = np.linalg.cholesky((n + lam) * self.P)

        sigma = np.zeros((2 * n + 1, n))
        sigma[0] = self.x
        for i in range(n):
            sigma[i + 1]     = self.x + S[:, i]
            sigma[i + 1 + n] = self.x - S[:, i]
        return sigma

    # ------------------------------------------------------------------
    def reset(self, x0: np.ndarray):
        x0 = np.array(x0, dtype=np.float64).flatten()
        assert x0.shape == (6,), f"x0 must be length-6, got {x0.shape}"

        if self.augment_bias:
            self.x    = np.zeros(9, dtype=np.float64)
            self.x[:6] = x0
            # bias initialised to zero — filter learns it online
            self.P    = np.diag([1.0, 1.0, 1.0,
                                10.0, 10.0, 10.0,
                                0.01, 0.01, 0.01]).astype(np.float64)
        else:
            self.x = x0.copy()
            self.P = np.diag([1.0, 1.0, 1.0,
                            10.0, 10.0, 10.0]).astype(np.float64)

    # ------------------------------------------------------------------
    def predict(self):
        """Propagate sigma points through F, compute predicted mean and cov."""
        sigma = self._sigma_points()                     # (2n+1, 6)
        sigma_pred = (self.F @ sigma.T).T                # (2n+1, 6)  linear F

        # Predicted mean
        x_pred = np.einsum("i,ij->j", self.Wm, sigma_pred)

        # Predicted covariance
        diff   = sigma_pred - x_pred                     # (2n+1, 6)
        P_pred = np.einsum("i,ij,ik->jk", self.Wc, diff, diff) + self.Q

        self.x = x_pred
        self.P = P_pred
        self._sigma_pred = sigma_pred   # cache for update step

    # ------------------------------------------------------------------
    def update(self, z: np.ndarray):
        """Update using measurement z (3,)."""
        z = np.asarray(z, dtype=np.float64).flatten()
        assert z.shape == (3,)

        sigma_pred = self._sigma_pred                    # (2n+1, 6)
        H          = self.H

        # Propagate sigma points through H (linear → direct matmul)
        z_sigma = (H @ sigma_pred.T).T                  # (2n+1, 3)

        # Predicted measurement mean
        z_pred = np.einsum("i,ij->j", self.Wm, z_sigma) # (3,)

        # Innovation covariance S and cross-covariance P_xz
        dz   = z_sigma - z_pred                          # (2n+1, 3)
        dx   = sigma_pred - self.x                       # (2n+1, 6)

        S    = np.einsum("i,ij,ik->jk", self.Wc, dz, dz) + self.R   # (3,3)
        Pxz  = np.einsum("i,ij,ik->jk", self.Wc, dx, dz)            # (6,3)

        K = Pxz @ np.linalg.inv(S)                      # (6,3)

        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ S @ K.T

        # Symmetrise P to prevent drift
        self.P = (self.P + self.P.T) / 2

    # ------------------------------------------------------------------
    def step(self, z: np.ndarray) -> np.ndarray:
        self.predict()
        self.update(z)
        return self.x.copy()