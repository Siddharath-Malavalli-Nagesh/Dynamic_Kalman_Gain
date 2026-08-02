"""
ekf.py
------
Extended Kalman Filter for constant-velocity state estimation on NCLT.

State:   x = [px, py, pz, vx, vy, vz]  (dim = 6)
Measure: z in R^3                        (dim = 3)

Based on the linear Wiener velocity model used in:
  Revach et al., "KalmanNet: Neural Network Aided Kalman Filtering
  for Partially Known Dynamics", IEEE TSP 2022.  (arXiv:2107.10043)

In the paper's NCLT experiment:
  - State per axis: x = (p, v)
  - Observations: noisy velocity readings, H = (0, 1) per axis
  - Q: kinematic noise covariance (dt^3/3, dt^2/2, dt) * accel_var
  - H and R are overridden externally from training-set least-squares fit
    in evaluate_models.py, so defaults here are just fallbacks.

Key design decisions vs previous broken versions:
  1. No sigma_y overwrite bug — parameter is used correctly.
  2. No H_bias term — adds complexity with no benefit when H is fitted.
  3. P reset uses matched uncertainty: high for velocity (unknown from x0),
     moderate for position (known from GT x0).
  4. Q uses the proper kinematic coupling from the Wiener model (eq. 23
     in the paper) rather than a diagonal approximation.
  5. accel_std is the key tuning parameter — set from data if possible.
"""

import numpy as np


class ExtendedKalmanFilter:
    """
    Constant-velocity KF/EKF for 6D state estimation.

    Since both F and H are linear, this is a standard KF.
    The EKF name is kept for interface consistency with ekf_gravity.py.

    H and R are typically overridden after construction:
        ekf.H = H_fit       # (3, 6) from training-set lstsq
        ekf.R = diag(s**2)  # (3, 3) from training-set residuals
    """

    def __init__(self,
                 dt:        float          = 0.01,
                 sigma_y:   np.ndarray     = None,
                 H_matrix:  np.ndarray     = None,
                 accel_std: float          = 0.5):
        """
        Args:
            dt        : timestep in seconds (must match KalmanNet: 0.01 s)
            sigma_y   : measurement noise std per channel (3,); sets R
            H_matrix  : observation matrix (3, 6); overrides default
            accel_std : process noise — std of acceleration (m/s²).
                        Default 0.5 m/s² is reasonable for slow vehicle/robot.
                        Increase for faster or more erratic motion.
        """
        self.dt = dt

        # ── State transition F (6×6) — constant velocity ─────────────
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # ── Measurement matrix H (3×6) ────────────────────────────────
        # Default: observe velocities (vx, vy, vz) — matches NCLT odometry.
        # Overridden externally by fitted H from training data.
        if H_matrix is not None:
            self.H = H_matrix.astype(np.float64)
        else:
            self.H = np.array([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ], dtype=np.float64)

        # ── Process noise Q (6×6) — Wiener kinematic model ───────────
        # From paper eq. (23): per-axis 2×2 block is:
        #   [[dt³/3, dt²/2],
        #    [dt²/2, dt   ]] * accel_var
        # Assembled into 6×6 for x, y, z axes independently.
        a2   = accel_std ** 2
        q_pp = (dt ** 3) / 3.0 * a2   # pos-pos
        q_pv = (dt ** 2) / 2.0 * a2   # pos-vel (symmetric)
        q_vv = dt * a2                  # vel-vel

        self.Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            self.Q[i,     i]     = q_pp
            self.Q[i,     i + 3] = q_pv
            self.Q[i + 3, i]     = q_pv
            self.Q[i + 3, i + 3] = q_vv

        # ── Measurement noise R (3×3) ─────────────────────────────────
        # Overridden externally by residual-based sigma_r from training data.
        if sigma_y is None:
            sigma_y = np.array([0.1, 0.1, 0.1])
        sigma_y = np.asarray(sigma_y, dtype=np.float64)
        self.R  = np.diag(sigma_y ** 2).astype(np.float64)

        # ── State and covariance ──────────────────────────────────────
        self.x = np.zeros(6, dtype=np.float64)
        # Initial P: moderate position uncertainty, larger velocity uncertainty
        # (GT x0 gives good position; velocity from GT is also good but less
        # trusted since first-step derivatives are noisy).
        self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]).astype(np.float64)

    # ------------------------------------------------------------------
    def reset(self, x0: np.ndarray):
        """Reset state and covariance for a new sequence."""
        self.x = np.array(x0, dtype=np.float64).flatten()
        assert self.x.shape == (6,), f"x0 must be length-6, got {self.x.shape}"
        # Reset P to same initial uncertainty as constructor
        self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]).astype(np.float64)

    # ------------------------------------------------------------------
    def predict(self):
        """x_prior = F @ x;   P_prior = F @ P @ F.T + Q"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    # ------------------------------------------------------------------
    def update(self, z: np.ndarray):
        """
        Standard KF update (linear H → EKF = KF here).
        Uses Joseph form for numerical stability of P.
        """
        z = np.array(z, dtype=np.float64).flatten()
        assert z.shape == (3,), f"Measurement must be length-3, got {z.shape}"

        H, R, P = self.H, self.R, self.P

        innov = z - H @ self.x                       # (3,)
        S     = H @ P @ H.T + R                      # (3,3)
        K     = P @ H.T @ np.linalg.inv(S)           # (6,3)

        self.x = self.x + K @ innov
        I_KH   = np.eye(6) - K @ H
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T    # Joseph form

    # ------------------------------------------------------------------
    def step(self, z: np.ndarray) -> np.ndarray:
        """Predict then update. Returns posterior state (6,)."""
        self.predict()
        self.update(z)
        return self.x.copy()