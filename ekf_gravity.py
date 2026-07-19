"""
ekf_gravity.py
--------------
Gravity-aware Extended Kalman Filter for IMU-based state estimation.

Two improvements over the base EKF:

1. Gravity compensation in the measurement preprocessing.
   The NCLT IMU y-channel (index 2) contains specific force ≈ -9.81 m/s²
   due to gravity. This constant bias has nothing to do with vz and causes
   the base EKF's innovation to be permanently large, driving state divergence.
   We subtract the gravity vector from the raw measurement before update.

2. Constant-acceleration state model (9D option, commented out).
   The 6D constant-velocity model accumulates position error when the
   platform accelerates. A 9D model [px,py,pz,vx,vy,vz,ax,ay,az] would
   track acceleration as a state, but requires more tuning and a 9D H
   matrix — left as a comment for future work.

Usage:
    from ekf_gravity import GravityAwareEKF

    ekf = GravityAwareEKF(dt=0.01)
    ekf.reset(x0)
    for t in range(1, T):
        x_post = ekf.step(y[t])

The gravity vector is estimated automatically from the first N_gravity_est
measurements (default 50) by averaging the raw z-channel, then subtracted
from all subsequent measurements. This avoids hard-coding 9.81 and adapts
to any sensor orientation.
"""

import numpy as np


class GravityAwareEKF:
    """
    Gravity-aware KF for state x = [px, py, pz, vx, vy, vz].

    Key addition over base EKF:
      - Estimates gravity offset from first N_gravity_est measurements
      - Subtracts it from z before each update step
      - H matrix is data-fitted (injected from outside, same as base EKF)
    """

    def __init__(self,
                 dt:              float = 0.01,
                 q_pos:           float = 0.01,
                 q_vel:           float = 0.1,
                 r_noise:         float = 0.5,
                 gravity_channel: int   = 2,
                 N_gravity_est:   int   = 50):
        """
        Args:
            dt              : timestep (s)
            q_pos           : process noise std for position
            q_vel           : process noise std for velocity
            r_noise         : default measurement noise std (overridden by fitted R)
            gravity_channel : which measurement channel contains gravity bias (default 2 = z)
            N_gravity_est   : how many initial measurements to average for gravity estimate
        """
        self.dt              = dt
        self.gravity_channel = gravity_channel
        self.N_gravity_est   = N_gravity_est

        # State transition — constant velocity
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # H: velocity observer default (overridden by fitted H from training data)
        self.H = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        # Process noise Q
        q_diag = [q_pos**2] * 3 + [q_vel**2] * 3
        self.Q = np.diag(q_diag).astype(np.float64)

        # Measurement noise R
        self.R = np.eye(3, dtype=np.float64) * (r_noise**2)

        # State and covariance
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)

        # Gravity compensation state
        self._gravity_offset  = np.zeros(3, dtype=np.float64)
        self._gravity_buffer  = []
        self._gravity_estimated = False

    # ------------------------------------------------------------------
    def reset(self, x0: np.ndarray):
        """Reset state, covariance, and gravity estimator for a new sequence."""
        self.x = np.array(x0, dtype=np.float64).flatten()
        assert self.x.shape == (6,), f"x0 must be length-6, got {self.x.shape}"
        self.P = np.eye(6, dtype=np.float64)

        # Reset gravity estimator — each sequence re-estimates gravity
        self._gravity_offset    = np.zeros(3, dtype=np.float64)
        self._gravity_buffer    = []
        self._gravity_estimated = False

    # ------------------------------------------------------------------
    def _update_gravity_estimate(self, z: np.ndarray):
        """
        Accumulate first N_gravity_est measurements to estimate gravity bias.
        After estimation, _gravity_offset[gravity_channel] ≈ -9.81 m/s²
        (or whatever the sensor reads when stationary).
        """
        if not self._gravity_estimated:
            self._gravity_buffer.append(z.copy())
            if len(self._gravity_buffer) >= self.N_gravity_est:
                buf = np.array(self._gravity_buffer)   # (N, 3)
                # Gravity only affects the designated channel
                # Other channels: zero offset (they contain dynamic signals)
                self._gravity_offset                  = np.zeros(3, dtype=np.float64)
                self._gravity_offset[self.gravity_channel] = float(np.mean(
                    buf[:, self.gravity_channel]))
                self._gravity_estimated = True

    # ------------------------------------------------------------------
    def _compensate_gravity(self, z: np.ndarray) -> np.ndarray:
        """Subtract estimated gravity offset from raw measurement."""
        if self._gravity_estimated:
            return z - self._gravity_offset
        return z   # during estimation window, use raw measurement

    # ------------------------------------------------------------------
    def predict(self):
        """Standard KF predict step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    # ------------------------------------------------------------------
    def update(self, z_raw: np.ndarray):
        """
        Update step with gravity-compensated measurement.
        z_raw: raw measurement from sensor (3,)
        """
        z_raw = np.array(z_raw, dtype=np.float64).flatten()
        assert z_raw.shape == (3,), f"Measurement must be length-3, got {z_raw.shape}"

        # Update gravity estimate during warm-up window
        self._update_gravity_estimate(z_raw)

        # Compensate gravity before computing innovation
        z = self._compensate_gravity(z_raw)

        H, R, P = self.H, self.R, self.P

        innovation = z - H @ self.x
        S          = H @ P @ H.T + R
        K          = P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        I_KH   = np.eye(6) - K @ H
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T   # Joseph form

    # ------------------------------------------------------------------
    def step(self, z_raw: np.ndarray) -> np.ndarray:
        """Predict then update. Returns posterior state."""
        self.predict()
        self.update(z_raw)
        return self.x.copy()