import numpy as np


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for constant-velocity state estimation.

    State:   x = [px, py, pz, vx, vy, vz]  (dim = 6)
    Measure: z = [px, py, pz]               (dim = 3)
    """

    def __init__(self, dt: float = 0.01,
             sigma_y: np.ndarray = None, H_matrix=None,vel_std=None):
        """
        Args:
            dt:     Time step (must match KalmanNet: 0.01 s)
            q_pos:  Process noise std for position states
            q_vel:  Process noise std for velocity states
            r_noise: Measurement noise std
        """
        self.dt = dt

        # --- State transition matrix F (6x6) ---
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # --- Measurement matrix H (3x6): observe position ---
        if H_matrix is not None:
            self.H = H_matrix.astype(np.float64)
        else:
            self.H = np.array([
                [0, 0, 0, 1, 0, 0],  # px
                [0, 1, 0, 0, 0, 0],  # py
                [0, 0, 1, 0, 0, 0],  # pz
            ], dtype=np.float64)

        # --- Process noise covariance Q (6x6) ---
        if vel_std is None:
            vel_std = np.array([0.1, 0.1, 0.1])

        q_pos = (0.5 * vel_std * dt) ** 2
        q_vel = vel_std ** 2

        self.Q = np.diag(np.concatenate([q_pos, q_vel])).astype(np.float64)

        # --- Measurement noise covariance R (3x3) ---
        if sigma_y is None:
            sigma_y = np.array([0.1, 0.1, 0.1])
        sigma_y = np.asarray(sigma_y, dtype=np.float64)
        self.R = np.diag(sigma_y ** 2).astype(np.float64)

        # --- State & covariance (initialised in reset) ---
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]).astype(np.float64)
    # ------------------------------------------------------------------
    def reset(self, x0: np.ndarray):
        """Initialise filter state from ground-truth first timestep."""
        self.x = np.array(x0, dtype=np.float64).flatten()
        assert self.x.shape == (6,), f"x0 must be length-6, got {self.x.shape}"
        self.P = np.eye(6, dtype=np.float64)

    # ------------------------------------------------------------------
    def predict(self):
        """
        Predict step:
            x_prior = F @ x_prev
            P_prior = F @ P @ F.T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    # ------------------------------------------------------------------
    def update(self, z: np.ndarray):
        """
        Update step (linear measurement model, so EKF = KF here):
            innovation  = z - H @ x_prior
            S           = H @ P @ H.T + R
            K           = P @ H.T @ inv(S)
            x_post      = x_prior + K @ innovation
            P_post      = (I - K @ H) @ P
        """
        z = np.array(z, dtype=np.float64).flatten()
        assert z.shape == (3,), f"Measurement must be length-3, got {z.shape}"

        H, R, P = self.H, self.R, self.P

        innovation = z - H @ self.x                     # (3,)
        S = H @ P @ H.T + R                             # (3,3)
        K = P @ H.T @ np.linalg.inv(S)                  # (6,3)

        self.x = self.x + K @ innovation                # (6,)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T       # Joseph form – numerically stable

    # ------------------------------------------------------------------
    def step(self, z: np.ndarray) -> np.ndarray:
        """Convenience: predict then update, return posterior state."""
        self.predict()
        self.update(z)
        return self.x.copy()