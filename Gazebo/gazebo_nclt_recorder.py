#!/usr/bin/env python3
"""
Gazebo recorder that produces data in the same shape as the NCLT
preprocess pipeline used by Model/train_model.py.

State (x): [px, py, pz, vx, vy, vz] in the world frame from /odom
Obs   (y): [acc_x, acc_y, acc_z]    in the body frame from /imu

It buffers raw odom + imu at their native rates, time-syncs them at
save time using nearest-neighbour matching against a uniform 100 Hz
grid, then chunks into SEQ_LEN=200 windows and writes
gazebo_train.npz / gazebo_val.npz / gazebo_test.npz with a 70/15/15
split (matching nclt_preprocess.py).

Usage:
  ros2 run kalmannet_ros2 gazebo_nclt_recorder
  # drive the robot for 2-3 minutes, then Ctrl+C
"""

import os
import sys
import math
import atexit
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

OUT_DIR  = os.path.expanduser("~/.ros")
SEQ_LEN  = 200
DT_TARGET = 0.01           # 100 Hz, same as NCLT
SPLIT = (0.70, 0.85)


class GazeboNCLTRecorder(Node):

    def __init__(self):
        super().__init__("gazebo_nclt_recorder")

        # Raw timestamped buffers
        self.odom_t   = []
        self.odom_x   = []   # [px, py, pz, vx, vy, vz] world frame
        self.imu_t    = []
        self.imu_acc  = []   # [ax, ay, az] body frame

        self.create_subscription(Odometry, "/odom", self._odom_cb, 50)
        self.create_subscription(Imu,      "/imu",  self._imu_cb,  100)
        self.create_timer(5.0, self._status)

        self.get_logger().info(
            "gazebo_nclt_recorder started. Drive the robot — Ctrl+C to save.\n"
            "  state (x) : [px, py, pz, vx, vy, vz]   from /odom\n"
            "  obs   (y) : [acc_x, acc_y, acc_z]      from /imu (body frame)"
        )

    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _odom_cb(self, msg: Odometry):
        # World-frame velocity by rotating body-frame twist using yaw.
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        vz_b = msg.twist.twist.linear.z
        vx = vx_b * math.cos(yaw) - vy_b * math.sin(yaw)
        vy = vx_b * math.sin(yaw) + vy_b * math.cos(yaw)
        vz = vz_b

        self.odom_t.append(self._stamp(msg))
        self.odom_x.append([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            vx, vy, vz,
        ])

    def _imu_cb(self, msg: Imu):
        a = msg.linear_acceleration
        self.imu_t.append(self._stamp(msg))
        self.imu_acc.append([a.x, a.y, a.z])

    def _status(self):
        self.get_logger().info(
            f"odom samples: {len(self.odom_t):6d}  |  "
            f"imu samples: {len(self.imu_t):6d}")

    # -------------------------------------------------------------- #
    def save(self):
        if getattr(self, "_saved", False):
            return
        self._saved = True

        if len(self.odom_t) < 200 or len(self.imu_t) < 200:
            print(f"[gazebo_nclt] Not enough samples "
                  f"(odom={len(self.odom_t)}, imu={len(self.imu_t)}). "
                  f"Drive longer.", flush=True)
            return

        odom_t = np.asarray(self.odom_t, dtype=np.float64)
        odom_x = np.asarray(self.odom_x, dtype=np.float64)
        imu_t  = np.asarray(self.imu_t,  dtype=np.float64)
        imu_a  = np.asarray(self.imu_acc, dtype=np.float64)

        # Common time window where both streams have data
        t0 = max(odom_t.min(), imu_t.min())
        t1 = min(odom_t.max(), imu_t.max())
        if t1 - t0 < SEQ_LEN * DT_TARGET:
            print(f"[gazebo_nclt] Overlap window too short "
                  f"({t1 - t0:.1f}s). Drive longer.", flush=True)
            return

        # Uniform 100 Hz grid
        grid = np.arange(t0, t1, DT_TARGET)

        # Nearest-neighbour resample for both streams
        def _nearest(t_src, x_src, t_grid):
            idx = np.searchsorted(t_src, t_grid)
            idx = np.clip(idx, 1, len(t_src) - 1)
            left = t_src[idx - 1]; right = t_src[idx]
            choose_left = (t_grid - left) < (right - t_grid)
            idx[choose_left] -= 1
            return x_src[idx]

        states = _nearest(odom_t, odom_x, grid)   # (N, 6)
        obs    = _nearest(imu_t,  imu_a,  grid)   # (N, 3)

        # Chunk to SEQ_LEN
        N = states.shape[0]
        n_seq = N // SEQ_LEN
        states = states[:n_seq * SEQ_LEN].reshape(n_seq, SEQ_LEN, 6)
        obs    = obs[:n_seq * SEQ_LEN].reshape(n_seq, SEQ_LEN, 3)

        # 70/15/15 split — matches nclt_preprocess.py
        s1 = int(SPLIT[0] * n_seq)
        s2 = int(SPLIT[1] * n_seq)

        os.makedirs(OUT_DIR, exist_ok=True)
        for name, sl in [("train", slice(0, s1)),
                         ("val",   slice(s1, s2)),
                         ("test",  slice(s2, n_seq))]:
            path = os.path.join(OUT_DIR, f"gazebo_{name}.npz")
            np.savez_compressed(path,
                                x=states[sl].astype(np.float32),
                                y=obs[sl].astype(np.float32))

        print(f"\n[gazebo_nclt] Saved {n_seq} sequences "
              f"(train={s1}, val={s2 - s1}, test={n_seq - s2}) "
              f"to {OUT_DIR}/gazebo_*.npz", flush=True)
        print(f"[gazebo_nclt]   x shape per file (train): "
              f"{states[:s1].shape}", flush=True)
        print(f"[gazebo_nclt]   y shape per file (train): "
              f"{obs[:s1].shape}", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboNCLTRecorder()
    atexit.register(node.save)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[gazebo_nclt] Ctrl+C — saving …", flush=True)
    finally:
        node.save()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
