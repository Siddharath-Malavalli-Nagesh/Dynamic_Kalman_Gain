#!/usr/bin/env python3
"""
Live evaluator for the Gazebo + KalmanNet pipeline.

Subscribes to:
  /odom                  — ground truth   [px, py, pz, vx, vy, vz]
  /kalmannet_nclt/odom   — KalmanNet est  (from gazebo_inference_node)

On Ctrl+C, time-aligns the two streams (nearest-neighbour) and prints
RMSE / MAE / % error / inlier precision per state, plus mean latency
(estimate timestamp − GT timestamp). Also writes online_eval.json.

Usage
-----
  ros2 run kalmannet_ros2 online_evaluator
  # in another terminal: drive the robot however you like
  # Ctrl+C when done

Tip: run ros2 launch kalmannet_ros2 gazebo_kalmannet.launch.py first,
then start this evaluator, then drive.
"""

import json
import math
import sys
import atexit
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

POS_FLOOR_M  = 0.5
VEL_FLOOR_MS = 0.05


class OnlineEvaluator(Node):

    def __init__(self):
        super().__init__("online_evaluator")
        self.gt_t,  self.gt_x  = [], []
        self.es_t,  self.es_x  = [], []

        self.create_subscription(Odometry, "/odom",
                                 self._gt_cb, 50)
        self.create_subscription(Odometry, "/kalmannet_nclt/odom",
                                 self._es_cb, 100)
        self.create_timer(5.0, self._status)

        self.get_logger().info(
            "online_evaluator up — listening to /odom and "
            "/kalmannet_nclt/odom. Ctrl+C to compute metrics.")

    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    @staticmethod
    def _state_world(msg: Odometry):
        # Rotate body-frame twist to world frame, mirroring the recorder.
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        vz_b = msg.twist.twist.linear.z
        vx = vx_b * math.cos(yaw) - vy_b * math.sin(yaw)
        vy = vx_b * math.sin(yaw) + vy_b * math.cos(yaw)
        return [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            vx, vy, vz_b,
        ]

    def _gt_cb(self, msg):
        self.gt_t.append(self._stamp(msg))
        self.gt_x.append(self._state_world(msg))

    def _es_cb(self, msg):
        # Estimator publishes already-world-frame velocity in twist.linear.
        self.es_t.append(self._stamp(msg))
        self.es_x.append([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ])

    def _status(self):
        self.get_logger().info(
            f"gt samples: {len(self.gt_t):6d}  |  est samples: {len(self.es_t):6d}")

    # --------------------------------------------------------------- #
    def evaluate(self):
        if getattr(self, "_done", False):
            return
        self._done = True

        if len(self.gt_t) < 50 or len(self.es_t) < 50:
            print(f"[online_evaluator] Not enough samples "
                  f"(gt={len(self.gt_t)}, est={len(self.es_t)}).",
                  flush=True)
            return

        gt_t = np.asarray(self.gt_t)
        gt_x = np.asarray(self.gt_x)
        es_t = np.asarray(self.es_t)
        es_x = np.asarray(self.es_x)

        # Common time window
        t0 = max(gt_t.min(), es_t.min())
        t1 = min(gt_t.max(), es_t.max())
        if t1 - t0 < 1.0:
            print("[online_evaluator] Overlap window <1s; aborting.", flush=True)
            return

        # For each estimate sample inside [t0, t1], find nearest GT.
        mask = (es_t >= t0) & (es_t <= t1)
        es_t_w = es_t[mask]
        es_x_w = es_x[mask]
        idx = np.searchsorted(gt_t, es_t_w)
        idx = np.clip(idx, 1, len(gt_t) - 1)
        left, right = gt_t[idx - 1], gt_t[idx]
        choose_left = (es_t_w - left) < (right - es_t_w)
        idx[choose_left] -= 1
        gt_aligned = gt_x[idx]
        gt_t_aligned = gt_t[idx]

        err = es_x_w - gt_aligned
        results = self._metrics(err, gt_aligned)
        results["latency_ms_mean"] = float((es_t_w - gt_t_aligned).mean() * 1000)
        results["n_aligned"] = int(len(es_t_w))
        results["window_seconds"] = float(t1 - t0)

        self._print(results)
        with open("online_eval.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved online_eval.json", flush=True)

    @staticmethod
    def _metrics(err, gt):
        names = ["px", "py", "pz", "vx", "vy", "vz"]
        per_rmse = np.sqrt(np.mean(err ** 2, axis=0))
        per_mae  = np.mean(np.abs(err), axis=0)
        pct = []
        for i, n in enumerate(names):
            floor = VEL_FLOOR_MS if n.startswith("v") else POS_FLOOR_M
            valid = np.abs(gt[:, i]) >= floor
            if valid.any():
                pct.append(float((np.abs(err[valid, i])
                                  / np.abs(gt[valid, i])).mean() * 100))
            else:
                pct.append(float("nan"))
        pos_err = err[:, :3]
        dists = np.sqrt(np.sum(pos_err ** 2, axis=1))
        return {
            "rmse_pos_m":     float(np.sqrt(np.mean(pos_err ** 2))),
            "mae_pos_m":      float(np.mean(np.abs(pos_err))),
            "precision_1m_pct": float((dists < 1.0).mean() * 100),
            "per_state_rmse": dict(zip(names, [float(v) for v in per_rmse])),
            "per_state_mae":  dict(zip(names, [float(v) for v in per_mae])),
            "per_state_pct_err": dict(zip(names, pct)),
        }

    @staticmethod
    def _print(r):
        print("\n=== Online evaluation ===", flush=True)
        print(f"  aligned samples : {r['n_aligned']}")
        print(f"  window          : {r['window_seconds']:.1f}s")
        print(f"  RMSE pos (m)    : {r['rmse_pos_m']:.4f}")
        print(f"  MAE  pos (m)    : {r['mae_pos_m']:.4f}")
        print(f"  Precision <1m   : {r['precision_1m_pct']:.2f} %")
        print(f"  est-to-gt lag   : {r['latency_ms_mean']:.2f} ms")
        print("\n  Per-state:")
        print(f"  {'state':<6}{'RMSE':>12}{'MAE':>12}{'%err':>10}")
        for n in ["px", "py", "pz", "vx", "vy", "vz"]:
            print(f"  {n:<6}{r['per_state_rmse'][n]:>12.4f}"
                  f"{r['per_state_mae'][n]:>12.4f}"
                  f"{r['per_state_pct_err'][n]:>10.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = OnlineEvaluator()
    atexit.register(node.evaluate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[online_evaluator] Ctrl+C — computing metrics …", flush=True)
    finally:
        node.evaluate()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
