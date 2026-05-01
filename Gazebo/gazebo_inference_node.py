#!/usr/bin/env python3
"""
ROS2 inference node for the NCLT-style KalmanNet running in Gazebo.

State (m=6) : [px, py, pz, vx, vy, vz]   world frame
Obs   (n=3) : [acc_x, acc_y, acc_z]      body frame, from /imu

Publishes nav_msgs/Odometry on /kalmannet_nclt/odom at IMU rate.

Usage:
  ros2 run kalmannet_ros2 gazebo_inference_node \
       --ros-args -p weights:=/path/to/best_knet_gazebo.pt
"""

import os
import math
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

# Import the architecture from the Model/ folder. Adjust path on the
# ROS2 box: copy Model/train_model.py beside this file, or extend
# PYTHONPATH so that `from train_model import KalmanNet` resolves.
try:
    from train_model import KalmanNet
except ImportError as e:
    raise ImportError(
        "Cannot import KalmanNet from train_model. Either copy "
        "Model/train_model.py next to this node, or add the Model/ "
        "directory to PYTHONPATH before running."
    ) from e


class GazeboInferenceNode(Node):

    def __init__(self):
        super().__init__("gazebo_inference_node")

        self.declare_parameter("weights", "best_knet_gazebo.pt")
        weights_path = self.get_parameter("weights").value
        if not os.path.exists(weights_path):
            self.get_logger().error(f"weights not found: {weights_path}")
            raise FileNotFoundError(weights_path)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = KalmanNet().to(self.device)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device))
        self.model.eval()

        # Persistent recurrent / posterior state across IMU callbacks.
        # Mirrors the loop body inside KalmanNet.forward but unrolled
        # one step at a time so we can run online.
        m, n, N = self.model.m, self.model.n, self.model.N_rnn
        self.h_Q     = torch.zeros(1, N, device=self.device)
        self.h_Sigma = torch.zeros(1, N, device=self.device)
        self.h_S     = torch.zeros(1, N, device=self.device)
        self.x_post   = None        # most recent posterior, shape (1, m)
        self.x_post_1 = None        # one step before
        self.y_prev   = None        # previous obs, shape (1, n)

        self.pub = self.create_publisher(Odometry, "/kalmannet_nclt/odom", 10)
        self.create_subscription(Imu, "/imu", self._imu_cb, 100)

        self.get_logger().info(
            f"gazebo_inference_node up. weights={weights_path}, "
            f"device={self.device}")

    @torch.no_grad()
    def _imu_cb(self, msg: Imu):
        a = msg.linear_acceleration
        y = torch.tensor([[a.x, a.y, a.z]],
                         dtype=torch.float32, device=self.device)

        # Cold start — wait for second sample so delta_y is defined,
        # and use raw zero state as prior.
        if self.y_prev is None:
            self.y_prev   = y
            self.x_post   = torch.zeros(1, self.model.m, device=self.device)
            self.x_post_1 = torch.zeros_like(self.x_post)
            return

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

        x_prior = torch.matmul(self.model.F,
                               self.x_post.unsqueeze(2)).squeeze(2)
        y_pred  = self.model.H(x_prior)
        innov   = y - y_pred
        x_new   = x_prior + torch.bmm(
            K, innov.unsqueeze(2)).squeeze(2)

        # Roll buffers
        self.x_post_1 = self.x_post
        self.x_post   = x_new
        self.y_prev   = y

        self._publish(msg.header.stamp, x_new[0].cpu().numpy())

    def _publish(self, stamp, x):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.position.x = float(x[0])
        odom.pose.pose.position.y = float(x[1])
        odom.pose.pose.position.z = float(x[2])
        odom.twist.twist.linear.x = float(x[3])
        odom.twist.twist.linear.y = float(x[4])
        odom.twist.twist.linear.z = float(x[5])
        self.pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
