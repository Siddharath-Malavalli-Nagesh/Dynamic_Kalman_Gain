#!/usr/bin/env python3
"""
Variable-gravity environment driver for Gazebo.

Periodically changes the world gravity vector via the
/gazebo/set_physics_properties service. Run this alongside the
recorder + auto_drive to collect training/eval data where the
IMU's acc_z baseline shifts mid-run.

Usage:
  ros2 run kalmannet_ros2 gravity_modulator
  # or directly:
  python3 gravity_modulator.py
  # optional: change interval and range
  python3 gravity_modulator.py --interval 8 --min-g -11 --max-g -7

Notes
-----
- This uses Gazebo Classic's /gazebo/set_physics_properties service.
  For Gazebo Sim/Ignition the service name and message type differ
  (use `gz physics ...` CLI instead).
- The IMU plugin in TurtleBot3 reports linear_acceleration that
  includes the gravity contribution from the physics engine, so
  changing gravity here directly shifts the recorded acc_z signal.
"""

import argparse
import random
import sys
import rclpy
from rclpy.node import Node

try:
    from gazebo_msgs.srv import SetPhysicsProperties
    from gazebo_msgs.msg  import ODEPhysics
    from geometry_msgs.msg import Vector3
    _HAS_GZ = True
except ImportError:
    _HAS_GZ = False


class GravityModulator(Node):

    def __init__(self, args):
        super().__init__("gravity_modulator")
        if not _HAS_GZ:
            raise ImportError(
                "gazebo_msgs not installed. apt install ros-${ROS_DISTRO}-gazebo-msgs")

        self.cli = self.create_client(
            SetPhysicsProperties, "/gazebo/set_physics_properties")
        self.get_logger().info("Waiting for /gazebo/set_physics_properties …")
        if not self.cli.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("Service not available — is Gazebo running?")

        self.interval = args.interval
        self.min_g    = args.min_g
        self.max_g    = args.max_g

        # Create timer that fires every interval seconds
        self.create_timer(self.interval, self._tick)
        self._tick()  # set immediately on start

        self.get_logger().info(
            f"gravity_modulator running. Will sample gz "
            f"~ Uniform([{self.min_g}, {self.max_g}]) every "
            f"{self.interval}s.")

    def _tick(self):
        gz = random.uniform(self.min_g, self.max_g)

        req = SetPhysicsProperties.Request()
        req.time_step = 0.001
        req.max_update_rate = 1000.0
        req.gravity = Vector3(x=0.0, y=0.0, z=gz)

        ode = ODEPhysics()
        ode.auto_disable_bodies = False
        ode.sor_pgs_precon_iters = 0
        ode.sor_pgs_iters = 50
        ode.sor_pgs_w = 1.3
        ode.sor_pgs_rms_error_tol = 0.0
        ode.contact_surface_layer = 0.001
        ode.contact_max_correcting_vel = 100.0
        ode.cfm = 0.0
        ode.erp = 0.2
        ode.max_contacts = 20
        req.ode_config = ode

        future = self.cli.call_async(req)
        future.add_done_callback(
            lambda f, gz=gz: self._on_set(f, gz))

    def _on_set(self, future, gz):
        res = future.result()
        if res is not None and res.success:
            self.get_logger().info(f"  gz set to {gz:+.3f} m/s²")
        else:
            self.get_logger().warn(
                f"  set_physics_properties failed (gz={gz:.3f}): {res}")


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=10.0,
                    help="Seconds between gravity changes.")
    ap.add_argument("--min-g", type=float, default=-11.0,
                    help="Most-negative gravity (heavier).")
    ap.add_argument("--max-g", type=float, default=-7.5,
                    help="Least-negative gravity (lighter).")
    cli_args, _ = ap.parse_known_args()

    rclpy.init(args=args)
    node = GravityModulator(cli_args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Restore Earth gravity on exit
        try:
            node.min_g = node.max_g = -9.81
            node._tick()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
