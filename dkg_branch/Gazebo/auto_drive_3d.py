#!/usr/bin/env python3
"""
3D auto-driver for a Gazebo quadrotor. Publishes 6-DOF /cmd_vel
(linear.x, linear.y, linear.z, angular.z) through an NCLT-style
schedule extended with gentle altitude variation.

Why 3D?
-------
KalmanNet was pretrained on NCLT, a ground-vehicle dataset whose z
coordinate varies smoothly with terrain. Fine-tuning on a strictly
planar TurtleBot (z ≡ 0) creates a train/deploy distribution mismatch
on the z axis — the model expects z to be informative and drifts.
A quadrotor with mild altitude variation closes that gap.

NCLT-realism notes
------------------
- z varies *slowly and smoothly* (like a vehicle going up a hill),
  not aggressive hover oscillations.
- |vz| <= 0.10 m/s — NCLT terrain-driven z is much slower than xy.
- Altitude stays within [0.5, 2.5] m so the drone never grounds out
  or drifts out of the world.
- xy motion is the same NCLT-style schedule (long straights, gentle
  arcs, rare sharp turns) as auto_drive.py.

Run alongside gazebo_nclt_recorder.py with a quadrotor model
(e.g. sjtu_drone, hector_quadrotor) that subscribes to /cmd_vel:

  Terminal A:  ros2 launch <your_drone_pkg> spawn.launch.py
  Terminal B:  python3 gazebo_nclt_recorder.py
  Terminal C:  python3 auto_drive_3d.py
"""

import math
import random
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

V_NOM = 0.15           # nominal cruising speed (m/s, xy)
V_MAX = 0.20
W_GENTLE = 0.40
W_SHARP  = 0.80

VZ_GENTLE = 0.05       # gentle climb/descent (m/s)
VZ_MAX    = 0.10       # cap on |vz|
Z_MIN, Z_MAX = 0.5, 2.5
Z_TARGET_NOM = 1.5     # cruise altitude


def _jitter(x, frac=0.2):
    return x * random.uniform(1 - frac, 1 + frac)


# Primitives return (vx, vy, vz, wz). vy stays 0 — we steer with yaw
# like a car. vz is the new degree of freedom.

def p_straight_long(t):   return (V_NOM, 0.0, 0.0,            0.0)
def p_straight_slow(t):   return (0.08,  0.0, 0.0,            0.0)
def p_cruise_burst(t):
    if t < 1.5:   return (V_NOM * (t / 1.5), 0.0, 0.0, 0.0)
    if t < 4.5:   return (V_NOM,             0.0, 0.0, 0.0)
    if t < 6.0:   return (V_NOM * max(0.0, 1.0 - (t - 4.5) / 1.5),
                          0.0, 0.0, 0.0)
    return (0.0, 0.0, 0.0, 0.0)

def p_gentle_arc_l(t):    return (V_NOM, 0.0, 0.0,  W_GENTLE * 0.5)
def p_gentle_arc_r(t):    return (V_NOM, 0.0, 0.0, -W_GENTLE * 0.5)
def p_curve_l(t):         return (V_NOM, 0.0, 0.0,  W_GENTLE)
def p_curve_r(t):         return (V_NOM, 0.0, 0.0, -W_GENTLE)
def p_tight_l(t):         return (0.08,  0.0, 0.0,  W_SHARP)
def p_tight_r(t):         return (0.08,  0.0, 0.0, -W_SHARP)
def p_serpentine(t):
    return (V_NOM, 0.0, 0.0, 0.4 * math.sin(2 * math.pi * t / 8.0))
def p_figure_8(t):
    w = W_GENTLE if int(t // 6.0) % 2 == 0 else -W_GENTLE
    return (0.12, 0.0, 0.0, w)
def p_pause(t):           return (0.0,   0.0, 0.0,  0.0)

# z-only primitives — smooth ramps that mimic terrain elevation
def p_climb(t):           return (V_NOM, 0.0,  VZ_GENTLE, 0.0)
def p_descend(t):         return (V_NOM, 0.0, -VZ_GENTLE, 0.0)
def p_climb_slow(t):      return (0.08,  0.0,  VZ_GENTLE * 0.6, 0.0)
def p_descend_slow(t):    return (0.08,  0.0, -VZ_GENTLE * 0.6, 0.0)
def p_hill(t):
    # Smooth up-and-down over the primitive duration (sin half-wave).
    # Matches a vehicle cresting a gentle hill.
    return (V_NOM, 0.0, VZ_GENTLE * math.sin(2 * math.pi * t / 16.0), 0.0)


SCHEDULE = [
    (p_straight_long,  20.0),
    (p_pause,           2.0),
    (p_climb,          10.0),         # gentle climb while cruising
    (p_gentle_arc_l,   12.0),
    (p_straight_long,  10.0),
    (p_descend,        10.0),         # gentle descent
    (p_gentle_arc_r,   12.0),
    (p_pause,           2.0),
    (p_cruise_burst,    8.0),
    (p_hill,           16.0),         # rolling-terrain analogue
    (p_curve_l,        14.0),
    (p_pause,           2.0),
    (p_curve_r,        14.0),
    (p_climb_slow,      8.0),
    (p_serpentine,     20.0),
    (p_descend_slow,    8.0),
    (p_pause,           2.0),
    (p_tight_l,         6.0),
    (p_straight_long,  15.0),
    (p_tight_r,         6.0),
    (p_pause,           2.0),
    (p_figure_8,       24.0),
    (p_hill,           16.0),
    (p_straight_long,  20.0),
    (p_pause,           3.0),
]


class AutoDriver3D(Node):

    def __init__(self):
        super().__init__("auto_drive_3d")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # Subscribe to ground truth so we can apply an altitude-keeping
        # bias when the drone drifts outside [Z_MIN, Z_MAX].
        self.sub = self.create_subscription(
            Odometry, "/odom", self._on_odom, 10)
        self.z = Z_TARGET_NOM

        self.create_timer(1.0 / 20.0, self._tick)
        self.create_timer(5.0, self._status)

        self.t_start = self._now()
        self.t_prim_start = 0.0
        self.idx = 0
        self.cycle = 0
        self._resample_cycle()

        self.get_logger().info(
            "auto_drive_3d started — NCLT-style 3D schedule, cycles forever. "
            "Ctrl+C to stop.")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg):
        self.z = msg.pose.pose.position.z

    def _resample_cycle(self):
        self._sched = [(fn, _jitter(d)) for fn, d in SCHEDULE]

    def _altitude_bias(self):
        # Soft pull-back when near altitude limits. Mimics a high-level
        # waypoint controller without overriding the schedule entirely.
        if self.z < Z_MIN + 0.2:
            return +VZ_GENTLE * 0.8
        if self.z > Z_MAX - 0.2:
            return -VZ_GENTLE * 0.8
        return 0.0

    def _tick(self):
        t_global = self._now() - self.t_start
        t_in_prim = t_global - self.t_prim_start
        fn, dur = self._sched[self.idx]

        if t_in_prim >= dur:
            self.idx = (self.idx + 1) % len(self._sched)
            if self.idx == 0:
                self.cycle += 1
                self._resample_cycle()
            self.t_prim_start = t_global
            t_in_prim = 0.0
            fn, dur = self._sched[self.idx]
            self.get_logger().info(
                f"cycle {self.cycle} → primitive '{fn.__name__}' "
                f"({dur:.1f}s)")

        vx, vy, vz, wz = fn(t_in_prim)
        vz = vz + self._altitude_bias()

        vx = float(np.clip(vx, -V_MAX,  V_MAX))
        vy = float(np.clip(vy, -V_MAX,  V_MAX))
        vz = float(np.clip(vz, -VZ_MAX, VZ_MAX))
        wz = float(np.clip(wz, -W_SHARP, W_SHARP))

        msg = Twist()
        msg.linear.x  = vx
        msg.linear.y  = vy
        msg.linear.z  = vz
        msg.angular.z = wz
        self.pub.publish(msg)

    def _status(self):
        t = self._now() - self.t_start
        fn, _ = self._sched[self.idx]
        self.get_logger().info(
            f"t={t:6.1f}s │ cycle={self.cycle} │ prim='{fn.__name__}' │ "
            f"z={self.z:.2f}m")

    def stop(self):
        try:
            self.pub.publish(Twist())
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = AutoDriver3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
