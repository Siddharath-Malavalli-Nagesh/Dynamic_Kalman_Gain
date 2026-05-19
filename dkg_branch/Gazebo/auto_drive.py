#!/usr/bin/env python3
"""
Auto-driver for TurtleBot3 in Gazebo. Publishes /cmd_vel through a
varied schedule designed to mimic the kind of motion present in the
NCLT dataset: long straights, gentle turns, occasional tight turns,
acceleration ramps, stops, and a small amount of random walk for
coverage. Cycles indefinitely until Ctrl+C.

Run alongside gazebo_nclt_recorder.py:

  Terminal A:  ros2 launch turtlebot3_gazebo empty_world.launch.py
  Terminal B:  python3 gazebo_nclt_recorder.py
  Terminal C:  python3 auto_drive.py

The schedule is randomised on each cycle (durations and parameters
jittered ±20%) so longer recordings produce more diverse data.

NCLT-realism notes
------------------
- Turn rates kept |w| <= 0.5 rad/s most of the time (gentle), with
  rare sharper turns at intersections.
- No jerky direction reversals.
- Brief stops between primitives (Segway-style pauses).
- Speed range 0.05 - 0.18 m/s (TurtleBot Waffle safe envelope).
"""

import math
import random
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

V_NOM = 0.15           # nominal cruising speed (m/s)
V_MAX = 0.20           # absolute cap
W_GENTLE = 0.40        # rad/s for normal turns
W_SHARP  = 0.80        # rad/s for occasional tight turns


def _jitter(x, frac=0.2):
    """Multiply x by uniform(1-frac, 1+frac)."""
    return x * random.uniform(1 - frac, 1 + frac)


# Each primitive is (name, default_duration_s, fn(t_local) -> (v, w)).
# Functions are pure given t_local; randomness lives in cycle-level
# parameter resampling below.

def p_straight_long(t):       return (V_NOM,  0.0)
def p_straight_slow(t):       return (0.08,   0.0)
def p_cruise_burst(t):        # accel ramp 0->V_NOM->cruise->decel
    if t < 1.5:   return (V_NOM * (t / 1.5), 0.0)
    if t < 4.5:   return (V_NOM, 0.0)
    if t < 6.0:   return (V_NOM * max(0.0, 1.0 - (t - 4.5) / 1.5), 0.0)
    return (0.0, 0.0)

def p_gentle_arc_l(t):        return (V_NOM,  W_GENTLE * 0.5)
def p_gentle_arc_r(t):        return (V_NOM, -W_GENTLE * 0.5)
def p_curve_l(t):             return (V_NOM,  W_GENTLE)
def p_curve_r(t):             return (V_NOM, -W_GENTLE)
def p_tight_l(t):             return (0.08,   W_SHARP)
def p_tight_r(t):             return (0.08,  -W_SHARP)

def p_serpentine(t):
    # Smooth sinusoidal weaving — exercises lateral accel
    return (V_NOM, 0.4 * math.sin(2 * math.pi * t / 8.0))

def p_figure_8(t):
    # Swap angular direction every 6 s
    w = W_GENTLE if int(t // 6.0) % 2 == 0 else -W_GENTLE
    return (0.12, w)

def p_pause(t):               return (0.0,    0.0)


# Schedule: (primitive_fn, nominal_duration_s).
# Bias toward "boring" straights and gentle curves so the recording
# matches NCLT-like distributions instead of being dominated by the
# unusual primitives.
SCHEDULE = [
    (p_straight_long,  20.0),
    (p_pause,           2.0),
    (p_gentle_arc_l,   12.0),
    (p_straight_long,  10.0),
    (p_gentle_arc_r,   12.0),
    (p_pause,           2.0),
    (p_cruise_burst,    8.0),
    (p_straight_slow,   8.0),
    (p_curve_l,        14.0),
    (p_pause,           2.0),
    (p_curve_r,        14.0),
    (p_serpentine,     20.0),
    (p_pause,           2.0),
    (p_tight_l,         6.0),     # rare sharp turn
    (p_straight_long,  15.0),
    (p_tight_r,         6.0),     # rare sharp turn
    (p_pause,           2.0),
    (p_figure_8,       24.0),
    (p_straight_long,  20.0),
    (p_pause,           3.0),
]


class AutoDriver(Node):

    def __init__(self):
        super().__init__("auto_drive")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # 20 Hz control loop
        self.create_timer(1.0 / 20.0, self._tick)
        self.create_timer(5.0, self._status)

        self.t_start = self._now()
        self.t_prim_start = 0.0
        self.idx = 0
        self.cycle = 0
        self._resample_cycle()

        self.get_logger().info(
            "auto_drive started — NCLT-style schedule, cycles forever. "
            "Ctrl+C to stop.")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _resample_cycle(self):
        """Jitter every duration in this cycle by ±20%."""
        self._sched = [(fn, _jitter(d)) for fn, d in SCHEDULE]

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

        v, w = fn(t_in_prim)
        v = float(np.clip(v, -V_MAX, V_MAX))
        w = float(np.clip(w, -W_SHARP, W_SHARP))

        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.pub.publish(msg)

    def _status(self):
        t = self._now() - self.t_start
        fn, _ = self._sched[self.idx]
        self.get_logger().info(
            f"t={t:6.1f}s │ cycle={self.cycle} │ prim='{fn.__name__}'")

    def stop(self):
        try:
            self.pub.publish(Twist())   # zero
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = AutoDriver()
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
