#!/usr/bin/env python3
"""
Inject time-varying gravity perturbations into recorded IMU data.

Use case: ConstructSim's locked-down Gazebo doesn't expose
/gazebo/set_physics_properties, so we can't change gravity at runtime
to test adaptivity. Instead, we record once with normal physics and
post-hoc add piecewise-constant offsets to acc_z. Motion, IMU noise,
and ground-truth states remain physically real; only the gravity
constant is perturbed.

This is equivalent to studying robustness against gravity-vector
miscalibration / temperature-induced IMU bias drift — a documented
real-world failure mode for inertial navigation.

Usage
-----
  python3 perturb_gravity.py \\
      --in     ~/.ros/gazebo_train.npz \\
      --out    ~/.ros/gazebo_train_vargrav.npz \\
      --interval-steps 50 \\
      --range  3.0 \\
      --seed   42

The output .npz has the same x and y arrays plus a 'gz_schedule'
array recording what offset was applied at each timestep (useful for
plotting / sanity checks).
"""

import argparse
import numpy as np


def perturb(in_path, out_path, interval_steps, g_range, seed):
    rng = np.random.default_rng(seed)
    data = np.load(in_path)
    x = data["x"]              # (N, T, 6)  states — untouched
    y = data["y"].copy()       # (N, T, 3)  obs — modify acc_z
    N, T, _ = y.shape

    # Build a piecewise-constant offset over the FLATTENED time axis,
    # so that perturbations can span chunk boundaries (the agent sees
    # transitions both within and between chunks).
    flat_len = N * T
    n_blocks = (flat_len + interval_steps - 1) // interval_steps
    block_offsets = rng.uniform(-g_range, g_range, size=n_blocks).astype(np.float32)

    # Expand to per-step schedule
    offsets_flat = np.repeat(block_offsets, interval_steps)[:flat_len]
    schedule = offsets_flat.reshape(N, T)

    # Apply to acc_z (channel index 2)
    y[..., 2] = y[..., 2] + schedule

    out = {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "gz_schedule": schedule.astype(np.float32),
        "gz_range": np.float32(g_range),
        "gz_interval_steps": np.int32(interval_steps),
    }
    np.savez_compressed(out_path, **out)

    print(f"in : {in_path}")
    print(f"     x={x.shape}  y={y.shape}")
    print(f"out: {out_path}")
    print(f"     interval_steps={interval_steps}  ({interval_steps * 0.01:.2f}s)")
    print(f"     range=±{g_range:.2f} m/s^2  seed={seed}")
    print(f"     {n_blocks} distinct gravity regimes; "
          f"applied offset stats min={schedule.min():+.3f} "
          f"max={schedule.max():+.3f} mean={schedule.mean():+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path",  required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--interval-steps", type=int, default=50,
                    help="Steps between gravity changes "
                         "(default 50 = 0.5s @ 100 Hz).")
    ap.add_argument("--range", type=float, default=3.0,
                    help="Half-range of uniform sampling, m/s^2 "
                         "(default 3.0 → gz drifts by up to ±3).")
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()
    perturb(args.in_path, args.out_path, args.interval_steps,
            args.range, args.seed)


if __name__ == "__main__":
    main()
