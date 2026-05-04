#!/usr/bin/env python3
"""
Concatenate per-gravity .npz recordings produced by
record_vargrav_session.sh into a single train/val/test set.

Each input file matches the pattern  <tag>_<split>.npz
where split ∈ {train, val, test} and tag identifies the gravity
session (e.g. g_8_5_train.npz, g_9_81_train.npz, ...).

Two modes
---------
1. Default: sweep all per-gravity files, concat into output
   train/val/test by their original split. Every gravity is present
   in every output split. Tests in-distribution generalisation only.

2. With --holdout-gravities: gravity values in the holdout list are
   pulled OUT of train/val and placed entirely into test. Output
   train/val will not contain those gravity values at all. This tests
   genuine out-of-distribution generalisation — does the RL meta-tuner
   adapt to gravity values it has never seen during training?

Usage
-----
  # Default (gravities in all splits)
  python3 concat_recordings.py --in-dir ~/.ros/vargrav --out-dir ~/.ros

  # Hold out -14 and -3 m/s² entirely from train; put them only in test
  python3 concat_recordings.py --in-dir ~/.ros/vargrav --out-dir ~/.ros \\
      --holdout-gravities -14 -3
"""

import argparse
import glob
import os
import re
import numpy as np


def parse_gz_from_filename(path):
    """g_8_5_train.npz -> -8.5; g8_5_train.npz -> 8.5 (positive); etc."""
    base = os.path.basename(path)
    m = re.match(r"g(_)?(-?\d+)(?:_(\d+))?_(train|val|test)\.npz", base)
    if not m:
        return float("nan")
    sign = -1.0 if m.group(1) == "_" else 1.0
    intp = m.group(2)
    frac = m.group(3) or "0"
    try:
        mag = float(f"{intp}.{frac}")
        return sign * mag
    except ValueError:
        return float("nan")


def load_npz(path):
    d = np.load(path)
    return d["x"], d["y"]


def gz_close(a, b, tol=1e-3):
    return abs(a - b) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix",  default="gazebo",
                    help="Output filename prefix (default 'gazebo').")
    ap.add_argument("--holdout-gravities", type=float, nargs="*",
                    default=[],
                    help="Gravity values (m/s^2) to hold out entirely "
                         "from train/val. All sequences recorded under "
                         "these gravities go into the test split only.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    holdouts = list(args.holdout_gravities)
    if holdouts:
        print(f"[concat] holdout gravities (test-only): "
              f"{', '.join(f'{g:+.3f}' for g in holdouts)}")

    # Index every input file by (gz, split)
    all_files = sorted(glob.glob(os.path.join(args.in_dir, "*.npz")))
    if not all_files:
        print(f"[concat] no .npz files found in {args.in_dir}")
        return

    # Buckets per output split
    out_buckets = {s: {"x": [], "y": [], "gz": []}
                   for s in ("train", "val", "test")}

    for f in all_files:
        base = os.path.basename(f)
        m = re.match(r".*_(train|val|test)\.npz$", base)
        if not m:
            continue
        split = m.group(1)
        gz = parse_gz_from_filename(f)

        x, y = load_npz(f)
        is_holdout = any(gz_close(gz, h) for h in holdouts)

        if is_holdout:
            target = "test"
            note = " [HOLDOUT → test]"
        else:
            target = split
            note = ""

        out_buckets[target]["x"].append(x)
        out_buckets[target]["y"].append(y)
        out_buckets[target]["gz"].append(
            np.full(x.shape[0], gz, dtype=np.float32))

        print(f"[concat] {base:30s} gz={gz:+.3f} "
              f"x={x.shape} -> {target}{note}")

    # Write out
    for split, bucket in out_buckets.items():
        if not bucket["x"]:
            print(f"[concat] no data for output split={split}")
            continue
        x = np.concatenate(bucket["x"], axis=0).astype(np.float32)
        y = np.concatenate(bucket["y"], axis=0).astype(np.float32)
        gz_per_seq = np.concatenate(bucket["gz"], axis=0)

        out = os.path.join(args.out_dir,
                           f"{args.prefix}_{split}_vargrav.npz")
        np.savez_compressed(out, x=x, y=y,
                            gz_per_seq=gz_per_seq,
                            holdout_gravities=np.asarray(holdouts,
                                                         dtype=np.float32))
        unique_gz = np.unique(gz_per_seq)
        print(f"[concat] -> {out}\n"
              f"            x={x.shape}, y={y.shape}, "
              f"{len(unique_gz)} unique gz: "
              f"[{', '.join(f'{g:+.2f}' for g in sorted(unique_gz))}]")


if __name__ == "__main__":
    main()
