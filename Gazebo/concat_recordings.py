#!/usr/bin/env python3
"""
Concatenate per-gravity .npz recordings produced by
record_vargrav_session.sh into a single train/val/test set.

Each input file matches the pattern  <tag>_<split>.npz
where split ∈ {train, val, test} and tag identifies the gravity
session (e.g. g_8_5_train.npz, g_9_81_train.npz, ...).

The concat order is alphabetical, which (for the default tag scheme)
sweeps gravity values in a deterministic order. The output also
records a per-sequence gravity label for analysis.

Usage:
  python3 concat_recordings.py --in-dir ~/.ros/vargrav --out-dir ~/.ros
"""

import argparse
import glob
import os
import re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix",  default="gazebo",
                    help="Output filename prefix; default 'gazebo' "
                         "(produces gazebo_train_vargrav.npz etc.)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split in ("train", "val", "test"):
        files = sorted(glob.glob(
            os.path.join(args.in_dir, f"*_{split}.npz")))
        if not files:
            print(f"[concat] no files for split={split}")
            continue

        xs, ys, gz_tags = [], [], []
        for f in files:
            d = np.load(f)
            xs.append(d["x"])
            ys.append(d["y"])
            base = os.path.basename(f)
            m = re.match(r"g(_?-?\d+(?:_\d+)?)_", base)
            if m:
                tag_raw = m.group(1).replace("_", ".")
                tag_raw = tag_raw.lstrip(".")
                try:
                    gz = float("-" + tag_raw) if not tag_raw.startswith("-") \
                         else float(tag_raw)
                except ValueError:
                    gz = float("nan")
            else:
                gz = float("nan")
            gz_tags.append(np.full(d["x"].shape[0], gz, dtype=np.float32))
            print(f"[concat] {os.path.basename(f):30s} "
                  f"x={d['x'].shape}  y={d['y'].shape}  gz={gz:+.3f}")

        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.float32)
        gz_per_seq = np.concatenate(gz_tags, axis=0)

        out = os.path.join(args.out_dir,
                           f"{args.prefix}_{split}_vargrav.npz")
        np.savez_compressed(out, x=x, y=y, gz_per_seq=gz_per_seq)
        print(f"[concat] -> {out}   (x={x.shape}, y={y.shape}, "
              f"{len(np.unique(gz_per_seq))} unique gz values)")


if __name__ == "__main__":
    main()
