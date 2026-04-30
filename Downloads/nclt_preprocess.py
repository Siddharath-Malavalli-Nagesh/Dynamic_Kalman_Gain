#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np

# Configuration
DATE = "2012-01-08"
BASE_DIR = "~/ros2_ws/src/Downloads"
IMU_FILE = os.path.join(BASE_DIR, "sensor_data", DATE, "ms25.csv")
GT_FILE = os.path.join(BASE_DIR, "ground_truth", f"groundtruth_{DATE}.csv")

SEQ_LEN = 200  # Sequence length for KalmanNet RNN chunks

def process_nclt_for_kalmannet():
    print("Loading Ground Truth and calculating velocity...")
    # 1. Load Ground Truth (m=6: pos_x, pos_y, pos_z, vel_x, vel_y, vel_z)
    gt_cols =['utime', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
    df_gt = pd.read_csv(GT_FILE, names=gt_cols, low_memory=False)
    
    # FIX: Force everything to numbers. If it's a string (like a header), it becomes NaN.
    for col in gt_cols:
        df_gt[col] = pd.to_numeric(df_gt[col], errors='coerce')
    
    # Drop the NaN rows (this removes the headers/text)
    df_gt = df_gt.dropna()
    
    # Convert microsec to sec to calculate true velocity (dx/dt)
    df_gt['time_sec'] = df_gt['utime'] / 1e6
    df_gt['dt'] = df_gt['time_sec'].diff()
    
    # Compute velocity
    df_gt['vx'] = df_gt['x'].diff() / df_gt['dt']
    df_gt['vy'] = df_gt['y'].diff() / df_gt['dt']
    df_gt['vz'] = df_gt['z'].diff() / df_gt['dt']
    df_gt = df_gt.bfill() # Fill the first NaN row
    
    # Keep only the state variables we need:[utime, x, y, z, vx, vy, vz]
    df_gt = df_gt[['utime', 'x', 'y', 'z', 'vx', 'vy', 'vz']].sort_values('utime')

    print("Loading IMU Observations...")
    # 2. Load IMU (n=3: acc_x, acc_y, acc_z)
    imu_cols =['utime', 'mag_x', 'mag_y', 'mag_z', 'acc_x', 'acc_y', 'acc_z', 'gyro_r', 'gyro_p', 'gyro_h']
    df_imu = pd.read_csv(IMU_FILE, names=imu_cols, low_memory=False)
    
    # FIX: Force IMU columns to numeric just in case IMU file also has headers
    for col in imu_cols:
        df_imu[col] = pd.to_numeric(df_imu[col], errors='coerce')
    df_imu = df_imu.dropna()
    
    # Keep only observation variables we need: [utime, acc_x, acc_y, acc_z]
    df_imu = df_imu[['utime', 'acc_x', 'acc_y', 'acc_z']].sort_values('utime')

    print("Synchronizing states and observations...")
    # 3. Synchronize IMU (100Hz) and GroundTruth using nearest timestamp
    # tolerance=50000 microseconds (50 ms) to ensure tight syncing
    df_synced = pd.merge_asof(df_imu, df_gt, on='utime', direction='nearest', tolerance=50000).dropna()

    # 4. Extract arrays
    observations = df_synced[['acc_x', 'acc_y', 'acc_z']].values      # Shape: (Total_steps, 3)
    states = df_synced[['x', 'y', 'z', 'vx', 'vy', 'vz']].values      # Shape: (Total_steps, 6)
    
    print(f"Total synchronized steps: {len(states)}")

    # 5. Chunking for RNN (e.g. 200 time steps per sequence)
    num_sequences = len(states) // SEQ_LEN
    
    # Truncate to make perfectly divisible by SEQ_LEN
    obs_chunks = observations[:num_sequences * SEQ_LEN].reshape((num_sequences, SEQ_LEN, 3))
    state_chunks = states[:num_sequences * SEQ_LEN].reshape((num_sequences, SEQ_LEN, 6))

    # 6. Split Train/Val/Test (70 / 15 / 15)
    split1 = int(0.70 * num_sequences)
    split2 = int(0.85 * num_sequences)

    train_x, train_y = state_chunks[:split1], obs_chunks[:split1]
    val_x, val_y     = state_chunks[split1:split2], obs_chunks[split1:split2]
    test_x, test_y   = state_chunks[split2:], obs_chunks[split2:]

    # 7. Save to compressed Numpy arrays
    np.savez_compressed('nclt_train.npz', x=train_x, y=train_y)
    np.savez_compressed('nclt_val.npz',   x=val_x,   y=val_y)
    np.savez_compressed('nclt_test.npz',  x=test_x,  y=test_y)

    print("\n--- Data Preparation Complete ---")
    print(f"Train shapes - States (x): {train_x.shape}, Obs (y): {train_y.shape}")
    print(f"Val shapes   - States (x): {val_x.shape},   Obs (y): {val_y.shape}")
    print(f"Test shapes  - States (x): {test_x.shape},  Obs (y): {test_y.shape}")
    print("Files saved: nclt_train.npz, nclt_val.npz, nclt_test.npz")

if __name__ == '__main__':
    process_nclt_for_kalmannet()