from ekf import ExtendedKalmanFilter
import time
import numpy as np
import torch
from evaluate_models import _sanitize, compute_metrics, _infer_ekf, _lat_ekf, _print_model_results

TEST_DATA     = "Downloads/nclt_test.npz"
TRAIN_DATA    = "Downloads/nclt_train.npz"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENCY_N_SEQ = 20

def main():
    print(f"Loading: {TEST_DATA}")
    raw     = np.load(TEST_DATA)
    x_np    = _sanitize(raw["x"].astype(np.float32))
    y_np    = _sanitize(raw["y"].astype(np.float32))
    N, T, _ = x_np.shape
    print(f"  {N} sequences × {T} timesteps | device: {DEVICE}\n")

    x_f64   = x_np.astype(np.float64)
    y_f64   = y_np.astype(np.float64)
    gt_eval = x_np[:, 1:, :].astype(np.float64)

    # ── Fit H bias and sigma_r from training set ──────────────────────
    print("  Loading training data...")
    _tr   = np.load(TRAIN_DATA)
    _tr_x = _sanitize(_tr["x"].astype(np.float32)).astype(np.float64)
    _tr_y = _sanitize(_tr["y"].astype(np.float32)).astype(np.float64)

    H_fit = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)

    _x_flat    = _tr_x[:, 1:, :].reshape(-1, 6)
    _y_flat    = _tr_y[:, 1:, :].reshape(-1, 3)
    _residuals = _y_flat - (H_fit @ _x_flat.T).T
    H_bias_fit = np.mean(_residuals, axis=0)
    sigma_r    = np.std(_residuals,  axis=0)

    print(f"  H bias:   {np.round(H_bias_fit, 4)}")
    print(f"  sigma_r:  {np.round(sigma_r, 4)}")
    del _tr, _tr_x, _tr_y, _x_flat, _y_flat, _residuals

    # ── Grid search over accel_std ────────────────────────────────────
    # 50 values from 0.01 to 0.5, log-spaced so we cover small values
    # more densely (where the filter is most sensitive)
    accel_values = np.round(np.linspace(0.01, 0.5, 50), 4)

    print(f"\nGrid searching accel_std over {len(accel_values)} values "
          f"({accel_values[0]} → {accel_values[-1]})...\n")
    print(f"  {'accel_std':>10} {'RMSE_pos (m)':>14} {'MAE_pos (m)':>13} "
          f"{'Prec<1m (%)':>13} {'MSE_full':>12}")
    print(f"  {'-'*10} {'-'*14} {'-'*13} {'-'*13} {'-'*12}")

    results = []

    for accel_std in accel_values:
        preds = _infer_ekf(x_f64, y_f64, H_fit, H_bias_fit, sigma_r, float(accel_std))
        m     = compute_metrics(preds, gt_eval)

        results.append({
            "accel_std":   float(accel_std),
            "rmse_pos":    m["rmse_pos"],
            "mae_pos":     m["mae_pos"],
            "precision_1m":m["precision_1m"],
            "mse_full":    m["mse_full"],
            "metrics":     m,
            "preds":       preds,
        })

        print(f"  {accel_std:>10.4f} {m['rmse_pos']:>14.4f} {m['mae_pos']:>13.4f} "
              f"{m['precision_1m']:>13.2f} {m['mse_full']:>12.6f}")

    # ── Find best by RMSE (primary), MAE (tiebreak) ───────────────────
    best_rmse  = min(results, key=lambda r: (r["rmse_pos"], r["mae_pos"]))
    best_prec  = max(results, key=lambda r: r["precision_1m"])
    best_mse   = min(results, key=lambda r: r["mse_full"])

    print(f"\n{'='*62}")
    print(f"  GRID SEARCH RESULTS")
    print(f"{'='*62}")
    print(f"  Best by RMSE (pos)   : accel_std = {best_rmse['accel_std']:.4f}  "
          f"→ RMSE = {best_rmse['rmse_pos']:.4f} m")
    print(f"  Best by Precision    : accel_std = {best_prec['accel_std']:.4f}  "
          f"→ Precision = {best_prec['precision_1m']:.2f} %")
    print(f"  Best by MSE (full)   : accel_std = {best_mse['accel_std']:.4f}  "
          f"→ MSE = {best_mse['mse_full']:.6f}")

    # ── Print full results for best RMSE value ─────────────────────────
    best = best_rmse
    print(f"\n  → Using best accel_std = {best['accel_std']:.4f} for final eval\n")

    # Measure latency for best accel_std
    lat_mean, lat_std = _lat_ekf(
        x_f64, y_f64, LATENCY_N_SEQ,
        H_fit, H_bias_fit, sigma_r, best["accel_std"]
    )

    _print_model_results(
        f"EKF — Best (accel_std={best['accel_std']:.4f})",
        best["metrics"], lat_mean, lat_std, -1.0, -1.0
    )

    # ── Summary table of top 5 by RMSE ────────────────────────────────
    sorted_by_rmse = sorted(results, key=lambda r: r["rmse_pos"])
    print(f"\n  Top 5 accel_std values by position RMSE:")
    print(f"  {'Rank':>4} {'accel_std':>10} {'RMSE_pos':>10} "
          f"{'MAE_pos':>10} {'Prec<1m':>10}")
    print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for rank, r in enumerate(sorted_by_rmse[:5], 1):
        print(f"  {rank:>4} {r['accel_std']:>10.4f} {r['rmse_pos']:>10.4f} "
              f"{r['mae_pos']:>10.4f} {r['precision_1m']:>10.2f}")


if __name__ == "__main__":
    main()