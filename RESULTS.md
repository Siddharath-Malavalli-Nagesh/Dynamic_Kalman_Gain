# Results

## KalmanNet vs Extended Kalman Filter (EKF)

### Evaluation Metrics

| Metric | EKF | KalmanNet |
|--------|-----|-----------|
| MSE (State overall) | 2.8149 | 0.2081 |
| RMSE (Position) | 1.8518 m | 0.5724 m |
| MAE (Position) | 1.2059 m | 0.2937 m |
| % Pos Error (relative mean) | 1.6022 % | 0.3991 % |
| Inlier Precision (< 1m) | 21.83 % | 79.90 % |
| Latency (ms/step) | 0.0146 | 0.0144 |

---

## Summary

- KalmanNet achieves significantly lower position error (RMSE and MAE).
- EKF shows higher error and lower inlier precision.
- Both models operate at similar real-time latency.
- KalmanNet provides substantially better tracking performance on this dataset.