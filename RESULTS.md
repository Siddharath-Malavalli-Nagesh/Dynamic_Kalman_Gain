# Results

## KalmanNet

### Position (px, py, pz)
- MSE: 0.327682 m²  
- RMSE: 0.9915 m  
- RMSE_x / y / z: 0.5627 / 0.8060 / 0.1293 m  
- MAE: 0.6455 m  
- MAE_x / y / z: 0.3334 / 0.4777 / 0.0699 m  
- Error Variance: 0.5663  
- Inlier Precision (<1m): 79.90 %  
- % Pos Error: 0.3991 %

### Velocity (vx, vy, vz)
- MSE: 0.088549 (m/s)²  
- RMSE: 0.5154 m/s  
- RMSE_vx / vy / vz: 0.3423 / 0.3747 / 0.0897 m/s  
- MAE: 0.3899 m/s  
- MAE_vx / vy / vz: 0.2344 / 0.2544 / 0.0472 m/s  
- Error Variance: 0.1136  

### Full State
- MSE: 0.208115  

### Latency (CPU, step-wise)
- Mean: 0.1003 ms / step  
- Std: 0.0876 ms  


---

## EKF

### Position (px, py, pz)
- MSE: 3.429029 m²  
- RMSE: 3.2073 m  
- RMSE_x / y / z: 2.6558 / 1.6217 / 0.7770 m  
- MAE: 2.5918 m  
- MAE_x / y / z: 1.8878 / 1.1614 / 0.5685 m  
- Error Variance: 3.5696  
- Inlier Precision (<1m): 21.83 %  
- % Pos Error: 1.6022 %

### Velocity (vx, vy, vz)
- MSE: 2.200786 (m/s)²  
- RMSE: 2.5695 m/s  
- RMSE_vx / vy / vz: 2.3253 / 1.0610 / 0.2639 m/s  
- MAE: 2.1436 m/s  
- MAE_vx / vy / vz: 1.8464 / 0.8412 / 0.1953 m/s  
- Error Variance: 2.0073  

### Full State
- MSE: 2.814907  

### Latency (CPU, step-wise)
- Mean: 0.0159 ms / step  
- Std: 0.0142 ms  


---

## Summary

- KalmanNet significantly outperforms EKF across all metrics.
- Position accuracy:
  - KalmanNet RMSE: 0.9915 m  
  - EKF RMSE: 3.2073 m  
- Velocity accuracy:
  - KalmanNet RMSE: 0.5154 m/s  
  - EKF RMSE: 2.5695 m/s  
- Full-state MSE:
  - KalmanNet: 0.2081  
  - EKF: 2.8149  
- Inlier precision (<1m):
  - KalmanNet: 79.90 %  
  - EKF: 21.83 %  

Per-dimension analysis shows EKF errors are highest in x and vx, while z and vz are comparatively lower. KalmanNet maintains lower and more consistent errors across all state dimensions.

- Latency:
  - EKF: 0.0159 ms / step  
  - KalmanNet: 0.1003 ms / step  

KalmanNet is slower but substantially more accurate.