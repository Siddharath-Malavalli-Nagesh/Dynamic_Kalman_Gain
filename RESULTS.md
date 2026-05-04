# EKF vs KalmanNet vs Shadow Student

## Overview
This report compares three state estimation methods:

- **EKF (classical baseline filter)**
- **KalmanNet (deep learning teacher model)**
- **Shadow Student (distilled lightweight model)**

All metrics are taken directly from evaluation outputs with consistent EKF baseline values.

---

# 1. Position Estimation Performance

| Model         | MSE (m²) | RMSE (m) | MAE (m) | Inlier Precision (<1m) | % Pos Error |
|---------------|----------|----------|---------|-------------------------|-------------|
| EKF           | 3.429029 | 3.2073   | 2.5918  | 21.83%                  | 1.6022%     |
| KalmanNet     | **0.327682** | **0.9915** | **0.6455** | **79.90%** | **0.3991%** |
| Student       | 0.592099 | 1.3328   | 0.9164  | 66.66%                  | 0.5665%     |

### Key Insights
- KalmanNet reduces EKF RMSE by **~69%**
- Student reduces EKF RMSE by **~58%**
- KalmanNet achieves **~3.2× higher precision than EKF**
- Student still maintains **~3× improvement over EKF**

---

# 2. Position Breakdown (Axis-wise RMSE)

| Model     | x RMSE | y RMSE | z RMSE |
|-----------|--------|--------|--------|
| EKF       | 2.6558 | 1.6217 | 0.7770 |
| KalmanNet | 0.5627 | 0.8060 | 0.1293 |
| Student   | 0.8946 | 0.9403 | 0.3031 |

### Key Insights
- EKF struggles heavily in **x and y axes**
- KalmanNet strongly reduces error across all axes
- Student shows moderate degradation but remains stable
- z-axis is easiest for all models

---

# 3. Velocity Estimation Performance

| Model         | MSE (m/s)² | RMSE (m/s) | MAE (m/s) |
|---------------|------------|------------|-----------|
| EKF           | 2.200786   | 2.5695     | 2.1436    |
| KalmanNet     | **0.088549** | **0.5154** | **0.3899** |
| Student       | 0.199972   | 0.7745     | 0.5923    |

### Key Insights
- KalmanNet improves EKF velocity RMSE by **~80%**
- Student still achieves **~70% reduction vs EKF**
- Neural models significantly outperform EKF in dynamics

---

# 4. Velocity Axis-wise RMSE

| Model     | vx RMSE | vy RMSE | vz RMSE |
|-----------|---------|---------|---------|
| EKF       | 2.3253  | 1.0610  | 0.2639  |
| KalmanNet | 0.3423  | 0.3747  | 0.0897  |
| Student   | 0.2991  | 0.6919  | 0.1781  |

### Key Insights
- EKF has severe vx and vy instability
- KalmanNet stabilizes all velocity dimensions
- Student retains strong vx performance but degrades in vy

---

# 5. Full State Error

| Model         | MSE |
|---------------|------|
| EKF           | 2.814907 |
| KalmanNet     | **0.208115** |
| Student       | 0.396036 |

### Key Insights
- KalmanNet is **~13× better than EKF**
- Student is **~7× better than EKF**
- Student retains ~50% of teacher improvement

---

# 6. Latency Comparison (CPU)

| Model         | Mean Latency (ms/step) | Std |
|---------------|------------------------|-----|
| EKF           | **0.0159**             | 0.0142 |
| KalmanNet     | 0.1019                 | 0.0342 |
| Student       | 0.0722                 | 0.0191 |

### Key Insights
- EKF is ~6× faster than Student
- Student is ~29% faster than KalmanNet
- Neural models trade speed for accuracy

---

# 7. Overall Comparison Summary

## EKF (Classical Baseline)
- Fastest model
- Weakest accuracy
- Poor handling of nonlinear motion
- High error in x and velocity components

## KalmanNet (Teacher Model)
- Best overall accuracy
- Strong improvements across all metrics:
  - ~3× better position accuracy
  - ~5× better velocity accuracy
  - ~13× lower full-state error
- Higher computational cost

## Shadow Student (Distilled Model)
- Intermediate performance
- Maintains major improvements over EKF:
  - ~58% reduction in position RMSE
  - ~65%+ reduction in velocity error
- Faster than teacher but slower than EKF
- Slight degradation in precision vs KalmanNet

---
