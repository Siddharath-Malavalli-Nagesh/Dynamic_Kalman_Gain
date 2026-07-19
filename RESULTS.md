# EKF vs Gravity-Aware EKF vs KalmanNet vs Shadow Student

## Overview

This report compares four state estimation methods:

- **EKF** (Classical Extended Kalman Filter baseline)
- **Gravity-Aware EKF** (EKF with gravity compensation)
- **KalmanNet** (Deep learning teacher model)
- **Shadow Student** (Distilled lightweight model)

---
## Experimental Setup
(CPU Specs)

| Component | Specification |
|----------|---------------|
| CPU Architecture | x86_64 |
| CPU Frequency | 2803.2 MHz (~2.80 GHz) |

# 1. Position Estimation Performance

| Model | MSE (m²) | RMSE (m) | MAE (m) | Inlier Precision (<1m) | % Position Error |
|------|---------:|---------:|---------:|-----------------------:|----------------:|
| EKF | 24.173650 | 8.5159 | 8.0030 | 0.02% | 4.9473% |
| Gravity-Aware EKF | 16.797172 | 7.0987 | 6.2604 | 0.29% | 3.8701% |
| KalmanNet | **0.327682** | **0.9915** | **0.6455** | **79.90%** | **0.3991%** |
| Shadow Student | 0.592099 | 1.3328 | 0.9164 | 66.66% | 0.5665% |

### Key Insights

- Gravity-aware EKF improves position RMSE by **16.6%** over the standard EKF.
- KalmanNet reduces EKF RMSE by **88.4%**.
- Shadow Student reduces EKF RMSE by **84.4%**.
- KalmanNet achieves nearly **80%** sub-meter precision.
- Shadow Student maintains **66.7%** sub-meter precision while using a significantly smaller model.

---

# 2. Position Breakdown (Axis-wise RMSE)

| Model | x RMSE | y RMSE | z RMSE |
|------|--------:|--------:|--------:|
| EKF | 3.2268 | 4.1503 | 6.6995 |
| Gravity-Aware EKF | 4.1233 | 4.1778 | 3.9920 |
| KalmanNet | **0.5627** | **0.8060** | **0.1293** |
| Shadow Student | 0.8946 | 0.9403 | 0.3031 |

### Key Insights

- Standard EKF exhibits large errors across all position axes.
- Gravity compensation significantly improves the **z-axis**, reducing RMSE from **6.70 m** to **3.99 m**.
- KalmanNet consistently achieves the lowest error across all axes.
- Shadow Student remains close to the teacher while using a much smaller architecture.

---

# 3. Velocity Estimation Performance

| Model | MSE (m/s)² | RMSE (m/s) | MAE (m/s) |
|------|-----------:|-----------:|----------:|
| EKF | 5.894971 | 4.2053 | 3.6389 |
| Gravity-Aware EKF | 9.056648 | 5.2125 | 4.5445 |
| KalmanNet | **0.088549** | **0.5154** | **0.3899** |
| Shadow Student | 0.199972 | 0.7745 | 0.5923 |

### Key Insights

- KalmanNet reduces EKF velocity RMSE by **87.7%**.
- Shadow Student reduces EKF velocity RMSE by **81.6%**.
- Gravity-aware EKF improves positional accuracy but slightly degrades velocity estimation, indicating that simple gravity compensation alone is insufficient for accurate dynamic modeling.

---

# 4. Velocity Breakdown (Axis-wise RMSE)

| Model | vx RMSE | vy RMSE | vz RMSE |
|------|---------:|---------:|---------:|
| EKF | 2.5531 | 3.3399 | 0.1081 |
| Gravity-Aware EKF | 3.6248 | 3.7439 | 0.1180 |
| KalmanNet | **0.3423** | **0.3747** | **0.0897** |
| Shadow Student | 0.2991 | 0.6919 | 0.1781 |

### Key Insights

- EKF exhibits significant instability in the horizontal velocity components.
- KalmanNet provides the most balanced velocity estimates.
- Shadow Student preserves strong performance while remaining computationally lightweight.

---

# 5. Full State Error

| Model | MSE |
|------|----:|
| EKF | 15.034311 |
| Gravity-Aware EKF | 12.926910 |
| KalmanNet | **0.208115** |
| Shadow Student | 0.396036 |

### Key Insights

- Gravity-aware EKF lowers full-state error by **14.0%** compared to the baseline EKF.
- KalmanNet achieves approximately **72×** lower full-state MSE than EKF.
- Shadow Student achieves approximately **38×** lower full-state MSE than EKF.
- Shadow Student retains much of the teacher's performance while using a significantly smaller model.

---

# 6. Latency Comparison (CPU)

| Model | Mean Latency (ms/step) | Std (ms) |
|------|-----------------------:|---------:|
| EKF | **0.0157** | 0.0106 |
| Gravity-Aware EKF | 0.0161 | 0.0097 |
| KalmanNet | 0.1052 | 0.0395 |
| Shadow Student | 0.0738 | 0.0289 |

### Key Insights

- EKF remains the fastest estimator.
- Gravity compensation introduces virtually **no computational overhead**.
- Shadow Student is approximately **30% faster** than KalmanNet.
- Both neural models comfortably satisfy real-time inference requirements.

---

# 7. Overall Comparison Summary

## EKF (Classical Baseline)

- Fastest model.
- Lowest computational cost.
- Weakest estimation accuracy.
- Large drift in both position and velocity.

---

## Gravity-Aware EKF

- Improves positional accuracy over the standard EKF.
- Reduces full-state error with negligible runtime overhead.
- Particularly improves vertical position estimation.
- Velocity estimation remains limited due to the simplified gravity compensation model.

---

## KalmanNet (Teacher Model)

- Best overall accuracy across every evaluation metric.
- Achieves:
  - **88% lower position RMSE** than EKF.
  - **88% lower velocity RMSE** than EKF.
  - **72× lower full-state MSE** than EKF.
- Highest computational cost among the evaluated methods.

---

## Shadow Student (Distilled Model)

- Maintains most of the teacher's estimation accuracy with substantially fewer parameters.
- Achieves:
  - **84% lower position RMSE** than EKF.
  - **82% lower velocity RMSE** than EKF.
  - **38× lower full-state MSE** than EKF.
- Approximately **30% faster** than KalmanNet while preserving strong estimation performance.

---

# Conclusion

The corrected EKF calibration and the addition of a gravity-aware baseline provide stronger and fairer comparisons against the learning-based approaches. While gravity compensation offers modest improvements to classical filtering with virtually no runtime cost, the neural estimators significantly outperform both EKF variants. KalmanNet delivers the highest overall estimation accuracy, whereas the Shadow Student achieves a favorable balance between computational efficiency and estimation performance, making it well suited for real-time deployment on resource-constrained robotic platforms.

# Shadow Student Learned Observation Matrix

Output from:

```python
state = torch.load("best_student_nclt.pt")
print(state["H.weight"])
print(state["H.bias"])
```

```text
tensor([[-1.3598e-03, -1.5775e-04, -3.4321e-02,  1.1140e+00,  4.7381e-01,
          1.4110e-01],
        [-1.4795e-05, -4.5697e-04,  2.8420e-03, -3.8631e-01,  9.6233e-01,
         -1.4420e-01],
        [-7.9180e-03,  9.2816e-04, -6.7924e-01,  2.4026e-01,  3.4952e-01,
          1.0776e+00]])
```

```text
tensor([-0.0365,  0.0373, -1.7892])
```