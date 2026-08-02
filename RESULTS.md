# EKF vs KalmanNet vs Shadow Student

## Overview

This report compares three state estimation methods:

- **EKF** (Classical Extended Kalman Filter baseline)
- **KalmanNet** (Deep learning teacher model)
- **Shadow Student** (Distilled lightweight model)

---

## Experimental Setup

(CPU Specs)

| Component | Specification |
|----------|---------------|
| CPU Architecture | x86_64 |
| CPU Frequency | 2803.2 MHz (~2.80 GHz) |

---

# 1. Position Estimation Performance

| Model | MSE (m²) | RMSE (m) | MAE (m) | Inlier Precision (<1m) | % Position Error |
|------|---------:|---------:|---------:|-----------------------:|----------------:|
| EKF | 3.178364 | 3.0879 | 2.5117 | 27.56% | 1.5527% |
| KalmanNet | **0.327682** | **0.9915** | **0.6455** | **79.90%** | **0.3991%** |
| Shadow Student | 0.592099 | 1.3328 | 0.9164 | 66.66% | 0.5665% |

### Key Insights

- KalmanNet reduces EKF position RMSE by **67.9%**.
- Shadow Student reduces EKF position RMSE by **56.8%**.
- KalmanNet achieves nearly **80%** sub-meter precision.
- Shadow Student maintains **66.7%** sub-meter precision while using a significantly smaller model.

---

# 2. Position Breakdown (Axis-wise RMSE)

| Model | x RMSE | y RMSE | z RMSE |
|------|--------:|--------:|--------:|
| EKF | 1.8958 | 2.4358 | 0.0881 |
| KalmanNet | **0.5627** | **0.8060** | 0.1293 |
| Shadow Student | 0.8946 | 0.9403 | 0.3031 |

### Key Insights

- KalmanNet provides the lowest horizontal position error.
- Shadow Student closely follows the teacher while remaining lightweight.
- EKF performs competitively on the z-axis but exhibits significantly larger horizontal drift.

---

# 3. Velocity Estimation Performance

| Model | MSE (m/s)² | RMSE (m/s) | MAE (m/s) |
|------|-----------:|-----------:|----------:|
| EKF | 0.658693 | 1.4057 | 1.3089 |
| KalmanNet | **0.088549** | **0.5154** | **0.3899** |
| Shadow Student | 0.199972 | 0.7745 | 0.5923 |

### Key Insights

- KalmanNet reduces EKF velocity RMSE by **63.3%**.
- Shadow Student reduces EKF velocity RMSE by **44.9%**.
- Both neural estimators substantially outperform the classical EKF.

---

# 4. Velocity Breakdown (Axis-wise RMSE)

| Model | vx RMSE | vy RMSE | vz RMSE |
|------|---------:|---------:|---------:|
| EKF | 0.8877 | 1.0890 | **0.0444** |
| KalmanNet | 0.3423 | **0.3747** | 0.0897 |
| Shadow Student | **0.2991** | 0.6919 | 0.1781 |

### Key Insights

- KalmanNet provides the most balanced velocity estimates.
- Shadow Student achieves the lowest x-axis velocity RMSE.
- EKF exhibits larger horizontal velocity errors than the learning-based models.

---

# 5. Full State Error

| Model | MSE |
|------|----:|
| EKF | 1.918528 |
| KalmanNet | **0.208115** |
| Shadow Student | 0.396036 |

### Key Insights

- KalmanNet achieves approximately **9.2×** lower full-state MSE than EKF.
- Shadow Student achieves approximately **4.8×** lower full-state MSE than EKF.
- The distilled model retains much of the teacher's estimation capability while remaining computationally efficient.

---

# 6. Latency Comparison (CPU)

| Model | Mean Latency (ms/step) | Std (ms) |
|------|-----------------------:|---------:|
| EKF | **0.0161** | 0.0037 |
| KalmanNet | 0.1134 | 0.2641 |
| Shadow Student | 0.0793 | 0.0559 |

### Key Insights

- EKF remains the fastest estimator.
- Shadow Student is approximately **30% faster** than KalmanNet.
- Both neural models comfortably satisfy real-time inference requirements.

---

# 7. Overall Comparison Summary

## EKF (Classical Baseline)

- Fastest model.
- Lowest computational cost.
- Moderate estimation accuracy.
- Larger position and velocity drift than the learning-based methods.

---

## KalmanNet (Teacher Model)

- Best overall estimation accuracy.
- Achieves:
  - **67.9% lower position RMSE** than EKF.
  - **63.3% lower velocity RMSE** than EKF.
  - **9.2× lower full-state MSE** than EKF.
- Highest computational cost among the evaluated models.

---

## Shadow Student (Distilled Model)

- Preserves most of the teacher's estimation accuracy while using substantially fewer parameters.
- Achieves:
  - **56.8% lower position RMSE** than EKF.
  - **44.9% lower velocity RMSE** than EKF.
  - **4.8× lower full-state MSE** than EKF.
- Approximately **30% faster** than KalmanNet, making it suitable for real-time deployment on resource-constrained robotic platforms.

---

# Ablation Study

| ALPHA_START | Position RMSE (m) | Velocity RMSE (m/s) | CPU Latency (ms/step) |
|-------------|------------------:|--------------------:|----------------------:|
| 0.50 | 4.7854 | 1.4180 | 0.0726 |
| 0.85 | 1.7734 | 1.4060 | 0.0679 |

---

# Conclusion

The updated EKF calibration provides a substantially stronger classical baseline for comparison. Both learning-based estimators significantly outperform the EKF in state estimation accuracy while maintaining real-time inference speed. KalmanNet achieves the highest overall accuracy, whereas the Shadow Student offers an excellent trade-off between computational efficiency and estimation performance, making it well suited for deployment on embedded and resource-constrained robotic platforms.

---

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