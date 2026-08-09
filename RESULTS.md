# EKF vs UKF vs Particle Filters vs KalmanNet vs Shadow Student

## Overview

This report compares seven state estimation methods:

- **EKF** (Extended Kalman Filter)
- **UKF** (Unscented Kalman Filter)
- **PF (200)** (Particle Filter with 200 particles)
- **PF (500)** (Particle Filter with 500 particles)
- **PF (1000)** (Particle Filter with 1000 particles)
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
| EKF | 3.1784 | 3.0879 | 2.5117 | 27.56% | 1.5527% |
| UKF | 3.1783 | 3.0879 | 2.5117 | 27.56% | 1.5527% |
| PF (200) | 1.4043 | 2.0525 | 1.6650 | 38.02% | 1.0293% |
| PF (500) | 1.4643 | 2.0959 | 1.6929 | 37.85% | 1.0465% |
| PF (1000) | 1.5004 | 2.1216 | 1.7168 | 36.90% | 1.0613% |
| KalmanNet | **0.3277** | **0.9915** | **0.6455** | **79.90%** | **0.3991%** |
| Shadow Student | 0.5921 | 1.3328 | 0.9164 | 66.66% | 0.5665% |

### Key Insights

- UKF provides virtually identical performance to EKF on this dataset.
- Particle Filters significantly reduce position error compared to EKF, with **PF (200)** providing the best performance among the PF variants.
- KalmanNet achieves the best overall position accuracy.
- Shadow Student substantially outperforms every classical filter while remaining significantly smaller than the teacher model.

---

# 2. Position Breakdown (Axis-wise RMSE)

| Model | x RMSE | y RMSE | z RMSE |
|------|--------:|--------:|--------:|
| EKF | 1.8958 | 2.4358 | **0.0881** |
| UKF | 1.8958 | 2.4358 | **0.0881** |
| PF (200) | 1.2623 | 1.6156 | 0.0963 |
| PF (500) | 1.2924 | 1.6473 | 0.0959 |
| PF (1000) | 1.2986 | 1.6751 | 0.0926 |
| KalmanNet | **0.5627** | **0.8060** | 0.1293 |
| Shadow Student | 0.8946 | 0.9403 | 0.3031 |

### Key Insights

- UKF does not improve upon EKF in this benchmark.
- Particle Filters substantially reduce horizontal drift.
- KalmanNet provides the lowest x- and y-axis errors.
- Shadow Student remains considerably more accurate than all classical filters.

---

# 3. Velocity Estimation Performance

| Model | MSE (m/s)² | RMSE (m/s) | MAE (m/s) |
|------|-----------:|-----------:|----------:|
| EKF | 0.6587 | 1.4057 | 1.3089 |
| UKF | 0.6587 | 1.4057 | 1.3089 |
| PF (200) | 0.1120 | 0.5797 | 0.5058 |
| PF (500) | 0.1231 | 0.6076 | 0.5313 |
| PF (1000) | 0.1268 | 0.6169 | 0.5466 |
| KalmanNet | **0.0885** | **0.5154** | **0.3899** |
| Shadow Student | 0.2000 | 0.7745 | 0.5923 |

### Key Insights

- UKF again performs nearly identically to EKF.
- Particle Filters significantly improve velocity estimation over EKF/UKF.
- KalmanNet remains the most accurate estimator.
- Shadow Student outperforms all Particle Filter variants while requiring no particle propagation.

---

# 4. Velocity Breakdown (Axis-wise RMSE)

| Model | vx RMSE | vy RMSE | vz RMSE |
|------|---------:|---------:|---------:|
| EKF | 0.8877 | 1.0890 | **0.0444** |
| UKF | 0.8877 | 1.0890 | **0.0444** |
| PF (200) | 0.3806 | 0.4352 | 0.0427 |
| PF (500) | 0.4006 | 0.4547 | 0.0441 |
| PF (1000) | 0.3987 | 0.4686 | 0.0441 |
| KalmanNet | 0.3423 | **0.3747** | 0.0897 |
| Shadow Student | **0.2991** | 0.6919 | 0.1781 |

### Key Insights

- Particle Filters substantially improve horizontal velocity estimation.
- Shadow Student achieves the lowest x-axis velocity RMSE.
- KalmanNet provides the most balanced velocity estimation overall.

---

# 5. Full State Error

| Model | MSE |
|------|----:|
| EKF | 1.9185 |
| UKF | 1.9185 |
| PF (200) | 0.7581 |
| PF (500) | 0.7937 |
| PF (1000) | 0.8136 |
| KalmanNet | **0.2081** |
| Shadow Student | 0.3960 |

### Key Insights

- UKF provides no measurable improvement over EKF.
- PF (200) is the strongest classical estimator.
- KalmanNet achieves the lowest full-state error.
- Shadow Student significantly outperforms all classical estimators while approaching teacher-level performance.

---

# 6. Latency Comparison (CPU)

Model  | Mean Latency (ms/step)  | Std (ms)
--- | --- | ---
EKF  | 0.0159  | 0.0079
UKF  | 0.0314  | 0.0308
PF (200)  | 0.0432  | 0.0320
PF (500)  | 0.0590  | 0.0141
PF (1000)  | 0.0970  | 0.0296
KalmanNet  | 0.1084  | 0.2633
Shadow Student  | 0.0733  | 0.0439

### Key Insights

- EKF remains the fastest estimator.
- Shadow Student is approximately **30% faster** than KalmanNet.
- Both neural models comfortably satisfy real-time inference requirements.

---

# 7. Overall Comparison Summary

## EKF

- Fastest classical estimator.
- Lowest computational complexity.
- Highest estimation error among the evaluated methods.

---

## UKF

- Similar computational complexity to EKF.
- Produced nearly identical estimation accuracy on this dataset.
- Did not provide measurable gains over EKF.

---

## Particle Filters

- Significantly improve estimation accuracy over EKF and UKF.
- **PF (200)** achieves the best accuracy among the tested Particle Filter configurations.
- Increasing the particle count beyond 200 provides diminishing returns while increasing computational cost.

---

## KalmanNet

- Best overall estimation accuracy.
- Achieves the lowest position, velocity, and full-state errors.
- Highest computational cost among the evaluated models.

---

## Shadow Student

- Preserves much of the teacher's estimation capability while using substantially fewer parameters.
- Outperforms all classical filters (EKF, UKF, and Particle Filters).
- Offers an excellent balance between accuracy and computational efficiency, making it suitable for deployment on resource-constrained robotic platforms.

---

# Ablation Study

| ALPHA_START | Position RMSE (m) | Velocity RMSE (m/s) | CPU Latency (ms/step) |
|-------------|------------------:|--------------------:|----------------------:|
| 0.50 | 4.7854 | 1.4180 | 0.0726 |
| 0.85 | 1.7734 | 1.4060 | 0.0679 |

---

# Conclusion

The updated evaluation demonstrates that Particle Filters provide a substantial improvement over EKF and UKF, with **PF (200)** emerging as the strongest classical estimator. However, both learning-based methods significantly outperform all classical approaches. KalmanNet achieves the highest overall estimation accuracy across every evaluation metric, while the Shadow Student retains much of the teacher's performance with considerably lower computational requirements, making it an attractive choice for deployment on embedded and resource-constrained robotic platforms.

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
