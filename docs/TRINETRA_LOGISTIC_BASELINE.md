# TRINETRA Logistic Regression Baseline & SIH Benchmarking

## 1. Purpose & SIH Requirement

The Smart India Hackathon (SIH) problem statement (**SIH26153 — AI based Network Attack Forecasting from Network Traffic Data**) explicitly mandates establishing a classical Machine Learning benchmark using **Logistic Regression** evaluated on four standardized metrics:
1. **Precision**
2. **Recall**
3. **F1 Score**
4. **False Positive Rate (FPR)**

---

## 2. Fundamental Task Distinction: Classification vs. Dynamics Forecasting

A critical conceptual error in machine learning benchmarking is conflating continuous dynamical system regression with static discrete pattern classification:

| Dimension | TRINETRA World Model (Track B) | Logistic Regression Baseline (Track A) |
|:---|:---|:---|
| **Core Task** | Continuous future-state trajectory prediction ($S_t \to S_{t+k}$) | Supervised attack / anomaly classification ($X_t \to y_t$) |
| **Output Type** | 280-dimensional continuous feature vectors in $\mathbb{R}^{280}$ | Discrete binary decision ($0 = \text{BENIGN}, 1 = \text{ATTACK}$) |
| **Primary Evaluation Metrics** | Mean Absolute Error (MAE), Mean Squared Error (MSE), Huber loss | Precision, Recall, F1 Score, False Positive Rate (FPR), ROC-AUC |
| **Temporal Role** | Long-horizon dynamical forecast ($k \in \{1, 3, 5\}$) | Point-in-time discriminatory classification |

> [!IMPORTANT]
> **Complementary, Not Interchangeable:**  
> Logistic Regression provides a classical supervised classification baseline, while the TRINETRA World Model evaluates temporal future-state prediction. These are complementary evaluation tasks rather than directly interchangeable metrics.

---

## 3. Supervised Classification Task Definition

- **Input:** Observation window or state feature vector $X_t$ representing network traffic dynamics over window $T$ (derived from the 78 canonical flow features).
- **Target:** Ground-truth classification label $y_t \in \{0, 1\}$:
  - `0`: **BENIGN** (Normal operational traffic)
  - `1`: **ATTACK** (Malicious intrusion pattern, e.g., DDoS, DoS, Brute Force, PortScan, Botnet)
- **Decision Threshold:** Default $\tau = 0.50$, where $\hat{y} = 1$ if $P(y=1 \mid X) \ge \tau$.

---

## 4. Leakage-Safe Chronological Partitioning

To avoid lookahead bias and temporal data leakage, random shuffling is strictly prohibited:
- **70% TRAIN:** Earlier chronological time segment ($t \in [t_0, t_{\text{train}}]$).
- **15% VALIDATION:** Intermediate time segment ($t \in [t_{\text{train}}, t_{\text{val}}]$).
- **15% TEST:** Latest time segment ($t \in [t_{\text{val}}, t_{\text{test}}]$).

### Leakage Safeguards
1. **Scaler Isolation:** `StandardScaler` is fitted **strictly on training samples** ($X_{\text{train}}$). Test and validation data are transformed using the training parameters; test distributions never contaminate preprocessing.
2. **Zero Boundary Overlap:** Partition boundaries are strictly monotonic in time ($\max(t_{\text{train}}) \le \min(t_{\text{val}}) \le \max(t_{\text{val}}) \le \min(t_{\text{test}})$).
3. **No Target Leakage:** Identifier columns (`src_ip`, `dst_ip`, `src_port`, `dst_port`, `flow_id`, `timestamp`) are excluded from feature matrices.

---

## 5. Mathematical Metric Definitions

For test predictions against ground-truth labels:
- **True Positives (TP):** Actual Attack correctly predicted as Attack.
- **True Negatives (TN):** Actual Benign correctly predicted as Benign.
- **False Positives (FP):** Actual Benign falsely flagged as Attack.
- **False Negatives (FN):** Actual Attack missed and classified as Benign.

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{False Positive Rate (FPR)} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

### Confusion Matrix Layout:
$$\begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}$$

---

## 6. Model Hyperparameters & Imbalance Strategy

- **Algorithm:** `sklearn.linear_model.LogisticRegression`
- **Class Imbalance Strategy:** `class_weight="balanced"` dynamically weights loss inversely proportional to class frequencies:
  $$w_c = \frac{N}{2 \cdot N_c}$$
- **Solver:** `liblinear` (or `lbfgs`)
- **Regularization:** $C = 1.0$ (L2 penalty)
- **Max Iterations:** 1000
- **Random State:** 42 (deterministic)

---

## 7. Artifact Storage Architecture

Artifacts are saved strictly in `artifacts/logistic_baseline/`:
- `model.joblib`: Serialized trained `LogisticRegression` model.
- `scaler.joblib`: Serialized `StandardScaler` fitted on training data.
- `model_config.json`: Hyperparameters and feature settings.
- `model_manifest.json`: Metadata, creation timestamp, feature dimensions, and test metrics.
- `metrics.json`: JSON output of `ClassificationMetricsReport`.
- `confusion_matrix.json`: Detailed TP, TN, FP, FN counts and FPR.

*Zero World Model Overwrite:* The production World Model (`artifacts/world_model/best_model.pt`) is completely separate and untouched.

---

## 8. Reproducibility & CLI Usage

Run the benchmark CLI:
```powershell
python scripts/run_logistic_baseline.py --output-dir artifacts/logistic_baseline --samples 300 --window-size 20
```
