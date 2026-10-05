# TRINETRA Unseen Attack Evaluation & Zero-Day Generalization

## 1. Objective & Scientific Grounding

The Smart India Hackathon (SIH) problem statement (**SIH26153 — AI based Network Attack Forecasting from Network Traffic Data**) mandates evaluating whether the predictive defense system can **generalize to unseen attack patterns**.

### Core Scientific Philosophy
> [!IMPORTANT]
> **What Unseen Attack Evaluation Proves:**  
> Unseen attack evaluation measures generalization under a controlled, out-of-distribution shift. It establishes quantitatively how model accuracy and lead time degrade when a novel attack category is encountered.  
> **What It Does NOT Prove:**  
> It does **not** prove universal zero-day detection capability, nor does it guarantee immunity to all unknown cyber threats.

---

## 2. Definition of "Unseen Attack"

The evaluation protocol enforces a strict three-way partitioning of attack categories:

| Data Subset | Category Composition | Role in Experiment |
|:---|:---|:---|
| **Training Set** | `BENIGN` + Known Attack Categories (e.g., DDoS, PortScan, FTP-Patator) | Optimization of model weights and feature scaling parameters |
| **Validation Set** | `BENIGN` + Known Attack Categories | Hyperparameter validation and early stopping (held-out attack barred) |
| **Known Test Set** | `BENIGN` + Known Attack Categories | Baseline performance benchmark on familiar attack signatures |
| **Unseen Test Set** | `BENIGN` + Withheld Attack Category (e.g., Bot, Infiltration) | Measurement of out-of-distribution generalization and degradation |

---

## 3. Why Random Row-Level Holdout Is Insufficient

Standard random train/test splitting (e.g. `train_test_split(shuffle=True)`) is fundamentally invalid for zero-day evaluation:
1. **Temporal Autocorrelation:** Network attack flows are temporally linked burst sequences. If 80% of flows from a DoS burst are placed in training and 20% in testing, the model merely memorizes the IP/port patterns of that specific attack episode.
2. **Category Contamination:** If the withheld category appears anywhere in training, the test is no longer an unseen attack test.
3. **Category-Level Holdout Required:** To evaluate genuine zero-day generalization, the **entire** attack category block must be isolated from the training distribution.

---

## 4. Leakage Safeguards & Integrity Enforcement

The `UnseenAttackDataset` and `UnseenAttackEvaluator` implement programmatic leakage guards:

1. **Intersection Disjointness Check:**
   $$\text{intersection}(\text{training\_attack\_categories}, \text{held\_out\_attack\_categories}) = \emptyset$$
   If any held-out sample or label is detected in training or validation splits, execution fails immediately (`ValueError: LEAKAGE DETECTED`).
2. **Scaler Isolation:**
   `StandardScaler` parameters ($\mu, \sigma^2$) are fitted **strictly on training samples** (`BENIGN` + known attacks). Held-out attack samples are transformed using the fitted training parameters; they never influence normalization.
3. **World Model Production Isolation:**
   Production World Model weights (`artifacts/world_model/best_model.pt`) are preserved and verified untouched during all holdout experiments.

---

## 5. Evaluation Tracks & Metrics

### Track A: Classical Classification (Logistic Regression)
- Evaluates the ability of linear decision boundaries to flag withheld attack patterns.
- Metrics measured on both Known Test and Unseen Test:
  - Precision: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
  - Recall: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
  - F1 Score: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
  - False Positive Rate (FPR): $\frac{\text{FP}}{\text{FP} + \text{TN}}$
- **Performance Degradation Deltas:**
  $$\Delta \text{F1} = \text{F1}_{\text{known}} - \text{F1}_{\text{unseen}}$$
  $$\Delta \text{Recall} = \text{Recall}_{\text{known}} - \text{Recall}_{\text{unseen}}$$
  $$\Delta \text{FPR} = \text{FPR}_{\text{unseen}} - \text{FPR}_{\text{known}}$$

### Track B: Dynamical Forecasting (TRINETRA World Model)
- Evaluates the Pre-LayerNorm Residual Dynamics MLP on trajectory sequences containing unseen attack transitions.
- Metrics: Mean Absolute Error ($\text{MAE}$) across forecast horizons $k \in \{1, 3, 5\}$.
- Compares trajectory drift on normal vs novel attack distributions.

---

## 6. CLI Execution

Run an unseen attack experiment withholding `Bot`:
```powershell
python scripts/run_unseen_attack_evaluation.py --holdout Bot --output-dir artifacts/unseen_attack
```

Output:
```
==========================================================
   TRINETRA AI: UNSEEN ATTACK GENERALIZATION EVALUATION   
==========================================================
EVALUATION PROTOCOL: Zero-Day-Style Category Holdout
STATUS: Synthetic Software Validation (Pipeline Verification)
HELD-OUT CATEGORIES: ['Bot']
----------------------------------------------------------
Configuration validated. Canonical held-out: ['Bot']
Training samples:   265 (Known Attacks + Benign)
Validation samples: 57 (Known Attacks + Benign)
Known Test samples: 43 (Known Attacks + Benign)
Unseen Test samples:75 (Unseen [Bot] + Benign)
LEAKAGE CHECK: PASSED. Zero held-out samples in training/validation splits.

--- GENERALIZATION BENCHMARK RESULTS ---
Metric                    | Known Attack    | Unseen Attack   | Degradation (Delta)
---------------------------------------------------------------------------
Precision                 | 0.9524          | 0.0000          | +0.7143
Recall                    | 0.7143          | 0.0000          | +0.7143
F1 Score                  | 0.8163          | 0.0000          | +0.8163
False Positive Rate       | 0.0667          | 0.0667          | +0.0000

Known Attack Confusion Matrix:  [[14, 1], [8, 20]]
Unseen Attack Confusion Matrix: [[14, 1], [60, 0]]

Artifacts saved successfully to: artifacts\unseen_attack\unseen_bot
==========================================================
```

---

## 7. Limitations & Honest Assessment
1. **Linear Classification Fragility:** Supervised linear classifiers without retraining typically suffer near-total recall degradation ($\text{Recall} \to 0.0$) when novel attack signatures do not project onto previously learned linear hyperplanes.
2. **Synthetic Software Validation:** The numbers above confirm the correctness of the holdout framework, metric pipelines, and leakage assertions; real dataset validation on physical CIC-IDS captures remains to be run in Phase 9.
