# TRINETRA Offline Learning Pipeline & Candidate Model Evaluation

## 1. Overview & Separation of Concerns

The TRINETRA RLM Learning Pipeline is strictly decoupled from the live operational response engine. Production model updates occur **only** through controlled, offline training and a formal evaluation gate.

```
                  artifacts/rlm/learning_records.jsonl
                                  │
                                  ▼
                     [1. Ingestion & Validation]
                                  │
                                  ▼
                   [2. Dataset Construction & Split]
                     (Train 70% / Val 15% / Test 15%)
                                  │
                                  ▼
                   [3. Candidate Model Training]
                     artifacts/world_model/candidates/<candidate_id>/
                                  │
                                  ▼
                   [4. Model Evaluation Gate]
                     (Candidate vs Production Baseline)
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
                 REJECT                      ACCEPT
          (Candidate MAE worse)       (Candidate MAE beats prod)
                    │                           │
                    ▼                           ▼
            Archived/Flagged            [5. Explicit Release Gate]
                                        Requires Human Approval
                                                │
                                                ▼
                                        Promoted to Production
                                        (Old model backed up)
```

---

## 2. Pipeline Execution Stages

### Stage 1: Ingestion & Validation
The pipeline reads all historical `LearningRecord` entries from `artifacts/rlm/learning_records.jsonl`. Each entry is validated for:
- Non-empty IDs (`record_id`, `incident_id`, `trace_id`).
- Absence of `NaN` or `Inf` in metrics or state vectors.
- Valid probability bounds $[0.0, 1.0]$.
- Valid enum memberships.

### Stage 2: Dataset Construction
Pairs of $(\mathbf{s}_{\text{curr}}, \mathbf{s}_{\text{target}})$ are extracted from records where real state references exist. If the number of real incident state pairs is limited, the dataset builder supplements them with deterministic baseline dynamics across multiple prediction horizons ($k \in \{1, 3, 5\}$). A chronological 70% / 15% / 15% train/val/test split is applied.

### Stage 3: Candidate Model Training
- **Candidate Isolation:** The model is trained and saved exclusively into an isolated candidate directory:
  `artifacts/world_model/candidates/<candidate_id>/`
- **Zero Production Overwrite:** The active production weights (`artifacts/world_model/best_model.pt`) are **never** modified or touched during training.
- **Architecture Integrity:** The model retains the canonical `StateResidualDynamicsModel` (Pre-LayerNorm Residual Dynamics MLP) with identical input dimension ($\text{STATE\_FEATURE\_DIM} = 280$).

### Stage 4: Model Evaluation Gate
The candidate model and the current production model are evaluated side-by-side on the identical test split using `WorldModelEvaluator`:
- **Candidate MAE:** Average Mean Absolute Error of candidate model across horizons.
- **Production Baseline MAE:** Average MAE of the active production model.
- **Decision Logic:**
  - **`ACCEPT`:** $\text{Candidate MAE} \le \text{Production MAE}$. The candidate demonstrates improved or equivalent predictive accuracy without regression.
  - **`REJECT`:** $\text{Candidate MAE} > \text{Production MAE}$. The candidate exhibits accuracy regression and is barred from release.
  - **`INCONCLUSIVE`:** Insufficient test pairs ($< 5$) to provide statistically meaningful validation.

### Stage 5: Explicit Release Gate
Candidate deployment is governed by strict safety checks:
1. **Human/Operator Approval Required:** Automated background jobs cannot deploy models. The operator must supply explicit confirmation (`--approve-release` or `operator_approved=True`).
2. **Evaluation Gate Clearance:** Releasing a candidate with decision `REJECT` raises an exception unless explicitly overridden with `--force-release`.
3. **Production Model Backup:** When promoted, the existing production weights, config, and manifest are archived to `artifacts/world_model/archive/world_model_backup_<timestamp>/`.
4. **Promotion:** Candidate artifacts are copied to `artifacts/world_model/best_model.pt`.

---

## 3. CLI Invocation (`scripts/run_rlm_learning.py`)

### Offline Candidate Training & Evaluation:
```powershell
python scripts/run_rlm_learning.py --epochs 5 --candidate-id cand-2026-v2
```
Output:
```
==========================================================
   TRINETRA AI: RLM OFFLINE CANDIDATE LEARNING PIPELINE   
==========================================================
Loading RLM learning records from artifacts/rlm/learning_records.jsonl...
Successfully loaded and validated 12 learning records.
Constructing candidate multi-horizon future datasets...
Dataset ready: 42 train pairs, 9 val pairs, 9 test pairs.

Training candidate model 'cand-2026-v2' for 5 epochs...
Candidate model saved to: artifacts/world_model/candidates/cand-2026-v2

Running Evaluation Gate (Candidate vs. Production Baseline)...

--- EVALUATION GATE REPORT ---
Candidate ID:        cand-2026-v2
Current Model MAE:   0.023771
Candidate Model MAE: 0.021450
Improvement:         +9.76%
DECISION:            ACCEPT
  * Candidate MAE (0.021450) meets or beats current production MAE (0.023771) by +9.76%.

NOTE: Candidate model remains isolated in artifacts/world_model/candidates/cand-2026-v2.
Production weights (artifacts/world_model/best_model.pt) were NOT modified.
To promote this candidate after verification, rerun with --approve-release.
==========================================================
```

### Explicit Promotion with Operator Approval:
```powershell
python scripts/run_rlm_learning.py --candidate-id cand-2026-v2 --approve-release
```
Promotes the candidate to `artifacts/world_model/best_model.pt` while backing up the original weights to `artifacts/world_model/archive/`.
