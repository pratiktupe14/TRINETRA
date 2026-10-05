# TRINETRA Predictive Explainability Specification

## 1. Purpose
The Predictive Explainability Engine implements deterministic **Forecast Driver Analysis** and **State Delta Attribution** for the TRINETRA system.

It answers the core SOC analyst questions:
- *"Why did the World Model predict a high-risk trajectory?"*
- *"Which specific network-state metrics are driving the forecast?"*
- *"Which architectural feature group (TCP Flags, Macro, Endpoints, Continuous, Shape) dominates the predicted deviation?"*
- *"What is the signed direction of change (surge vs drop)?"*

---

## 2. Honest XAI Disclosure: State Delta Attribution vs. SHAP
> [!NOTE]
> **No Fake SHAP Disclosure:**
> This module implements **State Delta Attribution**, a first-order trajectory decomposition method that measures the exact signed and absolute deviations between the observed state $S_t$ and the predicted state $\hat{S}_{t+k}$.
>
> It does **NOT** compute Shapley values (SHAP) or train auxiliary surrogate regression models. We avoid claiming SHAP when standard mathematical trajectory attribution is performed.

---

## 3. Inputs
- `current_state`: 1D NumPy array of shape (280,) representing $S_t$.
- `predicted_state`: 1D NumPy array of shape (280,) representing $\hat{S}_{t+k}$.
- `horizon`: Integer future prediction horizon $k \ge 1$.
- `top_k`: Number of top individual driving features to extract (default 5).

---

## 4. Processing Pipeline
1. **Validation:** Checks vector dimensions (280) and asserts finite numbers.
2. **Delta Computation:**
   $$\Delta S_i = \hat{S}_{t+k, i} - S_{t, i} \quad \forall i \in \{0, \dots, 279\}$$
   $$|\Delta S_i| = \left|\hat{S}_{t+k, i} - S_{t, i}\right|$$
   $$\text{Total Drift} = \sum_{i=0}^{279} |\Delta S_i|$$
3. **Signed Direction Classification:**
   - $\Delta S_i > 10^{-6} \implies \text{"INCREASE"}$
   - $\Delta S_i < -10^{-6} \implies \text{"DECREASE"}$
   - Otherwise $\implies \text{"UNCHANGED"}$
4. **Relative Share Allocation:**
   $$\text{relative\_share}_i = \frac{|\Delta S_i|}{\text{Total Drift} + \epsilon}$$
5. **Deterministic Lexicographical Ranking:**
   Features are sorted primarily by absolute delta magnitude descending, with ties broken deterministically by feature index ascending:
   $$\text{order} = \text{lexsort}\left(\text{index}, -|\Delta S|\right)$$
6. **Architectural Group Attribution:**
   Aggregates absolute deviations across the 5 canonical state groups:
   - Continuous Dynamics ($i \in [0, 150)$)
   - TCP Flags ($i \in [150, 198)$)
   - Distribution Shape ($i \in [198, 264)$)
   - Endpoint & Protocol Diversity ($i \in [264, 270)$)
   - Macro Indicators ($i \in [270, 280)$)
   Computes each group's relative share of the total drift and extracts its top 3 internal driving features.
7. **Synthesis of Human-Readable Narrative:**
   Generates a clear summary sentence highlighting the dominant group, its percentage contribution, and the top individual feature driver.

---

## 5. Outputs
- `ForecastExplanation`:
  - `explanation_method`: `"State Delta Attribution"`
  - `horizon`: int ($k$)
  - `total_drift_magnitude`: float
  - `top_features`: list of `FeatureChange`:
    - `feature_name`: str
    - `group`: str
    - `current_value`: float
    - `predicted_value`: float
    - `delta`: float (signed)
    - `magnitude`: float ($|\Delta|$)
    - `direction`: str (`INCREASE` / `DECREASE` / `UNCHANGED`)
    - `relative_share`: float in $[0.0, 1.0]$
  - `group_attributions`: list of `GroupAttribution` (sorted descending by absolute delta):
    - `group_name`: str
    - `absolute_delta_sum`: float
    - `relative_share`: float in $[0.0, 1.0]$
    - `dominant_features`: list of top 3 features in this group
  - `summary_text`: str
  - `to_dict()`: method returning a JSON-serializable dictionary.

---

## 6. Files
- Implementation: `src/shield_rca/world_model/explainability.py`
- Tests: `tests/test_explainability.py`

---

## 7. Configuration
Configurable via `ForecastExplainabilityEngine` constructor:
- `default_top_k`: Number of top features returned by default (default `5`, must be $\ge 1$).

---

## 8. Tests
Covered by 7 unit tests in `tests/test_explainability.py`:
- `test_zero_change_state`: Verifies zero drift produces `UNCHANGED` direction and quiescent summary.
- `test_non_zero_state_change`: Verifies correct signed detection (`INCREASE` vs `DECREASE`) and exact magnitudes.
- `test_top_k_selection`: Tests extracting custom $K$ features ($K=3, 8$).
- `test_group_attribution_completeness`: Confirms all 5 groups are present, group sums equal total drift, and relative shares sum to 1.0.
- `test_deterministic_ranking_and_stable_tie_breaking`: Verifies deterministic tie-breaking on tied deltas.
- `test_serialization_to_dict`: Verifies dictionary output for downstream UI and API endpoints.
- `test_invalid_inputs_rejection`: Rejects invalid parameters ($k < 1$, wrong dimensions, NaN/Inf).

---

## 9. Limitations
1. **First-Order Only:** Measures direct state coordinate deviations rather than non-linear higher-order interactions between features.
2. **Not Game-Theoretic:** Does not compute Shapley marginal contributions against an all-permutations baseline.
3. **Model-Agnostic to World Model Internals:** Explains the predicted output delta rather than inspecting internal neural network hidden activations.

---

## 10. Implementation Status
`IMPLEMENTED` — Fully implemented, type-hinted, and validated with 7 passing unit tests.
