# TRINETRA Safe Remediation Architecture

## 1. Overview & Core Philosophy

The **TRINETRA Safe Remediation Engine** is the defensive execution enforcement layer of TRINETRA and CyberHeal. Its core objective is to prevent uncontrolled autonomous system mutations while ensuring precise, automated mitigation of forecasted or active network threats.

### Safety Invariants
1. **Guardian Authority Is Inviolable:** Autonomous actions cannot bypass Guardian safety checks.
2. **Asset Criticality Enforced:** Operational stability takes precedence over aggressive remediation; critical infrastructure assets prohibit disruptive containment (e.g. direct service restarts or blind port closures).
3. **Deterministic Rollback Pairs:** Every proposed forward remediation action must generate a reversible, verified rollback plan before approval or execution.
4. **Zero Unintended Host Modification (`CYBERHEAL_SAFE_MODE`):** Default execution mode operates in non-destructive simulation or dry-run, ensuring developer environments and production host firewalls are never modified without explicit opt-in.
5. **No Direct Firewall Exposure:** TRINETRA does not act directly as a raw firewall interface; actions are validated, encapsulated in high-level intent objects (`RemediationAction`), verified against policy, and logged to an immutable ledger (`blockchain.json`).

---

## 2. Remediation Taxonomy

### 2.1 Action Types and Rollback Inverses

| Forward Remediation Action | Safety Level | Operational Impact | Deterministic Rollback Action |
|:---|:---|:---|:---|
| `MONITOR_TELEMETRY` | `SAFE` | Zero impact (telemetry observation only) | `NO_OP` |
| `RATE_LIMIT_IP` | `SAFE` / `RESTRICTED` | Low (traffic bandwidth throttling on target IP) | `REMOVE_RATE_LIMIT_IP` |
| `BLOCK_IP` | `RESTRICTED` | Medium (drops traffic from malicious IP) | `UNBLOCK_IP` |
| `QUARANTINE_SERVICE` | `RESTRICTED` | High (isolates compromised network service) | `UNQUARANTINE_SERVICE` |
| `RESTART_SERVICE` | `UNSAFE` | Disruptive (transient service unavailability) | `NO_OP` / Health Check |

### 2.2 Safety Classifications
- **`SAFE`:** Non-disruptive actions that do not affect legitimate client connectivity (e.g., telemetry observation, modest rate-limiting).
- **`RESTRICTED`:** Actions that drop packets or restrict specific remote endpoints; acceptable for automated response under high confidence and non-critical assets.
- **`UNSAFE`:** Potentially service-impacting interventions that interrupt availability; strictly require human authorization or low asset criticality.

---

## 3. Policy Enforcement Engine (`RemediationPolicyEngine`)

The `RemediationPolicyEngine` inspects proposed remediation plans and enforces asset-criticality constraints prior to execution:

```
Proposed Plan
      │
      ▼
┌───────────────────────────────────────────────┐
│ Policy Rule 1: CRITICAL Asset Protection      │
│  - Prohibits QUARANTINE_SERVICE               │
│  - Prohibits RESTART_SERVICE                  │
│  - Flags for HUMAN_REVIEW if attempted        │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ Policy Rule 2: HIGH Asset Containment Guard   │
│  - Prohibits unconfirmed RESTART_SERVICE      │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ Policy Rule 3: Automated Rollback Generation  │
│  - Synthesizes reverse actions for every step │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
                Compliant Plan + Rollback
```

### Criticality Mapping Table
| Asset Criticality | Allowed Autonomous Actions | Disallowed Autonomous Actions |
|:---|:---|:---|
| `LOW` | `MONITOR_TELEMETRY`, `RATE_LIMIT_IP`, `BLOCK_IP`, `QUARANTINE_SERVICE`, `RESTART_SERVICE` | None |
| `MEDIUM` | `MONITOR_TELEMETRY`, `RATE_LIMIT_IP`, `BLOCK_IP`, `QUARANTINE_SERVICE` | `RESTART_SERVICE` |
| `HIGH` | `MONITOR_TELEMETRY`, `RATE_LIMIT_IP`, `BLOCK_IP` | `QUARANTINE_SERVICE`, `RESTART_SERVICE` |
| `CRITICAL` | `MONITOR_TELEMETRY`, `RATE_LIMIT_IP` | `BLOCK_IP`, `QUARANTINE_SERVICE`, `RESTART_SERVICE` |

---

## 4. Execution Controller (`SafeRemediationExecutor`)

### 4.1 Execution Modes
1. **`SIMULATION`:**
   - Evaluates policy, registers simulated state transitions, and creates synthetic delta metrics.
   - Used for zero-risk test suites, demonstrations, and dry evaluations.
2. **`DRY_RUN`:**
   - Generates the exact CLI commands, firewall rules, and API payloads that would be executed on live hosts.
   - Emits payload records without executing network socket or host modifications.
3. **`LIVE`:**
   - Only accessible when `CYBERHEAL_SAFE_MODE=false`.
   - Modifies host packet filtering rules or container interfaces with strict exception handling and instant rollback triggers on failure.

### 4.2 Automated Rollback Execution
When a remediation fails or post-remediation verification determines the threat was not reduced, the executor invokes `execute_rollback_plan()`:
- Iterates the generated rollback plan in reverse order.
- Restores original firewall tables, removes throttle limiters, and re-enables quarantined services.
- Emits verification audit records to the local blockchain audit log.

---

## 5. Blockchain Audit Trail Integration

All remediation approvals, dry runs, live actions, and rollbacks are hashed and appended to `blockchain.json` via the internal `blockchain_sys`:
- Action type, target IP/service, rationale, and execution status are recorded.
- Cryptographic hash links guarantee tamper-evident incident response logs for SOC compliance and forensic audits.
