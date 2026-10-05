"""Unit tests for Safe Remediation Engine (Phase 4).

Verifies:
1. Policy validation for asset criticalities (CRITICAL, HIGH, MEDIUM, LOW).
2. Action construction with automated rollback plan generation.
3. Safe execution in SIMULATION mode without host modification.
4. Execution in DRY_RUN mode.
5. Reversal and rollback execution.
6. Blockchain audit trail linkage.
7. Serialization of actions and execution records.
"""

from __future__ import annotations

import json
import pytest

from shield_rca.safe_remediation import (
    ExecutionMode,
    RemediationAction,
    RemediationActionType,
    RemediationExecutionRecord,
    RemediationPolicyEngine,
    RemediationStatus,
    RollbackAction,
    SafeRemediationExecutor,
)
from shield_rca.world_model.trigger_gate import ActionSafety, AssetCriticality


@pytest.fixture
def policy_engine() -> RemediationPolicyEngine:
    return RemediationPolicyEngine()


@pytest.fixture
def simulation_executor() -> SafeRemediationExecutor:
    return SafeRemediationExecutor(mode=ExecutionMode.SIMULATION)


def test_asset_criticality_policy_enforcement(policy_engine: RemediationPolicyEngine):
    """1. Critical assets must prohibit disruptive restarts/quarantines but allow rate-limiting."""
    # CRITICAL asset prohibits QUARANTINE_SERVICE and RESTART_SERVICE
    ok1, reason1 = policy_engine.validate_action_for_asset(
        RemediationActionType.QUARANTINE_SERVICE, AssetCriticality.CRITICAL
    )
    assert not ok1
    assert "prohibited on CRITICAL asset" in reason1

    ok2, reason2 = policy_engine.validate_action_for_asset(
        RemediationActionType.RESTART_SERVICE, AssetCriticality.CRITICAL
    )
    assert not ok2
    assert "prohibited on CRITICAL asset" in reason2

    # CRITICAL asset permits non-disruptive RATE_LIMIT_IP and MONITOR_TELEMETRY
    ok3, _ = policy_engine.validate_action_for_asset(
        RemediationActionType.RATE_LIMIT_IP, AssetCriticality.CRITICAL
    )
    assert ok3

    ok4, _ = policy_engine.validate_action_for_asset(
        RemediationActionType.MONITOR_TELEMETRY, AssetCriticality.CRITICAL
    )
    assert ok4

    # HIGH asset permits BLOCK_IP but prohibits QUARANTINE_SERVICE
    ok5, _ = policy_engine.validate_action_for_asset(
        RemediationActionType.BLOCK_IP, AssetCriticality.HIGH
    )
    assert ok5

    ok6, _ = policy_engine.validate_action_for_asset(
        RemediationActionType.QUARANTINE_SERVICE, AssetCriticality.HIGH
    )
    assert not ok6


def test_action_creation_with_rollback(policy_engine: RemediationPolicyEngine):
    """2. Constructing an action must generate its exact automated rollback counterpart."""
    # BLOCK_IP forward action
    action = policy_engine.create_remediation_action(
        action_type=RemediationActionType.BLOCK_IP,
        target="192.168.1.150",
        asset_criticality=AssetCriticality.MEDIUM,
    )
    assert action.action_type == RemediationActionType.BLOCK_IP
    assert action.target == "192.168.1.150"
    assert action.status == RemediationStatus.PENDING
    assert action.safety_classification == ActionSafety.SAFE

    # Rollback must be UNBLOCK_IP
    rollback = action.rollback
    assert rollback.action_type == RemediationActionType.UNBLOCK_IP
    assert rollback.target == "192.168.1.150"

    # RATE_LIMIT_IP forward action -> rollback REMOVE_RATE_LIMIT
    rate_action = policy_engine.create_remediation_action(
        action_type=RemediationActionType.RATE_LIMIT_IP,
        target="10.0.0.8",
        parameters={"rate_limit_pps": 500},
    )
    assert rate_action.rollback.action_type == RemediationActionType.REMOVE_RATE_LIMIT
    assert rate_action.rollback.target == "10.0.0.8"


def test_safe_simulation_execution(
    policy_engine: RemediationPolicyEngine,
    simulation_executor: SafeRemediationExecutor,
):
    """3. Execution in simulation mode succeeds without altering host OS."""
    action = policy_engine.create_remediation_action(
        action_type=RemediationActionType.RATE_LIMIT_IP,
        target="198.51.100.5",
        parameters={"rate_limit_pps": 1000},
    )

    record = simulation_executor.execute_action(action, incident_id="INC-TEST-SIM-01")

    assert record.status == RemediationStatus.COMPLETED
    assert record.mode == ExecutionMode.SIMULATION
    assert "(SIMULATED)" in record.output_message
    assert record.blockchain_hash is not None
    assert record.blockchain_block is not None
    assert record.execution_time_ms >= 0.0


def test_dry_run_mode(policy_engine: RemediationPolicyEngine):
    """4. Execution in DRY_RUN mode validates plans without OS modifications."""
    executor = SafeRemediationExecutor(mode=ExecutionMode.DRY_RUN)
    action = policy_engine.create_remediation_action(
        action_type=RemediationActionType.BLOCK_IP,
        target="203.0.113.10",
    )

    record = executor.execute_action(action, incident_id="INC-DRYRUN-01")
    assert record.status == RemediationStatus.COMPLETED
    assert record.mode == ExecutionMode.DRY_RUN
    assert "(SIMULATED)" in record.output_message


def test_automated_rollback_execution(
    policy_engine: RemediationPolicyEngine,
    simulation_executor: SafeRemediationExecutor,
):
    """5. Rollback execution cleanly reverses previously completed action."""
    action = policy_engine.create_remediation_action(
        action_type=RemediationActionType.BLOCK_IP,
        target="192.0.2.1",
    )
    exec_record = simulation_executor.execute_action(action, incident_id="INC-REVERSE-01")
    assert exec_record.status == RemediationStatus.COMPLETED

    rollback_record = simulation_executor.execute_rollback(exec_record)
    assert rollback_record.status == RemediationStatus.ROLLED_BACK
    assert rollback_record.action.action_type == RemediationActionType.UNBLOCK_IP
    assert "[ROLLBACK]" in rollback_record.output_message
    assert rollback_record.blockchain_hash is not None


def test_remediation_serialization(
    policy_engine: RemediationPolicyEngine,
    simulation_executor: SafeRemediationExecutor,
):
    """6. All remediation actions and execution records are 100% JSON-serializable."""
    action = policy_engine.create_remediation_action(
        action_type=RemediationActionType.RESTART_SERVICE,
        target="nginx_proxy",
        asset_criticality=AssetCriticality.LOW,
    )
    record = simulation_executor.execute_action(action, incident_id="INC-SERIAL-01")

    serialized = json.dumps(record.to_dict())
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)

    assert deserialized["incident_id"] == "INC-SERIAL-01"
    assert deserialized["status"] == "COMPLETED"
    assert deserialized["action"]["action_type"] == "RESTART_SERVICE"
    assert deserialized["action"]["rollback"]["action_type"] == "MONITOR_TELEMETRY"
