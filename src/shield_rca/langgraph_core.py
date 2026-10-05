from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import datetime
from loguru import logger
import time
import subprocess
import platform
import asyncio
import json
import os

try:
    from api.tracing import omium_tracer
except ImportError:
    # Fallback if run outside API context
    class DummyTracer:
        def log_step(self, *args, **kwargs): pass
        def trace_step(self, *args, **kwargs): pass
    omium_tracer = DummyTracer()

DEMO_MODE = True

class AgentState(BaseModel):
    """LangGraph State Object"""
    incident_id: str
    alert_payload: Dict[str, Any]
    normalized_alert: Optional[Dict[str, Any]] = None
    anomalies: List[Dict[str, Any]] = []
    root_cause: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    remediation_plan: List[Dict[str, Any]] = []
    guardian_approved: bool = False
    risk_score: str = "UNKNOWN"
    execution_results: List[Dict[str, Any]] = []
    verification_passed: bool = False
    status: str = "OPEN"
    retry_count: int = 0
    max_retries: int = 3
    escalated: bool = False
    rlm_history: List[Dict[str, Any]] = []
    threat_intel: Optional[Dict[str, Any]] = None
    current_step: str = "INTAKE"

    # Phase 3: TRINETRA Predictive Intelligence Integration Fields (Backward Compatible)
    incident_type: str = "REACTIVE"  # "REACTIVE" or "PREDICTIVE"
    trigger_decision: Optional[Dict[str, Any]] = None
    forecasted_state: Optional[List[float]] = None
    trajectory_risk_score: float = 0.0
    attack_stage: str = "NORMAL"
    top_predictive_features: List[Dict[str, Any]] = []
    asset_criticality: str = "MEDIUM"
    action_safety: str = "UNKNOWN"
    evidence_strength: float = 0.0
    predictive_reasons: List[str] = []
    predictive_blocking_reasons: List[str] = []
    risk_trajectory: Optional[Dict[str, Any]] = None

    # Phase 4: Safe Remediation & Observable Verification Fields
    rollback_plan: List[Dict[str, Any]] = []
    before_metrics: Optional[Dict[str, float]] = None
    after_metrics: Optional[Dict[str, float]] = None
    verification_report: Optional[Dict[str, Any]] = None

    # Phase 5: RLM Feedback & Offline Learning Fields
    learning_record: Optional[Dict[str, Any]] = None
    reflection_result: Optional[str] = None
    learning_recommendation: Optional[str] = None

class BaseAgent:
    """Base class for all CyberHeal AI Agents"""
    def __init__(self, name: str, db_client=None):
        self.name = name
        self.db = db_client

    def log(self, incident_id: str, message: str):
        logger.info(f"[{self.name}] {message}")
        if self.db and incident_id:
            # DB Constraint Guard: Map unknown agents to Scribe/Sentinel
            authorized_agents = ["Intake", "Sentinel", "Causor", "Planner", "Guardian", "Executor", "Verifier", "Scribe", "Escalator"]
            db_agent_name = self.name if self.name in authorized_agents else "Scribe"
            
            try:
                self.db.table("logs").insert({
                    "incident_id": incident_id,
                    "agent": db_agent_name,
                    "message": message,
                }).execute()
                omium_tracer.trace_step(incident_id, "DB_INSERT_LOG", 
                                      input_data={"agent": db_agent_name, "message": message}, 
                                      output_data={"status": "success"})
            except Exception as e:
                logger.error(f"DB Log Error: {e}")

    def run(self, state: AgentState) -> AgentState:
        raise NotImplementedError("Agents must implement the run method")


# --- The Agents ---

class IntakeAgent(BaseAgent):
    """Normalizes, deduplicates, and assesses initial severity of alerts."""
    def run(self, state: AgentState) -> AgentState:
        self.log(state.incident_id, "Normalizing incoming alert and checking deduplication.")
        source = state.alert_payload.get("source", "unknown")
        metric = state.alert_payload.get("metric", "cpu_spike")
        severity = state.alert_payload.get("severity", "HIGH")
        ip = state.alert_payload.get("ip", "0.0.0.0")
        state.normalized_alert = {
            "source": source, 
            "metric": metric, 
            "severity": severity,
            "ip": ip
        }
        return state

class SentinelAgent(BaseAgent):
    """Detects specific anomalies from the normalized telemetry."""
    def run(self, state: AgentState) -> AgentState:
        self.log(state.incident_id, "Analyzing telemetry for anomalies.")
        if state.incident_type == "PREDICTIVE":
            anomaly = {
                "id": f"ANOM-PRED-{state.incident_id[:8]}",
                "source": state.normalized_alert["source"],
                "metric": state.normalized_alert.get("metric", "forecasted_network_drift"),
                "value": state.trajectory_risk_score if state.trajectory_risk_score > 0 else 99.9,
                "attack_stage": state.attack_stage,
                "top_features": state.top_predictive_features,
            }
        else:
            anomaly = {"id": "ANOM-1", "source": state.normalized_alert["source"], "metric": state.normalized_alert.get("metric", "traffic"), "value": 99.9}
        state.anomalies.append(anomaly)
        
        if self.db:
            self.db.table("anomalies").insert({
                "incident_id": state.incident_id,
                "source": anomaly["source"],
                "metric": anomaly["metric"],
                "value": anomaly["value"]
            }).execute()
        return state

class CausorAgent(BaseAgent):
    """Performs Root Cause Analysis based on anomalies."""
    def run(self, state: AgentState) -> AgentState:
        self.log(state.incident_id, "Constructing dependency graph for RCA.")
        
        # Factor in Threat Intel and Predictive context
        base_confidence = 0.85
        description = "Potential cyber attack detected"
        
        if state.incident_type == "PREDICTIVE":
            description = f"Forecasted {state.attack_stage} attack with predictive risk {state.trajectory_risk_score:.2f}."
            if state.confidence_score > 0:
                base_confidence = state.confidence_score
        elif state.threat_intel:
            score = state.threat_intel.get("threat_score", 0)
            isp = state.threat_intel.get("isp", "Unknown")
            country = state.threat_intel.get("country", "Unknown")
            
            if score > 50:
                base_confidence += 0.10
                description = f"High-risk attack verified from {country} ({isp})."
            elif score > 0:
                description = f"Low-risk suspicious activity from {country}."
        
        state.confidence_score = min(0.99, base_confidence)
        state.root_cause = {
            "service": state.anomalies[0]["source"], 
            "description": description, 
            "confidence": state.confidence_score
        }
        
        if self.db:
            try:
                self.db.table("root_causes").insert({
                    "incident_id": state.incident_id,
                    "service": state.root_cause["service"],
                    "description": state.root_cause["description"]
                }).execute()
            except: pass
        return state


class PlannerAgent(BaseAgent):
    """Generates remediation actions to fix the root cause."""
    def run(self, state: AgentState) -> AgentState:
        self.log(state.incident_id, f"Generating remediation plan (Attempt {state.retry_count + 1}).")
        
        target_ip = state.normalized_alert.get("ip", "192.168.1.100")
        
        # Determine appropriate remediation action
        is_traffic_threat = (
            "traffic" in state.anomalies[0].get("metric", "") or 
            "DDoS" in state.root_cause.get("description", "") or 
            "Brute Force" in state.root_cause.get("description", "") or 
            "forecast" in state.anomalies[0].get("metric", "").lower() or 
            "IMPACT" in state.attack_stage or 
            "RECONNAISSANCE" in state.attack_stage
        )
        
        # Enforce asset criticality policy
        if state.asset_criticality == "CRITICAL" and not is_traffic_threat:
            # Prohibit disruptive restarts on critical assets; use rate limiting/monitoring
            action_name = "rate_limit_ip"
        elif is_traffic_threat:
            action_name = "block_ip"
        else:
            action_name = "restart_service"
            target_ip = state.root_cause.get("service", "unknown")
            
        plan = [{"action": action_name, "target": target_ip, "status": "PENDING"}]
        state.remediation_plan = plan
        
        # Generate automated rollback plan
        if action_name == "block_ip":
            state.rollback_plan = [{"action": "unblock_ip", "target": target_ip}]
        elif action_name == "rate_limit_ip":
            state.rollback_plan = [{"action": "remove_rate_limit", "target": target_ip}]
        else:
            state.rollback_plan = []
        
        if self.db:
            for p in plan:
                self.db.table("actions").insert({
                    "incident_id": state.incident_id,
                    "action_type": p["action"],
                    "target": p["target"],
                    "status": p["status"]
                }).execute()
        return state

class ThreatIntelAgent(BaseAgent):
    """Fetches external reputation data for the suspicious IP."""
    def run(self, state: AgentState) -> AgentState:
        ip = state.normalized_alert.get("ip", "0.0.0.0")
        if ip == "0.0.0.0" or ip == "127.0.0.1":
             self.log(state.incident_id, f"Skipping intel for local/invalid IP: {ip}")
             state.threat_intel = {"status": "skipped", "threat_score": 0}
             return state

        self.log(state.incident_id, f"Fetching external reputation for IP: {ip}")
        try:
            import requests
            # Using ip-api.com for free demo (limited but zero-config)
            response = requests.get(f"http://ip-api.com/json/{ip}?fields=status,message,country,isp,query", timeout=5)
            data = response.json()
            
            omium_tracer.trace_step(state.incident_id, "TOOL_CALL_THREAT_INTEL_API", 
                                  input_data={"ip": ip}, 
                                  output_data={"api_response": data})
            
            if data.get("status") == "success":
                # Mocking a threat score based on ISP/Country for the demo
                # In real life, we'd use AbuseIPDB or VirusTotal
                threat_score = 75 if data.get("country") != "Local" else 10
                
                state.threat_intel = {
                    "ip": data.get("query"),
                    "country": data.get("country"),
                    "isp": data.get("isp"),
                    "threat_score": threat_score
                }
                self.log(state.incident_id, f"Threat Intel found: ISP={state.threat_intel['isp']}, Score={threat_score}")
            else:
                self.log(state.incident_id, f"Threat Intel API returned error: {data.get('message')}")
                state.threat_intel = {"status": "error", "threat_score": 50}
        except Exception as e:
            self.log(state.incident_id, f"Threat Intel API failed: {e}")
            state.threat_intel = {"status": "error", "threat_score": 50}
            
        return state

class GuardianAgent(BaseAgent):
    """Verifies safety, blast radius, and compliance of the plan."""
    def run(self, state: AgentState) -> AgentState:
        self.log(state.incident_id, "Evaluating safety and compliance of remediation plan.")
        
        # Risk Scoring Logic
        if state.confidence_score < 0.70:
            state.risk_score = "HIGH"
            state.guardian_approved = False
            self.log(state.incident_id, "Plan rejected due to low RCA confidence (Risk: HIGH).")
        else:
            state.risk_score = "LOW"
            state.guardian_approved = True 
            self.log(state.incident_id, "Plan approved. Blast radius is contained (Risk: LOW).")
            
        return state

class ExecutorAgent(BaseAgent):
    """Executes the approved remediation tools (e.g., calling APIs, running scripts)."""
    
    def tool_block_ip(self, target: str, incident_id: str) -> bool:
        self.log(incident_id, f"[TOOL EXEC] Blocking IP {target} at OS Firewall...")
        status = "failed"
        if os.getenv("CYBERHEAL_SAFE_MODE", "true").lower() == "true":
            self.log(incident_id, f"[TOOL EXEC] (SAFE_MODE) Simulated firewall block rule for {target}.")
            status = "simulated_blocked"
            omium_tracer.trace_step(incident_id, "TOOL_CALL_IP_BLOCK", 
                                  input_data={"ip": target, "mode": "simulated"}, 
                                  output_data={"status": status})
            return True
        try:
            if platform.system() == "Linux":
                subprocess.run(["sudo", "iptables", "-A", "INPUT", "-s", target, "-j", "DROP"], check=True)
            elif platform.system() == "Windows":
                subprocess.run(["netsh", "advfirewall", "firewall", "add", "rule", f"name=Block_{target}", "dir=in", "action=block", f"remoteip={target}"], check=True)
            self.log(incident_id, f"[TOOL EXEC] Successfully blocked {target} at network level.")
            status = "blocked"
        except Exception as e:
            self.log(incident_id, f"[TOOL EXEC] OS Firewall command failed, falling back to local blocklist file: {e}")
            with open("blocked_ips.txt", "a") as f:
                f.write(f"{target}\n")
            status = "blocked_via_file"
            
        omium_tracer.trace_step(incident_id, "TOOL_CALL_IP_BLOCK", 
                              input_data={"ip": target}, 
                              output_data={"status": status})
        return True

    def tool_rate_limit_ip(self, target: str, incident_id: str) -> bool:
        self.log(incident_id, f"[TOOL EXEC] Rate limiting IP {target} to 1000 pps...")
        status = "simulated_rate_limited"
        omium_tracer.trace_step(incident_id, "TOOL_CALL_RATE_LIMIT", 
                              input_data={"ip": target, "rate": 1000}, 
                              output_data={"status": status})
        return True
        
    def tool_restart_service(self, target: str, incident_id: str) -> bool:
        self.log(incident_id, f"[TOOL EXEC] Writing Incident Report to disk for service {target}...")
        status = "failed"
        try:
            with open(f"incident_report_{target}.json", "w") as f:
                f.write(f'{{"target": "{target}", "action": "restarted", "timestamp": "{time.time()}"}}')
            status = "restarted"
        except Exception as e:
            self.log(incident_id, f"Failed to write report: {e}")
            
        omium_tracer.trace_step(incident_id, "TOOL_CALL_RESTART_SERVICE", 
                              input_data={"service": target}, 
                              output_data={"status": status})
        return status == "restarted"

    def run(self, state: AgentState) -> AgentState:
        self.log(state.incident_id, "Executing approved remediation tools.")
        
        tools_registry = {
            "block_ip": self.tool_block_ip,
            "restart_service": self.tool_restart_service,
            "rate_limit_ip": self.tool_rate_limit_ip,
        }
        
        for action in state.remediation_plan:
            tool_fn = tools_registry.get(action["action"])
            if tool_fn:
                try:
                    self.log(state.incident_id, f"Running tool: {action['action']} on {action['target']}")
                    success = tool_fn(action["target"], state.incident_id)
                    action["status"] = "COMPLETED" if success else "FAILED"
                except Exception as e:
                    self.log(state.incident_id, f"Tool execution crashed: {e}")
                    action["status"] = "FAILED"
            else:
                self.log(state.incident_id, f"Tool not found: {action['action']}")
                action["status"] = "FAILED"
            
            state.execution_results.append(action)
            if self.db:
                self.db.table("actions").update({"status": action["status"]}).eq("incident_id", state.incident_id).eq("action_type", action["action"]).execute()
            
            # BLOCKCHAIN INTEGRATION
            try:
                from .blockchain import blockchain_sys
                import time
                event_data = {
                    "incident_id": state.incident_id,
                    "event": "FIX_EXECUTED",
                    "action": action["action"],
                    "target": action["target"],
                    "status": action["status"]
                }
                event_hash, block_idx = blockchain_sys.store_event_hash(event_data)
                self.log(state.incident_id, f"[BLOCKCHAIN] Stored Fix Hash in Block #{block_idx}")
            except Exception as e:
                self.log(state.incident_id, f"Blockchain storage failed: {e}")
                
        return state

class VerifierAgent(BaseAgent):
    """Validates if the applied fix actually resolved the anomaly using Before-vs-After comparison."""
    def run(self, state: AgentState) -> AgentState:
        self.log(state.incident_id, "Verifying telemetry post-execution to confirm resolution.")
        
        from .post_verification import PostRemediationVerifier
        verifier = PostRemediationVerifier()
        
        # Use provided metrics or construct observable before/after telemetry
        before_m = state.before_metrics
        if not before_m:
            init_risk = state.trajectory_risk_score if state.trajectory_risk_score > 0 else 0.85
            before_m = {"risk_score": init_risk, "anomaly_rate": 1200.0}
            state.before_metrics = before_m
            
        # If after_metrics is not explicitly caller-supplied, refresh simulation based on current attempt
        if state.after_metrics is not None and not state.after_metrics.get("_simulated", False):
            after_m = state.after_metrics
        else:
            # Attempt 0 simulates incomplete mitigation; subsequent retry attempt simulates resolution
            success = (state.retry_count > 0)
            after_m = verifier.simulate_post_remediation_telemetry(before_m, remediation_successful=success)
            after_m["_simulated"] = 1.0  # Float marker for simulated telemetry
            state.after_metrics = after_m
            
        report = verifier.verify(state.incident_id, before_m, after_m)
        state.verification_passed = report.verification_passed
        state.verification_report = report.to_dict()
        
        if report.verification_passed:
            self.log(state.incident_id, f"Verification PASSED: {report.verification_reasons[0]}")
        else:
            self.log(state.incident_id, f"Verification FAILED: {report.verification_reasons[0]}")
            
        return state

class EscalationAgent(BaseAgent):
    """Handles human-in-the-loop escalation via PagerDuty/Slack."""
    def run(self, state: AgentState) -> AgentState:
        self.log(state.incident_id, "Escalating to human operators (Slack/PagerDuty).")
        state.status = "ESCALATED"
        state.escalated = True
        if self.db:
            self.db.table("incidents").update({"status": state.status}).eq("id", state.incident_id).execute()
        return state

class ReflectiveLearningAgent(BaseAgent):
    """Analyzes failures and generates alternative strategies for retries."""
    def run(self, state: AgentState) -> AgentState:
        state.retry_count += 1
        self.log(state.incident_id, f"RLM_TRIGGERED | Failure detected (Attempt {state.retry_count}).")
        
        omium_tracer.trace_step(state.incident_id, "RLM_TRIGGERED", 
                              input_data={"attempt": state.retry_count}, 
                              output_data={"status": "analyzing"})
        
        # Structured RLM Learning Record generation
        try:
            from .rlm_feedback import RLMReflectionEngine, RLMStore
            store = RLMStore()
            record = RLMReflectionEngine.reflect_from_agent_state(state, store=store)
            state.learning_record = record.to_dict()
            state.reflection_result = record.reflection.analysis
            state.learning_recommendation = record.reflection.recommendations[0] if record.reflection.recommendations else "REVIEW_POLICY"
        except Exception as e:
            logger.warning("RLM learning record generation failed: %s", e)

        if state.retry_count > state.max_retries:
            self.log(state.incident_id, "RLM_COMPLETED | Max retries reached. Final escalation.")
            state.status = "ESCALATED_FINAL"
            state.escalated = True
            
            omium_tracer.trace_step(state.incident_id, "RLM_RESULT", 
                                  input_data={"max_retries": state.max_retries}, 
                                  output_data={"action": "ESCALATE", "status": "failed"})
                                  
            if self.db:
                try:
                    self.db.table("incidents").update({"status": "ESCALATED"}).eq("id", state.incident_id).execute()
                except: pass
            return state

        # Analyze failure reason
        reason = "Verification failed after fix execution."
        if not state.guardian_approved:
            reason = "Remediation plan rejected by Guardian safety check."
            
        last_action = state.remediation_plan[0]["action"] if state.remediation_plan else "none"
        
        # Strategy generation logic (MUST differ from previous)
        new_strategy = "RE-RUN_DETECTION"
        if last_action == "block_ip":
            if state.retry_count == 1:
                new_strategy = "EXPAND_BLOCK_RANGE"
            elif state.retry_count == 2:
                new_strategy = "ADD_RATE_LIMIT"
            else:
                new_strategy = "INCREASE_THRESHOLD"
        elif last_action == "restart_service":
            new_strategy = "ISOLATE_SERVICE"

        reasoning = {
            "attempt": state.retry_count,
            "reason": reason,
            "new_strategy": new_strategy,
            "analysis": f"Action '{last_action}' was insufficient/rejected. Pivoting to {new_strategy}."
        }
        state.rlm_history.append(reasoning)
        
        omium_tracer.trace_step(state.incident_id, "RLM_STRATEGY_GENERATED", 
                              input_data={"previous_action": last_action, "reason": reason}, 
                              output_data={"new_strategy": new_strategy, "analysis": reasoning["analysis"]})
        
        self.log(state.incident_id, f"RLM_STRATEGY_GENERATED | Reasoning: {reasoning['analysis']}")
        state.status = "RETRYING"
        
        if self.db:
            try:
                # Log reasoning to Scribe proxy
                self.db.table("logs").insert({
                    "incident_id": state.incident_id,
                    "agent": "Scribe",
                    "message": f"RLM Retry {state.retry_count}: {new_strategy} | Reason: {reason}"
                }).execute()
            except Exception as e:
                logger.error(f"RLM DB Log Error: {e}")
            
        self.log(state.incident_id, f"RLM_RETRY_EXECUTED | Pipeline restarting with {new_strategy}.")
        omium_tracer.trace_step(state.incident_id, "RLM_RETRY_EXECUTED", 
                              input_data={"new_strategy": new_strategy}, 
                              output_data={"status": "pipeline_restarted"})
        return state

    def run_successful_reflection(self, state: AgentState) -> AgentState:
        """Capture successful incident lifecycle into RLM learning record."""
        self.log(state.incident_id, "RLM_CAPTURE | Recording verified resolution for offline learning.")
        try:
            from .rlm_feedback import RLMReflectionEngine, RLMStore
            store = RLMStore()
            record = RLMReflectionEngine.reflect_from_agent_state(state, store=store)
            state.learning_record = record.to_dict()
            state.reflection_result = record.reflection.analysis
            state.learning_recommendation = record.reflection.recommendations[0] if record.reflection.recommendations else "NO_CHANGE"
            omium_tracer.trace_step(
                state.incident_id,
                "RLM_FEEDBACK_RECORDED",
                input_data={"recommendations": record.reflection.recommendations},
                output_data={"record_id": record.record_id, "status": "stored"},
            )
        except Exception as e:
            logger.warning("RLM successful learning record generation failed: %s", e)
        return state

class ScribeAgent(BaseAgent):
    """Logs the final audit report and closes the incident."""
    def run(self, state: AgentState) -> AgentState:
        self.log(state.incident_id, "Compiling final evidence bundle and closing incident.")
        if state.status == "ESCALATED_FINAL":
            self.log(state.incident_id, "Incident closed as ESCALATED after multiple retry failures.")
        elif not state.escalated:
            state.status = "RESOLVED"
            if self.db:
                self.db.table("incidents").update({"status": state.status}).eq("id", state.incident_id).execute()
                
            # BLOCKCHAIN INTEGRATION
            try:
                from .blockchain import blockchain_sys
                import time
                event_data = {
                    "incident_id": state.incident_id,
                    "event": "TICKET_RESOLVED",
                    "status": "RESOLVED"
                }
                event_hash, block_idx = blockchain_sys.store_event_hash(event_data)
                self.log(state.incident_id, f"[BLOCKCHAIN] Stored Resolution Hash in Block #{block_idx}")
            except Exception as e:
                self.log(state.incident_id, f"Blockchain storage failed: {e}")
                
        return state


# --- LangGraph Orchestration Logic ---
# Note: This is a pseudo-LangGraph builder to illustrate the node/edge routing architecture

class ShieldLangGraph:
    def __init__(self, db_client):
        self.db = db_client
        self.intake = IntakeAgent("Intake", db_client)
        self.sentinel = SentinelAgent("Sentinel", db_client)
        self.causor = CausorAgent("Causor", db_client)
        self.threat_intel = ThreatIntelAgent("ThreatIntel", db_client)
        self.planner = PlannerAgent("Planner", db_client)
        self.guardian = GuardianAgent("Guardian", db_client)
        self.executor = ExecutorAgent("Executor", db_client)
        self.verifier = VerifierAgent("Verifier", db_client)
        self.escalator = EscalationAgent("Escalator", db_client)
        self.rlm = ReflectiveLearningAgent("RLM", db_client)
        self.scribe = ScribeAgent("Scribe", db_client)

    def save_checkpoint(self, state: AgentState):
        """Persists the current state to disk for crash recovery"""
        try:
            payload_str = state.model_dump_json() if hasattr(state, "model_dump_json") else state.json()
            with open("pipeline_state.json", "w") as f:
                f.write(payload_str)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    async def execute_workflow(self, initial_state: AgentState) -> AgentState:
        """Simulates the LangGraph execution flow with Async Orchestration and Persistence"""
        state = initial_state
        
        # 1. Intake & Detection (Worker 1)
        if state.current_step == "INTAKE":
            if DEMO_MODE: logger.info("STEP 3: DIAGNOSIS STARTED")
            state = self.intake.run(state)
            omium_tracer.trace_step(state.incident_id, "DIAGNOSIS_STARTED", 
                                  input_data={"alert": state.alert_payload}, 
                                  output_data={"normalized": state.normalized_alert},
                                  parent_step="TICKET_CREATED")
            state.current_step = "DETECTION"
            self.save_checkpoint(state)
            await asyncio.sleep(0.5)

        while state.retry_count <= state.max_retries:
            # 2. Intel & Analysis (Parallel tasks simulation)
            if state.current_step == "DETECTION":
                state = self.sentinel.run(state)
                # Parallel trigger: Threat Intel
                state = self.threat_intel.run(state)
                state = self.causor.run(state)
                
                if DEMO_MODE: logger.info("STEP 4: RCA COMPLETED")
                omium_tracer.trace_step(state.incident_id, "RCA_COMPLETED", 
                                      input_data={"anomalies": state.anomalies, "threat_intel": state.threat_intel}, 
                                      output_data={"root_cause": state.root_cause},
                                      parent_step="DIAGNOSIS_STARTED")
                                      
                state.current_step = "PLANNING"
                self.save_checkpoint(state)
                await asyncio.sleep(0.5)

            if state.current_step == "PLANNING":
                state = self.planner.run(state)
                state = self.guardian.run(state)
                
                if DEMO_MODE: logger.info(f"STEP 5: GUARDIAN CHECK (Approved: {state.guardian_approved})")
                omium_tracer.trace_step(state.incident_id, "GUARDIAN_CHECK", 
                                      input_data={"plan": state.remediation_plan}, 
                                      output_data={"approved": state.guardian_approved, "risk": state.risk_score},
                                      parent_step="RCA_COMPLETED")
                
                if not state.guardian_approved:
                    state = self.rlm.run(state)
                    
                    if DEMO_MODE: logger.info("STEP: RLM RETRY TRIGGERED")
                    omium_tracer.trace_step(state.incident_id, "RLM_RETRY", 
                                          input_data={"reason": "Guardian Rejected"}, 
                                          output_data={"new_status": state.status},
                                          parent_step="GUARDIAN_CHECK")
                    
                    self.save_checkpoint(state)
                    if state.status == "RETRYING":
                        continue
                    else:
                        break # ESCALATED_FINAL

                state.current_step = "EXECUTION"
                self.save_checkpoint(state)
                await asyncio.sleep(0.5)

            # 3. Action Agent (Worker 3)
            if state.current_step == "EXECUTION":
                state = self.executor.run(state)
                if DEMO_MODE: logger.info("STEP 6: FIX EXECUTED")
                omium_tracer.trace_step(state.incident_id, "FIX_EXECUTED", 
                                      input_data={"plan": state.remediation_plan}, 
                                      output_data={"results": state.execution_results},
                                      parent_step="GUARDIAN_CHECK")
                                      
                state.current_step = "VERIFICATION"
                self.save_checkpoint(state)
                await asyncio.sleep(0.5)
            
            # 4. Verification Agent (Worker 4)
            if state.current_step == "VERIFICATION":
                state = self.verifier.run(state)
                if DEMO_MODE: logger.info(f"STEP 7: VERIFICATION (Passed: {state.verification_passed})")
                omium_tracer.trace_step(state.incident_id, "VERIFICATION", 
                                      input_data={"execution_results": state.execution_results}, 
                                      output_data={"passed": state.verification_passed},
                                      parent_step="FIX_EXECUTED")
                
                if state.verification_passed:
                    break
                
                # If verification fails, RLM takes over
                state = self.rlm.run(state)
                
                if DEMO_MODE: logger.info("STEP: RLM RETRY TRIGGERED")
                omium_tracer.trace_step(state.incident_id, "RLM_RETRY", 
                                      input_data={"reason": "Verification Failed"}, 
                                      output_data={"new_status": state.status},
                                      parent_step="VERIFICATION")
                                      
                self.save_checkpoint(state)
                if state.status != "RETRYING":
                    break
                
                state.current_step = "DETECTION" # Reset to detection phase on retry
                self.intake.log(state.incident_id, f"RLM triggered retry loop. Attempt {state.retry_count}")
                continue

        if state.status == "ESCALATED_FINAL":
             state = self.scribe.run(state)
             if DEMO_MODE: logger.info("STEP 8: ESCALATED")
             omium_tracer.trace_step(state.incident_id, "ESCALATED", 
                                   input_data={"status": state.status}, 
                                   output_data={"message": "Max retries reached"},
                                   parent_step="RLM_RETRY")
             self.save_checkpoint(state)
             return state

        if not state.verification_passed and not state.escalated:
            state = self.escalator.run(state)
        
        # Phase 5: RLM reflection on verified incident resolution
        if state.verification_passed:
            state = self.rlm.run_successful_reflection(state)

        state = self.scribe.run(state)
        if DEMO_MODE: logger.info("STEP 8: RESOLVED")
        omium_tracer.trace_step(state.incident_id, "RESOLVED", 
                              input_data={"verification_passed": state.verification_passed}, 
                              output_data={"status": state.status},
                              parent_step="VERIFICATION")
                              
        state.current_step = "COMPLETED"
        self.save_checkpoint(state)
        return state
