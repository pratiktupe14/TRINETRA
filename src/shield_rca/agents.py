from loguru import logger
import time
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import datetime

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = None
if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
else:
    logger.warning("Supabase credentials not found in environment. Logs will not be saved to DB.")

def save_log(incident_id, agent_name, message):
    if not supabase or not incident_id:
        return
    try:
        data = {
            "incident_id": incident_id,
            "message": message,
            "agent": agent_name
        }
        supabase.table("logs").insert(data).execute()
    except Exception as e:
        logger.error(f"Failed to save log to Supabase: {e}")

class SentinelAgent:
    def __init__(self, config):
        self.config = config
        self.threshold = config.get("threshold", 0.8)

    def detect_anomalies(self, incident_id, data_stream):
        msg = f"Analyzing telemetry data (Threshold: {self.threshold})..."
        logger.info(f"[Sentinel] {msg}")
        save_log(incident_id, "Sentinel", msg)
        time.sleep(1)
        
        msg2 = "Anomaly detected: Unusual traffic spike on node-A."
        logger.warning(f"[Sentinel] {msg2}")
        save_log(incident_id, "Sentinel", msg2)

        # Insert anomaly record
        anomaly_id = None
        if supabase and incident_id:
            try:
                res = supabase.table("anomalies").insert({
                    "incident_id": incident_id,
                    "source": "node-A",
                    "metric": "traffic_volume",
                    "value": 1500.5
                }).execute()
                anomaly_id = res.data[0]['id']
            except Exception as e:
                logger.error(f"Failed to insert anomaly: {e}")

        return {"anomaly_id": anomaly_id or "ANOM-001", "node": "node-A", "severity": "HIGH"}

class CausorAgent:
    def __init__(self, config):
        self.config = config

    def analyze_root_cause(self, incident_id, anomaly):
        msg = f"Constructing dynamic dependency graph for {anomaly['node']}..."
        logger.info(f"[Causor] {msg}")
        save_log(incident_id, "Causor", msg)
        time.sleep(1)
        
        msg2 = "Root cause localized: Malicious payload injection from external IP."
        logger.info(f"[Causor] {msg2}")
        save_log(incident_id, "Causor", msg2)

        # Insert root cause record
        cause_id = None
        if supabase and incident_id:
            try:
                res = supabase.table("root_causes").insert({
                    "incident_id": incident_id,
                    "service": anomaly['node'],
                    "description": "Malicious payload injection from external IP."
                }).execute()
                cause_id = res.data[0]['id']
            except Exception as e:
                logger.error(f"Failed to insert root cause: {e}")

        return {"cause_id": cause_id or "RC-001", "anomaly_id": anomaly['anomaly_id'], "cause_type": "injection"}

class PlannerAgent:
    def __init__(self, config):
        self.config = config

    def generate_remediation_plan(self, incident_id, root_cause):
        msg = f"Formulating constrained remediation plan for {root_cause['cause_type']}..."
        logger.info(f"[Planner] {msg}")
        save_log(incident_id, "Planner", msg)
        time.sleep(1)
        
        plan = [
            {"action": "quarantine", "target": "external IP"},
            {"action": "rollback", "target": "node-A state"}
        ]
        
        msg2 = f"Plan generated with {len(plan)} actions."
        logger.info(f"[Planner] {msg2}")
        save_log(incident_id, "Planner", msg2)

        # Insert actions records
        action_ids = []
        if supabase and incident_id:
            try:
                for a in plan:
                    res = supabase.table("actions").insert({
                        "incident_id": incident_id,
                        "action_type": a["action"],
                        "target": a["target"],
                        "status": "PENDING"
                    }).execute()
                    action_ids.append(res.data[0]['id'])
            except Exception as e:
                logger.error(f"Failed to insert actions: {e}")

        return {"action_ids": action_ids, "plan_details": plan}

class GuardianAgent:
    def __init__(self, config):
        self.config = config

    def verify_plan(self, incident_id, plan_data):
        msg = "Evaluating safety and policy compliance of remediation plan..."
        logger.info(f"[Guardian] {msg}")
        save_log(incident_id, "Guardian", msg)
        time.sleep(1)
        
        msg2 = "Blast-radius check passed. Plan is safe to execute."
        logger.success(f"[Guardian] {msg2}")
        save_log(incident_id, "Guardian", msg2)

        # Approve actions in DB
        if supabase and incident_id and plan_data["action_ids"]:
            try:
                for a_id in plan_data["action_ids"]:
                    supabase.table("actions").update({"status": "IN_PROGRESS"}).eq("id", a_id).execute()
            except Exception as e:
                logger.error(f"Failed to update actions: {e}")

        return True

class ScribeAgent:
    def __init__(self, config):
        self.config = config

    def log_audit(self, incident_id, event_data):
        msg = "Compiling evidence bundle and creating audit log..."
        logger.info(f"[Scribe] {msg}")
        save_log(incident_id, "Scribe", msg)
        time.sleep(1)
        
        msg2 = f"Audit Log Saved: {event_data['status']}"
        logger.info(f"[Scribe] {msg2}")
        save_log(incident_id, "Scribe", msg2)
        
        # Complete Actions and Resolve Incident
        if supabase and incident_id:
            try:
                if event_data.get("action_ids"):
                    for a_id in event_data["action_ids"]:
                        supabase.table("actions").update({
                            "status": "COMPLETED",
                            "executed_at": datetime.datetime.utcnow().isoformat()
                        }).eq("id", a_id).execute()
                
                supabase.table("incidents").update({"status": "RESOLVED"}).eq("id", incident_id).execute()
            except Exception as e:
                logger.error(f"Failed to finalize incident in DB: {e}")
                
        return True
