import argparse
import yaml
import sys
import os
from loguru import logger

# Add src to python path so we can import shield_rca
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from shield_rca.agents import SentinelAgent, CausorAgent, PlannerAgent, GuardianAgent, ScribeAgent, supabase

def main():
    parser = argparse.ArgumentParser(description="SHIELD-RCA Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration yaml")
    parser.add_argument("--mode", type=str, required=True, choices=["smoke", "eval_ids"], help="Execution mode")
    args = parser.parse_args()

    logger.info(f"Starting SHIELD-RCA Pipeline in '{args.mode}' mode")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.mode == "smoke":
        logger.info("Initializing Agents...")
        sentinel = SentinelAgent(config["agents"]["sentinel"])
        causor = CausorAgent(config["agents"]["causor"])
        planner = PlannerAgent(config["agents"]["planner"])
        guardian = GuardianAgent(config["agents"]["guardian"])
        scribe = ScribeAgent(config["agents"]["scribe"])

        logger.info("Creating New Incident in Database...")
        incident_id = None
        if supabase:
            try:
                res = supabase.table("incidents").insert({
                    "title": "Smoke Test Cyber Attack",
                    "severity": "CRITICAL",
                    "status": "INVESTIGATING"
                }).execute()
                incident_id = res.data[0]['id']
                logger.info(f"Created Incident ID: {incident_id}")
            except Exception as e:
                logger.error(f"Failed to create incident: {e}")

        logger.info("Running Smoke Test Pipeline...")
        
        # 1. Sentinel
        anomaly = sentinel.detect_anomalies(incident_id, data_stream="mock_stream")
        
        # 2. Causor
        root_cause = causor.analyze_root_cause(incident_id, anomaly)
        
        # 3. Planner
        plan_data = planner.generate_remediation_plan(incident_id, root_cause)
        
        # 4. Guardian
        is_safe = guardian.verify_plan(incident_id, plan_data)
        
        # 5. Scribe
        if is_safe:
            scribe.log_audit(incident_id, {"status": "SUCCESS", "action_ids": plan_data.get("action_ids", [])})
            logger.success("SHIELD-RCA Smoke Test Completed Successfully!")
        else:
            scribe.log_audit(incident_id, {"status": "FAILED", "reason": "Guardian rejected plan"})
            logger.error("SHIELD-RCA Smoke Test Failed due to Guardian rejection.")

if __name__ == "__main__":
    main()
