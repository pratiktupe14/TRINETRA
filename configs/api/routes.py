from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import List
import yaml
import sys
import os

# Import Supabase and Models
from shield_rca.agents import supabase
from shield_rca.langgraph_core import ShieldLangGraph, AgentState
from .models import (
    IncidentResponse, IncidentDetailsResponse, LogResponse, RunPipelineResponse
)
from .auth import verify_supabase_token
from fastapi import Depends
from loguru import logger
from .ingestion import simulation_engine, ingestor, ENABLE_SIMULATION
from .tracing import omium_tracer
import time

router = APIRouter()

from .worker import execute_pipeline_task

@router.post("/run", response_model=RunPipelineResponse)
async def run_pipeline(background_tasks: BackgroundTasks, current_user = Depends(verify_supabase_token)):
    """Triggers the multi-agent pipeline and returns the new incident ID instantly. Requires Authentication."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    try:
        res = supabase.table("incidents").insert({
            "title": "API Triggered Cyber Attack",
            "severity": "HIGH",
            "status": "INVESTIGATING",
            "user_id": current_user.id
        }).execute()
        
        incident_id = res.data[0]['id']
        
        # Dispatch the pipeline execution to the robust Celery Redis Queue
        execute_pipeline_task.delay(incident_id)
        
        return RunPipelineResponse(message="Pipeline queued successfully", incident_id=incident_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/incidents", response_model=List[IncidentResponse])
async def get_incidents():
    """Returns a list of all incidents."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")
        
    res = supabase.table("incidents").select("*").order("created_at", desc=True).execute()
    return res.data

@router.get("/incidents/{incident_id}", response_model=IncidentDetailsResponse)
async def get_incident_details(incident_id: str):
    """Returns full details of an incident including anomalies, root causes, and actions."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")
        
    incident_res = supabase.table("incidents").select("*").eq("id", incident_id).execute()
    if not incident_res.data:
        raise HTTPException(status_code=404, detail="Incident not found")
        
    anomalies = supabase.table("anomalies").select("*").eq("incident_id", incident_id).execute().data
    root_causes = supabase.table("root_causes").select("*").eq("incident_id", incident_id).execute().data
    actions = supabase.table("actions").select("*").eq("incident_id", incident_id).execute().data
    logs = supabase.table("logs").select("*").eq("incident_id", incident_id).order("timestamp", desc=False).execute().data

    return IncidentDetailsResponse(
        incident=incident_res.data[0],
        anomalies=anomalies,
        root_causes=root_causes,
        actions=actions,
        logs=logs
    )

@router.get("/logs/{incident_id}", response_model=List[LogResponse])
async def get_logs(incident_id: str):
    """Returns only the logs for a specific incident."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")
        
    res = supabase.table("logs").select("*").eq("incident_id", incident_id).order("timestamp", desc=False).execute()
    return res.data

# --- SIMULATION ENDPOINTS ---

@router.post("/simulate/bruteforce")
async def sim_bruteforce():
    incident_id = simulation_engine.simulate_brute_force()
    if not incident_id:
        raise HTTPException(status_code=400, detail="Simulation disabled or failed")
    return {"message": "Simulation triggered", "id": incident_id}

@router.post("/simulate/service-down")
async def sim_service_down():
    incident_id = simulation_engine.simulate_service_down()
    if not incident_id:
        raise HTTPException(status_code=400, detail="Simulation disabled or failed")
    return {"message": "Simulation triggered", "id": incident_id}

@router.post("/simulate/edge-anomaly")
async def sim_edge_anomaly():
    incident_id = simulation_engine.simulate_edge_anomaly()
    if not incident_id:
        raise HTTPException(status_code=400, detail="Simulation disabled or failed")
    return {"message": "Simulation triggered", "id": incident_id}

# --- TRACE EXPORT & REPLAY ENDPOINTS ---

@router.get("/trace/export/{incident_id}")
async def export_trace(incident_id: str):
    """Export a trace file by incident ID"""
    trace = trace_tracker.get_trace(incident_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found for this incident")
    return trace

@router.post("/trace/replay")
async def replay_trace(trace_data: dict):
    """Replay an uploaded trace file to test pipeline logic visually"""
    logger.info(f"Replaying trace: {trace_data.get('trace_id')}")
    
    # 1. Log simulation steps to backend console as requested
    for step in trace_data.get("pipeline", []):
        logger.info(f"[REPLAY] Execution Step: {step.get('step')} | Status: {step.get('status')}")
        time.sleep(0.5) # Emulate processing time visually in logs
        
    # 2. Re-inject into dashboard via simulation engine
    incident_id = ingestor._trigger_incident(
        title=f"[TRACE REPLAY] {trace_data.get('event_type', 'Unknown Event')}",
        issue_type=trace_data.get('event_type', 'Unknown'),
        severity="HIGH",
        source_ip="replay-sys",
        is_simulated=True
    )
    
    if not incident_id:
        raise HTTPException(status_code=500, detail="Failed to replay trace into dashboard")
        
    return {"message": "Trace replayed successfully into dashboard", "incident_id": incident_id}

# --- WEBHOOK ENDPOINT ---

@router.post("/webhook/security-event")
async def webhook_security_event(payload: dict):
    """External Webhook to trigger the Multi-Agent SOC Pipeline"""
    logger.info(f"Received Webhook Event: {payload}")
    
    # Extract data or fallback
    ip = payload.get("ip", "0.0.0.0")
    failed_attempts = payload.get("failedAttempts")
    
    if failed_attempts is not None:
        if failed_attempts > 5:
            event_type = "BRUTE_FORCE"
            severity = "HIGH"
        else:
            event_type = "LOW_RISK_EVENT"
            severity = "LOW"
    else:
        event_type = payload.get("type", "WEBHOOK_ALERT")
        severity = payload.get("severity", "HIGH")
        
    is_simulated = payload.get("is_simulated", False)
    
    incident_id = ingestor._trigger_incident(
        title=f"Webhook Alert: {event_type}",
        issue_type=event_type,
        severity=severity,
        source_ip=ip,
        is_simulated=is_simulated,
        source="webhook"
    )
    
    if not incident_id:
        raise HTTPException(status_code=500, detail="Failed to process webhook")
        
    return {"status": "accepted", "incident_id": incident_id, "message": "Multi-agent pipeline triggered"}



