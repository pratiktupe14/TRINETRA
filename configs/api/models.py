from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class IncidentResponse(BaseModel):
    id: str
    title: str
    severity: str
    status: str
    created_at: datetime

class AnomalyResponse(BaseModel):
    id: str
    incident_id: str
    source: str
    metric: str
    value: float
    timestamp: datetime

class RootCauseResponse(BaseModel):
    id: str
    incident_id: str
    service: str
    description: str
    created_at: datetime

class ActionResponse(BaseModel):
    id: str
    incident_id: str
    action_type: str
    target: str
    status: str
    executed_at: Optional[datetime] = None

class LogResponse(BaseModel):
    id: str
    incident_id: str
    message: str
    agent: str
    timestamp: datetime

class IncidentDetailsResponse(BaseModel):
    incident: IncidentResponse
    anomalies: List[AnomalyResponse]
    root_causes: List[RootCauseResponse]
    actions: List[ActionResponse]
    logs: List[LogResponse]

class RunPipelineResponse(BaseModel):
    message: str
    incident_id: str
