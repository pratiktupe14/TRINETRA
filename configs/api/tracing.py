import json
import uuid
from datetime import datetime
from loguru import logger
import os
import omium
from dotenv import load_dotenv

load_dotenv()

# Initialize Omium SDK with API key from environment
omium_api_key = os.getenv("OMIUM_API_KEY", "omium_demo_key_1234567890abcdef")
try:
    omium.init(api_key=omium_api_key)
except Exception as e:
    logger.warning(f"Omium cloud init skipped: {e}")

class OmiumTracer:
    def __init__(self, trace_dir="traces"):
        self.trace_dir = trace_dir
        os.makedirs(self.trace_dir, exist_ok=True)
    
    def trace_step(self, incident_id, step_name, input_data, output_data, parent_step=None):
        # Local JSON trace for debug/demo
        file_path = os.path.join(self.trace_dir, f"{incident_id}_trace.json")
        traces = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    traces = json.load(f)
            except Exception:
                pass
        
        trace = {
            "step": step_name,
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "parent": parent_step,
            "incident_id": incident_id
        }
        traces.append(trace)
        
        with open(file_path, "w") as f:
            json.dump(traces, f, indent=2)
            
        logger.debug(f"[Omium] Local trace recorded for {step_name}")
        
        # Official Omium SDK Direct API trace
        try:
            import requests
            import hashlib
            
            def get_deterministic_id(i_id, s_name):
                h = hashlib.md5(f"{i_id}_{s_name}".encode()).hexdigest()
                return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
            
            span_id = get_deterministic_id(incident_id, step_name)
            p_span_id = get_deterministic_id(incident_id, parent_step) if parent_step else None
            
            current_span_type = "workflow" if parent_step is None else "step"
            
            payload = {
                "project": "default",
                "execution_id": incident_id,
                "spans": [
                    {
                        "span_id": span_id,
                        "trace_id": incident_id,
                        "parent_span_id": p_span_id,
                        "name": step_name,
                        "service_name": "cyberheal",
                        "start_time": trace["timestamp"],
                        "status": "ok",
                        "attributes": {
                            "span_type": current_span_type,
                            "workflow_id": "cyberheal_rca",
                            "input": input_data,
                            "output": output_data
                        },
                        "events": []
                    }
                ],
                "sdk_version": "0.4.1",
                "metadata": {
                    "workflow_id": "cyberheal_rca",
                    "trace_id": incident_id,
                }
            }
            
            response = requests.post(
                "https://api.omium.ai/api/v1/traces/ingest",
                json=payload,
                headers={"X-API-Key": omium_api_key},
                timeout=5
            )
            
            if response.status_code >= 400:
                logger.debug(f"[Omium] API response: {response.status_code}")
            else:
                logger.debug(f"[Omium] API Trace sent successfully for {step_name}")
                
        except Exception as e:
            logger.debug(f"[Omium] Trace dispatch notice: {e}")

omium_tracer = OmiumTracer()
