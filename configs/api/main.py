from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import json
from loguru import logger

# Add src to Python path so shield_rca can be imported globally
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.append(src_path)
from api.routes import router as api_router
from api.ingestion import ingestor

app = FastAPI(
    title="CyberHeal AI Backend API",
    description="FastAPI Backend for CyberHeal AI multi-agent pipeline",
    version="1.0.0"
)

# Configure CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to the specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our routes
app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Welcome to CyberHeal AI Backend API. Go to /docs for the interactive documentation."}

@app.on_event("startup")
async def startup_event():
    # Start the background daemon to poll real OS logs
    ingestor.start_polling()
    
    # CRASH RECOVERY: Check for interrupted pipeline jobs
    if os.path.exists("pipeline_state.json"):
        try:
            with open("pipeline_state.json", "r") as f:
                data = json.load(f)
                incident_id = data.get("incident_id")
                if incident_id:
                    logger.warning(f"[RECOVERY] Found interrupted job for {incident_id}. Resuming...")
                    import threading
                    from api.ingestion import run_pipeline_thread
                    threading.Thread(target=run_pipeline_thread, args=(incident_id,)).start()
        except Exception as e:
            logger.error(f"Recovery failed: {e}")

