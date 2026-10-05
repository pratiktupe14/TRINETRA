from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import subprocess
import json
import logging

app = FastAPI(title="SOC Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_windows_logs(log_name="System", max_events=20):
    cmd = f'powershell.exe -Command "Get-WinEvent -LogName {log_name} -MaxEvents {max_events} -ErrorAction SilentlyContinue | Select-Object TimeCreated, Id, LevelDisplayName, Message, ProviderName | ConvertTo-Json"'
    try:
        result = subprocess.check_output(cmd, shell=True, text=True)
        if not result.strip():
            return []
        data = json.loads(result)
        if isinstance(data, dict):
            data = [data]
        return data
    except Exception as e:
        logging.error(f"Error fetching logs from {log_name}: {e}")
        return []

@app.get("/api/logs")
def get_recent_logs():
    # Attempt to read Security logs (requires Admin), fallback to System and Application
    logs = get_windows_logs("Security", 10)
    if not logs:
        logs = get_windows_logs("System", 20)
    return {"status": "success", "data": logs}

@app.websocket("/ws")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Poll every 5 seconds
            logs = get_windows_logs("System", 10) # Using System for broader accessibility
            security_logs = get_windows_logs("Security", 5)
            if security_logs:
                logs = security_logs + logs
            
            await websocket.send_json({"type": "LOG_UPDATE", "data": logs})
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass