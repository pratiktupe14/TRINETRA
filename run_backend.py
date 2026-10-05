import os
import sys
import uvicorn

# Add configs and src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'configs'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    from api.main import app
    print("Starting CyberHeal AI Multi-Agent Backend API on http://localhost:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
