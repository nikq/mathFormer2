import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import threading
import copy

app = FastAPI()

# Shared state
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.weights = {} # Structure: {layer_name: [[values...]]}
        self.metrics = {"loss": [], "step": 0}
        self.latest_update = 0
        self.command_queue = [] # List of commands from frontend

state = SharedState()

def update_state(weights, step, loss):
    """
    Called by the training loop to update visualization data.
    weights: dict of layer names to numpy arrays (or lists)
    """
    with state.lock:
        # Deep copy to avoid race conditions if training loop modifies it
        # For performance, we might want to optimize this (e.g. only copy what's needed)
        state.weights = weights
        state.metrics["loss"].append({"step": step, "value": loss})
        state.metrics["step"] = step
        state.metrics["step"] = step
        state.latest_update += 1

def reset_state():
    """
    Called to reset the visualization state (e.g. on restart).
    """
    with state.lock:
        state.metrics = {"loss": [], "step": 0}
        state.latest_update += 1

def get_latest_command():
    with state.lock:
        if state.command_queue:
            return state.command_queue.pop(0)
        return None

from pydantic import BaseModel

class ControlCommand(BaseModel):
    action: str
    config: dict = {}

@app.post("/api/control")
async def control(cmd: ControlCommand):
    print(f"DEBUG: Server received control command: {cmd}")
    with state.lock:
        state.command_queue.append(cmd.dict())
    return {"status": "ok"}

@app.get("/api/data")
async def get_data():
    with state.lock:
        return {
            "weights": state.weights,
            "metrics": state.metrics,
            "update_id": state.latest_update
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_sent_update = -1
    try:
        while True:
            # Poll for updates
            current_update = 0
            with state.lock:
                current_update = state.latest_update
            
            if current_update > last_sent_update:
                with state.lock:
                    data = {
                        "weights": state.weights,
                        "metrics": state.metrics,
                        "update_id": state.latest_update
                    }
                await websocket.send_json(data)
                last_sent_update = current_update
            
            await asyncio.sleep(0.1) # 10fps max
    except Exception as e:
        print(f"WebSocket disconnected: {e}")

# Serve static files
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

def start_server(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    start_server()
