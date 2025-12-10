import uvicorn
import time
import json
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# ==========================================
# 1. Initialize & Cap of logs
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"), 
        logging.StreamHandler()            
    ]
)
logger = logging.getLogger("backend")

app = FastAPI(title="Gomoku-AI (HeptagonPan)") 

# ==========================================
# 2. def data type
# ==========================================
class BoardState(BaseModel):
    board: List[List[int]]       
    current_player: int          
    algorithm: str = "minimax"   

class MoveResponse(BaseModel):
    x: int
    y: int
    processing_time: float
    debug_info: str

# ==========================================
# 3. API's acheivement
# ==========================================
@app.get("/")
def root():
    return {"message": "Gomoku AI Backend is Running", "docs_url": "http://127.0.0.1:8000/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/config")
def get_ai_config():
    config_path = "config/ai_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {"warning": "Config file not found"}

@app.post("/ai/predict", response_model=MoveResponse)
def predict_move(state: BoardState):
    start_time = time.time()
    logger.info(f"Received move request. Algo: {state.algorithm}")
    
    try:
        # Imitate the thinking process of AI
        time.sleep(0.5)
        best_x, best_y = 7, 7
        
        duration = time.time() - start_time
        logger.info(f"AI decided: ({best_x}, {best_y}) in {duration:.4f}s")
        
        return MoveResponse(
            x=best_x, y=best_y, processing_time=duration,
            debug_info="Mock AI (Integration Pending)"
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)