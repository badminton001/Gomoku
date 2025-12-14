import uvicorn
import time
import json
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from backend.engine.board import Board

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
        # Reconstruct Board
        # Note: Frontend sends list[list[int]], Board expects it internally or we set it manually
        # Board init creates empty.
        size = len(state.board)
        game_board = Board(size=size)
        game_board.board = state.board
        
        # Calculate move_count (important for some AIs)
        count = 0
        for r in range(size):
            for c in range(size):
                if state.board[r][c] != 0: count += 1
        game_board.move_count = count
        
        # Select Agent
        if state.algorithm.lower() == "greedy":
            from backend.ai.basic.classic_ai import GreedyAgent
            agent = GreedyAgent(distance=2)
        elif state.algorithm.lower() in ["alphabeta", "strong"]:
             from backend.ai.basic.strong_ai import AlphaBetaAgent
             # Use a reasonable default depth or config
             depth = 2 
             agent = AlphaBetaAgent(depth=depth, time_limit=2.0)
        elif state.algorithm.lower() == "random":
             from backend.ai.basic.classic_ai import RandomAgent
             agent = RandomAgent()
        else:
             # Default to Strong AI
             from backend.ai.basic.strong_ai import AlphaBetaAgent
             agent = AlphaBetaAgent(depth=2, time_limit=2.0)
             
        # Get Move
        move = agent.get_move(game_board, state.current_player)
        
        if move:
            best_x, best_y = move
        else:
            best_x, best_y = -1, -1 # Should not happen if board not full
            
        duration = time.time() - start_time
        logger.info(f"AI decided: ({best_x}, {best_y}) in {duration:.4f}s")
        
        return MoveResponse(
            x=best_x, y=best_y, processing_time=duration,
            debug_info=f"Agent: {state.algorithm}"
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)