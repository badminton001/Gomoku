"""
Gomoku AI API Backend.
"""
import uvicorn
import time
import json
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from backend.engine.board import Board

# 1. Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"), 
        logging.StreamHandler()            
    ]
)
logger = logging.getLogger("backend")

app = FastAPI(title="Gomoku-AI API") 

# 2. Data Models
class BoardState(BaseModel):
    board: List[List[int]]       
    current_player: int          
    algorithm: str = "minimax"   

class MoveResponse(BaseModel):
    x: int
    y: int
    processing_time: float
    debug_info: str

# 3. API Endpoints
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
        size = len(state.board)
        game_board = Board(size=size)
        game_board.board = state.board
        
        # Calculate move_count
        count = 0
        for r in range(size):
            for c in range(size):
                if state.board[r][c] != 0: count += 1
        game_board.move_count = count
        
        # Load Config
        config = {}
        config_path = "config/ai_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        
        # Select Agent
        if state.algorithm.lower() == "greedy":
            from backend.ai.baselines import GreedyAgent
            dist = config.get("greedy", {}).get("distance", 2)
            agent = GreedyAgent(distance=dist)
        elif state.algorithm.lower() == "hybrid":
            from backend.ai.hybrid import HybridAgent
            # Try to load best model, fallback to base
            model_path = "models/sl_policy_v2_kaggle.pth"
            if not os.path.exists(model_path): model_path = "models/sl_policy_v1_base.pth"
            agent = HybridAgent(model_path=model_path, device="cpu")
        elif state.algorithm.lower() == "dqn":
            from backend.ai.dqn import QLearningAgent
            model_path = "models/dqn_v1_final" 
            agent = QLearningAgent(model_path=model_path)
        elif state.algorithm.lower() in ["alphabeta", "strong"]:
             from backend.ai.minimax import AlphaBetaAgent
             cfg = config.get("alpha_beta", {})
             depth = cfg.get("depth", 2)
             time_limit = 2.0 # Keep safe limit for API
             agent = AlphaBetaAgent(depth=depth, time_limit=time_limit)
        elif state.algorithm.lower() == "mcts":
             from backend.ai.mcts import MCTSAgent
             agent = MCTSAgent(iteration_limit=500)
        else:
             # Default
             from backend.ai.baselines import GreedyAgent
             agent = GreedyAgent()

        # Get Move
        move = agent.get_move(game_board, state.current_player)
        
        if move:
            best_x, best_y = move
        else:
            best_x, best_y = -1, -1 
            
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