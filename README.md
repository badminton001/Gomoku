# Gomoku AI Battle Arena

A comprehensive Gomoku (Five-in-a-Row) game system featuring advanced AI agents, automated performance tuning, and cross-platform GUI.

## Project Structure

*   **backend/Engine**: Core game logic (15x15 Board, Rules).
*   **backend/AI**: Pluggable AI modules:
    *   **Greedy**: Heuristic-based fast agent.
    *   **AlphaBeta (Minimax)**: Strong rule-based agent with pruning and iterative deepening.
    *   **Hybrid**: Policy Network (SL) guided search with defense subsystems.
    *   **DQN**: Reinforcement Learning agent (Experimental).
*   **backend/API**: FastAPI service for AI inference and configuration.
*   **frontend**: Tkinter-based GUI for Human vs AI play.
*   **scripts**: Performance analysis, model training, and replay tools.

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run GUI (Play)
```bash
python frontend/gui.py
```

### 3. Run Backend API (Server)
```bash
python backend/api/main.py
```
*   API Docs: http://localhost:8000/docs

### 4. Performance Tuning
```bash
# Analyze AI performance
python scripts/analyze_performance.py

# Auto-tune AI hyperparameters
python backend/tune_ai.py
```

## Key Features
*   **Hybrid AI**: Combines Neural Network intuition with tactical search.
*   **Move Scoring**: Real-time evaluation and "Brilliant/Blunder" detection.
*   **Replay System**: Save and analyze full game records.
*   **Auto-Tuning**: Optuna integration for parameter optimization.
