# Gomoku AI Battle Arena

A comprehensive Gomoku (Five-in-a-Row) game system featuring advanced AI agents, automated performance tuning, and cross-platform GUI.

## Project Structure

*   **backend/**:
    *   **Engine**: Core logic (`board.py`, `game_engine.py`).
    *   **AI**: Pluggable agents:
        *   `minimax.py` (AlphaBeta), `mcts.py` (Monte Carlo Tree Search), `dqn.py` (Reinforcement Learning), `hybrid.py` (Policy+Search).
        *   `baselines.py` (Greedy, Random), `policy_network.py` (SL Model).
    *   **API**: FastAPI service for AI serving.
*   **frontend/**: Tkinter-based GUI (`gui.py`) for Human vs AI.
*   **scripts/**:
    *   **training/**: `train_*.py`, `tune_ai.py`, `generate_dataset.py`.
    *   **evaluation/**: `run_full_evaluation.py`, `watch_match.py`.
    *   **analysis/**: `generate_visualizations.py`, `test_replay_scoring.py`.
*   **notebooks/**:
    *   `train_kaggle.ipynb`: Self-contained notebook for Kaggle cloud training.
    *   `analysis.ipynb`: Data visualization dashboard.

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Play Game (GUI)
```bash
python frontend/main.py
```

### 3. Training
*   **Local Training**: `python scripts/training/train_dqn_15x15_final.py`
*   **Auto-Tuning**: `python scripts/training/tune_ai.py`
*   **Kaggle**: Upload `notebooks/train_kaggle.ipynb` to Kaggle.

### 4. Evaluation & Analysis
*   **Run Tournament**: `python scripts/evaluation/run_full_evaluation.py --games-per-pair 10 --threads 5`
*   **Watch Match**: `python scripts/evaluation/watch_match.py`
*   **Analyze Results**: Open `notebooks/analysis.ipynb`.

### 5. Backend API (Optional)
Start the AI server for external integrations:
```bash
python backend/api/main.py
# Docs: http://localhost:8000/docs
```

## Key Features
*   **Hybrid AI**: Combines Neural Network intuition (SL Policy) with tactical AlphaBeta search.
*   **Parallel Eval**: Multi-process tournament system for rapid ELO rating.
*   **Move Scoring**: Real-time "Brilliant/Blunder" analysis engine.
*   **Clean Architecture**: Modular design separating Engine, AI, and UI.
