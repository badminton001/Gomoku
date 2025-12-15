# Person F Report: UI & AI Integration

## 1. UI Architecture Design (UI 架构设计)

The **Gomoku AI Battle Arena** utilizes a **Tkinter-based** graphical user interface for maximum cross-platform compatibility (Windows/Linux/macOS) without requiring heavy external dependencies like Qt.

### 1.1 Architecture Overview
The code follows a simplified **Model-View-Controller (MVC)** pattern centralized in `frontend/gui.py`:

*   **Model (`backend.engine.game_engine.GameEngine`)**: 
    *   Manages the game state (15x15 board, move history, rules).
    *   Handles "Undo" logic (`undo_last_move`) and current player tracking.
    *   Strict separation of concerns: The engine knows nothing about the UI.
*   **View (`frontend.gui.GomokuGUI`)**:
    *   **Canvas (`tk.Canvas`)**: Renders the 15x15 grid, stars, and stone graphics using standard 2D primitives.
    *   **Control Panel (`tk.Frame`)**: Contains buttons (Start, Undo, Hint), Dropdowns (AI Selection), and Sliders (Difficulty).
    *   **Status Bar**: Feedback loop for "Thinking...", "Win/Loss", and "Hint Coordinates".
*   **Controller (Event Bindings in `GomokuGUI`)**:
    *   `on_board_click`: Translates mouse coordinates -> Grid coordinates -> Calls Engine -> Updates View.
    *   `ai_turn`: Scheduled via `root.after()` to allow the UI to refresh before the heavy blocking AI calculation starts.
    *   `get_hint`: Orchestrates the `HybridAgent` to fetch Top-5 moves and renders them as overlays.

### 1.2 Key UI Features
*   **Dynamic Board Rendering**: The board fully redraws on every state change, enabling clean "Undo" and "Reset" operations without graphical artifacts.
*   **Coordinate System**: Standard Gomoku coordinates (A-O, 15-1) displayed on axes.
*   **Responsiveness**: Uses `root.after(100)` scheduling to ensure button clicks register before processing starts.

---

## 2. AI Integration Workflow (AI 集成工作流)

The UI integrates with the backend AI module through a **Plug-and-Play Agent Interface**.

### 2.1 Agent Loading & Caching
To maintain high performance and avoid reloading heavy Neural Networks on every turn, the UI implements an **Agent Cache**:
```python
self.agent_cache = {
    "Hybrid_2": <HybridAgent Instance>,
    "Strong_4": <AlphaBetaAgent Instance>
}
```
*   **Lazy Loading**: Modules (`torch`, `backend.ai.advanced`) are only imported when a specific AI is selected.
*   **Parameter Injection**: Difficulty sliders (Depth 1-6) are injected into the Agent constructor at runtime.

### 2.2 The "AI Turn" Lifecycle
1.  **Trigger**: User makes a move -> `on_board_click` updates board -> Checks Game Over -> Calls `root.after(100, self.ai_turn)`.
2.  **Execution**: 
    *   Retrieves the selected Agent from cache.
    *   Calls `agent.get_move(board)`.
    *   **Hybrid Agent Flow**:
        *   Checks Policy Network (SL Model) for candidates.
        *   Verifies with Alpha-Beta Search (2-ply).
        *   Returns best coordinate.
3.  **Update**: UI applies the move, plays logic, and unlocks user input.

### 2.3 Intelligent Hint System
The **AI Hint** feature (`get_hint`) goes beyond simple move suggestion:
1.  **Top-5 Candidates**: Calls the specialized `HybridAgent.get_top_moves()` API.
2.  **Visualization**:
    *   **Best Move**: Thick Blue Circle + Text in Status Bar.
    *   **Alternatives**: Thinner Blue Circles on board + "Coordinate(Confidence%)" usage in logs.
3.  **Confidence Scoring**: Uses the Softmax probability from the Policy Network to show how confident the AI is (e.g., "J10(85%)").

### 2.4 Difficulty Adjustment
*   **Search Depth**: The slider directly controls the ply-depth of the Minimax/Alpha-Beta search.
*   **Model Switching**:
    *   **Easy**: Greedy Agent (Heuristic only).
    *   **Medium**: AlphaBeta (Depth 2-4).
    *   **Hard**: Hybrid Agent (SL Policy + Depth 2 Verification + Defense).

## 3. Configuration & Future Improvements
*   **Config**: AI parameters are loaded from `config/ai_config.json` where applicable (e.g. search distances).
*   **Future (Async)**: Currently, calculation blocks the main thread. A `threading.Thread` with a `queue` for UI callbacks would allow for a "Stop Thinking" button.
