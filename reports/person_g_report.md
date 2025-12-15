# Person G Report: Backend API & Performance Analysis

## 1. System Architecture (后端系统架构)

The Backend System is built on a **Microservices-ready** architecture using **FastAPI** for high-performance, asynchronous request handling.

### 1.1 Tech Stack
*   **Framework**: FastAPI (Python 3.10+) + Uvicorn (ASGI Server).
*   **Data Validation**: Pydantic models (`BoardState`, `MoveResponse`) implementation strict type checking.
*   **AI Integration**: Plug-and-play AI modules (`Greedy`, `AlphaBeta`, `Hybrid`, `DQN`, `Random`) dynamically loaded to minimize memory footprint.
*   **Logging**: Centralized system logging (`server.log`) tracking request attributes, processing time, and errors.

### 1.2 Core Components
*   **API Layer (`backend/api/main.py`)**: 
    *   Stateless connection handling.
    *   `/ai/predict`: Main inference endpoint accepting 15x15 matrix.
    *   `/config`: Exposes server-side AI configurations.
    *   `/health`: Heartbeat for load balancers.
*   **Engine Interface**: Adapts API JSON playloads into internal `Board` objects for the `GameEngine`.

---

## 2. Performance Analysis Workflow (性能分析工作流)

We implemented a multi-tiered performance verification strategy ranging from unit benchmarking to full-load stress testing.

### 2.1 Automated Load Testing (`Locust`)
*   **Tool**: Locust (Python-based load testing).
*   **Script**: `scripts/locustfile.py`
*   **Scenarios**:
    *   **High Concurrency**: Simulating 100+ concurrent games hitting the `/ai/predict` endpoint.
    *   **Mixed Workload**: 75% fast "Greedy" requests (simulating early game/easy bots) vs 25% heavy "AlphaBeta" requests.
*   **Metrics**: Latency (P50, P95, P99), RPS (Requests Per Second), Failure Rate.

### 2.2 Hyperparameter Tuning (`Optuna`)
*   **Tool**: Optuna (Automated Hyperparameter Optimization).
*   **Script**: `backend/tune_ai.py`
*   **Objective**: Maximize Win Rate against a baseline "Strong Agent" by tuning:
    *   Search Depth (2-6)
    *   Evaluation Weights (Shape scores)
    *   Search Width (Candidate Limits)
*   **Result**: Automatically found optimal config saved to `config/ai_config.json`.

### 2.3 Historical Analysis
*   **Tool**: Custom Pandas Pipeline (`scripts/analyze_performance.py`).
*   **Metrics**: Win Rate matrices, Move Time distributions, ELO rating updates.

---

## 3. Optimization & Bottleneck Analysis (优化方案)

### 3.1 Identified Bottlenecks
1.  **Serialization Overhead**: JSON parsing of 15x15 grid (225 ints) adds ~2ms latency per request.
2.  **GIL Blocking**: Python's Global Interpreter Lock prevents true parallelism during CPU-bound Alpha-Beta search.
3.  **Model Loading**: Loading PyTorch models (`HybridAgent`) on every request (if stateless) is prohibitively slow.

### 3.2 Implemented Optimizations
*   **FastAPI Async**: Endpoint defined as standard `def` (not `async def`) for CPU-bound tasks to allow thread-pooling by Starlette/Uvicorn, ensuring the main loop isn't blocked.
*   **Agent Caching**: `HybridAgent` and `QLearningAgent` are loaded once (Singleton pattern or Caching strategy in `gui.py` and planned for `main.py`) to avoid I/O latency.
*   **Search Pruning**: Alpha-Beta uses "Iterative Deepening" and "Candidate Restriction" (only top 10 moves) to keep response times < 2.0s.

### 3.3 Proposed Future Optimizations
*   **Binary Protocol**: Switch from JSON to **Protobuf** or raw bytes for board state transmission (est. 10x faster serialization).
*   **Batching**: Implement `Batch Inference` for the Neural Network if concurrent requests scale up (using `torch.utils.data.DataLoader`).
*   **MLflow Integration**: While Optuna handles tuning, full experiment tracking (MLflow) is planned for the next phase to version-control model binaries.
