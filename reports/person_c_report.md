# Person C – Classic Gomoku Algorithms (Work Package)

## Algorithm Principles (1 page)
- **Random Baseline**: Sample a legal move within a distance-limited neighborhood of existing stones. Purpose is to quantify improvement over chance.
- **Greedy Heuristic**: One-ply search that scores every candidate move using the shared heuristic and picks the max. Uses local neighborhood pruning to keep branching small.
- **Shared Heuristic**: Line-length scoring with openness weighting. Five-in-a-row is a terminal win; open-ended longer runs outrank closed ones. The evaluator returns `player_score - 1.1 * opponent_score` to bias toward blocking.
- **Minimax (Depth-Limited)**: Full game-tree expansion to a fixed depth using the shared heuristic on leaf nodes. Candidate ordering is based on the heuristic plus a large “block bonus” when a move stops an immediate loss, ensuring defensive consistency.
- **Alpha-Beta**: Minimax plus alpha-beta pruning with the same candidate ordering. This preserves optimality while reducing explored nodes.
- **Neighborhood Generator**: All algorithms reuse `get_neighbor_moves` (radius configurable) so the search space is consistent and centered on tactical regions.

## Configuration & Optimization (1 page)
- **Config file**: `backend/config/ai_config.json` (JSON chosen to avoid extra deps; YAML can be added if needed).
  - `neighbor_distance`: shared locality radius.
  - `greedy.distance`
  - `minimax.depth`, `distance`, `candidate_limit`
  - `alpha_beta.depth`, `distance`, `candidate_limit`
- **Hyperparameter choices**:
  - Distance radius `2`: balances tactical coverage and branching.
  - Candidate limits `10–12`: retain high-value moves while trimming noise.
  - Depths: Minimax=2 for responsiveness; Alpha-Beta=3 for a stronger search with pruning.
- **Implementation notes**:
  - All searchers record `SearchMetrics(elapsed_ms, explored_nodes, candidate_moves)` for benchmarking.
  - Heuristic and candidate ordering are identical across Greedy/Minimax/Alpha-Beta to ensure consistency.
  - Block bonus (900k) is injected into ordering to prioritize immediate threat blocking before deeper search.
- **How to tune**:
  - Edit `backend/config/ai_config.json` and re-run benchmarks/tests.
  - Increase depth or candidate_limit for strength; decrease for speed.
  - Adjust `neighbor_distance` if mid/late-game spaces feel too narrow or too broad.

## Performance Benchmark (mid-game, 10×10 board)
Generated via `python backend/benchmarks/run_classic_benchmark.py` (seed=42).

| Agent      | Depth | Candidates | Explored Nodes | Time (ms) | Move  | Heuristic After Move |
|------------|-------|------------|----------------|-----------|-------|----------------------|
| Random     | –     | 44         | 1              | 0.084     | (7,2) | 74.60                |
| Greedy     | 1     | 44         | 44             | 2.135     | (4,4) | 5201.60              |
| Minimax    | 2     | 44         | 157            | 63.771    | (6,7) | 5022.40              |
| Alpha-Beta | 3     | 44         | 228            | 156.848   | (4,4) | 5201.60              |

Observations:
- Greedy and Alpha-Beta both prefer (4,4) as a high-quality extension; Minimax d2 selects (6,7) defensively with similar heuristic strength.
- Alpha-Beta achieves deeper search with moderate node count, confirming pruning efficacy.
- Metrics provide a reproducible baseline for future tuning (e.g., depth vs. latency).
