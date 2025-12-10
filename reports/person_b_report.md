# Person B – Game Replay & AI Move Scoring System

## Data Management Workflow (Page 1)

### System Architecture

Person B's work focuses on **game replay data management** and **AI move scoring analysis**. The system provides comprehensive post-game analysis capabilities for Gomoku games.

#### Data Model Design

**Core Models** ([`backend/models/replay.py`](file:///Users/du/.gemini/antigravity/scratch/Gomoku/backend/models/replay.py)):

```python
class Move(BaseModel):
    step: int              # Move number (1-indexed)
    player: int            # Player ID (1=Black, 2=White)
    x: int                 # X coordinate
    y: int                 # Y coordinate
    timestamp: float       # Unix timestamp
    evaluation_score: Optional[float]  # AI evaluation
    is_critical: bool      # Critical move flag
```

```python
class GameReplay(BaseModel):
    game_id: str
    start_time: datetime
    end_time: Optional[datetime]
    winner: int
    moves: List[Move]
    stats: Optional[Dict]  # Statistics metadata
```

**Design Principles:**
- **JSON Schema Compliance**: Uses Pydantic for automatic schema generation and validation
- **Extensibility**: `Optional` fields allow future enhancements without breaking compatibility
- **Type Safety**: Strict typing ensures data integrity throughout the pipeline

#### Storage Architecture

**Directory Structure:**
```
data/
├── games/           # Raw replay JSON files
├── stats/           # Analyzed move statistics (CSV)
└── charts/          # Visualization outputs (PNG)
```

**File Naming Convention:**
- Game replays: `{game_id}.json`
- Statistics: `{game_id}_multi_algo_stats.csv`
- Charts: `{game_id}_multi_algo_analysis.png`

#### Data Persistence Workflow

**Replay Service** ([`backend/services/replay_service.py`](file:///Users/du/.gemini/antigravity/scratch/Gomoku/backend/services/replay_service.py)):

1. **Save Operation:**
   ```python
   replay_service.save_replay(game_replay)
   # → Serializes to JSON using Pydantic
   # → Saved to data/games/{game_id}.json
   ```

2. **Load Operation:**
   ```python
   game_data = replay_service.load_replay(game_id)
   # → Reads JSON file
   # → Returns dictionary for flexible processing
   ```

**Key Features:**
- Automatic directory creation
- UTF-8 encoding for international character support
- Error handling for missing files
- Pydantic validation ensures data integrity

#### Integration with Scoring System

**Data Flow:**
```
Game Completion → GameReplay → ReplayService.save()
                      ↓
                  JSON File
                      ↓
              MoveScorer.score_moves()
                      ↓
            ┌─────────┴─────────┐
            ↓                   ↓
     CSV Statistics      PNG Visualization
```

The scoring system consumes `Move` objects directly, allowing seamless integration between storage and analysis components.

---

## AI Move Scoring Algorithms (Page 2)

### Multi-Algorithm Evaluation System

The move scoring system ([`backend/services/move_scorer.py`](file:///Users/du/.gemini/antigravity/scratch/Gomoku/backend/services/move_scorer.py)) evaluates each move using multiple AI algorithms to provide comprehensive analysis.

#### Implemented Algorithms

**1. Greedy Heuristic** ([`GreedyAgent`](file:///Users/du/.gemini/antigravity/scratch/Gomoku/backend/algorithms/classic_ai.py#L102-L138))
- **Approach:** One-ply search with heuristic evaluation
- **Speed:** ⚡ Very fast (~2-3ms per move)
- **Evaluation Method:**
  ```python
  score = evaluate_board(board, player)
  # Scores sequences by length and openness
  # Normalizes to [0, 1] using sigmoid
  ```
- **Use Case:** Quick move assessment, baseline comparison

**2. Minimax Search** ([`MinimaxAgent`](file:///Users/du/.gemini/antigravity/scratch/Gomoku/backend/algorithms/classic_ai.py#L141-L234))
- **Approach:** Depth-limited minimax with candidate ordering
- **Configuration:**
  - Depth: 2 plies
  - Candidate limit: 12 moves (pruned by heuristic)
- **Performance:** ~50-100ms per position
- **Strength:** Considers opponent responses up to 2 moves ahead

**3. Alpha-Beta Pruning** ([`AlphaBetaAgent`](file:///Users/du/.gemini/antigravity/scratch/Gomoku/backend/algorithms/classic_ai.py#L237-L377))
- **Approach:** Optimized minimax with pruning and caching
- **Optimizations:**
  - Alpha-beta cutoffs reduce search space
  - Evaluation cache for repeated positions
  - Candidate ordering maximizes pruning
- **Performance:** ~100-150ms per position (depth 2-3)
- **Strength:** Same tactical depth as minimax but faster

**4. Monte Carlo Tree Search (MCTS)** *(Optional)*
- **Approach:** Simulation-based evaluation
- **Configuration:** 100 simulations per position
- **Performance:** ~3-5 seconds per position (slower)
- **Note:** Disabled by default; requires `monte-carlo-tree-search` package

#### Scoring Methodology

**Normalization:**
- Classic algorithms return raw heuristic scores
- Sigmoid normalization maps to [0, 1] range:
  ```python
  normalized = 1 / (1 + exp(-raw_score / 1000))
  ```

**Multi-Algorithm Aggregation:**
```python
# Calculate average and variance
avg_score = mean([greedy, minimax, alphabeta])
variance = var([greedy, minimax, alphabeta])
```

**Critical Move Identification:**
- **Brilliant Move**: `avg_score > 0.8` (very strong)
- **Blunder**: `avg_score < 0.2` (very weak)
- **Normal**: Otherwise

#### Output Format

**CSV Statistics** (`{game_id}_multi_algo_stats.csv`):
```
step,player,x,y,timestamp,greedy_score,minimax_score,alphabeta_score,
avg_score,score_variance,best_algorithm,move_type
```

**Analysis Chart:**
- Line plot showing all algorithm scores over time
- Average score trend (bold black line)
- Annotations: `!` for brilliant moves, `?` for blunders
- Color-coded by algorithm (Blue=Greedy, Orange=Minimax, Green=Alpha-Beta)

#### Performance Characteristics

| Algorithm | Speed | Accuracy | Use Case |
|-----------|-------|----------|----------|
| Greedy | ⚡⚡⚡ | ⭐⭐ | Fast baseline |
| Minimax | ⚡⚡ | ⭐⭐⭐ | Balanced |
| Alpha-Beta | ⚡⚡ | ⭐⭐⭐ | Best overall |
| MCTS | ⚡ | ⭐⭐⭐⭐ | Deep analysis |

#### Algorithm Comparison Metrics

**Consensus Measurement:**
- Low variance (< 0.1): All algorithms agree on move quality
- High variance (> 0.3): Algorithms disagree (interesting position)

**Example Use Cases:**
1. **Post-game review**: Identify critical moments where games were won/lost
2. **Player training**: Highlight blunders for improvement
3. **Algorithm evaluation**: Compare AI performance across different strategies
4. **Game database analysis**: Statistical analysis of move quality patterns

---

### Implementation Notes

**Dependencies:**
- Classic algorithms are self-contained (no external deps)
- MCTS requires `monte-carlo-tree-search` package
- Graceful degradation if MCTS unavailable

**Testing:**
- Integration tests in [`scripts/test_replay_scoring.py`](file:///Users/du/.gemini/antigravity/scratch/Gomoku/scripts/test_replay_scoring.py)
- Validates score ranges, CSV generation, chart creation
- Sample game with 10 moves completes in ~1 second

**Future Enhancements:**
- Parallel evaluation of moves (multiprocessing)
- Additional algorithms (Neural network-based scoring)
- Real-time analysis during gameplay
- Integration with replay API endpoints

---

*Technical Report – Person B | December 2025*
