"""Self-Play Engine Core Module

Responsible for automated battles between multiple AI algorithms, result collection, and performance metric tracking.
"""
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime
from pathlib import Path

from backend.engine.board import Board


@dataclass
class GameResult:
    """Single game result"""
    player1: str          # Name of first player (Black)
    player2: str          # Name of second player (White)
    winner: str           # Winner ('player1', 'player2', 'draw')
    total_moves: int      # Total moves
    player1_avg_time: float  # Avg time per move for P1
    player2_avg_time: float  # Avg time per move for P2
    player1_times: List[float]  # List of move times for P1
    player2_times: List[float]  # List of move times for P2
    move_history: List[Tuple[int, int]]  # Move history
    timestamp: str        # Timestamp
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'player1': self.player1,
            'player2': self.player2,
            'winner': self.winner,
            'total_moves': self.total_moves,
            'player1_avg_time': self.player1_avg_time,
            'player2_avg_time': self.player2_avg_time,
            'timestamp': self.timestamp
        }


class SelfPlayEngine:
    """Self-Play Engine
    
    Supports round-robin evaluation of multiple AI algorithms and collects performance metrics.
    """
    
    def __init__(self, board_size: int = 15, use_wandb: bool = False):
        """Initialize Self-Play Engine
        
        Args:
            board_size: Size of the board
            use_wandb: Whether to use Wandb for experiment tracking
        """
        self.board_size = board_size
        self.use_wandb = use_wandb
        self.ai_algorithms = {}
        self.checkpoint_path = "./data/results/self_play/checkpoint.json"
        
        # Wandb initialization (optional)
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project="gomoku-self-play",
                    config={
                        "board_size": board_size,
                        "evaluation_type": "round_robin"
                    }
                )
                print("[OK] Wandb initialized")
            except ImportError:
                print("[WARN] Wandb not available, skipping experiment tracking")
                self.use_wandb = False
    
    def register_ai(self, name: str, ai_instance):
        """Register AI algorithm
        
        Args:
            name: Algorithm name
            ai_instance: AI instance, must have get_move(board, player) method
        """
        self.ai_algorithms[name] = ai_instance
        print(f"[OK] Registered AI: {name}")
    
    def play_single_match(self, ai1_name: str, ai2_name: str, verbose: bool = False) -> GameResult:
        """Play a single match
        
        Args:
            ai1_name: Name of first player (Black)
            ai2_name: Name of second player (White)
            verbose: Whether to print detailed info (including every move)
            
        Returns:
            GameResult object
        """
        board = Board(self.board_size)
        ai1 = self.ai_algorithms[ai1_name]
        ai2 = self.ai_algorithms[ai2_name]
        
        move_history = []
        player1_times = []
        player2_times = []
        
        current_player = 1  # 1 = ai1 (Black), 2 = ai2 (White)
        move_count = 0
        max_moves = self.board_size * self.board_size
        winner = 'draw'
        
        if verbose:
            print(f"\n   {'='*50}")
            print(f"   Match: {ai1_name} (X) vs {ai2_name} (O)")
            print(f"   {'='*50}")
        
        while move_count < max_moves:
            # Select current AI
            current_ai = ai1 if current_player == 1 else ai2
            current_name = ai1_name if current_player == 1 else ai2_name
            player_symbol = "X" if current_player == 1 else "O"
            
            if verbose:
                print(f"\n   Move {move_count + 1}: {current_name} {player_symbol} thinking...", end="", flush=True)
            
            # Move with timing and retry logic
            max_retries = 3
            move = None
            retry_count = 0
            
            start_time = time.time()
            for retry_count in range(max_retries):
                try:
                    move = current_ai.get_move(board, current_player)
                except KeyboardInterrupt:
                    if verbose:
                        print(f"\n[WARN] Match interrupted by user (Ctrl+C)")
                    raise  # Re-raise to stop the tournament
                except Exception as e:
                    if verbose:
                        print(f" ERROR: {e}")
                        import traceback
                        print("\n[ERROR] Exception traceback:")
                        traceback.print_exc()
                    winner = 'player2' if current_player == 1 else 'player1'
                    break
                
                if move is None:  # No valid move
                    continue
                
                x, y = move
                
                # Check validity
                if board.is_valid_move(x, y):
                    # Valid move
                    break
                else:
                    if verbose:
                        print(f" INVALID: ({x}, {y}) - Retry {retry_count + 1}/{max_retries}", end="", flush=True)
                    move = None
            
            elapsed = time.time() - start_time
            
            # If all retries failed, try random valid move
            if move is None:
                from backend.ai.mcts import get_neighbor_moves
                candidates = get_neighbor_moves(board, distance=2)
                valid_candidates = [m for m in candidates if board.is_valid_move(m[0], m[1])]
                
                if valid_candidates:
                    import random
                    move = random.choice(valid_candidates)
                    x, y = move
                    if verbose:
                        print(f" - Auto-selected valid move: ({x}, {y})")
                else:
                    # No valid moves at all, loss
                    if verbose:
                        print(f" - No valid moves available")
                    winner = 'player2' if current_player == 1 else 'player1'
                    break
            
            move_history.append((x, y))
            
            # Record time
            if current_player == 1:
                player1_times.append(elapsed)
            else:
                player2_times.append(elapsed)
            
            # Make move
            board.place_stone(x, y, current_player)
            
            if verbose:
                print(f" ({x},{y}) [{elapsed:.2f}s]")
            
            # Check result
            result = board.get_game_result()
            if result == current_player:
                winner = 'player1' if current_player == 1 else 'player2'
                if verbose:
                    print(f"\n   [WIN] {current_name} WINS!")
                break
            elif result == -1:  # Draw
                winner = 'draw'
                if verbose:
                    print(f"\n   [DRAW] DRAW!")
                break
            
            # Switch player
            current_player = 3 - current_player
            move_count += 1
        
        if verbose:
            print(f"   {'='*50}\n")
        
        # Calculate average time
        avg_time_p1 = np.mean(player1_times) if player1_times else 0.0
        avg_time_p2 = np.mean(player2_times) if player2_times else 0.0
        
        return GameResult(
            player1=ai1_name,
            player2=ai2_name,
            winner=winner,
            total_moves=len(move_history),
            player1_avg_time=avg_time_p1,
            player2_avg_time=avg_time_p2,
            player1_times=player1_times,
            player2_times=player2_times,
            move_history=move_history,
            timestamp=datetime.now().isoformat()
        )
    
    def run_round_robin(self, num_games_per_pair: int = 10, verbose: bool = True, resume: bool = False) -> List[GameResult]:
        """Round Robin Tournament
        
        Args:
            num_games_per_pair: Number of games per pair
            verbose: Whether to print progress
            resume: Whether to resume from checkpoint
            
        Returns:
            List of all game results
        """
        ai_names = sorted(list(self.ai_algorithms.keys()))
        all_results = []
        
        total_matches = len(ai_names) * (len(ai_names) - 1) * num_games_per_pair
        completed = 0
        start_i, start_j, start_game = 0, 0, 0
        
        # Resume
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                all_results = checkpoint['results']
                start_i = checkpoint['current_i']
                start_j = checkpoint['current_j']
                start_game = checkpoint['current_game']
                completed = len(all_results)
                if verbose:
                    print(f"\n[INFO] Resuming from checkpoint...")
                    print(f"   Already completed: {completed}/{total_matches} games")
        
        if verbose and not resume:
            print(f"\n[START] Starting Round Robin Tournament")
            print(f"   Algorithms: {len(ai_names)}")
            print(f"   Total matches: {total_matches}\n")
        
        for i, ai1_name in enumerate(ai_names):
            if i < start_i:
                continue
            for j, ai2_name in enumerate(ai_names):
                if i == j:
                    continue  # Self-play excluded
                if i == start_i and j < start_j:
                    continue
                
                if verbose:
                    print(f"[MATCH] {ai1_name} vs {ai2_name}")
                
                game_start = start_game if (i == start_i and j == start_j) else 0
                for game_num in range(game_start, num_games_per_pair):
                    result = self.play_single_match(ai1_name, ai2_name, verbose=True)  # Enable detailed output
                    all_results.append(result)
                    completed += 1
                    
                    # Save checkpoint every 10 games
                    if completed % 10 == 0:
                        self.save_checkpoint(all_results, i, j, game_num + 1)
                    
                    # Wandb log
                    if self.use_wandb:
                        try:
                            import wandb
                            wandb.log({
                                f"{ai1_name}_vs_{ai2_name}/win": 1 if result.winner == 'player1' else 0,
                                f"{ai1_name}_vs_{ai2_name}/moves": result.total_moves,
                                f"{ai1_name}_vs_{ai2_name}/avg_time": (result.player1_avg_time + result.player2_avg_time) / 2,
                                "completed_games": completed
                            })
                        except:
                            pass
                    
                    if verbose:
                        print(f"   Game {game_num+1}/{num_games_per_pair}: "
                              f"Winner={result.winner}, Moves={result.total_moves}, "
                              f"Time={result.player1_avg_time:.3f}s/{result.player2_avg_time:.3f}s")
                
                if verbose:
                    print(f"   Progress: {completed}/{total_matches} ({100*completed/total_matches:.1f}%)\n")
        
        if verbose:
            print(f"[OK] Tournament completed! Total games: {len(all_results)}")
        
        # Clear checkpoint
        self.clear_checkpoint()
        
        return all_results
    
    def save_results(self, results: List[GameResult], output_dir: str = './data/results/self_play'):
        """Save results
        
        Args:
            results: List of game results
            output_dir: Output directory
            
        Returns:
            (Detailed JSON path, Aggregated CSV path)
        """
        import os
        import pandas as pd
        
        os.makedirs(f"{output_dir}/matches", exist_ok=True)
        os.makedirs(f"{output_dir}/aggregated", exist_ok=True)
        
        # Save detailed results (JSON)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detailed_path = f"{output_dir}/matches/results_{timestamp}.json"
        
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved detailed results to {detailed_path}")
        
        # Save aggregated CSV
        df = pd.DataFrame([r.to_dict() for r in results])
        csv_path = f"{output_dir}/aggregated/results_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"[OK] Saved aggregated results to {csv_path}")
        
        # Print basic stats
        print(f"\n[INFO] Quick Statistics:")
        print(f"   Total games: {len(results)}")
        print(f"   Average moves per game: {df['total_moves'].mean():.1f}")
        print(f"   Average time per move: {(df['player1_avg_time'] + df['player2_avg_time']).mean() / 2:.3f}s")
        
        return detailed_path, csv_path
    
    def save_checkpoint(self, results: List[GameResult], current_i: int, current_j: int, current_game: int):
        """Save Checkpoint
        
        Args:
            results: Current results
            current_i: Outer loop index
            current_j: Inner loop index
            current_game: Current game number
        """
        import os
        checkpoint = {
            'results': [r.to_dict() for r in results],
            'current_i': current_i,
            'current_j': current_j,
            'current_game': current_game,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load Checkpoint
        
        Returns:
            Checkpoint data or None
        """
        if not Path(self.checkpoint_path).exists():
            return None
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # Reconstruct GameResult objects
            results = []
            for r_dict in checkpoint_data['results']:
                results.append(GameResult(
                    player1=r_dict['player1'],
                    player2=r_dict['player2'],
                    winner=r_dict['winner'],
                    total_moves=r_dict['total_moves'],
                    player1_avg_time=r_dict['player1_avg_time'],
                    player2_avg_time=r_dict['player2_avg_time'],
                    player1_times=[],
                    player2_times=[],
                    move_history=[],
                    timestamp=r_dict['timestamp']
                ))
            
            return {
                'results': results,
                'current_i': checkpoint_data['current_i'],
                'current_j': checkpoint_data['current_j'],
                'current_game': checkpoint_data['current_game']
            }
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self):
        """Clear Checkpoint File"""
        if Path(self.checkpoint_path).exists():
            Path(self.checkpoint_path).unlink()
    
    def cleanup(self):
        """Cleanup Resources"""
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
                print("[OK] Wandb session finished")
            except:
                pass
