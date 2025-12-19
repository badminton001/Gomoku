"""Parallel Self-Play Evaluation"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd

from backend.api.services.winplay_service import SelfPlayEngine, GameResult
from backend.ai.baselines import GreedyAgent
from backend.ai.minimax import AlphaBetaAgent
from backend.ai.mcts import MCTSAgent
from backend.ai.dqn import QLearningAgent
from backend.ai.hybrid import HybridAgent


def get_output_dir(custom_dir: str = None) -> Path:
    """Get output directory."""
    if custom_dir:
        return Path(custom_dir)
    return Path("./data/results/self_play/5processes")


def create_ai_agents() -> Dict[str, any]:
    """Create agents."""
    agents = {}
    agents["Greedy"] = GreedyAgent(distance=2)
    agents["AlphaBeta"] = AlphaBetaAgent(depth=4, time_limit=4.0)
    agents["Minimax"] = AlphaBetaAgent(depth=2, time_limit=2.0)
    agents["MCTS-500"] = MCTSAgent(iteration_limit=500)
    
    try:
        agents["DQN"] = QLearningAgent(model_path="models/dqn_v1_final")
    except Exception as e:
        print(f"[WARNING] Failed to load DQN: {e}")
        agents["DQN"] = None
    
    try:
        agents["Hybrid"] = HybridAgent(model_path="models/sl_policy_v2_kaggle.pth", device="cpu")
    except Exception as e:
        print(f"[WARNING] Failed to load Hybrid: {e}")
        agents["Hybrid"] = None
    
    agents = {k: v for k, v in agents.items() if v is not None}
    return agents


def get_matchups(ai_names: List[str]) -> List[Tuple[str, str]]:
    """Generate matchups."""
    matchups = []
    for i, ai1 in enumerate(ai_names):
        for ai2 in ai_names[i+1:]:
            matchups.append((ai1, ai2))
    return matchups


def run_process_batch(process_id: int, matchups: List[Tuple[str, str]], games_per_pair: int, output_dir_str: str = None):
    """Run batch process."""
    print(f"[Process {process_id}] Started, processing {len(matchups)} matchups")
    
    # Create agents
    ai_agents = create_ai_agents()
    
    # Create engine
    engine = SelfPlayEngine(board_size=15, use_wandb=False)
    
    # Register AI
    for ai1, ai2 in matchups:
        if ai1 not in registered_ais:
            engine.register_ai(ai1, ai_agents[ai1])
            registered_ais.add(ai1)
        if ai2 not in registered_ais:
            engine.register_ai(ai2, ai_agents[ai2])
            registered_ais.add(ai2)
    
    # Run matchups
    batch_results = []
    total_games = len(matchups) * games_per_pair * 2
    completed = 0
    
    for idx, (ai1, ai2) in enumerate(matchups, 1):
        print(f"[Process {process_id}] Matchup {idx}/{len(matchups)}: {ai1} vs {ai2}")
        
        for game_num in range(games_per_pair):
            # P1 vs P2
            try:
                result1 = engine.play_single_match(ai1, ai2, verbose=False)
                batch_results.append(result1.to_dict())
                completed += 1
                if completed % 5 == 0:
                   print(f"[Process {process_id}]   > {completed}/{total_games} ...")
            except Exception as e:
                print(f"[Process {process_id}]   x Failed: {e}")
            
            # P2 vs P1
            try:
                result2 = engine.play_single_match(ai2, ai1, verbose=False)
                batch_results.append(result2.to_dict())
                completed += 1
                if completed % 5 == 0:
                   print(f"[Process {process_id}]   > {completed}/{total_games} ...")
            except Exception as e:
                print(f"[Process {process_id}]   x Failed: {e}")
    
    # Save results
    if output_dir_str:
        final_output_dir = Path(output_dir_str)
    else:
        final_output_dir = get_output_dir()
        
    final_output_dir.mkdir(parents=True, exist_ok=True)
    batch_file = final_output_dir / f"results_process_{process_id}.csv"
    
    if batch_results:
       df = pd.DataFrame(batch_results)
       df.to_csv(batch_file, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="5-Process Parallel Self-Play Evaluation")
    parser.add_argument('--games-per-pair', type=int, default=10, help='Games per pair (one way)')
    parser.add_argument('--threads', type=int, default=5, help='Number of processes')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    output_dir = get_output_dir(args.output_dir)
    print(f"Output Directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tasks
    temp_agents = create_ai_agents()
    ai_names = list(temp_agents.keys())
    matchups = get_matchups(ai_names)
    
    # 2. Distribute
    num_processes = min(args.threads, len(matchups)) or 1
    chunk_size = len(matchups) // num_processes
    chunks = [matchups[i:i + chunk_size] for i in range(0, len(matchups), chunk_size)]
    if len(chunks) > num_processes:
        last = chunks.pop()
        chunks[-1].extend(last)

    # 3. Run
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(
            target=run_process_batch,
            args=(i, chunks[i], args.games_per_pair, str(output_dir))
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    # 4. Aggregate
    print("Aggregating results...")
    all_dfs = []
    for f in output_dir.glob("results_process_*.csv"):
        try: all_dfs.append(pd.read_csv(f))
        except: pass
        
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = output_dir / f"results_{timestamp}.csv"
        final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"Generated summary file: {save_path}")
    else:
        print("No data generated")
