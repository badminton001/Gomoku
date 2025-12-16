import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.models.board import Board
from backend.algorithms.classic_ai import AlphaBetaAgent, random_move
from backend.algorithms.mcts_ai import MCTSAgent
from backend.algorithms.qlearning_ai import QLearningAgent

def test_agent_speed(agent, name, rounds=5):
    board = Board(size=15)
    board.place_stone(7, 7, 1)
    board.place_stone(7, 8, 2)
    board.place_stone(6, 6, 1)
    
    print(f"--- Benchmarking {name} ---")
    start_time = time.perf_counter()
    
    for i in range(rounds):
        move = agent.get_move(board, 1)
        print(f"  Round {i+1}: Move {move}")
        
    elapsed = (time.perf_counter() - start_time) / rounds
    print(f"  Average Time: {elapsed:.4f} seconds/move\n")

if __name__ == "__main__":
    mcts = MCTSAgent(time_limit=1000, iteration_limit=500)
    test_agent_speed(mcts, "MCTS (1s limit)")
    
    dqn = QLearningAgent(model_path="models/dqn_gomoku")
    test_agent_speed(dqn, "DQN Agent")
