"""自对弈引擎核心模块

负责6个AI算法的自动对战、结果收集和性能指标追踪
"""
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime
from pathlib import Path

from backend.models.board import Board


@dataclass
class GameResult:
    """单局游戏结果"""
    player1: str          # 先手算法名
    player2: str          # 后手算法名  
    winner: str           # 胜者 ('player1', 'player2', 'draw')
    total_moves: int      # 总步数
    player1_avg_time: float  # 先手平均每步耗时
    player2_avg_time: float  # 后手平均每步耗时
    player1_times: List[float]  # 先手每步耗时列表
    player2_times: List[float]  # 后手每步耗时列表
    move_history: List[Tuple[int, int]]  # 着法历史
    timestamp: str        # 时间戳
    
    def to_dict(self) -> Dict:
        """转换为字典（用于JSON序列化）"""
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
    """自对弈引擎
    
    支持多个AI算法的循环赛评估，收集性能指标
    """
    
    def __init__(self, board_size: int = 15, use_wandb: bool = False):
        """初始化自对弈引擎
        
        Args:
            board_size: 棋盘大小
            use_wandb: 是否使用Wandb进行实验追踪
        """
        self.board_size = board_size
        self.use_wandb = use_wandb
        self.ai_algorithms = {}
        self.checkpoint_path = "./data/results/self_play/checkpoint.json"
        
        # Wandb初始化（可选）
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
                print("✓ Wandb initialized")
            except ImportError:
                print("⚠ Wandb not available, skipping experiment tracking")
                self.use_wandb = False
    
    def register_ai(self, name: str, ai_instance):
        """注册AI算法
        
        Args:
            name: 算法名称
            ai_instance: AI实例，需要有get_move(board, player)方法
        """
        self.ai_algorithms[name] = ai_instance
        print(f"✓ Registered AI: {name}")
    
    def play_single_match(self, ai1_name: str, ai2_name: str, verbose: bool = False) -> GameResult:
        """单场对战
        
        Args:
            ai1_name: 先手AI名称
            ai2_name: 后手AI名称
            verbose: 是否打印详细信息
            
        Returns:
            GameResult对象
        """
        board = Board(self.board_size)
        ai1 = self.ai_algorithms[ai1_name]
        ai2 = self.ai_algorithms[ai2_name]
        
        move_history = []
        player1_times = []
        player2_times = []
        
        current_player = 1  # 1 = ai1 (黑), 2 = ai2 (白)
        move_count = 0
        max_moves = self.board_size * self.board_size
        winner = 'draw'
        
        while move_count < max_moves:
            # 选择当前AI
            current_ai = ai1 if current_player == 1 else ai2
            
            # 计时下棋
            start_time = time.time()
            try:
                move = current_ai.get_move(board, current_player)
            except Exception as e:
                if verbose:
                    print(f"  ⚠ AI error: {e}")
                winner = 'player2' if current_player == 1 else 'player1'
                break
                
            elapsed = time.time() - start_time
            
            if move is None:  # 无合法走法
                winner = 'draw'
                break
            
            x, y = move
            
            # 验证走法合法性
            if not board.is_valid_move(x, y):
                if verbose:
                    print(f"  ⚠ Invalid move: ({x}, {y})")
                winner = 'player2' if current_player == 1 else 'player1'
                break
            
            move_history.append((x, y))
            
            # 记录时间
            if current_player == 1:
                player1_times.append(elapsed)
            else:
                player2_times.append(elapsed)
            
            # 执行走法
            board.place_stone(x, y, current_player)
            
            # 检查胜负
            result = board.get_game_result()
            if result == current_player:
                winner = 'player1' if current_player == 1 else 'player2'
                break
            elif result == -1:  # 平局
                winner = 'draw'
                break
            
            # 切换玩家
            current_player = 3 - current_player
            move_count += 1
        
        # 计算平均时间
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
        """循环赛：每对AI互相对战多次
        
        Args:
            num_games_per_pair: 每对AI对战的场数
            verbose: 是否打印进度信息
            resume: 是否从断点继续
            
        Returns:
            所有对局结果列表
        """
        ai_names = sorted(list(self.ai_algorithms.keys()))
        all_results = []
        
        total_matches = len(ai_names) * (len(ai_names) - 1) * num_games_per_pair
        completed = 0
        start_i, start_j, start_game = 0, 0, 0
        
        # 断点续传
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                all_results = checkpoint['results']
                start_i = checkpoint['current_i']
                start_j = checkpoint['current_j']
                start_game = checkpoint['current_game']
                completed = len(all_results)
                if verbose:
                    print(f"\n🔄 Resuming from checkpoint...")
                    print(f"   Already completed: {completed}/{total_matches} games")
        
        if verbose and not resume:
            print(f"\n🎮 Starting Round Robin Tournament")
            print(f"   Algorithms: {len(ai_names)}")
            print(f"   Total matches: {total_matches}\n")
        
        for i, ai1_name in enumerate(ai_names):
            if i < start_i:
                continue
            for j, ai2_name in enumerate(ai_names):
                if i == j:
                    continue  # 不自己和自己对战
                if i == start_i and j < start_j:
                    continue
                
                if verbose:
                    print(f"⚔️  {ai1_name} vs {ai2_name}")
                
                game_start = start_game if (i == start_i and j == start_j) else 0
                for game_num in range(game_start, num_games_per_pair):
                    result = self.play_single_match(ai1_name, ai2_name, verbose=False)
                    all_results.append(result)
                    completed += 1
                    
                    # 每10场保存一次checkpoint
                    if completed % 10 == 0:
                        self.save_checkpoint(all_results, i, j, game_num + 1)
                    
                    # Wandb日志
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
            print(f"✅ Tournament completed! Total games: {len(all_results)}")
        
        # 清除checkpoint
        self.clear_checkpoint()
        
        return all_results
    
    def save_results(self, results: List[GameResult], output_dir: str = './data/results/self_play'):
        """保存结果
        
        Args:
            results: 对局结果列表
            output_dir: 输出目录
            
        Returns:
            (详细结果路径, 汇总CSV路径)
        """
        import os
        import pandas as pd
        
        os.makedirs(f"{output_dir}/matches", exist_ok=True)
        os.makedirs(f"{output_dir}/aggregated", exist_ok=True)
        
        # 保存每局详细结果（JSON）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detailed_path = f"{output_dir}/matches/results_{timestamp}.json"
        
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved detailed results to {detailed_path}")
        
        # 保存汇总CSV
        df = pd.DataFrame([r.to_dict() for r in results])
        csv_path = f"{output_dir}/aggregated/results_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"✓ Saved aggregated results to {csv_path}")
        
        # 打印基础统计
        print(f"\n📊 Quick Statistics:")
        print(f"   Total games: {len(results)}")
        print(f"   Average moves per game: {df['total_moves'].mean():.1f}")
        print(f"   Average time per move: {(df['player1_avg_time'] + df['player2_avg_time']).mean() / 2:.3f}s")
        
        return detailed_path, csv_path
    
    def save_checkpoint(self, results: List[GameResult], current_i: int, current_j: int, current_game: int):
        """保存断点
        
        Args:
            results: 当前所有结果
            current_i: 当前外层循环索引
            current_j: 当前内层循环索引
            current_game: 当前游戏编号
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
        """加载断点
        
        Returns:
            断点数据或None
        """
        if not Path(self.checkpoint_path).exists():
            return None
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # 重建GameResult对象
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
            print(f"⚠ Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self):
        """清除断点文件"""
        if Path(self.checkpoint_path).exists():
            Path(self.checkpoint_path).unlink()
    
    def cleanup(self):
        """清理资源"""
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
                print("✓ Wandb session finished")
            except:
                pass
