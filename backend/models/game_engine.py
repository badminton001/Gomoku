from typing import List, Tuple, Dict, Any, Optional
from backend.models.board import Board

class GameEngine:

    # 构造函数
    def __init__(self, size: int = 15, first_player: int = 1) -> None:

        # 棋盘对象
        self.board: Board = Board(size=size)

        # 当前该谁落子
        self.current_player: int = first_player
        self.first_player: int = first_player  # 记录先手

        # 游戏状态，0=未结束，1/2=某方胜，3=平局
        self.game_over: bool = False
        self.winner: int = 0

        # 落子历史：[(x, y, player), ...]
        self.move_history: List[Tuple[int, int, int]] = []

        # 统计数据
        self.total_games: int = 0
        self.black_wins: int = 0
        self.white_wins: int = 0
        self.draws: int = 0

    # 重置一局游戏，但保留整体统计
    def reset_game(self, first_player: Optional[int] = None) -> None:

        self.board.reset()
        if first_player is not None:
            self.first_player = first_player
        self.current_player = self.first_player

        self.game_over = False
        self.winner = 0
        self.move_history.clear()

    # 切换当前玩家
    def _switch_player(self) -> None:
        self.current_player = 3 - self.current_player

    # 落子逻辑
    def make_move(self, x: int, y: int) -> bool:

        if self.game_over:
            return False

        # 尝试在棋盘上落子
        success = self.board.place_stone(x, y, self.current_player)
        if not success:
            return False

        # 记录历史
        self.move_history.append((x, y, self.current_player))

        # 判断游戏结果
        result = self.board.get_game_result(with_line=False)
        if result != 0:
            # 游戏结束
            self.game_over = True
            self.winner = result
            self._update_statistics(result)
        else:
            # 游戏未结束，轮到另一个玩家
            self._switch_player()

        return True

    # 由指定玩家在 (x, y) 落子，用于算法自对弈
    def make_move_for(self, x: int, y: int, player: int) -> bool:

        if self.game_over:
            return False

        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")

        success = self.board.place_stone(x, y, player)
        if not success:
            return False

        self.move_history.append((x, y, player))

        result = self.board.get_game_result(with_line=False)
        if result != 0:
            self.game_over = True
            self.winner = result
            self._update_statistics(result)

        return True

    # 更新统计数据
    def _update_statistics(self, result: int) -> None:

        if result == 0:
            return

        self.total_games += 1

        if result == 1:
            self.black_wins += 1
        elif result == 2:
            self.white_wins += 1
        elif result == 3:
            self.draws += 1

    # ---------- 状态查询接口 ----------

    # 返回当前游戏的整体状态
    def get_status(self) -> Dict[str, Any]:
        return {
            "board_size": self.board.size,
            "board": self.board.board,  # 直接给出二维数组
            "current_player": self.current_player,
            "game_over": self.game_over,
            "winner": self.winner,  # 0/1/2/3
            "move_count": self.board.move_count,
            "move_history": list(self.move_history),
            "statistics": {
                "total_games": self.total_games,
                "black_wins": self.black_wins,
                "white_wins": self.white_wins,
                "draws": self.draws,
            },
        }

    # 返回最近一次落子
    def get_last_move(self) -> Optional[Tuple[int, int, int]]:
        if not self.move_history:
            return None
        return self.move_history[-1]

    # 简单打印当前棋盘到终端，仅调试用
    def debug_print_board(self) -> None:
        size = self.board.size
        for y in range(size):
            row = []
            for x in range(size):
                v = self.board.board[x][y]
                if v == 0:
                    row.append(".")
                elif v == 1:
                    row.append("●")  # 黑子
                else:
                    row.append("○")  # 白子
            print(" ".join(row))
        print(f"current_player = {self.current_player}, game_over = {self.game_over}, winner = {self.winner}")