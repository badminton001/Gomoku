from typing import List, Optional, Tuple

class Board:

    # 构造函数
    def __init__(self, size: int = 15) -> None:
        self.size: int = size
        self.board: List[List[int]] = [
            [0 for _ in range(size)] for _ in range(size)
        ] # 棋盘
        self.move_count: int = 0 # 步数

    # 重置棋盘到初始状态
    def reset(self) -> None:
        for x in range(self.size):
            for y in range(self.size):
                self.board[x][y] = 0
        self.move_count = 0

    # 判断坐标是否在棋盘范围内
    def is_inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    # 判断该位置是否为空
    def is_empty(self, x: int, y: int) -> bool:
        return self.board[x][y] == 0

    # 判断一个落子是否合法
    def is_valid_move(self, x: int, y: int) -> bool:
        return self.is_inside(x, y) and self.is_empty(x, y)

    # 执行落子
    def place_stone(self, x: int, y: int, player: int) -> bool:
        # 检查玩家是否合法
        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")
        # 检查是否是有效落子
        if not self.is_valid_move(x, y):
            return False
        self.board[x][y] = player
        self.move_count += 1
        return True

    # 获取棋盘上 (x, y) 的值
    def get_cell(self, x: int, y: int) -> int:
        if not self.is_inside(x, y):
            raise ValueError(f"({x}, {y}) is outside the board")
        return self.board[x][y]

    # 胜负平局判定
    def check_five_in_a_row(
        self, player: int
    ) -> Optional[List[Tuple[int, int]]]:
        n = self.size

        # 1. 横向（x 增加，y 不变）
        for x in range(n - 4):
            for y in range(n):
                if (
                    self.board[x][y] == player
                    and self.board[x + 1][y] == player
                    and self.board[x + 2][y] == player
                    and self.board[x + 3][y] == player
                    and self.board[x + 4][y] == player
                ):
                    return [(x + i, y) for i in range(5)]

        # 2. 纵向（y 增加，x 不变）
        for x in range(n):
            for y in range(n - 4):
                if (
                    self.board[x][y] == player
                    and self.board[x][y + 1] == player
                    and self.board[x][y + 2] == player
                    and self.board[x][y + 3] == player
                    and self.board[x][y + 4] == player
                ):
                    return [(x, y + i) for i in range(5)]

        # 3. 左上 → 右下 斜线
        for x in range(n - 4):
            for y in range(n - 4):
                if (
                    self.board[x][y] == player
                    and self.board[x + 1][y + 1] == player
                    and self.board[x + 2][y + 2] == player
                    and self.board[x + 3][y + 3] == player
                    and self.board[x + 4][y + 4] == player
                ):
                    return [(x + i, y + i) for i in range(5)]
        # 4. 右上 → 左下 斜线

        for x in range(n - 4):
            for y in range(n - 4):
                if (
                    self.board[x + 4][y] == player
                    and self.board[x + 3][y + 1] == player
                    and self.board[x + 2][y + 2] == player
                    and self.board[x + 1][y + 3] == player
                    and self.board[x][y + 4] == player
                ):
                    return [(x + 4 - i, y + i) for i in range(5)]
        return None

    # 判断棋盘是否已经下满
    def is_full(self) -> bool:
        return self.move_count >= self.size * self.size

    # 判断当前棋局状态
    def get_game_result(
        self, with_line: bool = False
    ):
        # 玩家 1 是否赢
        line1 = self.check_five_in_a_row(1)
        if line1 is not None:
            return (1, line1) if with_line else 1

        # 玩家 2 是否赢
        line2 = self.check_five_in_a_row(2)
        if line2 is not None:
            return (2, line2) if with_line else 2

        # 是否平局
        if self.is_full():
            return (3, [(-1, -1)]) if with_line else 3

        # 游戏进行中
        return (0, [(-1, -1)]) if with_line else 0

    # 棋盘序列化，将当前棋盘编码为字符串
    def to_string(self) -> str:
        codes: List[str] = []
        for x in range(self.size):
            for y in range(self.size):
                codes.append(str(self.board[x][y]))
        return "".join(codes)
