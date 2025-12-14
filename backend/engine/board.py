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
    
    # --------禁手检测--------

    # 四个基本方向：横、竖、主对角线、副对角线
    _DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]

    # 统计某个方向上的最大连续同色长度（包含 (x, y)）
    def _max_run_len(self, x: int, y: int, dx: int, dy: int, player: int) -> int:
        if not self.is_inside(x, y) or self.board[x][y] != player:
            return 0
        length = 1
        cx, cy = x + dx, y + dy
        while self.is_inside(cx, cy) and self.board[cx][cy] == player:
            length += 1
            cx += dx
            cy += dy
        cx, cy = x - dx, y - dy
        while self.is_inside(cx, cy) and self.board[cx][cy] == player:
            length += 1
            cx -= dx
            cy -= dy
        return length

    # 整盘有没有某方的“五连”（恰好 5 个连续）
    def _has_five_anywhere(self, player: int) -> bool:
        n = self.size
        for x in range(n):
            for y in range(n):
                if self.board[x][y] != player:
                    continue
                for dx, dy in self._DIRS:
                    if self._max_run_len(x, y, dx, dy, player) == 5:
                        return True
        return False

    # 判断包含 (x, y) 的某方向上是否有“直四”（两头空的连续 4 子）
    def _has_straight_four_including(self, x: int, y: int, player: int) -> bool:
        for dx, dy in self._DIRS:
            # 以 (x, y) 为中心，找到这条连续同色段的两端
            if self.board[x][y] != player:
                continue

            # 向正方向
            forward = 0
            cx, cy = x + dx, y + dy
            while self.is_inside(cx, cy) and self.board[cx][cy] == player:
                forward += 1
                cx += dx
                cy += dy
            end_fx, end_fy = cx, cy  # 正向第一个非 player 的格子

            # 向反方向
            backward = 0
            cx, cy = x - dx, y - dy
            while self.is_inside(cx, cy) and self.board[cx][cy] == player:
                backward += 1
                cx -= dx
                cy -= dy
            end_bx, end_by = cx, cy  # 反向第一个非 player 的格子

            run_len = 1 + forward + backward
            if run_len != 4:
                continue

            # 两端必须在棋盘内且为空，才是“直四”
            if not (self.is_inside(end_fx, end_fy) and self.is_inside(end_bx, end_by)):
                continue
            if self.board[end_fx][end_fy] == 0 and self.board[end_bx][end_by] == 0:
                return True
        return False

    # 统计：在已经把 (x, y) 落子为 player 的前提下，这一手产生了多少个“四”（方向数）。
    # 定义：存在某个空位 q，使得在 q 落一子后，可以得到一个“五连”，且这 5 个子中包含 (x, y)。
    def _count_fours_from_move(self, x: int, y: int, player: int) -> int:
        count = 0

        for dx, dy in self._DIRS:
            has_four_in_dir = False

            # 先收集这条线上所有坐标
            coords = []
            # 先往负方向
            cx, cy = x, y
            while self.is_inside(cx, cy):
                coords.append((cx, cy))
                cx -= dx
                cy -= dy
            coords.reverse()  # 现在 coords 从负端到 (x, y)

            # 再往正方向补全
            cx, cy = x + dx, y + dy
            while self.is_inside(cx, cy):
                coords.append((cx, cy))
                cx += dx
                cy += dy

            # 对这条线上的每一个空格，尝试落子，看是否会产生“包含 (x, y) 的五连”
            for (qx, qy) in coords:
                if self.board[qx][qy] != 0:
                    continue
                self.board[qx][qy] = player  # 假想第二手
                # 在这个假想局面下，看看 (x, y) 是否在某个方向形成恰好 5 连
                run_len = self._max_run_len(x, y, dx, dy, player)
                self.board[qx][qy] = 0       # 撤回

                if run_len == 5:
                    has_four_in_dir = True
                    break  # 这一方向已经确定有一个“四”了

            if has_four_in_dir:
                count += 1

        return count

    def _find_threes_from_move(self, x: int, y: int, player: int) -> List[List[Tuple[int, int]]]:
        """
        返回一个列表 threes：
        threes 中的每个元素都是一个三，对应一条方向；
        其中存的是这个“三”的所有 vital points（可能 1 个，也可能 2 个）。
        """
        threes: List[List[Tuple[int, int]]] = []

        for dx, dy in self._DIRS:
            vital_points: List[Tuple[int, int]] = []

            # 收集这一方向上的所有格子坐标（含当前落子）
            coords: List[Tuple[int, int]] = []
            cx, cy = x, y
            while self.is_inside(cx, cy):
                coords.append((cx, cy))
                cx -= dx
                cy -= dy
            coords.reverse()
            cx, cy = x + dx, y + dy
            while self.is_inside(cx, cy):
                coords.append((cx, cy))
                cx += dx
                cy += dy

            # 在这条线上找“能让 (x,y) 参与直四、且无五连”的 vital point
            for (qx, qy) in coords:
                if self.board[qx][qy] != 0:
                    continue

                # 假想在 (qx, qy) 再下一个子，看看是否构成“直四”但不五连
                self.board[qx][qy] = player

                # 不能产生五连（否则不算三，而是“直接赢或四四/长连相关”）
                if self._has_five_anywhere(player):
                    self.board[qx][qy] = 0
                    continue

                # 必须产生一个包含 (x, y) 的直四
                if self._has_straight_four_including(x, y, player):
                    vital_points.append((qx, qy))

                self.board[qx][qy] = 0

            if vital_points:
                threes.append(vital_points)

        return threes

    def _three_is_real(self, vital_points: List[Tuple[int, int]], depth: int) -> bool:
        """
        一个三要想算“真的三”，只要它的 vital_points 里
        至少有一个点，是接下来“允许落子”的（考虑递归）。
        """
        for (qx, qy) in vital_points:
            if self.board[qx][qy] != 0:
                continue
            if self._is_vital_point_legal(qx, qy, depth):
                return True
        return False

    def _is_vital_point_legal(self, px: int, py: int, depth: int) -> bool:
        """
        判断在当前局面下，黑棋在 vital point (px, py) 落子是否“允许”。
        这里考虑了：
        - 五连（允许）
        - 长连 / 四四（直接禁）
        - 三三：如果是“真实 double-three”，则禁；否则允许。
        depth：递归深度，防止无限展开，通常设为 2 或 3。
        """
        assert self.board[px][py] == 0
        self.board[px][py] = 1  # 假想在 vital point 落子

        # 1) 检查五连 / 长连
        has_five = False
        has_overline = False
        for dx, dy in self._DIRS:
            run_len = self._max_run_len(px, py, dx, dy, 1)
            if run_len == 5:
                has_five = True
            elif run_len >= 6:
                has_overline = True

        if has_five:
            # 五连优先：这步是赢棋，不看禁手
            self.board[px][py] = 0
            return True

        if has_overline:
            self.board[px][py] = 0
            return False

        # 2) 四四：如果形成 double-four，则这个 vital point 不合法
        fours = self._count_fours_from_move(px, py, 1)
        if fours >= 2:
            self.board[px][py] = 0
            return False

        # 3) 三三（可能需要递归）
        threes = self._find_threes_from_move(px, py, 1)

        if len(threes) < 2:
            # 不是 double-three，这步就算合法
            self.board[px][py] = 0
            return True

        # 现在：这步看起来是 double-three，需要判断是“真三三”还是“假三三”

        if depth <= 0:
            # 深度耗尽，保守起见：把它当“真三三”（禁手）
            self.board[px][py] = 0
            return False

        # 递归地检查这步形成的每一个“三”是否“真的三”
        real_three_count = 0
        for vital_points in threes:
            if self._three_is_real(vital_points, depth - 1):
                real_three_count += 1
                if real_three_count >= 2:
                    # 真正的 double-three：这个 vital point 不合法
                    self.board[px][py] = 0
                    return False

        # 形成的“真的三”不足两个 → 虽然表面看是 double-three，其实是假禁手，这步允许
        self.board[px][py] = 0
        return True

    def _is_forbidden_move_black(self, x: int, y: int) -> bool:
        assert self.board[x][y] == 0
        self.board[x][y] = 1  # 假想落子

        # ---- 1. 五连 / 长连 ----
        has_five = False
        has_overline = False
        for dx, dy in self._DIRS:
            run_len = self._max_run_len(x, y, dx, dy, 1)
            if run_len == 5:
                has_five = True
            elif run_len >= 6:
                has_overline = True

        if has_five:
            self.board[x][y] = 0
            return False  # 赢棋优先，不看禁手

        if has_overline:
            self.board[x][y] = 0
            return True

        # ---- 2. 四四禁手 ----
        fours = self._count_fours_from_move(x, y, 1)
        if fours >= 2:
            self.board[x][y] = 0
            return True

        # ---- 3. 三三禁手（含递归）----
        threes = self._find_threes_from_move(x, y, 1)

        if len(threes) < 2:
            self.board[x][y] = 0
            return False  # 不构成 double-three

        # 用递归逻辑，只统计“真的三”的数量
        real_three_count = 0
        for vital_points in threes:
            if self._three_is_real(vital_points, depth=2):  # depth 可以调大一点，比如 3
                real_three_count += 1
                if real_three_count >= 2:
                    self.board[x][y] = 0
                    return True

        self.board[x][y] = 0
        return False


    def is_valid_move(self, x: int, y: int) -> bool:
        if not (self.is_inside(x, y) and self.is_empty(x, y)):
            return False

        # 根据步数判断当前轮到谁走：
        current_player = 1 if (self.move_count % 2 == 0) else 2

        # 禁手只对黑棋（先手）生效
        if current_player == 1:
            if self._is_forbidden_move_black(x, y):
                return False

        return True

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
