"""
Board Module.

Handles the 15x15 Gomoku board state, move validation, and game rules.
Includes Renju-style forbidden move detection (3-3, 4-4, overline) for the black player.
"""
from typing import List, Optional, Tuple

class Board:
    def __init__(self, size: int = 15) -> None:
        self.size: int = size
        # 0: Empty, 1: Black, 2: White
        self.board: List[List[int]] = [[0] * size for _ in range(size)]
        self.move_count: int = 0

    def reset(self) -> None:
        for x in range(self.size):
            for y in range(self.size):
                self.board[x][y] = 0
        self.move_count = 0

    def is_inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def is_empty(self, x: int, y: int) -> bool:
        return self.board[x][y] == 0
    
    # -------- Forbidden Move Logic --------
    _DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]

    def _max_run_len(self, x: int, y: int, dx: int, dy: int, player: int) -> int:
        """Count consecutive stones including (x, y)."""
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

    def _has_five_anywhere(self, player: int) -> bool:
        """Check if any 5-in-a-row exists for player."""
        n = self.size
        for x in range(n):
            for y in range(n):
                if self.board[x][y] != player:
                    continue
                for dx, dy in self._DIRS:
                    if self._max_run_len(x, y, dx, dy, player) == 5:
                        return True
        return False

    def _has_straight_four_including(self, x: int, y: int, player: int) -> bool:
        """Check if (x,y) forms a 'straight four' (open ends)."""
        for dx, dy in self._DIRS:
            if self.board[x][y] != player:
                continue

            forward = 0
            cx, cy = x + dx, y + dy
            while self.is_inside(cx, cy) and self.board[cx][cy] == player:
                forward += 1
                cx += dx
                cy += dy
            end_fx, end_fy = cx, cy

            backward = 0
            cx, cy = x - dx, y - dy
            while self.is_inside(cx, cy) and self.board[cx][cy] == player:
                backward += 1
                cx -= dx
                cy -= dy
            end_bx, end_by = cx, cy

            if (1 + forward + backward) == 4:
                if (self.is_inside(end_fx, end_fy) and self.board[end_fx][end_fy] == 0 and
                    self.is_inside(end_bx, end_by) and self.board[end_bx][end_by] == 0):
                    return True
        return False

    def _count_fours_from_move(self, x: int, y: int, player: int) -> int:
        """Count potential 4s created by this move."""
        count = 0

        for dx, dy in self._DIRS:
            has_four_in_dir = False

            # Collect all coordinates on this line
            coords = []
            # Backward
            cx, cy = x, y
            while self.is_inside(cx, cy):
                coords.append((cx, cy))
                cx -= dx
                cy -= dy
            coords.reverse() 

            # Forward
            cx, cy = x + dx, y + dy
            while self.is_inside(cx, cy):
                coords.append((cx, cy))
                cx += dx
                cy += dy

            # Try placing stone at each empty spot to see if it creates a 'five'
            for (qx, qy) in coords:
                if self.board[qx][qy] != 0:
                    continue
                self.board[qx][qy] = player 
                # Check for 5 length run
                run_len = self._max_run_len(x, y, dx, dy, player)
                self.board[qx][qy] = 0

                if run_len == 5:
                    has_four_in_dir = True
                    break

            if has_four_in_dir:
                count += 1

        return count

    def _find_threes_from_move(self, x: int, y: int, player: int) -> List[List[Tuple[int, int]]]:
        """
        Returns a list of 'threes'.
        Each element contains the vital points of that three.
        """
        threes: List[List[Tuple[int, int]]] = []

        for dx, dy in self._DIRS:
            vital_points: List[Tuple[int, int]] = []

            # Collect line coordinates
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

            # Find vital points that make this a 'straight four' but not a 'five'
            for (qx, qy) in coords:
                if self.board[qx][qy] != 0:
                    continue

                # Hypothesis: Place stone at qx, qy
                self.board[qx][qy] = player

                # Must not form 5 (creates 5 -> creates 4 is wrong logic)
                if self._has_five_anywhere(player):
                    self.board[qx][qy] = 0
                    continue

                # Must form straight four including (x,y)
                if self._has_straight_four_including(x, y, player):
                    vital_points.append((qx, qy))

                self.board[qx][qy] = 0

            if vital_points:
                threes.append(vital_points)

        return threes

    def _three_is_real(self, vital_points: List[Tuple[int, int]], depth: int) -> bool:
        """
        A Three is 'Real' if at least one vital point is a legal move.
        """
        for (qx, qy) in vital_points:
            if self.board[qx][qy] != 0:
                continue
            if self._is_vital_point_legal(qx, qy, depth):
                return True
        return False

    def _is_vital_point_legal(self, px: int, py: int, depth: int) -> bool:
        """
        Check if placing at vital point is legal (Forbidden move check).
        Recursively checks for 3-3, 4-4, Long-5.
        """
        assert self.board[px][py] == 0
        self.board[px][py] = 1

        # 1) Check Five / Long Five
        has_five = False
        has_overline = False
        for dx, dy in self._DIRS:
            run_len = self._max_run_len(px, py, dx, dy, 1)
            if run_len == 5:
                has_five = True
            elif run_len >= 6:
                has_overline = True

        if has_five:
            # Win takes precedence
            self.board[px][py] = 0
            return True

        if has_overline:
            self.board[px][py] = 0
            return False

        # 2) 4-4 Forbidden
        fours = self._count_fours_from_move(px, py, 1)
        if fours >= 2:
            self.board[px][py] = 0
            return False

        # 3) 3-3 Forbidden (Recursive)
        threes = self._find_threes_from_move(px, py, 1)

        if len(threes) < 2:
            self.board[px][py] = 0
            return True

        if depth <= 0:
            # Depth exhausted, assume forbidden
            self.board[px][py] = 0
            return False

        # Count 'Real' Threes
        real_three_count = 0
        for vital_points in threes:
            if self._three_is_real(vital_points, depth - 1):
                real_three_count += 1
                if real_three_count >= 2:
                    self.board[px][py] = 0
                    return False

        self.board[px][py] = 0
        return True

    def _is_forbidden_move_black(self, x: int, y: int) -> bool:
        assert self.board[x][y] == 0
        self.board[x][y] = 1

        # ---- 1. Five / Overline ----
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
            return False

        if has_overline:
            self.board[x][y] = 0
            return True

        # ---- 2. 4-4 Forbidden ----
        fours = self._count_fours_from_move(x, y, 1)
        if fours >= 2:
            self.board[x][y] = 0
            return True

        # ---- 3. 3-3 Forbidden ----
        threes = self._find_threes_from_move(x, y, 1)

        if len(threes) < 2:
            self.board[x][y] = 0
            return False

        real_three_count = 0
        for vital_points in threes:
            if self._three_is_real(vital_points, depth=2):
                real_three_count += 1
                if real_three_count >= 2:
                    self.board[x][y] = 0
                    return True

        self.board[x][y] = 0
        return False


    def is_valid_move(self, x: int, y: int) -> bool:
        if not (self.is_inside(x, y) and self.is_empty(x, y)):
            return False

        # Determine player
        current_player = 1 if (self.move_count % 2 == 0) else 2

        # Forbidden moves enabled (currently disabled in logic below returning True immediately)
        # Standard Gomoku Rules (Forbidden Moves Disabled)

        return True

    def place_stone(self, x: int, y: int, player: int) -> bool:
        """Execute move."""
        if player not in (1, 2):
            raise ValueError("Player must be 1 or 2")
        
        if not self.is_valid_move(x, y):
            return False
            
        self.board[x][y] = player
        self.move_count += 1
        return True

    def get_cell(self, x: int, y: int) -> int:
        if not self.is_inside(x, y):
            raise ValueError(f"({x}, {y}) is outside the board")
        return self.board[x][y]

    def check_five_in_a_row(
        self, player: int
    ) -> Optional[List[Tuple[int, int]]]:
        """Check for win condition."""
        n = self.size

        # 1. Horizontal
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

        # 2. Vertical
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

        # 3. Diagonal (Top-Left to Bottom-Right)
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
        
        # 4. Anti-Diagonal (Top-Right to Bottom-Left)
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

    def is_full(self) -> bool:
        return self.move_count >= self.size * self.size

    def get_game_result(
        self, with_line: bool = False
    ):
        """Return 1/2 for win, 3 for draw, 0 for ongoing."""
        # Check Player 1
        line1 = self.check_five_in_a_row(1)
        if line1 is not None:
            return (1, line1) if with_line else 1

        # Check Player 2
        line2 = self.check_five_in_a_row(2)
        if line2 is not None:
            return (2, line2) if with_line else 2

        # Check Draw
        if self.is_full():
            return (3, [(-1, -1)]) if with_line else 3

        # Ongoing
        return (0, [(-1, -1)]) if with_line else 0

    def to_string(self) -> str:
        codes: List[str] = []
        for x in range(self.size):
            for y in range(self.size):
                codes.append(str(self.board[x][y]))
        return "".join(codes)
