from typing import List, Tuple, Dict, Any, Optional
from backend.engine.board import Board

class GameEngine:
    def __init__(self, size: int = 15, first_player: int = 1) -> None:
        self.board: Board = Board(size=size)
        self.current_player: int = first_player
        self.first_player: int = first_player
        self.game_over: bool = False
        self.winner: int = 0
        self.move_history: List[Tuple[int, int, int]] = []

        # Statistics
        self.total_games: int = 0
        self.black_wins: int = 0
        self.white_wins: int = 0
        self.draws: int = 0

    def reset_game(self, first_player: Optional[int] = None) -> None:
        """Reset game state while keeping statistics."""
        self.board.reset()
        if first_player is not None:
            self.first_player = first_player
        self.current_player = self.first_player

        self.game_over = False
        self.winner = 0
        self.move_history.clear()

    def _switch_player(self) -> None:
        self.current_player = 3 - self.current_player

    def make_move(self, x: int, y: int) -> bool:
        """Attempt to place a stone at (x, y) for the current player."""
        if self.game_over:
            return False

        success = self.board.place_stone(x, y, self.current_player)
        if not success:
            return False

        self.move_history.append((x, y, self.current_player))

        result = self.board.get_game_result(with_line=False)
        if result != 0:
            self.game_over = True
            self.winner = result
            self._update_statistics(result)
        else:
            self._switch_player()

        return True

    def make_move_for(self, x: int, y: int, player: int) -> bool:
        """Force a move for a specific player (used for self-play/simulation)."""
        if self.game_over:
            return False

        if player not in (1, 2):
            raise ValueError("Player must be 1 or 2")

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

    def undo_last_move(self) -> bool:
        """Undo the last move and revert game state."""
        if not self.move_history:
            return False
            
        x, y, player = self.move_history.pop()
        
        self.board.board[x][y] = 0
        self.board.move_count -= 1
        
        self.game_over = False
        self.winner = 0
        self.current_player = player
        
        return True

    def _update_statistics(self, result: int) -> None:
        if result == 0: return

        self.total_games += 1
        if result == 1:
            self.black_wins += 1
        elif result == 2:
            self.white_wins += 1
        elif result == 3:
            self.draws += 1

    def get_status(self) -> Dict[str, Any]:
        """Return full game state dictionary."""
        return {
            "board_size": self.board.size,
            "board": self.board.board,
            "current_player": self.current_player,
            "game_over": self.game_over,
            "winner": self.winner,
            "move_count": self.board.move_count,
            "move_history": list(self.move_history),
            "statistics": {
                "total_games": self.total_games,
                "black_wins": self.black_wins,
                "white_wins": self.white_wins,
                "draws": self.draws,
            },
        }

    def get_last_move(self) -> Optional[Tuple[int, int, int]]:
        if not self.move_history:
            return None
        return self.move_history[-1]

    def debug_print_board(self) -> None:
        """Print board to console for debugging."""
        size = self.board.size
        for y in range(size):
            row = []
            for x in range(size):
                v = self.board.board[x][y]
                if v == 0: row.append(".")
                elif v == 1: row.append("X") # Black
                else: row.append("O") # White
            print(" ".join(row))
        print(f"Player: {self.current_player}, Game Over: {self.game_over}, Winner: {self.winner}")