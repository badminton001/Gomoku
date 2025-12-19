import tkinter as tk
from tkinter import messagebox
import sys
import os

# --- Import UI Components ---
from .difficulty_control import DifficultyControlUI
from .ai_recommender import AiRecommender

# --- Import Backend ---
from backend.engine.game_engine import GameEngine
from backend.ai.minimax import AlphaBetaAgent
from backend.ai.baselines import GreedyAgent
from backend.ai.hybrid import HybridAgent


class BoardUI(tk.Frame):
    def __init__(self, master=None, board_size=15, cell_size=60):
        super().__init__(master)
        self.master = master
        self.board_size = board_size
        self.cell_size = cell_size

        # 1. Initialize Game Engine
        self.engine = GameEngine(size=board_size)

        # 2. AI Agent
        self.ai_agent = AlphaBetaAgent(depth=2)

        # 3. Player Identity (1=Black, 2=White)
        self.human_player = 1

        self.pack(fill="both", expand=True)

        # --- Layout ---
        self.left_panel = tk.Frame(self, bg="#E3C088")
        self.left_panel.pack(side="left", fill="both", expand=True)

        self.right_panel = tk.Frame(self, bg="#f0f0f0", width=280)
        self.right_panel.pack(side="right", fill="y")
        self.right_panel.pack_propagate(False)

        # --- Canvas ---
        self.padding = 20
        self.canvas_width = cell_size * (board_size) + 2 * self.padding
        self.canvas_height = cell_size * (board_size) + 2 * self.padding
        
        self.canvas = tk.Canvas(
            self.left_panel,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#E3C088",
            highlightthickness=0
        )
        self.canvas.pack(padx=20, pady=20)

        # --- Components ---
        self.recommender = AiRecommender(self.canvas, self.cell_size)

        self.control_panel = DifficultyControlUI(
            self.right_panel,
            on_settings_change=self.handle_settings_change,
            on_hint=self.handle_hint_request,
            on_reset=self.reset_game
        )
        self.control_panel.pack(fill="both", expand=True)

        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_grid()

        self.info_label = tk.Label(self.left_panel, text="Current Turn: Black",
                                   font=("Arial", 12, "bold"), bg="#E3C088")
        self.info_label.pack(pady=10)

    def reset_game(self):
        """Reset game"""
        print("UI: Resetting game")
        self.engine.reset_game()
        self.canvas.delete("piece")
        self.recommender.clear_hint()
        self.info_label.config(text="Current Turn: Black")
        self.canvas.delete("all")
        self.draw_grid()

        # Check who goes first
        # Check who goes first
        if self.human_player == 2:
            print("UI: Human chose White. AI (Black) moves first.")
            self.after(500, self.computer_move)

    def handle_hint_request(self):
        """Handle hint (Async)"""
        if self.engine.game_over:
            return

        # Human turn only
        if self.engine.current_player != self.human_player:
            print("UI: It's not your turn!")
            return

        self.info_label.config(text="AI Thinking for Hints...")
        self.update() # Force update immediately

        # Run in thread
        import threading
        t = threading.Thread(target=self._calculate_hint)
        t.daemon = True
        t.start()

    def _calculate_hint(self):
        """Hint worker"""
        try:
            candidates = []
            # Top-5 API
            if hasattr(self.ai_agent, 'get_top_moves'):
                 candidates = self.ai_agent.get_top_moves(
                        self.engine.board,
                        self.engine.current_player,
                        limit=5
                    )
            elif hasattr(self.ai_agent, '_ordered_candidates'):
                 candidates = self.ai_agent._ordered_candidates(
                        self.engine.board,
                        self.engine.current_player,
                        depth=1
                    )

            if candidates:
                # Show Top 5
                top_5 = candidates[:5]
                self.after(0, lambda: self.recommender.show_top5(top_5))
                self.after(0, lambda: self.info_label.config(text="Hints AI: Here are the top moves."))
            else:
                # Fallback: Single move
                move = self.ai_agent.get_move(self.engine.board, self.engine.current_player)
                if move:
                     self.after(0, lambda: self.recommender.show_hint(move[0], move[1]))
                     self.after(0, lambda: self.info_label.config(text="Hints AI: Best move shown."))
                else:
                     self.after(0, lambda: self.info_label.config(text="Hints AI: No moves found."))

        except Exception as e:
            print(f"AI Hint Error: {e}")
            self.after(0, lambda: messagebox.showerror("Hint Error", str(e)))
            self.after(0, lambda: self.info_label.config(text="Hints AI: Error occurred."))

    def handle_settings_change(self, color, diff, time_str="5s"):
        """Handle settings change"""
        print(f"Settings changed -> Color: {color}, Difficulty: {diff}, Time: {time_str}")
        
        try:
            time_limit = float(time_str.replace("s", ""))
        except:
            time_limit = 5.0

        # 1. Update Human Identity
        self.human_player = 1 if color == 'black' else 2

        # 2. Update AI Difficulty
        if diff == 'easy':
            self.ai_agent = GreedyAgent(distance=2)
            print("AI: Switch to GreedyAgent")
        elif diff == 'medium':
            self.ai_agent = AlphaBetaAgent(depth=2, time_limit=time_limit)
            print(f"AI: Switch to AlphaBeta (Depth 2, {time_limit}s)")
        elif diff == 'hard':
            self.ai_agent = AlphaBetaAgent(depth=3, time_limit=time_limit)
            print(f"AI: Switch to AlphaBeta (Depth 3, {time_limit}s)")
        elif diff == 'expert':
            # Load Hybrid model
            model_path = "models/sl_policy_v2.pth"
            if not os.path.exists(model_path): 
                model_path = "models/sl_model_kaggle.pth"
                if not os.path.exists(model_path):
                    model_path = "models/sl_model_v1.pth"
            
            # HybridAgent
            self.ai_agent = HybridAgent(model_path=model_path) 
            # Set time limit if supported
            if hasattr(self.ai_agent, 'time_limit'):
                 self.ai_agent.time_limit = time_limit
            print(f"AI: Switch to HybridAgent ({model_path})")

        # 3. Reset Game
        self.reset_game()

    def computer_move(self):
        """Computer move logic"""
        if self.engine.game_over:
            return

        # Double check: Is it actually AI's turn?
        # AI plays when current_player is NOT human_player
        if self.engine.current_player == self.human_player:
            return

        # 1. AI thinking
        move = self.ai_agent.get_move(self.engine.board, self.engine.current_player)

        if move:
            x, y = move
            ai_player_code = self.engine.current_player

            # 2. Engine execute move
            success = self.engine.make_move(x, y)
            if success:
                # 3. Draw piece
                color = "black" if ai_player_code == 1 else "white"
                self.draw_piece(x, y, color)

                # 4. Check game over
                if self.engine.game_over:
                    self.update() # Force redraw
                    self.check_game_over()
                else:
                    self.check_game_over()
                    # If human is supposed to play next, allow it.

    def on_click(self, event):
        """Player click event"""
        if self.engine.game_over:
            return

        # --- Lock UI for AI ---
        if self.engine.current_player != self.human_player:
            print("UI: Please wait for AI to move.")
            return

        self.recommender.clear_hint()

        # 1. Calculate coordinates
        margin = self.padding
        x = int(round((event.x - margin) / self.cell_size))
        y = int(round((event.y - margin) / self.cell_size))

        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return

        # 2. Player attempts move
        current_color_code = self.engine.current_player
        success = self.engine.make_move(x, y)

        if success:
            color = "black" if current_color_code == 1 else "white"
            self.draw_piece(x, y, color)

            # 3. Check game over
            is_over = self.check_game_over()

            # 4. Trigger AI
            if not is_over:
                self.info_label.config(text="AI Thinking...")
                self.update()
                self.after(100, self.computer_move)
        else:
            print("Invalid move")

    def check_game_over(self):
        """Check game over"""
        if self.engine.game_over:
            winner = self.engine.winner
            if winner == 1:
                messagebox.showinfo("Game Over", "Black wins!")
            elif winner == 2:
                messagebox.showinfo("Game Over", "White wins!")
            elif winner == 3:
                messagebox.showinfo("Game Over", "Draw!")
            return True
        else:
            next_player_text = "Black" if self.engine.current_player == 1 else "White"
            self.info_label.config(text=f"Current Turn: {next_player_text} (You are {'Black' if self.human_player==1 else 'White'})")
            return False

    def draw_grid(self):
        """Draw board grid"""
        margin = self.padding
        width = self.cell_size * (self.board_size - 1)
        self.canvas.create_rectangle(0, 0, self.canvas_width, self.canvas_height, fill="#E3C088")
        
        # Draw lines
        for i in range(self.board_size):
            start = margin + i * self.cell_size
            end = margin + width
            # Horizontal (y varies)
            self.canvas.create_line(margin, start, end + margin, start)
            # Vertical (x varies)
            self.canvas.create_line(start, margin, start, end + margin)
            
            # Labels
            font = ("Arial", 12)
            # Row Numbers
            self.canvas.create_text(margin/2, start, text=str(15-i), fill="black", font=font)
            # Col Letters
            col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
            if i < len(col_labels):
                 self.canvas.create_text(start, margin/2, text=col_labels[i], fill="black", font=font)

        # Star points
        star_points = [(3, 3), (11, 3), (7, 7), (3, 11), (11, 11)]
        size_stars = [3, 7, 11] # Indices
        
        for x in size_stars:
            for y in size_stars:
                cx = margin + x * self.cell_size
                cy = margin + y * self.cell_size
                r = 3
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="black")

    def draw_piece(self, x, y, color):
        margin = self.padding
        cx = margin + x * self.cell_size
        cy = margin + y * self.cell_size
        r = self.cell_size * 0.4
        # Slight shadow
        self.canvas.create_oval(cx - r+2, cy - r+2, cx + r+2, cy + r+2, fill="#aaaaaa", outline="")
        # Piece
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="gray")