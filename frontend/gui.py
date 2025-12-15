import tkinter as tk
from tkinter import messagebox, ttk
import sys
import os
import threading
import time

# Add root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine.board import Board
from backend.engine.game_engine import GameEngine

class GomokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gomoku AI Battle Arena (Hybrid/DQN)")
        self.root.geometry("900x700")
        
        # Game State
        self.board_size = 15
        self.cell_size = 40
        self.engine = GameEngine()
        self.human_player = 1 # Human is Black
        self.ai_player = 2    # AI is White
        self.game_active = False
        self.last_move = None
        self.agent_cache = {} # Cache for AI agents
        
        # Layout
        self.main_frame = tk.Frame(root)

        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Board
        self.canvas = tk.Canvas(self.main_frame, width=600, height=600, bg="#E8C98F")
        self.canvas.pack(side=tk.LEFT, padx=10)
        self.canvas.bind("<Button-1>", self.on_board_click)
        
        # Right: Controls
        self.control_frame = tk.Frame(self.main_frame, width=250)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        
        self.create_controls()
        self.draw_board()
        self.reset_game()

    def create_controls(self):
        # Title
        tk.Label(self.control_frame, text="Control Panel", font=("Arial", 16, "bold")).pack(pady=10)
        
        # AI Selection
        tk.Label(self.control_frame, text="Select Opponent:", font=("Arial", 10)).pack(anchor="w")
        self.ai_var = tk.StringVar(value="Hybrid")
        # Removed DQN to avoid environment dependency issues
        ai_options = ["Greedy", "Strong (AlphaBeta)", "Hybrid (SL+Search)"]
        self.ai_combo = ttk.Combobox(self.control_frame, textvariable=self.ai_var, values=ai_options, state="readonly")
        self.ai_combo.pack(fill=tk.X, pady=5)
        self.ai_combo.current(2)
        
        # Difficulty
        tk.Label(self.control_frame, text="Search Depth (Difficulty):").pack(anchor="w", pady=(10,0))
        self.depth_var = tk.IntVar(value=2)
        self.depth_scale = tk.Scale(self.control_frame, from_=1, to=6, orient=tk.HORIZONTAL, variable=self.depth_var)
        self.depth_scale.pack(fill=tk.X)
        
        # Buttons
        tk.Button(self.control_frame, text="Start New Game", command=self.reset_game, bg="#4CAF50", fg="white", font=("Arial", 12)).pack(fill=tk.X, pady=20)
        
        # Undo Button
        tk.Button(self.control_frame, text="Undo Last Move", command=self.undo_move, bg="#FF9800", fg="white").pack(fill=tk.X, pady=5)
        
        tk.Button(self.control_frame, text="AI Hint (Best Move)", command=self.get_hint, bg="#2196F3", fg="white").pack(fill=tk.X, pady=5)
        
        # Status Label
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.control_frame, textvariable=self.status_var, wraplength=200, fg="#666").pack(side=tk.BOTTOM, pady=20)

    def undo_move(self):
        # 1. If currently AI's turn (thinking), do not allow undo for safety or just return
        # Actually, if AI is calculating, shared state might be messy.
        # But our AI is synchronous in this simple GUI (frozen UI). So we can't click anyway?
        # If we use thread in future, we need lock.
        # For now, if game_active is False (Game Over), we can undo 1 step to resume?
        
        # Strategy:
        # If Game Over: Undo 1 step (Winning move). Result: Game continues, Human turn (if human won) or AI turn (if AI won).
        # If Game Active:
        #   If it's Human Turn: Undo 2 moves (AI's last move + Human's last move) to retry.
        #   If it's AI Turn: (We can't click).
        
        if not self.engine.move_history:
            return
            
        steps = 0
        if self.engine.game_over:
            # Undo once to resume game
            self.engine.undo_last_move()
            self.game_active = True
            self.status_var.set("Game Resumed (Undo)")
            steps = 1
        else:
            # Normal undo during play
            if self.engine.current_player == self.human_player:
                # Undo AI move AND Human move
                self.engine.undo_last_move() # Undo AI
                self.engine.undo_last_move() # Undo Human
                self.status_var.set("Undo successful")
                steps = 2
            else:
                # AI turn (rarely reached in sync mode)
                self.engine.undo_last_move()
                steps = 1
        
        # Update last move visual
        if self.engine.move_history:
            last = self.engine.move_history[-1]
            self.last_move = (last[0], last[1])
        else:
            self.last_move = None
            
        self.draw_stones()

    def draw_board(self):
        self.canvas.delete("all")
        # Draw grid
        padding = 20
        for i in range(self.board_size):
            start = padding + i * self.cell_size
            end = padding + (self.board_size - 1) * self.cell_size
            # Vertical
            self.canvas.create_line(start, padding, start, end)
            # Horizontal
            self.canvas.create_line(padding, start, end, start)
            
        # Coordinates
        col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        for i in range(self.board_size):
            pos = padding + i * self.cell_size
            # Top X-axis (Letters)
            self.canvas.create_text(pos, 10, text=col_labels[i], font=("Arial", 10, "bold"))
            # Left Y-axis (Numbers 15->1)
            row_label = str(15 - i)
            self.canvas.create_text(10, pos, text=row_label, font=("Arial", 10, "bold"))

        # Star points
        stars = [3, 7, 11]
        for x in stars:
            for y in stars:
                cx = padding + x * self.cell_size
                cy = padding + y * self.cell_size
                self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill="black")

    def draw_stones(self):
        self.canvas.delete("stone")
        padding = 20
        for x in range(self.board_size):
            for y in range(self.board_size):
                val = self.engine.board.board[x][y]
                if val != 0:
                    cx = padding + x * self.cell_size
                    cy = padding + y * self.cell_size
                    color = "black" if val == 1 else "white"
                    # Shadow
                    self.canvas.create_oval(cx-18, cy-18, cx+20, cy+20, fill="#aaa", outline="", tags="stone")
                    # Stone
                    self.canvas.create_oval(cx-18, cy-18, cx+18, cy+18, fill=color, outline="gray", tags="stone")
                    
                    if self.last_move == (x, y):
                        self.canvas.create_oval(cx-5, cy-5, cx+5, cy+5, fill="red", outline="", tags="stone")

    def reset_game(self):
        self.engine = GameEngine()
        self.game_active = True
        self.last_move = None
        self.draw_board()
        self.status_var.set(f"Game Started! Black (You) vs White ({self.ai_var.get()}).")

    def on_board_click(self, event):
        if not self.game_active or self.engine.current_player != self.human_player:
            return
            
        padding = 20
        # Calculate grid intersection
        x = int(round((event.x - padding) / self.cell_size))
        y = int(round((event.y - padding) / self.cell_size))
        
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            if self.engine.board.is_valid_move(x, y):
                self.make_move(x, y)
                self.root.update()
                
                # Check Win
                if not self.game_active: return
                
                # Trigger AI Turn
                self.status_var.set("AI thinking...")
                self.root.after(100, self.ai_turn)

    def make_move(self, x, y):
        success = self.engine.make_move(x, y)
        if success:
            self.last_move = (x, y)
            self.draw_stones()
            
            if self.engine.game_over:
                self.game_active = False
                w = "Black" if self.engine.winner == 1 else "White"
                if self.engine.winner == 0:
                    msg = "Draw!"
                else:
                    msg = f"{w} Wins!"
                self.status_var.set(msg)
                messagebox.showinfo("Game Over", msg)

    def _get_coord_str(self, x, y):
        col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        if 0 <= x < len(col_labels):
            col_str = col_labels[x]
        else:
            col_str = "?"
        return f"{col_str}{15-y}"

    def ai_turn(self):
        if not self.game_active: return
        
        algo_name = self.ai_var.get()
        depth = self.depth_var.get()
        
        try:
            agent = self._get_agent(algo_name, depth)
            
            start = time.time()
            move = agent.get_move(self.engine.board, self.ai_player)
            duration = time.time() - start
            
            if move:
                self.make_move(move[0], move[1])
                coord_str = self._get_coord_str(move[0], move[1])
                self.status_var.set(f"AI played {coord_str} in {duration:.2f}s")
            else:
                 # Should not happen
                 self.game_active = False
                 self.status_var.set("AI Resigned (No moves)")
                 
        except Exception as e:
            self.status_var.set(f"AI Error: {e}")
            print(e)

    def _get_agent(self, name, depth):
        cache_key = f"{name}_{depth}"
        if cache_key in self.agent_cache:
            return self.agent_cache[cache_key]

        agent = None
        # Lazy loading to avoid imports if not needed
        if "Greedy" in name:
            from backend.ai.basic.classic_ai import GreedyAgent
            agent = GreedyAgent(distance=2)
        elif "AlphaBeta" in name:
            from backend.ai.basic.strong_ai import AlphaBetaAgent
            agent = AlphaBetaAgent(depth=depth, time_limit=3.0)
        elif "Hybrid" in name:
             from backend.ai.advanced.hybrid_ai import HybridAgent
             # Try kaggle model first
             model = "models/sl_model_kaggle.pth"
             if not os.path.exists(model): model = "models/sl_model_v1.pth"
             agent = HybridAgent(model_path=model)
        
        if agent:
            self.agent_cache[cache_key] = agent
        return agent

    def get_hint(self):
        if not self.game_active: return
        
        self.status_var.set("Calculating Hint...")
        self.root.update()
        
        # Use Hybrid Agent for hints as it's strongest
        try:
             from backend.ai.advanced.hybrid_ai import HybridAgent
             model = "models/sl_model_kaggle.pth"
             if not os.path.exists(model): model = "models/sl_model_v1.pth"
             agent = HybridAgent(model_path=model)
             
             move = agent.get_move(self.engine.board, self.engine.current_player)
             
             padding = 20
             cx = padding + move[0] * self.cell_size
             cy = padding + move[1] * self.cell_size
             
             # Clear old hint
             self.canvas.delete("hint")
             
             # Draw a blue circle hint
             # Flash effect: Solid -> Dash
             self.canvas.create_oval(cx-20, cy-20, cx+20, cy+20, outline="#007BFF", width=4, tags="hint")
             
             # Coordinate Fix: Skip 'I'
             col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
             
             # Draw primary hint
             self.canvas.create_oval(cx-20, cy-20, cx+20, cy+20, outline="#007BFF", width=4, tags="hint")
             col_str_best = col_labels[move[0]]
             msg = f"Best: {col_str_best}{15-move[1]}"

             # Try to get Top-5 if available
             if hasattr(agent, 'get_top_moves'):
                 tops = agent.get_top_moves(self.engine.board, self.engine.current_player, limit=5)
                 if tops:
                     msg += " | Top 5: "
                     for score, (tx, ty) in tops:
                         t_col = col_labels[tx]
                         # Draw smaller hints
                         tcx = padding + tx * self.cell_size
                         tcy = padding + ty * self.cell_size
                         if (tx, ty) != move: # Don't redraw best
                             self.canvas.create_oval(tcx-10, tcy-10, tcx+10, tcy+10, outline="#007BFF", width=2, tags="hint")
                         
                         msg += f"{t_col}{15-ty}({int(score*100)}%) "
             
             self.status_var.set(msg)
             
             # Remove hint after 5 seconds (Slow fade out feeling)
             self.root.after(5000, lambda: self.canvas.delete("hint"))
             
        except Exception as e:
             self.status_var.set(f"Hint Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()
