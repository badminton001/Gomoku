# File location: frontend/ui/board_ui.py
import tkinter as tk
from tkinter import messagebox

# --- Import Backend Logic ---
# Ensure GameEngine is saved in backend/models/game_engine.py
from backend.models.game_engine import GameEngine
# Import Algorithms (Using AlphaBeta as default AI)
from backend.algorithms.classic_ai import AlphaBetaAgent

# --- Import UI Components (Relative imports) ---
from .difficulty_control import DifficultyControlUI
from .ai_recommender import AiRecommender


class BoardUI(tk.Frame):
    def __init__(self, master=None, board_size=15, cell_size=40):
        super().__init__(master)
        self.master = master
        self.board_size = board_size
        self.cell_size = cell_size

        # 1. Initialize Game Engine
        self.engine = GameEngine(size=board_size)

        # 2. Initialize AI (Default medium difficulty, depth=2)
        self.ai_agent = AlphaBetaAgent(depth=2)

        self.pack(fill="both", expand=True)

        # --- Layout ---
        self.left_panel = tk.Frame(self, bg="#E3C088")
        self.left_panel.pack(side="left", fill="both", expand=True)

        self.right_panel = tk.Frame(self, bg="#f0f0f0", width=200)
        self.right_panel.pack(side="right", fill="y")
        self.right_panel.pack_propagate(False)

        # --- Canvas ---
        self.canvas_width = cell_size * (board_size + 1)
        self.canvas_height = cell_size * (board_size + 1)
        self.canvas = tk.Canvas(
            self.left_panel,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#E3C088",
            highlightthickness=0
        )
        self.canvas.pack(padx=20, pady=20)

        # --- Initialize Components ---
        self.recommender = AiRecommender(self.canvas, self.cell_size)

        self.control_panel = DifficultyControlUI(
            self.right_panel,
            on_mode_change=self.handle_settings_change,
            on_hint=self.handle_hint_request,
            on_reset=self.reset_game  # <--- 直接在这里传进去！
        )
        self.control_panel.pack(fill="both", expand=True)

        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_grid()

        self.info_label = tk.Label(self.left_panel, text="Current Turn: Black",
                                   font=("Arial", 12, "bold"), bg="#E3C088")
        self.info_label.pack(pady=10)

    def reset_game(self):
        """Reset the game"""
        print("UI: Resetting game")
        self.engine.reset_game()
        self.canvas.delete("piece")  # Clear pieces
        self.recommender.clear_hint()  # Clear hints
        self.info_label.config(text="Current Turn: Black")
        self.canvas.delete("all")  # Refresh canvas
        self.draw_grid()

    def handle_hint_request(self):
        """Handle hint request"""
        if self.engine.game_over:
            return

        print("UI: Thinking for hint...")
        # Calculate best move using AI
        try:
            move = self.ai_agent.get_move(self.engine.board, self.engine.current_player)
            if move:
                self.recommender.show_hint(move[0], move[1])
        except Exception as e:
            print(f"AI Error: {e}")

    def handle_settings_change(self, mode, diff):
        """Handle settings change"""
        print(f"Settings changed -> Mode: {mode}, Difficulty: {diff}")
        # Update AI depth based on difficulty
        if diff == 'easy':
            self.ai_agent = AlphaBetaAgent(depth=1)
        elif diff == 'medium':
            self.ai_agent = AlphaBetaAgent(depth=2)
        elif diff == 'hard':
            self.ai_agent = AlphaBetaAgent(depth=3)

        self.reset_game()

    def computer_move(self):
        """Computer move logic"""
        if self.engine.game_over:
            return

        # 1. AI thinking
        move = self.ai_agent.get_move(self.engine.board, self.engine.current_player)

        if move:
            x, y = move

            # --- 核心修复：在落子前，先记录当前是谁 (AI 的身份) ---
            ai_player_code = self.engine.current_player
            # ------------------------------------------------

            # 2. Engine execute move
            success = self.engine.make_move(x, y)

            if success:
                # 3. Draw piece
                # 直接用刚才记录的身份来决定颜色，不再依赖引擎是否切换了回合
                color = "black" if ai_player_code == 1 else "white"
                self.draw_piece(x, y, color)

                # 4. Check game over
                # (这里加入 self.update() 是为了防止弹窗太快遮住棋子，建议加上)
                if self.engine.game_over:
                    self.update()  # 强制刷新界面显示
                    self.check_game_over()
                else:
                    self.check_game_over()

    def on_click(self, event):
        """Player click event"""
        if self.engine.game_over:
            return

        self.recommender.clear_hint()

        # 1. Calculate coordinates
        margin = self.cell_size
        x = round((event.x - margin) / self.cell_size)
        y = round((event.y - margin) / self.cell_size)

        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return

        # 2. Player attempts move
        # Record current player for drawing color
        current_color_code = self.engine.current_player

        success = self.engine.make_move(x, y)

        if success:
            color = "black" if current_color_code == 1 else "white"
            self.draw_piece(x, y, color)

            # 3. Check game over
            is_over = self.check_game_over()

            # 4. If not over, and in PvE mode, trigger AI
            current_mode = self.control_panel.mode_var.get()
            if not is_over and current_mode == "pve":
                # Delay 0.5s for better UX
                self.after(500, self.computer_move)
        else:
            print("Invalid move")

    def check_game_over(self):
        """检查并处理游戏结束"""
        if self.engine.game_over:
            # --- 新增这一行：强制刷新界面，确保最后一颗子画出来后再弹窗 ---
            self.update()
            # --------------------------------------------------------

            winner = self.engine.winner
            if winner == 1:
                messagebox.showinfo("Game Over", "Black wins!")
            elif winner == 2:
                messagebox.showinfo("Game Over", "White wins!")
            elif winner == 3:
                messagebox.showinfo("Game Over", "Draw!")
            return True
        else:
            # Update bottom info label
            next_player_text = "Black" if self.engine.current_player == 1 else "White"
            self.info_label.config(text=f"Current Turn: {next_player_text}")
            return False

    def draw_grid(self):
        """Draw board grid"""
        margin = self.cell_size
        width = self.cell_size * (self.board_size - 1)

        # Background
        self.canvas.create_rectangle(0, 0, self.canvas_width, self.canvas_height, fill="#E3C088")

        for i in range(self.board_size):
            start = margin + i * self.cell_size
            end = margin + width
            self.canvas.create_line(margin, start, end + margin, start)
            self.canvas.create_line(start, margin, start, end + margin)
            self.canvas.create_text(start, margin / 2, text=str(i), fill="gray")
            self.canvas.create_text(margin / 2, start, text=str(i), fill="gray")

        star_points = [(3, 3), (11, 3), (7, 7), (3, 11), (11, 11)]
        for x, y in star_points:
            cx = margin + x * self.cell_size
            cy = margin + y * self.cell_size
            r = 3
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="black")

    def draw_piece(self, x, y, color):
        margin = self.cell_size
        cx = margin + x * self.cell_size
        cy = margin + y * self.cell_size
        r = self.cell_size * 0.4
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline=color, tags="piece")