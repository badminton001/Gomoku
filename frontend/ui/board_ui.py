import tkinter as tk
from tkinter import messagebox
from backend.models.board import Board
from ui.difficulty_control import DifficultyControlUI
from ui.ai_recommender import AiRecommender  # 引入新组件


class BoardUI(tk.Frame):
    def __init__(self, master=None, board_size=15, cell_size=40):
        super().__init__(master)
        self.master = master
        self.board_size = board_size
        self.cell_size = cell_size

        self.logic_board = Board()
        self.current_player = 1
        self.game_over = False

        self.pack(fill="both", expand=True)

        # 布局
        self.left_panel = tk.Frame(self, bg="#E3C088")
        self.left_panel.pack(side="left", fill="both", expand=True)
        self.right_panel = tk.Frame(self, bg="#f0f0f0", width=200)
        self.right_panel.pack(side="right", fill="y")
        self.right_panel.pack_propagate(False)

        # 画布
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

        # --- 初始化 AI 推荐器 ---
        self.recommender = AiRecommender(self.canvas, self.cell_size)

        # 加载控制面板 (传入了 on_hint 回调)
        self.control_panel = DifficultyControlUI(
            self.right_panel,
            on_mode_change=self.handle_settings_change,
            on_hint=self.handle_hint_request  # 绑定求助按钮
        )
        self.control_panel.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_grid()

        self.info_label = tk.Label(self.left_panel, text="当前回合: 黑棋 (Black)",
                                   font=("Arial", 12, "bold"), bg="#E3C088")
        self.info_label.pack(pady=10)

    def handle_hint_request(self):
        """处理求助请求"""
        if self.game_over:
            return
        # 这里模拟 AI 算出的最佳位置 (比如中心点 7,7)
        # 真正合并后，这里会调用 Person C 的算法
        print("请求 AI 提示...")
        self.recommender.show_hint(7, 7)  # 测试：在天元位置画红框

    def handle_settings_change(self, mode, diff):
        print(f"设置变更: {mode} - {diff}")

    def draw_grid(self):
        margin = self.cell_size
        width = self.cell_size * (self.board_size - 1)
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

    def on_click(self, event):
        if self.game_over:
            return

        # 每次下棋前，清除之前的提示框
        self.recommender.clear_hint()

        margin = self.cell_size
        x = round((event.x - margin) / self.cell_size)
        y = round((event.y - margin) / self.cell_size)

        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return

        success = self.logic_board.place_stone(x, y, self.current_player)

        if success:
            color = "black" if self.current_player == 1 else "white"
            self.draw_piece(x, y, color)

            try:
                if hasattr(self.logic_board, 'check_winner') and self.logic_board.check_winner(x, y):
                    messagebox.showinfo("游戏结束", f"{color} 获胜!")
                    self.game_over = True
                    return
            except:
                pass

            self.current_player = 3 - self.current_player
            next_color = "黑棋 (Black)" if self.current_player == 1 else "白棋 (White)"
            self.info_label.config(text=f"当前回合: {next_color}")

    def draw_piece(self, x, y, color):
        margin = self.cell_size
        cx = margin + x * self.cell_size
        cy = margin + y * self.cell_size
        r = self.cell_size * 0.4
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline=color)