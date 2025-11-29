import tkinter as tk
from tkinter import messagebox
from backend.models.board import Board
# 引入刚才写好的控制面板组件
from ui.difficulty_control import DifficultyControlUI


class BoardUI(tk.Frame):
    def __init__(self, master=None, board_size=15, cell_size=40):
        super().__init__(master)
        self.master = master
        self.board_size = board_size
        self.cell_size = cell_size

        # 逻辑初始化
        self.logic_board = Board()
        self.current_player = 1
        self.game_over = False

        self.pack(fill="both", expand=True)

        # --- 布局核心：分为左边和右边 ---

        # 1. 创建左侧容器 (放棋盘)
        self.left_panel = tk.Frame(self, bg="#E3C088")
        self.left_panel.pack(side="left", fill="both", expand=True)

        # 2. 创建右侧容器 (放控制面板)
        self.right_panel = tk.Frame(self, bg="#f0f0f0", width=200)
        self.right_panel.pack(side="right", fill="y")
        self.right_panel.pack_propagate(False)  # 固定宽度，不被内容撑开

        # 3. 在左边画棋盘
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

        # 4. 在右边加载控制面板
        self.control_panel = DifficultyControlUI(self.right_panel, on_mode_change=self.handle_settings_change)
        self.control_panel.pack(fill="both", expand=True)

        # 绑定事件
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_grid()

        # 底部信息改放在左侧面板下面
        self.info_label = tk.Label(self.left_panel, text="当前回合: 黑棋 (Black)",
                                   font=("Arial", 12, "bold"), bg="#E3C088")
        self.info_label.pack(pady=10)

    def handle_settings_change(self, mode, diff):
        """处理设置变更"""
        print(f"主界面收到设置变更: {mode} - {diff}")
        # 后续这里会连接真正的 AI 切换逻辑

    def draw_grid(self):
        """绘制网格"""
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
        """点击事件"""
        if self.game_over:
            return

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
        else:
            print("这里不能下")

    def draw_piece(self, x, y, color):
        margin = self.cell_size
        cx = margin + x * self.cell_size
        cy = margin + y * self.cell_size
        r = self.cell_size * 0.4
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline=color)