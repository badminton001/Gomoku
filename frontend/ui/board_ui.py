import tkinter as tk
from tkinter import messagebox
# 引入 Person A 写的棋盘逻辑
from backend.models.board import Board


class BoardUI(tk.Frame):
    def __init__(self, master=None, board_size=15, cell_size=40):
        super().__init__(master)
        self.master = master
        self.board_size = board_size
        self.cell_size = cell_size

        # --- 1. 初始化游戏逻辑 ---
        self.logic_board = Board()  # 实例化 Person A 的棋盘
        self.current_player = 1  # 1代表黑棋(Black), 2代表白棋(White)
        self.game_over = False
        # -----------------------

        self.pack()
        self.canvas_width = cell_size * (board_size + 1)
        self.canvas_height = cell_size * (board_size + 1)

        self.canvas = tk.Canvas(
            self,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#E3C088"
        )
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_grid()

        # 底部显示当前是谁的回合
        self.info_label = tk.Label(self, text="当前回合: 黑棋 (Black)", font=("Arial", 12, "bold"))
        self.info_label.pack(pady=10)

    def draw_grid(self):
        """绘制网格 (保持不变)"""
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
        """点击事件 (接入逻辑判断)"""
        if self.game_over:
            return

        margin = self.cell_size
        # 1. 计算坐标
        x = round((event.x - margin) / self.cell_size)
        y = round((event.y - margin) / self.cell_size)

        # 2. 检查点击是否在棋盘内
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return

        # 3. 调用 Person A 的逻辑：尝试落子
        # logic_board.place_stone 会返回 True(成功) 或 False(位置已有子)
        success = self.logic_board.place_stone(x, y, self.current_player)

        if success:
            # 4. 如果落子成功，画出棋子
            color = "black" if self.current_player == 1 else "white"
            self.draw_piece(x, y, color)
            print(f"落子成功: ({x}, {y}) - {color}")

            # 5. 检查是否获胜 (Person A 的 check_winner 方法)
            # 注意：根据之前的文档，如果Person A还没写check_winner，这步可能会报错。
            # 我们先尝试调用，如果报错说明Person A还没写好这部分。
            try:
                if hasattr(self.logic_board, 'check_winner') and self.logic_board.check_winner(x, y):
                    messagebox.showinfo("游戏结束", f"{color} 获胜!")
                    self.game_over = True
                    return
            except Exception as e:
                print(f"胜负判定暂未实现或出错: {e}")

            # 6. 切换回合 (1变2，2变1)
            self.current_player = 3 - self.current_player
            next_color = "黑棋 (Black)" if self.current_player == 1 else "白棋 (White)"
            self.info_label.config(text=f"当前回合: {next_color}")
        else:
            print("落子失败：这里已经有棋子了！")

    def draw_piece(self, x, y, color):
        margin = self.cell_size
        cx = margin + x * self.cell_size
        cy = margin + y * self.cell_size
        r = self.cell_size * 0.4
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline=color)