# frontend/ui/ai_recommender.py
class AiRecommender:
    def __init__(self, canvas, cell_size):
        self.canvas = canvas
        self.cell_size = cell_size
        self.hint_tag = "ai_hint_marker"

    def show_hint(self, x, y):
        """(旧功能) 显示单个红框"""
        self.clear_hint()
        margin = self.cell_size
        cx = margin + x * self.cell_size
        cy = margin + y * self.cell_size
        r = self.cell_size * 0.45
        self.canvas.create_rectangle(
            cx - r, cy - r, cx + r, cy + r,
            outline="red", width=3, tags=self.hint_tag
        )

    def show_top5(self, moves):
        """
        (新功能) 显示 Top-5 推荐点 - 字体加大版
        """
        self.clear_hint()

        # 颜色梯度：第1名红色，第2-3名橙色，第4-5名灰色
        colors = ["#FF0000", "#FF8C00", "#FF8C00", "#808080", "#808080"]

        for i, (x, y) in enumerate(moves):
            rank = i + 1  # 排名 1-5
            color = colors[i] if i < len(colors) else "gray"

            margin = self.cell_size
            cx = margin + x * self.cell_size
            cy = margin + y * self.cell_size
            r = self.cell_size * 0.4

            # 1. 画圆圈背景 (线条加粗到 3)
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=None, outline=color, width=3, tags=self.hint_tag
            )

            # 2. 画数字 (字体加大到 16)
            self.canvas.create_text(
                cx, cy, text=str(rank), fill=color,
                font=("Arial", 16, "bold"), tags=self.hint_tag
            )

            # 3. 如果是第一名，额外加个外框强调
            if rank == 1:
                r2 = r + 4
                self.canvas.create_rectangle(
                    cx - r2, cy - r2, cx + r2, cy + r2,
                    outline="red", width=3, tags=self.hint_tag
                )

    def clear_hint(self):
        """清除屏幕上所有的提示标记"""
        self.canvas.delete(self.hint_tag)