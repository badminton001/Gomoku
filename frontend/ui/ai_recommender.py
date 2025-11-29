# 文件位置: frontend/ui/ai_recommender.py
class AiRecommender:
    def __init__(self, canvas, cell_size):
        self.canvas = canvas
        self.cell_size = cell_size
        self.hint_tag = "ai_hint_marker"  # 专门给提示框打个标签，方便删除

    def show_hint(self, x, y):
        """在指定位置画一个闪烁的红色方框"""
        self.clear_hint()  # 先清除旧的提示

        margin = self.cell_size
        cx = margin + x * self.cell_size
        cy = margin + y * self.cell_size

        # 画一个比格子稍微小一点的红框
        r = self.cell_size * 0.45
        self.canvas.create_rectangle(
            cx - r, cy - r, cx + r, cy + r,
            outline="red", width=3, tags=self.hint_tag
        )
        print(f"UI层：已在 ({x}, {y}) 显示推荐红框")

    def clear_hint(self):
        """清除棋盘上的提示框"""
        self.canvas.delete(self.hint_tag)