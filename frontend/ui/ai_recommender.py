"""
AI Recommender UI Helper.

Handles the rendering of AI move suggestions (hints/Top-5) on the game board.
"""

class AiRecommender:
    def __init__(self, canvas, cell_size):
        self.canvas = canvas
        self.cell_size = cell_size
        self.hint_tag = "ai_hint_marker"

    def show_hint(self, x, y):
        """Show a single move hint (Red box)."""
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
        Show Top-5 recommended moves with rank indicators.
        Colors: 1=Red, 2-3=Orange, 4-5=Gray.
        """
        self.clear_hint()
        colors = ["#FF0000", "#FF8C00", "#FF8C00", "#808080", "#808080"]

        for i, (x, y) in enumerate(moves):
            rank = i + 1
            color = colors[i] if i < len(colors) else "gray"

            margin = self.cell_size
            cx = margin + x * self.cell_size
            cy = margin + y * self.cell_size
            r = self.cell_size * 0.4

            # 1. Circle Background
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=None, outline=color, width=3, tags=self.hint_tag
            )

            # 2. Rank Number
            self.canvas.create_text(
                cx, cy, text=str(rank), fill=color,
                font=("Arial", 16, "bold"), tags=self.hint_tag
            )

            # 3. Emphasis for Rank 1
            if rank == 1:
                r2 = r + 4
                self.canvas.create_rectangle(
                    cx - r2, cy - r2, cx + r2, cy + r2,
                    outline="red", width=3, tags=self.hint_tag
                )

    def clear_hint(self):
        """Clear all hint markers."""
        self.canvas.delete(self.hint_tag)