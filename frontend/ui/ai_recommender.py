# File location: frontend/ui/ai_recommender.py
class AiRecommender:
    def __init__(self, canvas, cell_size):
        self.canvas = canvas
        self.cell_size = cell_size
        self.hint_tag = "ai_hint_marker"  # Special tag for hint box, easy to delete

    def show_hint(self, x, y):
        """Draw a flashing red square at the specified position"""
        self.clear_hint()  # Clear old hint first

        margin = self.cell_size
        cx = margin + x * self.cell_size
        cy = margin + y * self.cell_size

        # Draw a red square slightly smaller than the grid
        r = self.cell_size * 0.45
        self.canvas.create_rectangle(
            cx - r, cy - r, cx + r, cy + r,
            outline="red", width=3, tags=self.hint_tag
        )
        print(f"UI layer: Recommended red square displayed at ({x}, {y})")

    def clear_hint(self):
        """Clear the hint box on the chessboard"""
        self.canvas.delete(self.hint_tag)