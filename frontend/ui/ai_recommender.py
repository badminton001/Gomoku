import tkinter as tk

class AiRecommender:
    def __init__(self, canvas: tk.Canvas, cell_size: int):
        self.canvas = canvas
        self.cell_size = cell_size
        self.padding = 20 # Hardcoded matching BoardUI

    def clear_hint(self):
        self.canvas.delete("hint")

    def show_hint(self, x, y):
        """Show a single best move hint (Blue circle)"""
        self.clear_hint()
        cx = self.padding + x * self.cell_size
        cy = self.padding + y * self.cell_size
        
        # Draw animated-like blue circle
        self.canvas.create_oval(cx-15, cy-15, cx+15, cy+15, outline="#007BFF", width=3, tags="hint")
        self.canvas.create_oval(cx-18, cy-18, cx+18, cy+18, outline="#007BFF", width=1, dash=(2,2), tags="hint")

    def show_top5(self, candidates):
        """
        Visualize Top 5 moves with Rank coloring.
        candidates: List of (score, (x, y)) tuples, sorted best to worst.
        """
        self.clear_hint()
        
        # Colors for Ranks: 1=Red, 2-3=Orange, 4-5=Gray
        colors = ["red", "orange", "orange", "gray", "gray"]
        
        for i, (score, move) in enumerate(candidates):
            if i >= 5: break
            x, y = move
            cx = self.padding + x * self.cell_size
            cy = self.padding + y * self.cell_size
            
            color = colors[i]
            rank = i + 1
            
            # Special styling for Rank 1 (Best Move): Square box + Double thickness
            if i == 0:
                self.canvas.create_rectangle(cx-18, cy-18, cx+18, cy+18, outline=color, width=2, tags="hint")
            
            # Draw marker (Circle)
            self.canvas.create_oval(cx-14, cy-14, cx+14, cy+14, outline=color, width=2, tags="hint")
            
            # Rank Number (Centered)
            self.canvas.create_text(cx, cy, text=str(rank), fill=color, font=("Arial", 10, "bold"), tags="hint")