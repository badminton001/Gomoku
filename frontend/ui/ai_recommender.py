import tkinter as tk

class AiRecommender:
    def __init__(self, canvas: tk.Canvas, cell_size: int):
        self.canvas = canvas
        self.cell_size = cell_size
        self.padding = 20 # Match BoardUI

    def clear_hint(self):
        self.canvas.delete("hint")

    def show_hint(self, x, y):
        """Show best move hint."""
        self.clear_hint()
        cx = self.padding + x * self.cell_size
        cy = self.padding + y * self.cell_size
        
        r = self.cell_size * 0.35
        # Draw hint circle
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#007BFF", width=3, tags="hint")
        self.canvas.create_oval(cx-(r+3), cy-(r+3), cx+(r+3), cy+(r+3), outline="#007BFF", width=1, dash=(2,2), tags="hint")

    def show_top5(self, candidates):
        """Visualize Top 5 moves."""
        self.clear_hint()
        
        # Rank colors
        colors = ["red", "orange", "orange", "gray", "gray"]
        r = self.cell_size * 0.35
        
        for i, (score, move) in enumerate(candidates):
            if i >= 5: break
            x, y = move
            cx = self.padding + x * self.cell_size
            cy = self.padding + y * self.cell_size
            
            color = colors[i]
            rank = i + 1
            
            # Rank 1 styling
            if i == 0:
                self.canvas.create_rectangle(cx-(r+4), cy-(r+4), cx+(r+4), cy+(r+4), outline=color, width=2, tags="hint")
            
            # Draw marker
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline=color, width=2, tags="hint")
            
            # Rank number
            self.canvas.create_text(cx, cy, text=str(rank), fill=color, font=("Arial", int(r*0.8), "bold"), tags="hint")