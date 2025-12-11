import tkinter as tk
from tkinter import ttk


class DifficultyControlUI(tk.Frame):
    def __init__(self, master=None, on_settings_change=None, on_hint=None, on_reset=None):
        super().__init__(master, bg="#f0f0f0", padx=10, pady=10)
        self.on_settings_change = on_settings_change
        self.on_hint = on_hint
        self.on_reset = on_reset

        tk.Label(self, text="Game Settings", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=(0, 20))

        # --- Change 1: Player Color Selection (Replaces Battle Mode) ---
        color_frame = tk.LabelFrame(self, text="Your Color", padx=10, pady=10, bg="#f0f0f0")
        color_frame.pack(fill="x", pady=10)

        self.color_var = tk.StringVar(value="black")  # Default: Player is Black

        # Option 1: Player Black (AI White)
        tk.Radiobutton(color_frame, text="Black (Player First)", variable=self.color_var, value="black",
                       bg="#f0f0f0", command=self.notify_change).pack(anchor="w")

        # Option 2: Player White (AI Black)
        tk.Radiobutton(color_frame, text="White (AI First)", variable=self.color_var, value="white",
                       bg="#f0f0f0", command=self.notify_change).pack(anchor="w")
        # ---------------------------------------------------------------

        # --- Difficulty Selection ---
        diff_frame = tk.LabelFrame(self, text="AI Difficulty", padx=10, pady=10, bg="#f0f0f0")
        diff_frame.pack(fill="x", pady=10)
        self.diff_var = tk.StringVar(value="medium")  # default medium
        modes = [("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard")]
        for text, val in modes:
            tk.Radiobutton(diff_frame, text=text, variable=self.diff_var, value=val,
                           bg="#f0f0f0", command=self.notify_change).pack(anchor="w")

        # --- Buttons ---
        btn_frame = tk.Frame(self, bg="#f0f0f0")
        btn_frame.pack(fill="x", pady=20)

        tk.Button(btn_frame, text="💡 Ask AI for Hint", bg="#fff0b3", font=("Arial", 10),
                  command=self.request_hint).pack(fill="x", pady=5)

        tk.Button(btn_frame, text="🔄 Restart Game", bg="#ffcccc", font=("Arial", 10),
                  command=self.reset_game).pack(fill="x", pady=5)

    def notify_change(self):
        """Notify parent when settings (Color or Difficulty) change"""
        if self.on_settings_change:
            # Pass (color, difficulty) instead of (mode, difficulty)
            self.on_settings_change(self.color_var.get(), self.diff_var.get())

    def request_hint(self):
        if self.on_hint:
            self.on_hint()

    def reset_game(self):
        print("Control Panel: Restart clicked")
        if self.on_reset:
            self.on_reset()