import tkinter as tk
from tkinter import ttk


class DifficultyControlUI(tk.Frame):
    def __init__(self, master=None, on_settings_change=None, on_hint=None, on_reset=None):
        super().__init__(master, bg="#f0f0f0", padx=10, pady=10)
        self.on_settings_change = on_settings_change
        self.on_hint = on_hint
        self.on_reset = on_reset

        tk.Label(self, text="Game Settings", font=("Arial", 24, "bold"), bg="#f0f0f0").pack(pady=(0, 20))

        # Player Color
        color_frame = tk.LabelFrame(self, text="Your Color", font=("Arial", 14, "bold"), padx=10, pady=10, bg="#f0f0f0")
        color_frame.pack(fill="x", pady=10)

        self.color_var = tk.StringVar(value="black")  # Default Black

        # Option 1: Player Black
        tk.Radiobutton(color_frame, text="Black (Player First)", variable=self.color_var, value="black",
                       bg="#f0f0f0", font=("Arial", 12), command=self.notify_change).pack(anchor="w")

        # Option 2: Player White
        tk.Radiobutton(color_frame, text="White (AI First)", variable=self.color_var, value="white",
                       bg="#f0f0f0", font=("Arial", 12), command=self.notify_change).pack(anchor="w")
        # ---------------------------------------------------------------

        # Algorithm Selection
        diff_frame = tk.LabelFrame(self, text="AI Strategy", font=("Arial", 14, "bold"), padx=10, pady=10, bg="#f0f0f0")
        diff_frame.pack(fill="x", pady=10)
        self.diff_var = tk.StringVar(value="medium")  # default
        
        # Strategy labels
        modes = [
            ("Greedy Algorithm (Fast)", "easy"),
            ("Minimax Strategy (Depth 2)", "medium"),
            ("AlphaBeta Search (Depth 3)", "hard"),
            ("Hybrid Agent (Neural Net)", "expert")
        ]
        
        for text, val in modes:
            tk.Radiobutton(diff_frame, text=text, variable=self.diff_var, value=val,
                           bg="#f0f0f0", font=("Arial", 12), command=self.notify_change).pack(anchor="w")

        # Time Limit
        time_frame = tk.LabelFrame(self, text="Response Time Limit", font=("Arial", 14, "bold"), padx=10, pady=10, bg="#f0f0f0")
        time_frame.pack(fill="x", pady=10)
        
        self.time_var = tk.StringVar(value="5s")
        time_options = ["2s", "5s", "10s", "30s"]
        self.time_combo = ttk.Combobox(time_frame, textvariable=self.time_var, values=time_options, state="readonly", font=("Arial", 12))
        self.time_combo.pack(fill="x")
        self.time_combo.bind("<<ComboboxSelected>>", lambda e: self.notify_change())

        # --- Buttons ---
        btn_frame = tk.Frame(self, bg="#f0f0f0")
        btn_frame.pack(fill="x", pady=20)

        tk.Button(btn_frame, text="Ask AI for Hint", bg="#fff0b3", font=("Arial", 14),
                  command=self.request_hint).pack(fill="x", pady=5)

        tk.Button(btn_frame, text="Restart Game", bg="#ffcccc", font=("Arial", 14),
                  command=self.reset_game).pack(fill="x", pady=5)

    def notify_change(self):
        """Notify settings change"""
        if self.on_settings_change:
            # Pass (color, difficulty, time_limit)
            self.on_settings_change(self.color_var.get(), self.diff_var.get(), self.time_var.get())

    def request_hint(self):
        if self.on_hint:
            self.on_hint()

    def reset_game(self):
        print("Control Panel: Restart clicked")
        if self.on_reset:
            self.on_reset()