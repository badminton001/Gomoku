import tkinter as tk
from tkinter import ttk


class DifficultyControlUI(tk.Frame):
    # 修改点 1: 在这里增加了 on_reset 参数
    def __init__(self, master=None, on_mode_change=None, on_hint=None, on_reset=None):
        super().__init__(master, bg="#f0f0f0", padx=10, pady=10)
        self.on_mode_change = on_mode_change
        self.on_hint = on_hint
        self.on_reset = on_reset  # 修改点 2: 保存这个回调函数

        tk.Label(self, text="Game Settings", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=(0, 20))

        # Battle Mode
        mode_frame = tk.LabelFrame(self, text="Battle Mode", padx=10, pady=10, bg="#f0f0f0")
        mode_frame.pack(fill="x", pady=10)
        self.mode_var = tk.StringVar(value="pve")
        tk.Radiobutton(mode_frame, text="Player vs AI (PvE)", variable=self.mode_var, value="pve", bg="#f0f0f0",
                       command=self.notify_change).pack(anchor="w")
        tk.Radiobutton(mode_frame, text="Player vs Player (PvP)", variable=self.mode_var, value="pvp", bg="#f0f0f0",
                       command=self.notify_change).pack(anchor="w")

        # Difficulty
        diff_frame = tk.LabelFrame(self, text="Difficulty", padx=10, pady=10, bg="#f0f0f0")
        diff_frame.pack(fill="x", pady=10)
        self.diff_var = tk.StringVar(value="medium")  # default medium
        modes = [("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard")]
        for text, val in modes:
            tk.Radiobutton(diff_frame, text=text, variable=self.diff_var, value=val, bg="#f0f0f0",
                           command=self.notify_change).pack(anchor="w")

        # Buttons
        btn_frame = tk.Frame(self, bg="#f0f0f0")
        btn_frame.pack(fill="x", pady=20)

        tk.Button(btn_frame, text="💡 Ask AI for Hint", bg="#fff0b3", font=("Arial", 10),
                  command=self.request_hint).pack(fill="x", pady=5)

        # 这里的 command 绑定的是下面的 reset_game 方法
        tk.Button(btn_frame, text="🔄 Restart Game", bg="#ffcccc", font=("Arial", 10),
                  command=self.reset_game).pack(fill="x", pady=5)

    def notify_change(self):
        if self.on_mode_change:
            self.on_mode_change(self.mode_var.get(), self.diff_var.get())

    def request_hint(self):
        if self.on_hint:
            self.on_hint()

    def reset_game(self):
        """内部的重置逻辑，去调用外部传入的函数"""
        print("Control Panel: Restart clicked")
        # 修改点 3: 如果有外部传入的重置函数，就执行它
        if self.on_reset:
            self.on_reset()