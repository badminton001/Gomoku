# 文件位置: frontend/ui/difficulty_control.py
import tkinter as tk
from tkinter import ttk


class DifficultyControlUI(tk.Frame):
    def __init__(self, master=None, on_mode_change=None, on_hint=None):
        super().__init__(master, bg="#f0f0f0", padx=10, pady=10)
        self.on_mode_change = on_mode_change
        self.on_hint = on_hint  # 新增：接收求助函数

        tk.Label(self, text="游戏设置", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=(0, 20))

        # 模式
        mode_frame = tk.LabelFrame(self, text="对战模式", padx=10, pady=10, bg="#f0f0f0")
        mode_frame.pack(fill="x", pady=10)
        self.mode_var = tk.StringVar(value="pve")
        tk.Radiobutton(mode_frame, text="人机对战 (PvE)", variable=self.mode_var, value="pve", bg="#f0f0f0",
                       command=self.notify_change).pack(anchor="w")
        tk.Radiobutton(mode_frame, text="双人对战 (PvP)", variable=self.mode_var, value="pvp", bg="#f0f0f0",
                       command=self.notify_change).pack(anchor="w")

        # 难度
        diff_frame = tk.LabelFrame(self, text="AI 难度", padx=10, pady=10, bg="#f0f0f0")
        diff_frame.pack(fill="x", pady=10)
        self.diff_var = tk.StringVar(value="easy")
        modes = [("简单 (Easy)", "easy"), ("中等 (Medium)", "medium"), ("困难 (Hard)", "hard")]
        for text, val in modes:
            tk.Radiobutton(diff_frame, text=text, variable=self.diff_var, value=val, bg="#f0f0f0",
                           command=self.notify_change).pack(anchor="w")

        # --- 新增功能按钮区 ---
        btn_frame = tk.Frame(self, bg="#f0f0f0")
        btn_frame.pack(fill="x", pady=20)

        # 求助按钮
        tk.Button(btn_frame, text="💡 求助 AI", bg="#fff0b3", font=("Arial", 10),
                  command=self.request_hint).pack(fill="x", pady=5)

        # 重开按钮
        tk.Button(btn_frame, text="🔄 重新开始", bg="#ffcccc", font=("Arial", 10),
                  command=self.reset_game).pack(fill="x", pady=5)

    def notify_change(self):
        if self.on_mode_change:
            self.on_mode_change(self.mode_var.get(), self.diff_var.get())

    def request_hint(self):
        """点击求助时触发"""
        if self.on_hint:
            self.on_hint()

    def reset_game(self):
        print("点击了重新开始")