# 文件位置: frontend/ui/difficulty_control.py
import tkinter as tk
from tkinter import ttk


class DifficultyControlUI(tk.Frame):
    def __init__(self, master=None, on_mode_change=None):
        super().__init__(master, bg="#f0f0f0", padx=10, pady=10)
        self.on_mode_change = on_mode_change

        # 1. 标题
        tk.Label(self, text="游戏设置", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=(0, 20))

        # 2. 模式选择区域 (LabelFrame是带边框的框)
        mode_frame = tk.LabelFrame(self, text="对战模式", padx=10, pady=10, bg="#f0f0f0")
        mode_frame.pack(fill="x", pady=10)

        self.mode_var = tk.StringVar(value="pve")  # 默认人机

        # 单选按钮: 人机 vs 双人
        tk.Radiobutton(mode_frame, text="人机对战 (PvE)", variable=self.mode_var,
                       value="pve", bg="#f0f0f0", command=self.notify_change).pack(anchor="w")
        tk.Radiobutton(mode_frame, text="双人对战 (PvP)", variable=self.mode_var,
                       value="pvp", bg="#f0f0f0", command=self.notify_change).pack(anchor="w")

        # 3. 难度选择区域
        diff_frame = tk.LabelFrame(self, text="AI 难度", padx=10, pady=10, bg="#f0f0f0")
        diff_frame.pack(fill="x", pady=10)

        self.diff_var = tk.StringVar(value="easy")

        modes = [("简单 (Easy)", "easy"), ("中等 (Medium)", "medium"), ("困难 (Hard)", "hard")]
        for text, val in modes:
            tk.Radiobutton(diff_frame, text=text, variable=self.diff_var,
                           value=val, bg="#f0f0f0", command=self.notify_change).pack(anchor="w")

        # 4. 功能按钮
        tk.Button(self, text="重新开始", bg="#ffcccc", command=self.reset_game).pack(fill="x", pady=20)

    def notify_change(self):
        """当用户改了选项，打印一下（后续这里会通知主程序）"""
        mode = self.mode_var.get()
        diff = self.diff_var.get()
        print(f"设置已更改 -> 模式: {mode}, 难度: {diff}")
        if self.on_mode_change:
            self.on_mode_change(mode, diff)

    def reset_game(self):
        print("点击了重新开始")
        # 这里预留给 reset 逻辑