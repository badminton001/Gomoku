import sys
import os

# --- 核心修改：把项目根目录加入到 Python 的搜索路径中 ---
# 获取当前文件 main.py 的所在的目录 (frontend)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (Gomoku) - 即 frontend 的上一级
root_dir = os.path.dirname(current_dir)
# 把根目录加进去，这样 Python 就能找到 'backend' 了
sys.path.append(root_dir)
# ---------------------------------------------------

import tkinter as tk
from ui.board_ui import BoardUI


def main():
    root = tk.Tk()
    root.title("Gomoku - 五子棋对战 (Person F)")
    root.geometry("900x750")  # 稍微拉高一点，底部留点空间显示提示信息

    app = BoardUI(master=root)

    root.mainloop()


if __name__ == "__main__":
    main()