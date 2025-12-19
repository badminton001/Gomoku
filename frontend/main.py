import sys
import os

# --- Add project root to path ---
# Current dir
current_dir = os.path.dirname(os.path.abspath(__file__))
# Project root
root_dir = os.path.dirname(current_dir)
# Add root to sys.path
sys.path.append(root_dir)
# --------------------------------

import tkinter as tk
from ui.board_ui import BoardUI


def main():
    root = tk.Tk()
    root.title("Gomoku - AI Battle")
    root.geometry("900x750")  # Adjust height for status

    app = BoardUI(master=root)

    root.mainloop()


if __name__ == "__main__":
    main()