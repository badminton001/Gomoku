import sys
import os

# --- Core Fix: Add project root to Python search path ---
# Get current file directory (frontend)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get project root (Gomoku) - parent of frontend
root_dir = os.path.dirname(current_dir)
# Add root to sys.path so 'backend' can be found
sys.path.append(root_dir)
# ---------------------------------------------------

import tkinter as tk
from ui.board_ui import BoardUI


def main():
    root = tk.Tk()
    root.title("Gomoku - AI Battle")
    root.geometry("900x750")  # Increase height slightly for status messages

    app = BoardUI(master=root)

    root.mainloop()


if __name__ == "__main__":
    main()