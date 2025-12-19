import tkinter as tk
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from frontend.ui.board_ui import BoardUI
except ImportError:
    # Run from frontend/
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from frontend.ui.board_ui import BoardUI

def main():
    root = tk.Tk()
    root.title("Gomoku AI Battle Arena (Modular Edition)")
    root.geometry("1400x1000")
    
    try:
        # High DPI (Windows)
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    # Maximize window
    root.state('zoomed')

    # Center/Maximize
    root.update_idletasks()
    
    # --- Dynamic Scaling ---
    screen_height = root.winfo_screenheight()
    screen_width = root.winfo_screenwidth()
    
    if screen_height > 1800: # 4K Screens
        print(f"Detected 4K Screen ({screen_width}x{screen_height}). Using Ultra-Large Scale.")
        cell_size = 90
        # Large default window
        root.geometry(f"{int(screen_width*0.8)}x{int(screen_height*0.9)}")
    elif screen_height > 1200: # 1440p Screens
        print(f"Detected 2K Screen ({screen_width}x{screen_height}). Using Large Scale.")
        cell_size = 70
        root.geometry("1600x1200")
    else: # Standard Screens
        print(f"Detected Standard Screen ({screen_width}x{screen_height}). Using Standard Scale.")
        cell_size = 50 # Smaller for laptops
        root.geometry("1400x1000")

    app = BoardUI(root, cell_size=cell_size)
    # BoardUI auto-packs

    root.mainloop()

if __name__ == "__main__":
    main()
