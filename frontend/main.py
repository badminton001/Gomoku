import tkinter as tk

class GomokuUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gomoku AI")
        self.root.geometry("800x600")
        
        # Game initialization
        label = tk.Label(root, text="Gomoku AI System", font=("Arial", 20))
        label.pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuUI(root)
    root.mainloop()
