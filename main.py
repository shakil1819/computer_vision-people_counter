import tkinter as tk
from src.gui.gui import PedestrianDetectionApp

def main():
    """
    Entry point for the Pedestrian Detection Application.
    """
    root = tk.Tk()
    app = PedestrianDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()