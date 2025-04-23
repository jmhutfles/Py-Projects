import tkinter as tk
from tkinter import messagebox
import subprocess

def RunTool(Tool):
  subprocess.run(["Python", Tool])

root = tk.Tk()
root.state("zoomed")
root.configure(bg='black')
root.title("Choose a Tool")

tk.Button(root, 
          text="FlySight Wind Correction", 
          command=lambda: RunTool("WindCompensation.py"), 
          height=2, 
          width=20
          ).pack(pady=20)

root.mainloop()