import tkinter as tk
from tkinter import messagebox
import subprocess
from PIL import Image, ImageTk
import os

def RunTool(Tool):
  subprocess.run(["Python", Tool])

#Configure Window
root = tk.Tk()
root.configure(bg='black')
root.title("Choose a Tool")
#Add Background Image
BackgroundImagePath = os.path.join("Pictures", "Test Session Pictures.jpg")
BackgroundImage = Image.open(BackgroundImagePath)
Background = ImageTk.PhotoImage(BackgroundImage)
bg_label = tk.Label(root, image=Background)
bg_label.place(relwidth=1, relheight=1)
width, height = BackgroundImage.size
root.geometry(f"{width}x{height}")




tk.Button(root, 
          text="FlySight Wind Correction", 
          command=lambda: RunTool("WindCompensation.py"), 
          height=2, 
          width=20
          ).pack(pady=20)

root.mainloop()