import tkinter as tk
from tkinter import messagebox
import subprocess
from PIL import Image, ImageTk
import os
from Conversions import FeetToMeters
import subprocess

def RunTool(Tool):
  subprocess.run(["Python", Tool])


def clear_root_window(root):
  for widget in root.winfo_children():
      widget.destroy()
    
  BackgroundImagePath = os.path.join("Pictures", "Test Session Pictures.jpg")
  BackgroundImage = Image.open(BackgroundImagePath)
  Background = ImageTk.PhotoImage(BackgroundImage)
  bg_label = tk.Label(root, image=Background)
  bg_label.place(relwidth=1, relheight=1)
  width, height = BackgroundImage.size
  root.geometry(f"{width}x{height}")


def ConversionsWindow(root):
  clear_root_window(root)
  new_label = tk.Label(root, text="Conversions", bg="gray")
  new_label.pack(pady=50)
  FeetToMetersButton = tk.Button(root,
                                 text="Convert Entered Value from Feet to Meters", 
                                 command=lambda: messagebox.showinfo("Result", f"{FeetToMeters(5)} meters"))
  FeetToMetersButton.pack(pady=20)