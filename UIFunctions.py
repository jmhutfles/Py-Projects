import tkinter as tk
from tkinter import messagebox
import subprocess
from PIL import Image, ImageTk
import os
from Conversions import FeetToMeters
import sys

def RunTool(Tool, location):
    # Always use the directory of the running executable or script
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(sys.argv[0])))
    exe_name = os.path.splitext(Tool)[0] + ".exe"
    exe_path = os.path.join(base_dir, exe_name)
    if os.path.exists(exe_path):
        subprocess.run([exe_path], cwd=base_dir)
    else:
        # fallback for development
        subprocess.run(["python", Tool], cwd=base_dir)


def clear_root_window(root):
    for widget in root.winfo_children():
        widget.destroy()
    
    # Get the directory where the .exe or script is located
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    BackgroundImagePath = os.path.join(base_dir, "Pictures", "Test Session Pictures.jpg")
    BackgroundImage = Image.open(BackgroundImagePath)
    Background = ImageTk.PhotoImage(BackgroundImage)
    bg_label = tk.Label(root, image=Background)
    bg_label.place(relwidth=1, relheight=1)
    width, height = BackgroundImage.size
    root.geometry(f"{width}x{height}")


def ConversionsWindow(root):
  clear_root_window(root)

  #Create new Window
  new_label = tk.Label(root, text="Conversions", bg="gray")
  new_label.pack(pady=50)

  #Entry Feild Label
  EntryLabel = tk.Label(root, text="Enter value you want to convert.")
  EntryLabel.pack()

  #Entry Feild
  FeetInput = tk.Entry(root)
  FeetInput.pack()

  def CalculateShow():
      try:
        FeetValue = float(FeetInput.get())
        meters = FeetToMeters(FeetValue)
        messagebox.showinfo("Result", f"{FeetValue} feet = {meters:.2f} meters")
      except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number.")


  FeetToMetersButton = tk.Button(root,
                                 text="Convert Entered Value from Feet to Meters", 
                                 command=CalculateShow)
  FeetToMetersButton.pack(pady=20)