import tkinter as tk
from tkinter import messagebox
import subprocess
from PIL import Image, ImageTk
import os
from Conversions import FeetToMeters
import sys

def RunTool(Tool, location, root=None):
    def hide_and_run(func):
        if root is not None:
            root.withdraw()
        func()
        if root is not None:
            root.deiconify()

    if Tool == "ABT_Quick_View.py":
        from ABT_Quick_View import run_abt_quick_view
        hide_and_run(run_abt_quick_view)
    elif Tool == "ABTVideo.py":
        from ABTVideo import run_abt_video_overlay
        hide_and_run(run_abt_video_overlay)
    elif Tool == "IMUQuickView.py":
        from IMUQuickView import IMUQuickView
        hide_and_run(IMUQuickView)
    elif Tool == "IMUVideo.py":
        from IMUVideo import IMUVideo
        hide_and_run(IMUVideo)
    elif Tool == "FlySightVideo.py":
        from FlySightVideo import FlySightVideo
        hide_and_run(FlySightVideo)
    elif Tool == "WindCompensation.py":
        from WindCompensation import WindCompensation
        hide_and_run(WindCompensation)
    elif Tool == "FlysightDisplay.py":
        from FlysightDisplay import run_FlysightDisplay
        hide_and_run(run_FlysightDisplay)
    else:
        # fallback for development
        import sys
        import os
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(sys.argv[0])))
        exe_name = os.path.splitext(Tool)[0] + ".exe"
        exe_path = os.path.join(base_dir, exe_name)
        if os.path.exists(exe_path):
            if root is not None:
                root.withdraw()
            subprocess.run([exe_path], cwd=base_dir)
            if root is not None:
                root.deiconify()
        else:
            if root is not None:
                root.withdraw()
            subprocess.run(["python", Tool], cwd=base_dir)
            if root is not None:
                root.deiconify()


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