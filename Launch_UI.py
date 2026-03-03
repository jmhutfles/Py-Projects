import sys
import os
import tkinter as tk
from tkinter import messagebox
import subprocess
from PIL import Image, ImageTk
from UIFunctions import RunTool
from UIFunctions import ConversionsWindow
from UIFunctions import get_resource_base_dir



# Configure Window
root = tk.Tk()
root.configure(bg='black')
root.title("Choose a Tool")

# Add Background Image
base_dir = get_resource_base_dir()

BackgroundImagePath = os.path.join(base_dir, "Pictures", "Test Session Pictures.jpg")
BackgroundImage = Image.open(BackgroundImagePath)
Background = ImageTk.PhotoImage(BackgroundImage)
bg_label = tk.Label(root, image=Background)
bg_label.place(relwidth=1, relheight=1)
width, height = BackgroundImage.size
root.geometry(f"{width}x{height}")



#Buttons that do things
button_specs = [
    ("FlySight Wind Compensation", "WindCompensation.py"),
    ("FlySight Video Overlay", "FlySightVideo.py"),
    ("ABT Video Overlay", "ABTVideo.py"),
    ("ABT Video Overlay (Special)", "SpecialRequestABTVideo.py"),
    ("IMU Video Overlay", "IMUVideo.py"),
    ("ABT Quick View", "ABT_Quick_View.py"),
    ("IMU Quick View", "IMUQuickView.py"),
    ("IMU Cone Viewer", "OrentationQuickView.py"),
    ("Flysight Sensor Fusion", "FlysightDisplay.py"),
    ("Flysight Sensor Quick View", "FlysightQuickView.py"),
    ("Ballistic Calculator DART", "DART_Timer_Simulation.py"),
]

for text, script in button_specs:
    tk.Button(
        root,
        text=text,
        command=lambda s=script: RunTool(s, base_dir, root),
        height=2,
        width=23
    ).pack(pady=15)

root.mainloop()