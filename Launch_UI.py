import tkinter as tk
from tkinter import messagebox
import subprocess
from PIL import Image, ImageTk
import os
from UIFunctions import RunTool
from UIFunctions import ConversionsWindow



#Configure Window
root = tk.Tk()
root.configure(bg='black')
root.title("Choose a Tool")
#Add Background Image
script_dir = os.path.dirname(os.path.abspath(__file__))
BackgroundImagePath = os.path.join(script_dir, "Pictures", "Test Session Pictures.jpg")
BackgroundImage = Image.open(BackgroundImagePath)
Background = ImageTk.PhotoImage(BackgroundImage)
bg_label = tk.Label(root, image=Background)
bg_label.place(relwidth=1, relheight=1)
width, height = BackgroundImage.size
root.geometry(f"{width}x{height}")



#Buttons that do things
#Run Wind Correction
tk.Button(root, 
          text="FlySight Wind Correction", 
          command=lambda: RunTool("WindCompensation.py", script_dir), 
          height=2, 
          width=20
          ).pack(pady=20)
#Open Conversiosn Page
tk.Button(root,
          text="Open Conversions Calculator",
          command=lambda: ConversionsWindow(root), 
          height=2, 
          width=23).pack(pady=15)
#Open ABT Quick View
tk.Button(root,
          text="ABT Quick View",
          command=lambda: RunTool("ABT_Quick_View.py", script_dir),
          height=2, 
          width=23).pack(pady=15)

#Open IMU Quick View
tk.Button(root,
          text="IMU Quick View",
          command=lambda: RunTool("IMUQuickView.py", script_dir),
          height=2, 
          width=23).pack(pady=15)


root.mainloop()