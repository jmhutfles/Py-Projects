import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import math
import tkinter as tk
from scipy.interpolate import UnivariateSpline
import ReadRawData
from tkinter import filedialog
import Conversions
import time


#Get Data
root = tk.Tk()
root.withdraw()
Data = ReadRawData.FlySightSensorRead("Select the Sensor FLysight file.")

#CLeaning up the data
accel_data = Data.dropna(subset=["Time (s)", "Ax (g)", "Ay (g)", "Az (g)"])
print(Data)
# Drop rows with missing time or acceleration data
plt.figure(figsize=(10, 5))
plt.plot(accel_data["Time (s)"], 
         np.sqrt((accel_data["Ax (g)"])**2 + 
                 (accel_data["Ay (g)"])**2 + 
                 (accel_data["Az (g)"])**2), 
        label="Accleration (g)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (g)")
plt.title("Acceleration vs. Time")
plt.show()
