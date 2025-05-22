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
print(Data)

output_file = filedialog.asksaveasfilename(
    title="Save the processed data as CSV",
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if output_file:
    Data.to_csv(output_file, index=False)
    print(f"Data exported successfully to {output_file}")
else:
    print("Export canceled.")

#CLeaning up the data
accel_data = Data.dropna(subset=["Time (s)", "Ax (g)", "Ay (g)", "Az (g)"])
print(accel_data)

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
