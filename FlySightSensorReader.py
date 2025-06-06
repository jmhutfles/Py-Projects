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
GPSData = ReadRawData.LoadFlysightData("Select the GPS Flysight file.")

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
for col in ["Ax (g)", "Ay (g)", "Az (g)"]:
    Data[col] = pd.to_numeric(Data[col], errors="coerce")
accel_data = Data.dropna(subset=["Time (s)", "Ax (g)", "Ay (g)", "Az (g)"])

#Get GPS Altitude Data in Same Time Scale
GPSData["Time"] = GPSData["Time"].apply(lambda x: Conversions.iso_to_gps_week_seconds(x)[1])

plt.figure(figsize=(10, 5))
plt.plot(accel_data["Time (s)"], 
         np.sqrt((accel_data["Ax (g)"])**2 + 
                 (accel_data["Ay (g)"])**2 + 
                 (accel_data["Az (g)"])**2), 
        label="Accleration (g)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (g)")
plt.title("Acceleration vs. Time")

plt.figure(figsize=(10, 5))
plt.plot(GPSData["Time"], GPSData["Altitude MSL"], 
        label="Altitude")
plt.xlabel("Time (s)")
plt.ylabel("Altitude MSL (m)")
plt.title("Altitude vs. Time")
plt.show()
