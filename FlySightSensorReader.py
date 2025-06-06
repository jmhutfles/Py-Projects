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
from matplotlib.ticker import AutoLocator


#Get Data
root = tk.Tk()
root.withdraw()
Data = ReadRawData.FlySightSensorRead("Select the Sensor FLysight file.")
Data = Conversions.convert_sensor_time_to_utc(Data)
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
accel_data = Data.dropna(subset=["UTC", "Ax (g)", "Ay (g)", "Az (g)"])

# Ensure both UTC columns are datetime and timezone-naive
accel_data.loc[:, "UTC"] = pd.to_datetime(accel_data["UTC"]).dt.tz_localize(None)
GPSData["UTC"] = pd.to_datetime(GPSData["UTC"]).dt.tz_localize(None)

# Find the earliest UTC across both datasets
min_utc = min(accel_data["UTC"].min(), GPSData["UTC"].min())

# Create elapsed seconds columns
accel_data["Elapsed (s)"] = (accel_data["UTC"] - min_utc).dt.total_seconds()
GPSData["Elapsed (s)"] = (GPSData["UTC"] - min_utc).dt.total_seconds()

# Plot both on the same figure with elapsed seconds
plt.figure(figsize=(12, 6))
ax1 = plt.gca()

# Acceleration
accel_mag = np.sqrt(
    accel_data["Ax (g)"]**2 +
    accel_data["Ay (g)"]**2 +
    accel_data["Az (g)"]**2
)
line1, = ax1.plot(accel_data["Elapsed (s)"], accel_mag, color='tab:blue', label="Acceleration (g)")
ax1.set_xlabel("Elapsed Time (s)")
ax1.set_ylabel("Acceleration (g)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Altitude
ax2 = ax1.twinx()
line2, = ax2.plot(GPSData["Elapsed (s)"], GPSData["Altitude MSL"], color='tab:orange', label="Altitude MSL (m)")
ax2.set_ylabel("Altitude MSL (m)", color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Vertical Speed (third axis)
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.12))  # Offset the third axis
line3, = ax3.plot(GPSData["Elapsed (s)"], GPSData["Down Velocity"], color='tab:green', label="Vertical Speed (m/s)")
ax3.set_ylabel("Vertical Speed (m/s)", color='tab:green')
ax3.tick_params(axis='y', labelcolor='tab:green')
ax3.yaxis.set_major_locator(AutoLocator())

# Combine legends
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Acceleration, Altitude, and GPS Vertical Speed vs. Elapsed Time")
plt.tight_layout()
plt.show()
