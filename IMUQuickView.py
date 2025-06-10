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
Data = ReadRawData.ReadIMU("Select the IMU file.")

#Turn Data into Proprt Units
DataUnits = pd.DataFrame()
DataUnits["Time (s)"] = Data["Time"]

#Altitude Data Formatting
DataUnits["Altitude MSL (m)"] = 44330 * (1-(Data["P"] / 101325)**(1/5.255))
DataUnits["Altitude MSL (m)"] = DataUnits["Altitude MSL (m)"].ffill()

#Temperature Data Formatting
DataUnits["T (deg C)"] = Data["T"] / 1000
DataUnits["T (deg C)"] = DataUnits["T (deg C)"].ffill()

#Accleration Data Formatting
DataUnits["Acceleration (g)"] = np.sqrt((np.square(Data["Ax"])) + np.square(Data["Ay"]) + np.square(Data["Az"])) / 2048

# Drop rows where Time is missing or invalid
DataUnits = DataUnits.dropna(subset=["Time (s)"])

# Sort by Time (s) to ensure monotonic index
DataUnits = DataUnits.sort_values("Time (s)")

# Set index to Timedelta for time-based rolling
DataUnits = DataUnits.set_index(pd.to_timedelta(DataUnits["Time (s)"], unit='s'))

# Smoothing using time-based rolling
SmoothnessAlt = float(input("Enter filtering value for Altitude data in ms, recommend 500ms: "))
SmoothnessAcc = float(input("Enter filtering value for acceleration in ms, recommend 100ms: "))
SmoothnessROD = float(input("Enter filtering value for ROD in ms, recommend 500ms: "))

DataUnits["Smoothed Altitude MSL (ft)"] = Conversions.MetersToFeet(
    DataUnits["Altitude MSL (m)"].rolling(f"{int(SmoothnessAlt)}ms", min_periods=1).mean()
)
DataUnits["Smoothed Acceleration (g)"] = DataUnits["Acceleration (g)"].rolling(
    f"{int(SmoothnessAcc)}ms", min_periods=1
).mean()

# Calc ROD
DataUnits["altitude_diff"] = DataUnits["Smoothed Altitude MSL (ft)"].diff()
DataUnits["time_diff"] = DataUnits["Time (s)"].diff()
DataUnits["rate_of_descent_ftps"] = -DataUnits["altitude_diff"] / DataUnits["time_diff"]
DataUnits["rate_of_descent_ftps"] = DataUnits["rate_of_descent_ftps"].fillna(np.nan)

# Smooth ROD
DataUnits["rate_of_descent_ftps"] = DataUnits["rate_of_descent_ftps"].rolling(
    f"{int(SmoothnessROD)}ms", min_periods=1
).mean()

# Reset index to keep "Time (s)" as a column
DataUnits = DataUnits.reset_index(drop=True)

# Create figure and axis
fig, ax1 = plt.subplots()

# Plot the first Y-axis (Acceleration)
ax1.plot(DataUnits["Time (s)"], DataUnits["Smoothed Acceleration (g)"], label="Acceleration (g)", color='g')
ax1.set_xlabel("Time (s)")  # X-axis label
ax1.set_ylabel("Acceleration (g)", color='g')  # First Y-axis label
ax1.tick_params(axis='y', labelcolor='g')

# Create the second Y-axis for Altitude
ax2 = ax1.twinx()  # Create a second Y-axis that shares the same X-axis
ax2.plot(DataUnits["Time (s)"], DataUnits["Smoothed Altitude MSL (ft)"], label="Altitude (ft)", color='b')
ax2.set_ylabel("Altitude MSL (ft)", color='b')  # Second Y-axis label
ax2.tick_params(axis='y', labelcolor='b')

# Create the third Y-axis for Rate of Descent
ax3 = ax1.twinx()  # Create another Y-axis that shares the same X-axis
ax3.spines['right'].set_position(('outward', 60))  # Position the third Y-axis to the right
ax3.plot(DataUnits["Time (s)"], DataUnits["rate_of_descent_ftps"], label="Rate of Descent (ft/s)", color='r')
ax3.set_ylabel("Rate of Descent (ft/s)", color='r')  # Third Y-axis label
ax3.tick_params(axis='y', labelcolor='r')

# Adding title and grid
plt.title("Acceleration, Altitude and ROD Smoothed Data")
fig.tight_layout()  # Adjust layout to prevent overlap
plt.show()