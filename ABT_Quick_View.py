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
Data = ReadRawData.ReadABT("Select the ABT file.")


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


# # Create figure and axis
# fig, ax1 = plt.subplots()

# # Plot the first Y-axis (Acceleration)
# ax1.plot(DataUnits["Time (s)"], DataUnits["Acceleration (g)"], label="Acceleration (g)", color='g')
# ax1.set_xlabel("Time (s)")  # X-axis label
# ax1.set_ylabel("Acceleration (g)", color='g')  # First Y-axis label
# ax1.tick_params(axis='y', labelcolor='g')

# # Create the second Y-axis for Altitude
# ax2 = ax1.twinx()  # Create a second Y-axis that shares the same X-axis
# ax2.plot(DataUnits["Time (s)"], DataUnits["Altitude MSL (m)"], label="Altitude (m)", color='b')
# ax2.set_ylabel("Altitude MSL (m)", color='b')  # Second Y-axis label
# ax2.tick_params(axis='y', labelcolor='b')

# # Adding title and grid
# plt.title("Acceleration and Altitude Raw Data")
# fig.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

#Smoothing
SmoothnessAlt = 500
SmoothnessAcc = float(input("Enter filtering value for accleration in ms: "))
SmoothnessAcc = int(SmoothnessAcc / 2.5)
DataUnits["Smoothed Altitude MSL (ft)"] = Conversions.MetersToFeet(DataUnits["Altitude MSL (m)"].rolling(window=SmoothnessAlt, min_periods=1).mean())
DataUnits["Smoothed Accleration (g)"] = DataUnits["Acceleration (g)"].rolling(window=SmoothnessAcc, min_periods=1).mean()

#Calc ROD
DataUnits["altitude_diff"] = DataUnits["Smoothed Altitude MSL (ft)"].diff()  # ft/s
DataUnits["time_diff"] = DataUnits["Time (s)"].diff()  # s
DataUnits["rate_of_descent_ftps"] = -DataUnits["altitude_diff"] / DataUnits["time_diff"]  # Negative for descent

# Ensure the length of rate_of_descent matches the length of DataUnits
DataUnits["rate_of_descent_ftps"] = DataUnits["rate_of_descent_ftps"].fillna(np.nan)  # Append NaN to make the length match

#Smooth ROD
SmoothnessROD = 500
DataUnits["rate_of_descent_ftps"] = DataUnits["rate_of_descent_ftps"].rolling(window=SmoothnessROD, min_periods=1).mean()

# Create figure and axis
fig, ax1 = plt.subplots()

# Plot the first Y-axis (Acceleration)
ax1.plot(DataUnits["Time (s)"], DataUnits["Smoothed Accleration (g)"], label="Acceleration (g)", color='g')
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