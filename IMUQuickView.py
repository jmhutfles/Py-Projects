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

while True:
    # Get Data
    root = tk.Tk()
    root.withdraw()
    Data = ReadRawData.ReadIMU("Select the IMU file.")

    # Use the format_and_smooth_imu_data function from Conversions
    DataUnits = Conversions.format_and_smooth_imu_data(Data)

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

    again = input("Process another IMU file? (y/n): ").strip().lower()
    if again != 'y':
        break