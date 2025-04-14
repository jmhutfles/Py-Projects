#The flysight data must be from flysight 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import math
import tkinter as tk
from tkinter import filedialog

#Dev note: msl alt range for this data is 895 475

#Importing Data
root = tk.Tk()
root.withdraw()
PathWindPack = filedialog.askopenfilename(title="Select Wind pack Data")
PathJumperRaw = filedialog.askopenfilename(title="Select Jumper Data")


# PathWindPack = Path(r'C:\Users\jhutfles\Desktop\Glide Sensor\Glide Analysis\March 2025 Test Session\L2P1\FS 11.CSV')
# PathJumperRaw = Path(r'C:\Users\jhutfles\Desktop\Glide Sensor\Glide Analysis\March 2025 Test Session\L2P1\FS 10.CSV')

DataHeaders = ["Idk What this means", "Time", "Latitude", "Longitude", "Altitude MSL", "North Velocity", "East Velocity", "Down Velocity", "hAcc", "vAcc", "sAcc", "numSV"]
WindPack = pd.read_csv(PathWindPack, names=DataHeaders, skiprows = 7)
JumperRaw = pd.read_csv(PathJumperRaw, names=DataHeaders, skiprows = 7)



#Collect User Input about Height Wind Pack was Dropped & Ground Level
DropAlt = float(input("What Altitude MSL was the WindPack Dropped? (m) : "))
GroundAlt = float(input("What Altitude is ground level MSL? (m) : "))



#Mask Data Outside Altitude Window
mask = (WindPack["Altitude MSL"] > GroundAlt) & (WindPack["Altitude MSL"] < DropAlt)
filteredWindPack = WindPack[mask]

mask = (JumperRaw["Altitude MSL"] > GroundAlt) & (JumperRaw["Altitude MSL"] < DropAlt)
filteredJumperRaw = JumperRaw[mask]




#Creating a function that describes wind as a function of altitude
Eastcoeff = np.polyfit(filteredWindPack["Altitude MSL"], filteredWindPack["East Velocity"], deg=10)
East = np.poly1d(Eastcoeff)
Northcoeff = np.polyfit(filteredWindPack["Altitude MSL"], filteredWindPack["North Velocity"], deg=10)
North = np.poly1d(Northcoeff)


#Subtract Wind from jumper data creating new data file
JumperCorrected = filteredJumperRaw
JumperCorrected.loc[:, "East Velocity"] -= East(JumperCorrected["Altitude MSL"])
JumperCorrected.loc[:, "North Velocity"] -= North(JumperCorrected["Altitude MSL"])

hspeed = np.sqrt(JumperCorrected["East Velocity"]**2 + JumperCorrected["North Velocity"]**2)
GlideRatio = pd.DataFrame({
        "Altitude MSL" : JumperCorrected["Altitude MSL"],
         "Glide Ratio" : hspeed / JumperCorrected["Down Velocity"] 
})


#Smoothing Glide Ratio Data

GlideRatio["Glide Ratio"] = GlideRatio["Glide Ratio"].rolling(window=10).mean()





#Plot the two datasets
plt.figure()
plt.scatter(filteredWindPack["Altitude MSL"], filteredWindPack["North Velocity"], label="North Wind", color = "r")
plt.plot(filteredWindPack["Altitude MSL"], North(filteredWindPack["Altitude MSL"]), label = "Polynomial Fit", color = "g")
plt.xlabel("Altitude MSL (M)")
plt.ylabel("Velocity (m/s)")
plt.title("North Wind Velocity form WindPack")
plt.legend()

plt.figure()
plt.scatter(filteredWindPack["Altitude MSL"], filteredWindPack["East Velocity"], label="East Wind", color = "r")
plt.plot(filteredWindPack["Altitude MSL"], East(filteredWindPack["Altitude MSL"]), label = "Polynomial Fit", color = "g")
plt.xlabel("Altitude MSL (M)")
plt.ylabel("Velocity (m/s)")
plt.title("East Wind Velocity form WindPack")
plt.legend()

plt.figure()
plt.plot(GlideRatio["Altitude MSL"], GlideRatio["Glide Ratio"], label = "Glide Ratio", color = "r")
plt.xlabel("Altitude MSL")
plt.ylabel("Glide Ratio")
plt.title("Glide Ratio vs Altitude")

plt.show()





