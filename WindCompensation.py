#The flysight data must be from flysight 2

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



#Importing Data
root = tk.Tk()
root.withdraw()
WindPack = ReadRawData.LoadFlysightData("Select Wind Pack data")
JumperRaw = ReadRawData.LoadFlysightData("Select Jumper Flysight data")



#Use graph to Click ground and drop altitude form WindPack data
ClickedPoints = []

def OnClick(event):
    global ClickedPoints
    #Ignore clicks while zooming or panning
    if plt.get_current_fig_manager().toolbar.mode != '':
        return

    if event.xdata is None or event.ydata is None:
        return

    ClickedPoints.append(event.xdata)

    if len(ClickedPoints) == 1:
        print(f"Ground Altitude Selected MSL= {event.xdata:.2f} (m)")

    if len(ClickedPoints) == 2:
        print(f"Wind Pack Drop Altitude MSL= {event.xdata:.2f} (m)")
    
    # Stop after 2 points
    if len(ClickedPoints) == 2:
        plt.close()

#Plot Raw Wind Pack Data
plt.figure()
plt.scatter(WindPack["Altitude MSL"], WindPack["Down Velocity"], label="Down Velocity", color="r")
plt.scatter(WindPack["Altitude MSL"], WindPack["North Velocity"], label="North Velocity", color="g")
plt.scatter(WindPack["Altitude MSL"], WindPack["East Velocity"], label="East Velocity", color="b")
plt.xlabel("Altitude MSL (m)")
plt.ylabel("Velocity (m)")
plt.title("Click to select Ground Altitude then Drop Altitude")
plt.legend()

plt.gcf().canvas.mpl_connect('button_press_event', OnClick)

plt.show()

#Assign Clicked Points to variables
GroundAlt = ClickedPoints[0]
DropAlt = ClickedPoints[1]



#Mask Data Outside Altitude Window
mask = (WindPack["Altitude MSL"] > GroundAlt) & (WindPack["Altitude MSL"] < DropAlt)
filteredWindPack = WindPack[mask]

mask = (JumperRaw["Altitude MSL"] > GroundAlt) & (JumperRaw["Altitude MSL"] < DropAlt)
filteredJumperRaw = JumperRaw[mask]



#Creating a function that describes wind(Altitude)

#Clean Up data to allow spline fit to work
filteredWindPack = filteredWindPack.sort_values("Altitude MSL")
filteredWindPack = filteredWindPack.dropna(subset=["Altitude MSL", "East Velocity", "North Velocity"])

East = UnivariateSpline(
    filteredWindPack["Altitude MSL"],
    filteredWindPack["East Velocity"],
    k=3,
    s=10
)

North = UnivariateSpline(
    filteredWindPack["Altitude MSL"],
    filteredWindPack["North Velocity"],
    k=3,
    s=10
)



#Subtract Wind from jumper data creating new data file
JumperCorrected = filteredJumperRaw
JumperCorrected.loc[:, "East Velocity"] -= East(JumperCorrected["Altitude MSL"])
JumperCorrected.loc[:, "North Velocity"] -= North(JumperCorrected["Altitude MSL"])


#Create Glide Ratio Graph
hspeed = np.sqrt(JumperCorrected["East Velocity"]**2 + JumperCorrected["North Velocity"]**2)
GlideRatio = pd.DataFrame({
        "Altitude MSL" : JumperCorrected["Altitude MSL"],
         "Glide Ratio" : hspeed / JumperCorrected["Down Velocity"] 
})

#Smoothing Glide Ratio Data
GlideRatio["Glide Ratio"] = GlideRatio["Glide Ratio"].rolling(window=10).mean()

#Create Glide Ratio Data File With Altitude in ft AGL for Graphing
GlideAGLft = GlideRatio.copy()
GlideAGLft["Altitude MSL"] = GlideAGLft["Altitude MSL"] - GroundAlt
GlideAGLft["Altitude MSL"] = Conversions.MetersToFeet(GlideAGLft["Altitude MSL"])
GlideAGLft = GlideAGLft.rename(columns={"Altitude MSL": "Altitude AGL (ft)"})



#Export CSV Readable by Flysight Viewer
ExportPath = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetype=[("CSV files", "*.csv")],
        title="Save CSV File as"
)
if ExportPath:
    JumperCorrected.insert(0, "$GNSS", "$GNSS")
    JumperCorrected.to_csv(ExportPath, index=False, header=False)
    print(f"File saved to: {ExportPath}")
else:
    print("Save cancelled.")


#Create Glide Ratio Data File With Altitude in ft AGL for Graphing
GlideAGLft = GlideRatio.copy()
GlideAGLft["Altitude MSL"] = GlideAGLft["Altitude MSL"] - GroundAlt
GlideAGLft["Altitude MSL"] = Conversions.MetersToFeet(GlideAGLft["Altitude MSL"])
GlideAGLft = GlideAGLft.rename(columns={"Altitude MSL": "Altitude AGL (ft)"})





#Plot the two datasets
plt.figure()
plt.scatter(filteredWindPack["Altitude MSL"], filteredWindPack["North Velocity"], label="North Wind", color = "r")
plt.plot(filteredWindPack["Altitude MSL"], North(filteredWindPack["Altitude MSL"]), label = "Spline Fit", color = "g")
plt.xlabel("Altitude MSL (m)")
plt.ylabel("Velocity (m/s)")
plt.title("North Wind Velocity form WindPack")
plt.legend()

plt.figure()
plt.scatter(filteredWindPack["Altitude MSL"], filteredWindPack["East Velocity"], label="East Wind", color = "r")
plt.plot(filteredWindPack["Altitude MSL"], East(filteredWindPack["Altitude MSL"]), label = "Spline Fit", color = "g")
plt.xlabel("Altitude MSL (m)")
plt.ylabel("Velocity (m/s)")
plt.title("East Wind Velocity form WindPack")
plt.legend()

# plt.figure()
# plt.plot(GlideRatio["Altitude MSL"], GlideRatio["Glide Ratio"], label = "Glide Ratio", color = "r")
# plt.xlabel("Altitude MSL")
# plt.ylabel("Glide Ratio")
# plt.title("Glide Ratio vs Altitude")

plt.figure()
plt.plot(GlideAGLft["Altitude AGL (ft)"], GlideAGLft["Glide Ratio"], label = "Glide Ratio", color = "r")
plt.xlabel("Altitude AGL (ft)")
plt.ylabel("Glide Ratio")
plt.title("Glide Ratio vs Altitude")

plt.show()





