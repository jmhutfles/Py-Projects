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
import time



def main():
    #Importing Data
    root = tk.Tk()
    root.withdraw()
    WindPack = ReadRawData.LoadFlysightData("Select Wind Pack data")
    JumperRaw = ReadRawData.LoadFlysightData("Select Jumper Flysight data")

    #Creating Data Frame Time Format Column
    WindPack["FormattedTime"] = pd.to_datetime(WindPack["Time"])
    WindPack["TimeSinceStart (s)"] = (WindPack["FormattedTime"] - WindPack["FormattedTime"].iloc[0]).dt.total_seconds()

    JumperRaw["FormattedTime"] = pd.to_datetime(JumperRaw["Time"])
    JumperRaw["TimeSinceStart (s)"] = (JumperRaw["FormattedTime"] - JumperRaw["FormattedTime"].iloc[0]).dt.total_seconds()



    #Use graph to Click ground and drop altitude form WindPack data
    ClickedPointsY = []
    ClickedPointsX = []

    def OnClick(event):
        #Ignore clicks while zooming or panning
        if plt.get_current_fig_manager().toolbar.mode != '':
            return

        if event.xdata is None or event.ydata is None:
            return

        ClickedPointsY.append(event.ydata)
        ClickedPointsX.append(event.xdata)


        if len(ClickedPointsY) == 1:
            print(f"Impact Altitude MSL = {event.ydata:.2f} (m)")

        if len(ClickedPointsY) == 2:
            print(f"Drop Altitude MSL = {event.ydata:.2f} (m)")
        
        # Stop after 2 points
        if len(ClickedPointsY) == 2:
            plt.close()

    #Plot Raw Wind Pack Data
    fig, ax1 = plt.subplots()

    ax1.scatter(WindPack["TimeSinceStart (s)"], WindPack["Down Velocity"], label="Down Velocity", color="r")
    ax1.scatter(WindPack["TimeSinceStart (s)"], WindPack["North Velocity"], label="North Velocity", color="g")
    ax1.scatter(WindPack["TimeSinceStart (s)"], WindPack["East Velocity"], label="East Velocity", color="b")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.tick_params(axis="y", labelcolor="black")

    ax2 = ax1.twinx()
    ax2.scatter(WindPack["TimeSinceStart (s)"], WindPack["Altitude MSL"], label="Altitude MSL", color='black')
    ax2.set_ylabel("Altitude MSL (m)")
    ax2.tick_params(axis="y", labelcolor="black")

    lines_1, lables_1 = ax1.get_legend_handles_labels()
    lines_2, lables_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, lables_1 + lables_2, loc="upper right")

    plt.title("Click to select impact then drop altitude. Must click on altitude data.")

    plt.gcf().canvas.mpl_connect('button_press_event', OnClick)

    plt.show()

    #Assign Clicked Points to variables
    AltLow = min(ClickedPointsY)
    AltHigh = max(ClickedPointsY)
    DropTime = pd.to_timedelta(min(ClickedPointsX), unit="s") + WindPack["FormattedTime"].iloc[0]
    ImpactTime = pd.to_timedelta(max(ClickedPointsX), unit="s") + WindPack["FormattedTime"].iloc[0]


    #Mask Data Outside Altitude Window
    mask = (WindPack["Altitude MSL"] > AltLow) & (WindPack["Altitude MSL"] < AltHigh) & (WindPack["FormattedTime"] < (ImpactTime + pd.Timedelta(minutes=5)))
    filteredWindPack = WindPack[mask]

    mask = (JumperRaw["Altitude MSL"] > AltLow) & (JumperRaw["Altitude MSL"] < AltHigh) & (WindPack["FormattedTime"] < (ImpactTime + pd.Timedelta(minutes=5)))
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

    print("A .csv file readable by flysight viewer has been exported. Make sure the spline fits the data nicley in the supplied graphs. When confirmed the spline fits the data close the plots to continue.")

    #Plot the three datasets
    plt.figure()
    plt.scatter(filteredWindPack["Altitude MSL"], filteredWindPack["North Velocity"], label="North Wind", color = "r")
    plt.plot(filteredWindPack["Altitude MSL"], North(filteredWindPack["Altitude MSL"]), label = "Spline Fit", color = "g")
    plt.xlabel("Altitude MSL (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title("North Wind Velocity from WindPack")
    plt.gca().invert_xaxis()
    plt.legend()

    plt.figure()
    plt.scatter(filteredWindPack["Altitude MSL"], filteredWindPack["East Velocity"], label="East Wind", color = "r")
    plt.plot(filteredWindPack["Altitude MSL"], East(filteredWindPack["Altitude MSL"]), label = "Spline Fit", color = "g")
    plt.xlabel("Altitude MSL (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title("East Wind Velocity from WindPack")
    plt.gca().invert_xaxis()
    plt.legend()


    # plt.figure()
    # plt.plot(GlideRatio["Altitude MSL"], GlideRatio["Glide Ratio"], label = "Glide Ratio", color = "r")
    # plt.xlabel("Altitude MSL (m)")
    # plt.ylabel("Glide Ratio")
    # plt.title("Glide Ratio vs Altitude")
    # plt.gca().invert_xaxis()

    plt.show()


while True:
        main()
        choice = input("Would you like to process another FlySight file? (y/n): ").strip().lower()
        if choice != 'y':
            print("Goodbye!")
            time.sleep(3)
            break
