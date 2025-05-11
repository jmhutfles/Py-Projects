import pandas as pd
import tkinter as tk
from tkinter import filedialog

def LoadFlysightData(prompt):
    #Loads flysight data into pandas data frame and copensates for it it is FS 1 or FS 2
    #Note: this deletes the first few lines of data to account for some data having headers
    
    
    #Importing Data
    
    
    #Defing Columns
    DataHeaders = ["Time", "Latitude", "Longitude", "Altitude MSL", "North Velocity", "East Velocity", "Down Velocity", "hAcc", "vAcc", "sAcc", "heading", "cAcc", "gpsFix", "numSV"]
    DataHeaders2 = ["Time", "Latitude", "Longitude", "Altitude MSL", "North Velocity", "East Velocity", "Down Velocity", "hAcc", "vAcc", "sAcc", "numSV"]
    
    
    #File dialog
    Path = filedialog.askopenfilename(title=prompt)

    #Read CSV File
    Data = pd.read_csv(Path,skiprows = 7)


    #Accound for extra first colloumn in new Flysights & Naming Columns
    if Data.iloc[0,0] == "$GNSS":
        Data.drop(Data.columns[0], axis = 1, inplace = True)
        Data.columns = DataHeaders2
    else:
        Data.columns = DataHeaders
    
    return Data


def FlySightSensorRead(prompt):    
    # Loads the Sensor data from a FlySight 2
    
    # Define Column Names
    DataHeaders = ["Time (s)", "Pressure (Pa)", "Temperature (deg C)", 
                   "Relative Humidity (%)", "X Mag (gauss)", "Y Mag (gauss)", 
                   "Z Mag (gauss)", "Wx (deg/s)", "Wy (deg/s)", "Wz (deg/s)", 
                   "Ax (g)", "Ay (g)", "Az (g)", "tow (s)", "week", "voltage (V)"]
    NData = pd.DataFrame(columns=DataHeaders)
    
    # File Dialog
    Path = filedialog.askopenfilename(title=prompt)

    # Read CSV File    
    Data = pd.read_csv(Path, skiprows=17)

    for index, row in Data.iterrows():  # Loop over rows using iterrows()
        if row[0] == "$IMU":
            NData.at[index, "Time (s)"] = row[1]
            NData.at[index, "Wx (deg/s)"] = row[2]
            NData.at[index, "Wy (deg/s)"] = row[3]
            NData.at[index, "Wz (deg/s)"] = row[4]
            NData.at[index, "Ax (g)"] = row[5]
            NData.at[index, "Ay (g)"] = row[6]
            NData.at[index, "Az (g)"] = row[7]
            NData.at[index, "Temperature (deg C)"] = row[8]
        elif row[0] == "$BARO":
            NData.at[index, "Time (s)"] = row[1]
            NData.at[index, "Pressure (Pa)"] = row[2]
        elif row[0] == "$MAG":
            NData.at[index, "Time (s)"] = row[1]
            NData.at[index, "X Mag (gauss)"] = row[2]
            NData.at[index, "Y Mag (gauss)"] = row[3]
            NData.at[index, "Z Mag (gauss)"] = row[4]
        elif row[0] == "$HUM":
            NData.at[index, "Time (s)"] = row[1]
            NData.at[index, "Relative Humidity (%)"] = row[2]
        elif row[0] == "$TIME":
            NData.at[index, "Time (s)"] = row[1]
            NData.at[index, "tow (s)"] = row[2]
            NData.at[index, "week"] = row[3]
        elif row[0] == "$VBAT":
            NData.at[index, "Time (s)"] = row[1]
            NData.at[index, "voltage (V)"] = row[2]

    return NData





def ReadABT(prompt):
    DataHeaders = ["Time", "Ax", "Ay", "Az", "P", "T"]
    
    Path = filedialog.askopenfilename(title=prompt)

    try:
        Data = pd.read_csv(Path, skiprows=11, header=None, names=DataHeaders)
        Data = Data.apply(pd.to_numeric, errors='coerce')
        
        return Data
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None


def ReadIMU(prompt):
    DataHeaders = ["Time", "Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Qw", "Qx", "Qy", "Qz", "Mx", "My", "Mz", "P", "T"]

    Path = filedialog.askopenfilename(title=prompt)

    try:
        Data = pd.read_csv(Path, skiprows=10, header=None, names=DataHeaders)
        Data = Data.apply(pd.to_numeric, errors='coerce')

        return Data
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None