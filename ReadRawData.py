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
    # Define column names
    DataHeaders = ["Time (s)", "Pressure (Pa)", "Temperature (deg C)", 
                   "Relative Humidity (%)", "X Mag (gauss)", "Y Mag (gauss)", 
                   "Z Mag (gauss)", "Wx (deg/s)", "Wy (deg/s)", "Wz (deg/s)", 
                   "Ax (g)", "Ay (g)", "Az (g)", "tow (s)", "week", "voltage (V)"]

    # File dialog
    Path = filedialog.askopenfilename(title=prompt)
    if not Path:
        return pd.DataFrame(columns=DataHeaders)

    # Read file (skip header lines)
    Data = pd.read_csv(Path, skiprows=17, header=None, engine="python", on_bad_lines="skip")

    # Accumulate parsed rows
    parsed_rows = []

    for _, row in Data.iterrows():
        row_data = {key: None for key in DataHeaders}  # Blank row
        tag = row.iloc[0]

        try:
            if tag == "$IMU":
                row_data["Time (s)"] = row.iloc[1]
                row_data["Wx (deg/s)"] = row.iloc[2]
                row_data["Wy (deg/s)"] = row.iloc[3]
                row_data["Wz (deg/s)"] = row.iloc[4]
                row_data["Ax (g)"] = row.iloc[5]
                row_data["Ay (g)"] = row.iloc[6]
                row_data["Az (g)"] = row.iloc[7]
                row_data["Temperature (deg C)"] = row.iloc[8]
            elif tag == "$BARO":
                row_data["Time (s)"] = row.iloc[1]
                row_data["Pressure (Pa)"] = row.iloc[2]
            elif tag == "$MAG":
                row_data["Time (s)"] = row.iloc[1]
                row_data["X Mag (gauss)"] = row.iloc[2]
                row_data["Y Mag (gauss)"] = row.iloc[3]
                row_data["Z Mag (gauss)"] = row.iloc[4]
            elif tag == "$HUM":
                row_data["Time (s)"] = row.iloc[1]
                row_data["Relative Humidity (%)"] = row.iloc[2]
            elif tag == "$TIME":
                row_data["Time (s)"] = row.iloc[1]
                row_data["tow (s)"] = row.iloc[2]
                row_data["week"] = row.iloc[3]
            elif tag == "$VBAT":
                row_data["Time (s)"] = row.iloc[1]
                row_data["voltage (V)"] = row.iloc[2]
            parsed_rows.append(row_data)
        except IndexError:
            continue  # Skip malformed rows
    
    NData = pd.DataFrame(parsed_rows, columns=DataHeaders)
    NData["Time (s)"] = pd.to_numeric(NData["Time (s)"], errors="coerce")
    NData = NData.sort_values("Time (s)").reset_index(drop=True)
    
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