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

    # Read file line by line
    parsed_rows = []
    with open(Path, "r") as file:
        lines = file.readlines()[17:]  # Skip the first 17 lines

        for line in lines:
            columns = line.strip().split(",")  # Split line into columns
            row_data = {key: None for key in DataHeaders}  # Blank row
            tag = columns[0] if len(columns) > 0 else None  # Get the tag

            try:
                if tag == "$IMU" and len(columns) >= 9:
                    row_data["Time (s)"] = columns[1]
                    row_data["Wx (deg/s)"] = columns[2]
                    row_data["Wy (deg/s)"] = columns[3]
                    row_data["Wz (deg/s)"] = columns[4]
                    row_data["Ax (g)"] = columns[5]
                    row_data["Ay (g)"] = columns[6]
                    row_data["Az (g)"] = columns[7]
                    row_data["Temperature (deg C)"] = columns[8]
                elif tag == "$BARO" and len(columns) >= 3:
                    row_data["Time (s)"] = columns[1]
                    row_data["Pressure (Pa)"] = columns[2]
                elif tag == "$MAG" and len(columns) >= 5:
                    row_data["Time (s)"] = columns[1]
                    row_data["X Mag (gauss)"] = columns[2]
                    row_data["Y Mag (gauss)"] = columns[3]
                    row_data["Z Mag (gauss)"] = columns[4]
                elif tag == "$HUM" and len(columns) >= 3:
                    row_data["Time (s)"] = columns[1]
                    row_data["Relative Humidity (%)"] = columns[2]
                elif tag == "$TIME" and len(columns) >= 4:
                    row_data["Time (s)"] = columns[1]
                    row_data["tow (s)"] = columns[2]
                    row_data["week"] = columns[3]
                elif tag == "$VBAT" and len(columns) >= 3:
                    row_data["Time (s)"] = columns[1]
                    row_data["voltage (V)"] = columns[2]
                parsed_rows.append(row_data)
            except IndexError:
                print(f"Skipping malformed row: {line.strip()}")
                continue

    # Convert parsed rows to DataFrame
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