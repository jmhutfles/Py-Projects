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


def ReadABT(prompt):
    DataHeaders = ["Time", "Ax", "Ay", "Az", "P", "T"]
    
    Path = filedialog.askopenfilename(title=prompt)

    try:
        Data = pd.read_csv(Path, skiprows=10, header=None, names=DataHeaders)
        
        return Data
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None


def ReadIMU(prompt):
    DataHeaders = ["Time", "Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Qw", "Qx", "Qy", "Qz", "Mx", "My", "Mz", "P", "T"]

    Path = filedialog.askopenfilename(title=prompt)

    try:
        Data = pd.read_csv(Path, skiprows=10, header=None, names=DataHeaders)

        return Data
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None