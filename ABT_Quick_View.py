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

plt.figure()
plt.scatter(Data["Time"], Data["Ax"], label="Ax", color="g")
plt.scatter(Data["Time"], Data["Ay"], label="Ay", color="r")
plt.scatter(Data["Time"], Data["Az"], label="Az", color="b")
plt.scatter(Data["Time"], Data["P"], label="Pressure", color="black")
plt.scatter(Data["Time"], Data["T"], label="Temperature", color="y")
plt.legend()
plt.show()
