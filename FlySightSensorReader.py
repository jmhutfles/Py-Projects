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
Data = ReadRawData.FlySightSensorRead("Select the Sensor FLysight file.")

print(Data)