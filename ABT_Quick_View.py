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
Data = ReadRawData.ReadABT("Select the ABT file.")

plt.figure()
plt.scatter(Data["Time"], Data["Ax"], label="Ax", color="g")
plt.show()
