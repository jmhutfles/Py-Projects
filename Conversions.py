from datetime import datetime, timedelta
import numpy as np
import pandas as pd

#Feet to meters
def FeetToMeters (feet):
    meters = feet / 3.28284
    return meters

#meteres to feet
def MetersToFeet(meters):
    feet = meters * 3.28084
    return feet

def gps_to_utc(gps_week, gps_seconds):
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    leap_seconds = 18  # Update if leap seconds change
    return gps_epoch + timedelta(weeks=int(gps_week), seconds=float(gps_seconds) - leap_seconds)

def convert_sensor_time_to_utc(df):
    # Find the first row with both 'Time (s)', 'tow (s)', and 'week' filled (from $TIME row)
    ref_row = df.dropna(subset=["Time (s)", "tow (s)", "week"]).iloc[0]
    t_sensor_ref = float(ref_row["Time (s)"])
    t_gps_ref = float(ref_row["tow (s)"])
    gps_week = int(ref_row["week"])
    # Compute UTC for each row
    def row_to_utc(t_sensor):
        t_gps = t_gps_ref + (t_sensor - t_sensor_ref)
        return gps_to_utc(gps_week, t_gps)
    df["UTC"] = df["Time (s)"].apply(row_to_utc)
    return df

# Add this function to Conversions.py or a new utils file



def format_and_smooth_abt_data(Data, 
                               smoothness_alt_ms=500, 
                               smoothness_acc_ms=100, 
                               smoothness_rod_ms=1500):
    """
    Formats and smooths ABT data, returning a DataFrame with all derived columns.
    """
    import Conversions

    DataUnits = pd.DataFrame()
    DataUnits["Time (s)"] = Data["Time"]

    # --- Drop rows where Time is missing or invalid ---
    DataUnits = DataUnits.dropna(subset=["Time (s)"])

    # --- Sort by Time (s) to ensure monotonic index ---
    DataUnits = DataUnits.sort_values("Time (s)")

    # Altitude Data Formatting
    DataUnits["Altitude MSL (m)"] = 44330 * (1 - (Data["P"] / 101325) ** (1 / 5.255))
    DataUnits["Altitude MSL (m)"] = DataUnits["Altitude MSL (m)"].ffill()

    # Temperature Data Formatting
    DataUnits["T (deg C)"] = Data["T"] / 1000
    DataUnits["T (deg C)"] = DataUnits["T (deg C)"].ffill()

    # Acceleration Data Formatting
    DataUnits["Acceleration (g)"] = np.sqrt(
        np.square(Data["Ax"]) + np.square(Data["Ay"]) + np.square(Data["Az"])
    ) / 2048

    # --- Set index to Timedelta for time-based rolling ---
    DataUnits = DataUnits.set_index(pd.to_timedelta(DataUnits["Time (s)"], unit='s'))

    # Smoothing using time-based rolling
    DataUnits["Smoothed Altitude MSL (ft)"] = Conversions.MetersToFeet(
        DataUnits["Altitude MSL (m)"].rolling(f"{smoothness_alt_ms}ms", min_periods=1).mean()
    )
    DataUnits["Smoothed Acceleration (g)"] = DataUnits["Acceleration (g)"].rolling(
        f"{smoothness_acc_ms}ms", min_periods=1
    ).mean()

    # Calc ROD
    DataUnits["altitude_diff"] = DataUnits["Smoothed Altitude MSL (ft)"].diff()
    DataUnits["time_diff"] = DataUnits["Time (s)"].diff()
    DataUnits["rate_of_descent_ftps"] = -DataUnits["altitude_diff"] / DataUnits["time_diff"]
    DataUnits["rate_of_descent_ftps"] = DataUnits["rate_of_descent_ftps"].fillna(np.nan)

    # Smooth ROD
    DataUnits["rate_of_descent_ftps"] = DataUnits["rate_of_descent_ftps"].rolling(
        f"{smoothness_rod_ms}ms", min_periods=1
    ).mean()

    # Reset index to keep "Time (s)" as a column
    DataUnits = DataUnits.reset_index(drop=True)

    return DataUnits