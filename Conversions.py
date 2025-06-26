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



import numpy as np
import pandas as pd

def format_and_smooth_abt_data(Data):
    """
    Formats and smooths ABT data, resampling all channels to 400 Hz and smoothing as requested.
    """
    import Conversions

    # User input for smoothing windows
    smoothness_alt_ms = int(input("Enter smoothing window for altitude (ms, default 500): ") or 500)
    smoothness_acc_ms = int(input("Enter smoothing window for acceleration (ms, default 100): ") or 100)
    smoothness_rod_ms = int(input("Enter smoothing window for rate of descent (ms, default 1500): ") or 1500)

    # Clean and sort
    Data = Data.dropna(subset=["Time"])
    Data = Data.sort_values("Time")
    Data = Data.drop_duplicates(subset=["Time"], keep="first")

    # --- Create master time grid at 400 Hz ---
    t_min = Data["Time"].min()
    t_max = Data["Time"].max()
    new_time = np.arange(t_min, t_max, 1/400)

    # --- Interpolate acceleration onto new_time ---
    ax_interp = np.interp(new_time, Data["Time"], Data["Ax"])
    ay_interp = np.interp(new_time, Data["Time"], Data["Ay"])
    az_interp = np.interp(new_time, Data["Time"], Data["Az"])

    # --- Interpolate pressure and temperature only at their valid points ---
    valid_p = Data[~Data["P"].isna()]
    valid_t = Data[~Data["T"].isna()]

    p_interp = np.interp(new_time, valid_p["Time"], valid_p["P"])
    t_interp = np.interp(new_time, valid_t["Time"], valid_t["T"])

    # --- Calculate derived columns ---
    altitude_msl_m = 44330 * (1 - (p_interp / 101325) ** (1 / 5.255))
    t_deg_c = t_interp / 1000

    # --- Build DataFrame ---
    DataUnits = pd.DataFrame({
        "Time (s)": new_time,
        "Ax": ax_interp,
        "Ay": ay_interp,
        "Az": az_interp,
        "Altitude MSL (m)": altitude_msl_m,
        "T (deg C)": t_deg_c
    })

    # --- Set index to Timedelta for time-based rolling ---
    DataUnits = DataUnits.set_index(pd.to_timedelta(DataUnits["Time (s)"], unit='s'))

    # Smoothing using time-based rolling
    DataUnits["Smoothed Altitude MSL (ft)"] = Conversions.MetersToFeet(DataUnits["Altitude MSL (m)"].rolling(f"{smoothness_alt_ms}ms", min_periods=1).mean())
    DataUnits["Smoothed Ax"] = DataUnits["Ax"].rolling(f"{smoothness_acc_ms}ms", min_periods=1).mean()
    DataUnits["Smoothed Ay"] = DataUnits["Ay"].rolling(f"{smoothness_acc_ms}ms", min_periods=1).mean()
    DataUnits["Smoothed Az"] = DataUnits["Az"].rolling(f"{smoothness_acc_ms}ms", min_periods=1).mean()

    #Calculate RMS
    DataUnits["Smoothed Acceleration (g)"] = np.sqrt(DataUnits["Smoothed Ax"]**2 + DataUnits["Smoothed Ay"]**2 + DataUnits["Smoothed Az"]**2) / 2048

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



def format_and_smooth_imu_data(Data):
    """
    Formats and smooths IMU data, resampling all channels to 400 Hz and smoothing as requested.
    """
    import Conversions

    # User input for smoothing windows
    smoothness_alt_ms = int(input("Enter smoothing window for altitude (ms, default 500): ") or 500)
    smoothness_acc_ms = int(input("Enter smoothing window for acceleration (ms, default 100): ") or 100)
    smoothness_rod_ms = int(input("Enter smoothing window for rate of descent (ms, default 1500): ") or 1500)

    # Clean and sort
    Data = Data.dropna(subset=["Time"])
    Data = Data.sort_values("Time")
    Data = Data.drop_duplicates(subset=["Time"], keep="first")

    # --- Create master time grid at 400 Hz ---
    t_min = Data["Time"].min()
    t_max = Data["Time"].max()
    new_time = np.arange(t_min, t_max, 1/400)

    # --- Interpolate acceleration onto new_time ---
    ax_interp = np.interp(new_time, Data["Time"], Data["Ax"])
    ay_interp = np.interp(new_time, Data["Time"], Data["Ay"])
    az_interp = np.interp(new_time, Data["Time"], Data["Az"])

    # --- Interpolate pressure and temperature only at their valid points ---
    valid_p = Data[~Data["P"].isna()]
    valid_t = Data[~Data["T"].isna()]

    p_interp = np.interp(new_time, valid_p["Time"], valid_p["P"])
    t_interp = np.interp(new_time, valid_t["Time"], valid_t["T"])

    # --- Calculate derived columns ---
    altitude_msl_m = 44330 * (1 - (p_interp / 101325) ** (1 / 5.255))
    t_deg_c = t_interp / 1000

    # --- Build DataFrame ---
    DataUnits = pd.DataFrame({
        "Time (s)": new_time,
        "Ax": ax_interp,
        "Ay": ay_interp,
        "Az": az_interp,
        "Altitude MSL (m)": altitude_msl_m,
        "T (deg C)": t_deg_c
    })

    # --- Set index to Timedelta for time-based rolling ---
    DataUnits = DataUnits.set_index(pd.to_timedelta(DataUnits["Time (s)"], unit='s'))

    # Smoothing using time-based rolling
    DataUnits["Smoothed Altitude MSL (ft)"] = Conversions.MetersToFeet(DataUnits["Altitude MSL (m)"].rolling(f"{smoothness_alt_ms}ms", min_periods=1).mean())
    DataUnits["Smoothed Ax"] = DataUnits["Ax"].rolling(f"{smoothness_acc_ms}ms", min_periods=1).mean()
    DataUnits["Smoothed Ay"] = DataUnits["Ay"].rolling(f"{smoothness_acc_ms}ms", min_periods=1).mean()
    DataUnits["Smoothed Az"] = DataUnits["Az"].rolling(f"{smoothness_acc_ms}ms", min_periods=1).mean()

    #Calculate RMS
    DataUnits["Smoothed Acceleration (g)"] = np.sqrt(DataUnits["Smoothed Ax"]**2 + DataUnits["Smoothed Ay"]**2 + DataUnits["Smoothed Az"]**2) / 2048

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

def align_sensor_to_gps_end(accel_data, gps_data):
    """
    Shifts the sensor UTC times so that the end of the sensor data matches the end of the GPS data.
    Returns a copy of accel_data with shifted UTC.
    """
    accel_data = accel_data.copy()
    # Find the last UTC in each dataset
    sensor_end_utc = accel_data["UTC"].max()
    gps_end_utc = gps_data["UTC"].max()
    # Compute the offset needed to align ends
    offset = gps_end_utc - sensor_end_utc
    # Shift all sensor UTCs
    accel_data["UTC"] = accel_data["UTC"] + offset
    return accel_data