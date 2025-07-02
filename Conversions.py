from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R

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


import tkinter as tk
import tkinter.simpledialog
import ReadRawData

def format_and_smooth_FS_data():
    #Get Data
    root = tk.Tk()
    root.withdraw()
    Data = ReadRawData.FlySightSensorRead("Select the Sensor FLysight file.")
    Data = convert_sensor_time_to_utc(Data)
    GPSData = ReadRawData.LoadFlysightData("Select the GPS Flysight file.")
    
    # Ask user for filter window sizes in ms (command prompt)

    while True:
        try:
            accel_window_ms = int(input("Enter acceleration filter window (ms, default 100ms): "))
            break
        except ValueError:
            print("Please enter a valid integer.")

    while True:
        try:
            pressure_window_ms = int(input("Enter pressure altitude filter window (ms default 1500ms): "))
            break
        except ValueError:
            print("Please enter a valid integer.")

    # Ensure both UTC columns are datetime and timezone-naive
    Data["UTC"] = pd.to_datetime(Data["UTC"]).dt.tz_localize(None)
    GPSData["UTC"] = pd.to_datetime(GPSData["UTC"]).dt.tz_localize(None)

    # Align sensor data so its end matches GPS data end
    Data = align_sensor_to_gps_end(Data, GPSData)

    # Sort by UTC before merging
    Data = Data.sort_values("UTC")
    GPSData = GPSData.sort_values("UTC")

    # Merge GPS data onto sensor data (keeps all sensor data rows)
    combined = pd.merge_asof(
        Data,
        GPSData,
        on="UTC",
        direction="nearest",
        suffixes=('', '_gps')
    )

    # Calculate elapsed time since start (based on UTC)
    start_utc = combined["UTC"].min()
    combined["Elapsed (s)"] = (combined["UTC"] - start_utc).dt.total_seconds()

    # Drop all other time-related columns except 'Elapsed (s)'
    time_cols = [col for col in combined.columns if col.lower().startswith("time") or col in ["tow (s)", "week", "utc", "UTC"]]
    time_cols = [col for col in time_cols if col != "Elapsed (s)"]  # keep only Elapsed (s)
    combined = combined.drop(columns=time_cols)

    # Move 'Elapsed (s)' to the first column
    cols = combined.columns.tolist()
    if "Elapsed (s)" in cols:
        cols.insert(0, cols.pop(cols.index("Elapsed (s)")))
        combined = combined[cols]


    # Ensure all sensor columns are numeric for interpolation
    sensor_cols = ["Ax (g)", "Ay (g)", "Az (g)", "Pressure (Pa)", "Temperature (deg C)", "Relative Humidity (%)", "X Mag (gauss)", "Y Mag (gauss)", "Z Mag (gauss)", "voltage (V)", "Wx (deg/s)", "Wy (deg/s)", "Wz (deg/s)"]
    for col in sensor_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Infer dtypes for all columns except 'Elapsed (s)'
    combined.iloc[:, 1:] = combined.iloc[:, 1:].infer_objects(copy=False)


    # Interpolate all columns except 'Elapsed (s)'
    combined.iloc[:, 1:] = combined.iloc[:, 1:].interpolate(method='linear', limit_direction='both')

    # Fill any remaining NaNs at the start/end
    combined = combined.bfill().ffill()

    # Now 'combined' has only 'Elapsed (s)' as the time column

    # 1. Average all duplicate rows for each 'Elapsed (s)'
    combined = combined.groupby("Elapsed (s)", as_index=False).mean(numeric_only=True)

    # 2. Set 'Elapsed (s)' as the index for resampling
    combined = combined.set_index("Elapsed (s)")

    # 3. Create a new index at 100 Hz (every 0.01 seconds)
    elapsed_min = combined.index.min()
    elapsed_max = combined.index.max()
    new_index = np.arange(elapsed_min, elapsed_max, 0.01)

    # 4. Reindex and interpolate to fill in missing values at 100 Hz
    combined_interp = pd.DataFrame({'Elapsed (s)': new_index})

    for col in combined.columns:
        # Only interpolate numeric columns, skip the index itself
        if col != 'Elapsed (s)' and np.issubdtype(combined[col].dtype, np.number):
            combined_interp[col] = np.interp(new_index, combined.index, combined[col])
        elif col != 'Elapsed (s)':
            # For non-numeric columns, just forward-fill the nearest value
            combined_interp[col] = pd.Series(combined[col].values, index=combined.index).reindex(new_index, method='nearest')

    combined_interp = combined_interp.set_index('Elapsed (s)')

    combined = combined_interp.reset_index()

    # Convert ms to samples (100 Hz = 10 ms per sample)
    accel_window_samples = max(1, int(accel_window_ms / 10))
    pressure_window_samples = max(1, int(pressure_window_ms / 10))

    #Save a dataframe before time filtering
    rawcombined = combined.copy()

    # Apply rolling mean filter
    for col in ["Ax (g)", "Ay (g)", "Az (g)"]:
        if col in combined.columns:
            combined[col + " (filtered)"] = combined[col].rolling(window=accel_window_samples, center=True, min_periods=1).mean()

    # Apply rolling mean filter to gyro columns as well
    for col in ["Wx (deg/s)", "Wy (deg/s)", "Wz (deg/s)"]:
        if col in combined.columns:
            combined[col + " (filtered)"] = combined[col].rolling(window=accel_window_samples, center=True, min_periods=1).mean()

    if "Pressure (Pa)" in combined.columns:
        combined["Pressure (Pa) (filtered)"] = combined["Pressure (Pa)"].rolling(window=pressure_window_samples, center=True, min_periods=1).mean()

    combined["Baro Altitude (m)"] = 44330 * (1 - (combined["Pressure (Pa) (filtered)"] / 101325) ** (1 / 5.255))

    return combined, Data, GPSData, rawcombined



import numpy as np
import pandas as pd

def kalman_fuse_gps_baro(
    df,
    gps_col="Altitude MSL",
    baro_col="Baro Altitude (m)",
    dt=0.01,
    R_gps=4,      # GPS measurement noise (variance, meters^2)
    R_baro=4,     # Baro measurement noise (variance, meters^2)
    Q=[[0.1, 0.0], [0.0, 0.1]]  # Process noise
):
    """
    Simple 1D Kalman filter fusing GPS and barometric altitude.
    """
    df = df.copy()

    N = len(df)
    alt_gps = df[gps_col].values if gps_col in df.columns else np.full(N, np.nan)
    alt_baro = df[baro_col].values if baro_col in df.columns else np.full(N, np.nan)

    # Kalman filter setup
    first_alt = alt_gps[0] if not np.isnan(alt_gps[0]) else alt_baro[0]
    x = np.array([[first_alt], [0]])  # [altitude, vertical_speed]
    P = np.eye(2) * 10
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.array(Q)

    kf_alt = []
    kf_vspeed = []

    for i in range(N):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update with baro
        if not np.isnan(alt_baro[i]):
            z_baro = alt_baro[i]
            y = z_baro - (H @ x)[0]
            S = H @ P @ H.T + R_baro
            K = P @ H.T / S
            x = x + K * y
            P = (np.eye(2) - K @ H) @ P

        # Update with GPS
        if not np.isnan(alt_gps[i]):
            z_gps = alt_gps[i]
            y = z_gps - (H @ x)[0]
            S = H @ P @ H.T + R_gps
            K = P @ H.T / S
            x = x + K * y
            P = (np.eye(2) - K @ H) @ P

        kf_alt.append(x[0, 0])
        kf_vspeed.append(x[1, 0])

    df["KF Altitude (m)"] = kf_alt
    df["KF Vertical Speed (m/s)"] = kf_vspeed
    return df

import numpy as np
import pandas as pd

def align_baro_to_gps(
    df,
    gps_col="Altitude MSL",
    pressure_col="Baro Altitude (m)",
    elapsed_col="Elapsed (s)",
    align_seconds=3
):
    """
    Convert pressure to altitude, then align baro altitude to GPS altitude
    using the mean offset over the last `align_seconds` seconds.
    Adds/updates 'Pressure Altitude (m)' in the DataFrame.
    """
    df = df.copy()

    # Align baro altitude to GPS altitude using the last `align_seconds` seconds
    if "Baro Altitude (m)" in df.columns and gps_col in df.columns and elapsed_col in df.columns:
        last_time = df[elapsed_col].max()
        mask = df[elapsed_col] >= (last_time - align_seconds)
        valid = mask & (~df["Baro Altitude (m)"].isna()) & (~df[gps_col].isna())
        if valid.sum() > 0:
            offset = df.loc[valid, "Baro Altitude (m)"].mean() - df.loc[valid, gps_col].mean()
            df["Baro Altitude (m)"] = df["Baro Altitude (m)"] - offset
    return df