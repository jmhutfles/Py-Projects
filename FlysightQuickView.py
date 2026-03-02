import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import tkinter as tk
import ReadRawData
import Conversions
import mplcursors
import os
import math

def format_and_smooth_flysight_sensor_data(Data):
    """
    Formats and smooths Flysight sensor data for quick view display.
    Similar to ABT formatting but adapted for Flysight sensor structure.
    """
    
    # Default values for smoothing windows
    default_alt_ms = 500
    default_acc_ms = 100
    default_rod_ms = 1500

    use_defaults = input(
        f"Use default smoothing windows? (altitude={default_alt_ms} ms, acceleration={default_acc_ms} ms, ROD={default_rod_ms} ms) [y/n]: "
    ).strip().lower()

    if use_defaults == "y":
        smoothness_alt_ms = default_alt_ms
        smoothness_acc_ms = default_acc_ms
        smoothness_rod_ms = default_rod_ms
    else:
        smoothness_alt_ms = int(input("Enter smoothing window for altitude (ms, default 500): ") or default_alt_ms)
        smoothness_acc_ms = int(input("Enter smoothing window for acceleration (ms, default 100): ") or default_acc_ms)
        smoothness_rod_ms = int(input("Enter smoothing window for rate of descent (ms, default 1500): ") or default_rod_ms)

    # Clean and sort by Time (s)
    Data = Data.dropna(subset=["Time (s)"])
    
    # Convert all numeric columns to proper numeric types with error handling
    numeric_columns = ["Time (s)", "Ax (g)", "Ay (g)", "Az (g)", "Pressure (Pa)", "Temperature (deg C)",
                      "Relative Humidity (%)", "X Mag (gauss)", "Y Mag (gauss)", "Z Mag (gauss)", 
                      "Wx (deg/s)", "Wy (deg/s)", "Wz (deg/s)", "voltage (V)"]
    
    for col in numeric_columns:
        if col in Data.columns:
            Data[col] = pd.to_numeric(Data[col], errors='coerce')
    
    Data = Data.sort_values("Time (s)")
    Data = Data.drop_duplicates(subset=["Time (s)"], keep="first")
    
    # Remove rows where Time is NaN after conversion
    Data = Data.dropna(subset=["Time (s)"])
    
    if len(Data) == 0:
        raise ValueError("No valid time data found after cleaning")
    
    if len(Data) < 2:
        raise ValueError(f"Not enough data points ({len(Data)}) for interpolation. Need at least 2 points.")

    # --- Create master time grid at 100 Hz (Flysight typical rate) ---
    t_min = Data["Time (s)"].min()
    t_max = Data["Time (s)"].max()
    
    if pd.isna(t_min) or pd.isna(t_max) or t_max <= t_min:
        raise ValueError(f"Invalid time range: {t_min} to {t_max}")
    
    time_span = t_max - t_min
    if time_span < 0.01:  # Less than 10ms
        raise ValueError(f"Time span too short: {time_span:.6f} seconds")
    
    # Use relative time for better interpolation precision with large absolute times
    Data_relative = Data.copy()
    Data_relative["Time (s)"] = Data["Time (s)"] - t_min
    
    new_time_relative = np.arange(0, time_span, 1/100)  # 100 Hz from 0 to time_span

    # --- Interpolate acceleration data ---
    # Only use valid IMU data points, don't fill NaN with zeros
    
    # Get only rows with valid acceleration data (actual IMU measurements)
    valid_imu_data = Data_relative[
        Data_relative["Ax (g)"].notna() & 
        Data_relative["Ay (g)"].notna() & 
        Data_relative["Az (g)"].notna()
    ].copy()
    
    if len(valid_imu_data) < 2:
        raise ValueError(f"Not enough valid IMU data points ({len(valid_imu_data)}) for interpolation")
    
    # Convert to numeric for valid data only
    valid_imu_data["Ax (g)"] = pd.to_numeric(valid_imu_data["Ax (g)"], errors='coerce')
    valid_imu_data["Ay (g)"] = pd.to_numeric(valid_imu_data["Ay (g)"], errors='coerce')
    valid_imu_data["Az (g)"] = pd.to_numeric(valid_imu_data["Az (g)"], errors='coerce')
    
    # Remove any rows that became NaN after numeric conversion
    valid_imu_data = valid_imu_data.dropna(subset=["Ax (g)", "Ay (g)", "Az (g)"])
    
    if len(valid_imu_data) < 2:
        raise ValueError(f"Not enough valid numeric IMU data ({len(valid_imu_data)}) after conversion")
    
    # Interpolate using ONLY valid IMU data points
    ax_interp = np.interp(new_time_relative, valid_imu_data["Time (s)"], valid_imu_data["Ax (g)"])
    ay_interp = np.interp(new_time_relative, valid_imu_data["Time (s)"], valid_imu_data["Ay (g)"])
    az_interp = np.interp(new_time_relative, valid_imu_data["Time (s)"], valid_imu_data["Az (g)"])

    # --- Interpolate pressure data ---
    if "Pressure (Pa)" in Data_relative.columns:
        valid_pressure_data = Data_relative[Data_relative["Pressure (Pa)"].notna()].copy()
        if len(valid_pressure_data) > 0:
            valid_pressure_data["Pressure (Pa)"] = pd.to_numeric(valid_pressure_data["Pressure (Pa)"], errors='coerce')
            valid_pressure_data = valid_pressure_data.dropna(subset=["Pressure (Pa)"])
            if len(valid_pressure_data) > 0:
                p_interp = np.interp(new_time_relative, valid_pressure_data["Time (s)"], valid_pressure_data["Pressure (Pa)"])
            else:
                p_interp = np.full_like(new_time_relative, 101325)
        else:
            p_interp = np.full_like(new_time_relative, 101325)
    else:
        p_interp = np.full_like(new_time_relative, 101325)

    # --- Interpolate temperature data ---
    if "Temperature (deg C)" in Data_relative.columns:
        valid_temp_data = Data_relative[Data_relative["Temperature (deg C)"].notna()].copy()
        if len(valid_temp_data) > 0:
            valid_temp_data["Temperature (deg C)"] = pd.to_numeric(valid_temp_data["Temperature (deg C)"], errors='coerce')
            valid_temp_data = valid_temp_data.dropna(subset=["Temperature (deg C)"])
            if len(valid_temp_data) > 0:
                t_interp = np.interp(new_time_relative, valid_temp_data["Time (s)"], valid_temp_data["Temperature (deg C)"])
            else:
                t_interp = np.full_like(new_time_relative, 15)
        else:
            t_interp = np.full_like(new_time_relative, 15)
    else:
        t_interp = np.full_like(new_time_relative, 15)

    # --- Calculate barometric altitude ---
    altitude_msl_m = 44330 * (1 - (p_interp / 101325) ** (1 / 5.255))

    # --- Build DataFrame ---
    # Keep time starting at 0
    DataUnits = pd.DataFrame({
        "Time (s)": new_time_relative,  # Already starts at 0
        "Ax (g)": ax_interp,
        "Ay (g)": ay_interp,
        "Az (g)": az_interp,
        "Altitude MSL (m)": altitude_msl_m,
        "Temperature (deg C)": t_interp,
        "Pressure (Pa)": p_interp
    })

    # --- Set index to Timedelta for time-based rolling ---
    # Use relative time for timedelta calculation to avoid precision issues
    DataUnits = DataUnits.set_index(pd.to_timedelta(new_time_relative, unit='s'))

    # --- Apply smoothing using time-based rolling ---
    DataUnits["Smoothed Altitude MSL (ft)"] = Conversions.MetersToFeet(
        DataUnits["Altitude MSL (m)"].rolling(f"{smoothness_alt_ms}ms", min_periods=1).mean()
    )
    
    DataUnits["Smoothed Ax"] = DataUnits["Ax (g)"].rolling(f"{smoothness_acc_ms}ms", min_periods=1).mean()
    DataUnits["Smoothed Ay"] = DataUnits["Ay (g)"].rolling(f"{smoothness_acc_ms}ms", min_periods=1).mean()
    DataUnits["Smoothed Az"] = DataUnits["Az (g)"].rolling(f"{smoothness_acc_ms}ms", min_periods=1).mean()

    # Calculate acceleration magnitude (RMS) - this should be around 1g when stationary
    DataUnits["Smoothed Acceleration (g)"] = np.sqrt(
        DataUnits["Smoothed Ax"]**2 + 
        DataUnits["Smoothed Ay"]**2 + 
        DataUnits["Smoothed Az"]**2
    )
    


    # Calculate Rate of Descent (ROD)
    DataUnits["altitude_diff"] = DataUnits["Smoothed Altitude MSL (ft)"].diff()
    DataUnits["time_diff"] = DataUnits["Time (s)"].diff()
    DataUnits["rate_of_descent_ftps"] = -DataUnits["altitude_diff"] / DataUnits["time_diff"]
    DataUnits["rate_of_descent_ftps"] = DataUnits["rate_of_descent_ftps"].fillna(0)

    # Smooth ROD
    DataUnits["rate_of_descent_ftps"] = DataUnits["rate_of_descent_ftps"].rolling(
        f"{smoothness_rod_ms}ms", min_periods=1
    ).mean()

    # Reset index to keep "Time (s)" as a column
    DataUnits = DataUnits.reset_index(drop=True)

    return DataUnits


def run_flysight_sensor_quick_view():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    while True:
        # Get Flysight sensor data using ReadRawData's dialog
        try:
            Data = ReadRawData.FlySightSensorRead("Select Flysight sensor file")
            if Data is None or Data.empty:
                print("No file selected. Exiting.")
                break
                
            # Get a simple display name
            file_name_display = "Flysight Sensor Data"
            
        except Exception as e:
            print(f"Error reading file: {e}")
            break

        # Format and smooth the Flysight sensor data
        try:
            DataUnits = format_and_smooth_flysight_sensor_data(Data)
        except Exception as e:
            print(f"Error processing data: {e}")
            continue

        # Ask user which plot to show
        print("Choose plot type:")
        print("1: Altitude and Acceleration")
        print("2: Altitude and Rate of Descent (ROD)")
        choice = input("Enter 1 or 2: ").strip()

        fig, ax1 = plt.subplots()
        fig.canvas.manager.set_window_title(file_name_display)

        if choice == "1":
            print("Left Click to add annotation, click and hold middle mouse to move annotation, right click to delete annotation.")
            
            # Plot Altitude and Acceleration
            line1, = ax1.plot(DataUnits["Time (s)"], DataUnits["Smoothed Altitude MSL (ft)"], color='g', label="Alt (ft)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Altitude MSL (ft)", color='g')
            ax1.tick_params(axis='y', labelcolor='g')

            ax2 = ax1.twinx()
            line2, = ax2.plot(DataUnits["Time (s)"], DataUnits["Smoothed Acceleration (g)"], color='b', label="Acc Magnitude (g)")
            ax2.set_ylabel("Acceleration Magnitude (g)", color='b')  # Make it clear this is magnitude
            ax2.tick_params(axis='y', labelcolor='b')

            # Add a constant 1G dashed line on the acceleration axis
            ax2.axhline(y=1, color='gray', linestyle='--', linewidth=3, label='1G Reference')
            
            # Set reasonable y-axis limits for acceleration to ensure 1G line is visible
            acc_min = DataUnits["Smoothed Acceleration (g)"].min()
            acc_max = DataUnits["Smoothed Acceleration (g)"].max()
            margin = (acc_max - acc_min) * 0.1 + 0.1  # Add 10% margin plus 0.1g
            ax2.set_ylim(max(0, acc_min - margin), acc_max + margin)

            # Add grid lines
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)

            plt.title(file_name_display + f" (Acc: {DataUnits['Smoothed Acceleration (g)'].mean():.2f}g avg)")
            fig.tight_layout()

            # Interactive annotations for both lines
            cursor = mplcursors.cursor([line1, line2], hover=False, multiple=True)

            @cursor.connect("add")
            def on_add(sel):
                x, y = sel.target
                idx = (np.abs(DataUnits["Time (s)"] - x)).idxmin()
                t = DataUnits["Time (s)"].iloc[idx]
                acc = DataUnits["Smoothed Acceleration (g)"].iloc[idx]
                alt = DataUnits["Smoothed Altitude MSL (ft)"].iloc[idx]
                rod = DataUnits["rate_of_descent_ftps"].iloc[idx]
                sel.annotation.set(
                    text=f"Time: {t:.2f}s\nAcc: {acc:.2f} g\nAlt: {alt:.2f} ft",
                    bbox=dict(boxstyle="round", fc="yellow", alpha=0.8)
                )

        elif choice == "2":
            print("Left Click to add annotation, click and hold middle mouse to move annotation, right click to delete annotation.")
            print("Click two points on the ROD axis to average between ROD calculation.")

            # Altitude on left, ROD on right
            line1, = ax1.plot(DataUnits["Time (s)"], DataUnits["Smoothed Altitude MSL (ft)"], color='b', label="Alt (ft)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Altitude MSL (ft)", color='b')
            ax1.tick_params(axis='y', labelcolor='b')

            ax2 = ax1.twinx()
            line2, = ax2.plot(DataUnits["Time (s)"], DataUnits["rate_of_descent_ftps"], color='r', label="ROD (ft/s)")
            ax2.set_ylabel("Rate of Descent (ft/s)", color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            plt.title(file_name_display)
            fig.tight_layout()

            # Add grid lines
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)

            # Interactive annotations for both lines
            cursor = mplcursors.cursor([line1, line2], hover=False, multiple=True)

            @cursor.connect("add")
            def on_add(sel):
                x, y = sel.target
                idx = (np.abs(DataUnits["Time (s)"] - x)).idxmin()
                t = DataUnits["Time (s)"].iloc[idx]
                acc = DataUnits["Smoothed Acceleration (g)"].iloc[idx]
                alt = DataUnits["Smoothed Altitude MSL (ft)"].iloc[idx]
                rod = DataUnits["rate_of_descent_ftps"].iloc[idx]
                sel.annotation.set(
                    text=f"Time: {t:.2f}s\nAcc: {acc:.2f} g\nAlt: {alt:.2f} ft\nROD: {rod:.2f} ft/s",
                    bbox=dict(boxstyle="round", fc="yellow", alpha=0.8)
                )

            # ROD averaging functionality (same as ABT Quick View)
            rod_points = []
            interval_artists = []

            def on_click(event):
                nonlocal rod_points, interval_artists
                if event.inaxes != ax2 or event.button != 1:
                    return
                xdata = DataUnits["Time (s)"].values
                ydata = DataUnits["rate_of_descent_ftps"].values
                distances = np.sqrt((xdata - event.xdata)**2 + (ydata - event.ydata)**2)
                idx = np.argmin(distances)
                if len(rod_points) < 2:
                    rod_points.append(idx)
                    xlim = ax2.get_xlim()
                    ylim = ax2.get_ylim()
                    marker, = ax2.plot(xdata[idx], ydata[idx], 'ko')
                    interval_artists.append(marker)
                    fig.canvas.draw_idle()
                    ax2.set_xlim(xlim)
                    ax2.set_ylim(ylim)
                    fig.canvas.draw_idle()
                    if len(rod_points) == 2:
                        idx1, idx2 = sorted(rod_points)
                        line = ax2.plot(xdata[[idx1, idx2]], ydata[[idx1, idx2]], 'm--', lw=2)[0]
                        interval_artists.append(line)
                        avg_rod = np.mean(ydata[idx1:idx2+1])
                        
                        # Calculate SDSL ROD correction using barometric altitude
                        if "Altitude MSL (m)" in DataUnits.columns:
                            mean_altitude_m = np.mean(DataUnits["Altitude MSL (m)"].iloc[idx1:idx2+1])
                        else:
                            mean_altitude_m = 0.0
                        pressure0_pa = 101325
                        scale_height = 8434
                        pressure_pa = pressure0_pa * math.exp(-mean_altitude_m / scale_height)
                        avg_SDSL_ROD = avg_rod * math.sqrt(pressure_pa / pressure0_pa)
                        
                        mid_time = (xdata[idx1] + xdata[idx2]) / 2
                        mid_rod = (ydata[idx1] + ydata[idx2]) / 2
                        annotation = ax2.annotate(
                            f"Avg ROD: {avg_rod:.2f} ft/s\nAvg SDSL ROD: {avg_SDSL_ROD:.2f} ft/s",
                            xy=(mid_time, mid_rod),
                            xytext=(0, 30),
                            textcoords="offset points",
                            ha='center',
                            bbox=dict(boxstyle="round", fc="yellow", alpha=0.8),
                            arrowprops=dict(arrowstyle="->", color='magenta')
                        )
                        interval_artists.append(annotation)
                        fig.canvas.draw_idle()
                        rod_points.clear()

            def on_key(event):
                nonlocal rod_points, interval_artists
                if event.key == 'c':
                    for artist in interval_artists:
                        try:
                            artist.remove()
                        except ValueError:
                            pass
                    interval_artists.clear()
                    rod_points.clear()
                    fig.canvas.draw_idle()

            fig.canvas.mpl_connect("button_press_event", on_click)
            fig.canvas.mpl_connect("key_press_event", on_key)
            print("Press 'c' to clear all interval selections and annotations from the plot.")

        else:
            print("Invalid choice. Please enter 1 or 2.")
            continue

        plt.show()

        again = input("Process another Flysight sensor file? (y/n): ").strip().lower()
        if again != 'y':
            break


if __name__ == "__main__":
    run_flysight_sensor_quick_view()