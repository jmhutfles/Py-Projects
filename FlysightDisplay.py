import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.simpledialog
import pandas as pd
import ReadRawData
import Conversions
import mplcursors

def run_FlysightDisplay():

    #Pull in data
    combined, Data, GPSData, rawcombined = Conversions.format_and_smooth_FS_data()

    combined = Conversions.align_baro_to_gps(combined)

    KulCombined = Conversions.kalman_fuse_gps_baro(combined)

    # Convert GPS altitude and pressure to feet if needed

    combined["GPS Altitude (ft)"] = combined["Altitude MSL"] * 3.28084  # meters to feet
    
    combined["Baro Altitude (ft)"] = combined["Baro Altitude (m)"] * 3.28084

    # Calculate acceleration magnitude (g)
    accel_cols = ["Ax (g) (filtered)", "Ay (g) (filtered)", "Az (g) (filtered)"]
    if all(col in combined.columns for col in accel_cols):
        combined["Accel Mag (g)"] = np.sqrt(
            combined["Ax (g) (filtered)"]**2 + combined["Ay (g) (filtered)"]**2 + combined["Az (g) (filtered)"]**2
        )

    if "Down Velocity" in combined.columns:
        combined["Down Velocity (ft/s)"] = combined["Down Velocity"] * 3.28084

    # Plot
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()

    # Acceleration magnitude
    lines, labels = [], []
    if "Accel Mag (g)" in combined.columns:
        l1, = ax1.plot(combined["Elapsed (s)"], combined["Accel Mag (g)"], color='tab:blue', label="Acceleration Magnitude (g)")
        ax1.set_ylabel("Acceleration Magnitude (g)", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        lines.append(l1)
        labels.append("Acceleration Magnitude (g)")

    # Altitude (GPS and Pressure)
    ax2 = ax1.twinx()
    if "GPS Altitude (ft)" in combined.columns:
        l2, = ax2.plot(combined["Elapsed (s)"], combined["GPS Altitude (ft)"], color='tab:orange', label="GPS Altitude (ft)")
        lines.append(l2)
        labels.append("GPS Altitude (ft)")
    if "Baro Altitude (ft)" in combined.columns:
        l3, = ax2.plot(combined["Elapsed (s)"], combined["Baro Altitude (ft)"], color='tab:green', label="Baro Altitude (ft)")
        lines.append(l3)
        labels.append("Baro Altitude (ft)")

    # --- Add Kalman Fused Altitude (in feet) to the same plot ---
    if "Elapsed (s)" in KulCombined.columns and "KF Altitude (m)" in KulCombined.columns:
        KulCombined["KF Altitude (ft)"] = KulCombined["KF Altitude (m)"] * 3.28084
        l_kf, = ax2.plot(KulCombined["Elapsed (s)"], KulCombined["KF Altitude (ft)"], color='tab:purple', label="KF Altitude (ft)", linewidth=2)
        lines.append(l_kf)
        labels.append("KF Altitude (ft)")

    ax2.set_ylabel("Altitude (ft)", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Add a third axis for vertical speed (GPS) in ft/s
    if "Down Velocity (ft/s)" in combined.columns:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.12))  # Offset the third axis
        l4, = ax3.plot(combined["Elapsed (s)"], combined["Down Velocity (ft/s)"], color='tab:red', label="GPS Vertical Speed (ft/s)")
        ax3.set_ylabel("GPS Vertical Speed (ft/s)", color='tab:red')
        ax3.tick_params(axis='y', labelcolor='tab:red')
        lines.append(l4)
        labels.append("GPS Vertical Speed (ft/s)")

    # Add Kalman (Kulman) vertical speed (in ft/s) to the vertical speed axis
    if "Elapsed (s)" in KulCombined.columns and "KF Vertical Speed (m/s)" in KulCombined.columns:
        KulCombined["KF Vertical Speed (ft/s)"] = KulCombined["KF Vertical Speed (m/s)"] * -3.28084
        l_kf_vs, = ax3.plot(KulCombined["Elapsed (s)"], KulCombined["KF Vertical Speed (ft/s)"], color='tab:brown', label="KF Vertical Speed (ft/s)", linewidth=2)
        lines.append(l_kf_vs)
        labels.append("KF Vertical Speed (ft/s)")

    # Add horizontal line at 25 ft/s vertical speed
    ax3.axhline(25, color='purple', linestyle='--', linewidth=1, label='25 ft/s Vertical Speed')

    # Update legend to include all lines
    ax1.legend(lines, labels, loc='upper right')

    plt.xlabel("Elapsed Time (s)")
    plt.title("Acceleration Magnitude, GPS Altitude, Pressure Altitude, and Vertical Speed vs. Elapsed Time")
    plt.tight_layout()

    # Enable interactive cursor for all lines (bubble appears on click)
    all_lines = lines  # lines contains all plotted Line2D objects
    cursor = mplcursors.cursor(all_lines, hover=False, multiple=True)

    @cursor.connect("add")
    def on_add(sel):
        # Get the x-value (Elapsed (s)) from the selected point
        x = sel.target[0]
        # Find the closest row in the DataFrame
        idx = (np.abs(combined["Elapsed (s)"] - x)).idxmin()
        # Only show selected columns
        show_cols = [
            "Elapsed (s)",
            "GPS Altitude (ft)",
            "Pressure Altitude (ft)",
            "Accel Mag (g)",
            "Down Velocity (ft/s)",
            "KF Vertical Speed (ft/s)",
            "KF Altitude (ft)"
        ]
        stats = []
        for col in show_cols:
            if col in combined.columns:
                val = combined.loc[idx, col]
                stats.append(f"{col}: {val:.3f}" if isinstance(val, (float, int, np.floating, np.integer)) else f"{col}: {val}")
            elif col in KulCombined.columns:
                # Find the closest row in KulCombined for the same elapsed time
                idx_kul = (np.abs(KulCombined["Elapsed (s)"] - x)).idxmin()
                val = KulCombined.loc[idx_kul, col]
                stats.append(f"{col}: {val:.3f}" if isinstance(val, (float, int, np.floating, np.integer)) else f"{col}: {val}")
        sel.annotation.set_text("\n".join(stats))
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

    plt.show()


if __name__ == "__main__":
    run_FlysightDisplay()