import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.simpledialog
import pandas as pd
import ReadRawData
import Conversions
import mplcursors

def run_FlysightDisplay():
    # Pull in data
    combined, Data, GPSData, rawcombined = Conversions.format_and_smooth_FS_data()
    combined = Conversions.align_baro_to_gps(combined)
    KulCombined = Conversions.kalman_fuse_gps_baro(combined)

    # Convert Kalman outputs to feet
    KulCombined["KF Altitude (ft)"] = KulCombined["KF Altitude (m)"] * 3.28084
    KulCombined["KF Vertical Speed (ft/s)"] = KulCombined["KF Vertical Speed (m/s)"] * 3.28084

    # Always use GPS/baro altitude from combined, not KulCombined
    KulCombined["GPS Altitude (ft)"] = combined["Altitude MSL (m) (filtered)"] * 3.28084
    KulCombined["Baro Altitude (ft)"] = combined["Baro Altitude (m)"] * 3.28084

    # Prepare legend lists before plotting
    lines = []
    labels = []

    # Plot KF Altitude and KF Vertical Speed
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Elapsed Time (s)")
    ax1.set_ylabel("KF Altitude (ft)", color="tab:blue")
    l1, = ax1.plot(KulCombined["Elapsed (s)"], KulCombined["KF Altitude (ft)"], color="tab:blue", label="KF Altitude (ft)")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    lines.append(l1)
    labels.append(l1.get_label())

    ax2 = ax1.twinx()
    ax2.set_ylabel("KF Vertical Speed (ft/s)", color="tab:red")
    l2, = ax2.plot(KulCombined["Elapsed (s)"], -1 * KulCombined["KF Vertical Speed (ft/s)"], color="tab:red", label="KF Vertical Speed (ft/s)")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    lines.append(l2)
    labels.append(l2.get_label())

    # Add a third y-axis for acceleration magnitude
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))
    l3, = ax3.plot(combined["Elapsed (s)"], KulCombined["Amag (g)"], color="tab:green", label="Accel Magnitude (g)", alpha=0.7)
    ax3.set_ylabel("Accel Magnitude (g)", color="tab:green")
    ax3.tick_params(axis='y', labelcolor="tab:green")
    lines.append(l3)
    labels.append("Accel Magnitude (g)")

    # Add a horizontal line at 25 ft/s on the vertical speed axis
    ax2.axhline(25, color="tab:orange", linestyle="--", linewidth=1.5, label="25 ft/s")
    lines.append(ax2.lines[-1])
    labels.append("25 ft/s")

    # Combine legends
    ax1.legend(lines, labels, loc="upper right")
    plt.title("Kalman Fused Altitude, Vertical Speed, and Accel Magnitude")
    plt.tight_layout()

    # --- Interactive annotation bubbles ---
    import mplcursors

    df = pd.DataFrame({
        "Elapsed (s)": KulCombined["Elapsed (s)"],
        "KF Altitude (ft)": KulCombined["KF Altitude (ft)"],
        "KF Vertical Speed (ft/s)": KulCombined["KF Vertical Speed (ft/s)"],
        "Accel Magnitude (g)": KulCombined["Amag (g)"]
    })

    cursor = mplcursors.cursor([l1, l2, l3], hover=False, multiple=True)

    # Store last two selected indices
    selected_points = []

    @cursor.connect("add")
    def on_add(sel):
        t = sel.target[0]
        i = (np.abs(df["Elapsed (s)"] - t)).argmin()
        row = df.iloc[i]
        sel.annotation.set(text=(
            f"Time: {row['Elapsed (s)']:.2f} s\n"
            f"KF Altitude: {row['KF Altitude (ft)']:.1f} ft\n"
            f"KF VSpeed: {row['KF Vertical Speed (ft/s)']:.1f} ft/s\n"
            f"Accel Mag: {row['Accel Magnitude (g)']:.2f} g"
        ))
        sel.annotation.draggable(True)

        # Track selected points
        selected_points.append(i)
        if len(selected_points) > 2:
            selected_points.pop(0)
        if len(selected_points) == 2:
            idx1, idx2 = selected_points
            alt1 = df.iloc[idx1]["KF Altitude (ft)"]
            alt2 = df.iloc[idx2]["KF Altitude (ft)"]
            delta = alt2 - alt1
            # Display as a popup annotation at the second point
            sel.annotation.set_text(sel.annotation.get_text() + f"\nΔ Altitude: {delta:.1f} ft")

    plt.show()

    # Plot all three on the same axis for comparison
    fig, ax = plt.subplots()
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Altitude (ft)")
    l_kf, = ax.plot(KulCombined["Elapsed (s)"], KulCombined["KF Altitude (ft)"], label="KF Altitude (ft)", color="tab:blue")
    l_gps, = ax.plot(combined["Elapsed (s)"], combined["Altitude MSL (m) (filtered)"] * 3.28084, label="GPS Altitude (ft)", color="tab:orange", alpha=0.6)
    l_baro, = ax.plot(combined["Elapsed (s)"], combined["Baro Altitude (m)"] * 3.28084, label="Baro Altitude (ft)", color="tab:green", alpha=0.6)
    ax.legend(loc="upper right")
    plt.title("Kalman Filter vs. Raw GPS and Baro Altitude")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_FlysightDisplay()