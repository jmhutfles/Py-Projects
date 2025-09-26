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

def IMUQuickView():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    while True:
        # Get Data and filenames using ReadRawData's dialog
        Data, file_paths = ReadRawData.ReadIMU("Select one or more IMU file(s).")
        if Data is None or file_paths is None:
            print("No file selected. Exiting.")
            break
        file_name = os.path.basename(file_paths[0])

        # Use the format_and_smooth_imu_data function from Conversions
        DataUnits = Conversions.format_and_smooth_imu_data(Data)

        # Ask user which plot to show
        print("Choose plot type:")
        print("1: Altitude and Acceleration")
        print("2: Altitude and Rate of Descent (ROD)")
        choice = input("Enter 1 or 2: ").strip()

        fig, ax1 = plt.subplots()
        fig.canvas.manager.set_window_title(file_name)

        if choice == "1":
            print("Left Click to add annotation, click and hold middle mouse to move annotation, right click to delete annotation.")
            # Plot Altitude and Acceleration
            line1, = ax1.plot(DataUnits["Time (s)"], DataUnits["Smoothed Altitude MSL (ft)"], color='g', label="Alt (ft)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Altitude MSL (ft)", color='g')
            ax1.tick_params(axis='y', labelcolor='g')

            ax2 = ax1.twinx()
            line2, = ax2.plot(DataUnits["Time (s)"], DataUnits["Smoothed Acceleration (g)"], color='b', label="Acc (g)")
            ax2.set_ylabel("Acceleration (g)", color='b')
            ax2.tick_params(axis='y', labelcolor='b')

            # Add a constant 1G dashed line on the acceleration axis
            ax2.axhline(y=1, color='gray', linestyle='--', linewidth=3, label='1G Reference')

            plt.title(file_name)
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

            plt.title(file_name)
            fig.tight_layout()

            # Interactive annotations for both lines
            cursor = mplcursors.cursor([line1, line2], hover=False, multiple=True)

            @cursor.connect("add")
            def on_add(sel):
                x, y = sel.target
                idx = (np.abs(DataUnits["Time (s)"] - x)).idxmin()
                alt = DataUnits["Smoothed Altitude MSL (ft)"].iloc[idx]
                rod = DataUnits["rate_of_descent_ftps"].iloc[idx]
                t = DataUnits["Time (s)"].iloc[idx]

                # Convert altitude (m) to pressure (Pa)
                altitude_m = DataUnits["Altitude MSL (m)"].iloc[idx] if "Altitude MSL (m)" in DataUnits.columns else 0.0
                pressure0_pa = 101325
                scale_height = 8434  # meters
                pressure_pa = pressure0_pa * math.exp(-altitude_m / scale_height)

                # SDSL correction using pressure in Pa
                SDSL_ROD = rod * math.sqrt(pressure_pa / pressure0_pa)

                sel.annotation.set(
                    text=f"Time: {t:.2f}s\nAlt: {alt:.2f} ft\nROD: {rod:.2f} ft/s\nSDSL ROD: {SDSL_ROD:.2f} ft/s",
                    bbox=dict(boxstyle="round", fc="yellow", alpha=0.8)
                )

            # --- Custom ROD interval selection ---
            rod_points = []
            interval_artists = []

            def clear_selection(event=None):
                rod_points.clear()
                # Remove all interval lines, annotations, and markers
                for artist in interval_artists:
                    try:
                        artist.remove()
                    except Exception:
                        pass
                interval_artists.clear()
                fig.canvas.draw_idle()
                print("ROD interval selection cleared. You can select new points.")

            def on_key(event):
                if event.key == "c":
                    clear_selection()

            def on_click(event):
                toolbar = plt.get_current_fig_manager().toolbar
                if toolbar.mode != '':
                    return

                if event.inaxes == ax2 and event.button == 1:
                    xdata = DataUnits["Time (s)"].values
                    ydata = DataUnits["rate_of_descent_ftps"].values
                    if event.xdata is None:
                        return
                    idx = (np.abs(xdata - event.xdata)).argmin()
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

            fig.canvas.mpl_connect("button_press_event", on_click)
            fig.canvas.mpl_connect("key_press_event", on_key)
            print("Press 'c' to clear all interval selections and annotations from the plot.")

        else:
            print("Invalid choice. Please enter 1 or 2.")
            continue

        plt.show()

        again = input("Process another IMU file? (y/n): ").strip().lower()
        if again != 'y':
            break

if __name__ == "__main__":
    IMUQuickView()