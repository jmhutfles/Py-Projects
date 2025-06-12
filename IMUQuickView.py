import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import ReadRawData
import Conversions
import mplcursors
import os

root = tk.Tk()
root.withdraw()  # Hide the root window

while True:
    # Get Data and filename using ReadRawData's dialog
    Data, file_path = ReadRawData.ReadIMU("Select the IMU file.")
    if Data is None or file_path is None:
        print("No file selected. Exiting.")
        break
    file_name = os.path.basename(file_path)

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
        line1, = ax1.plot(DataUnits["Time (s)"], DataUnits["Smoothed Acceleration (g)"], color='g', label="Acc (g)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Acceleration (g)", color='g')
        ax1.tick_params(axis='y', labelcolor='g')

        ax2 = ax1.twinx()
        line2, = ax2.plot(DataUnits["Time (s)"], DataUnits["Smoothed Altitude MSL (ft)"], color='b', label="Alt (ft)")
        ax2.set_ylabel("Altitude MSL (ft)", color='b')
        ax2.tick_params(axis='y', labelcolor='b')

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
                text=f"Time: {t:.2f}s\nAcc: {acc:.2f} g\nAlt: {alt:.2f} ft (MSL)",
                bbox=dict(boxstyle="round", fc="yellow", alpha=0.8)
            )

    elif choice == "2":
        # Plot Altitude and ROD
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
            sel.annotation.set(
                text=f"Time: {t:.2f}s\nAlt: {alt:.2f} ft (MSL)\nROD: {rod:.2f} ft/s",
                bbox=dict(boxstyle="round", fc="yellow", alpha=0.8)
            )

        # --- Custom ROD interval selection ---
        rod_points = []

        def on_click(event):
            # Prevent selection while zooming or panning
            toolbar = plt.get_current_fig_manager().toolbar
            if toolbar.mode != '':
                return

            if event.inaxes == ax2 and event.button == 1:  # Left click on ROD axis
                xdata = DataUnits["Time (s)"].values
                ydata = DataUnits["rate_of_descent_ftps"].values
                if event.xdata is None:
                    return
                idx = (np.abs(xdata - event.xdata)).argmin()
                rod_points.append(idx)
                ax2.plot(xdata[idx], ydata[idx], 'ko')  # Mark the point
                fig.canvas.draw_idle()
                if len(rod_points) == 2:
                    idx1, idx2 = sorted(rod_points)
                    # Draw line between points
                    ax2.plot(xdata[[idx1, idx2]], ydata[[idx1, idx2]], 'm--', lw=2)
                    # Calculate average ROD
                    avg_rod = np.mean(ydata[idx1:idx2+1])
                    # Annotate average ROD
                    mid_time = (xdata[idx1] + xdata[idx2]) / 2
                    mid_rod = (ydata[idx1] + ydata[idx2]) / 2
                    ax2.annotate(
                        f"Avg ROD: {avg_rod:.2f} ft/s",
                        xy=(mid_time, mid_rod),
                        xytext=(0, 30),
                        textcoords="offset points",
                        ha='center',
                        bbox=dict(boxstyle="round", fc="yellow", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", color='magenta')
                    )
                    fig.canvas.draw_idle()
                    rod_points.clear()  # Reset for next selection

        fig.canvas.mpl_connect("button_press_event", on_click)

    else:
        print("Invalid choice. Please enter 1 or 2.")
        continue

    plt.show()

    again = input("Process another IMU file? (y/n): ").strip().lower()
    if again != 'y':
        break