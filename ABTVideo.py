import cv2
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
import ReadRawData
import Conversions
from tqdm import tqdm

# --- FILE DIALOG PROMPTS ---
root = Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Select the parachute drop video",
    filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")]
)
output_path = filedialog.asksaveasfilename(
    title="Save output video as",
    defaultextension=".mp4",
    filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
)
plot_window = 10  # Seconds to show in the plot window

if not video_path or not output_path:
    print("Operation cancelled.")
    exit()

# --- LOAD DATA ---
raw_df = ReadRawData.ReadABT("Select the ABT file.")
df = Conversions.format_and_smooth_abt_data(raw_df)

# --- Let user pick landing in ABT data by clicking the graph interactively ---
import matplotlib.pyplot as plt

clicked_time = []

def on_click(event):
    if plt.get_current_fig_manager().toolbar.mode != '':
        return
    if event.xdata is None or event.ydata is None:
        return
    clicked_time.append(event.xdata)
    print(f"Selected landing time: {event.xdata:.2f} s")
    plt.close()

fig, ax1 = plt.subplots()
ax1.plot(df['Time (s)'], df['Smoothed Altitude MSL (ft)'], label='Smoothed Altitude (ft)', color='b')
ax2 = ax1.twinx()
ax2.plot(df['Time (s)'], df['Smoothed Accleration (g)'], label='Smoothed Acceleration (g)', color='g')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Altitude MSL (ft)", color='b')
ax2.set_ylabel("Acceleration (g)", color='g')
fig.suptitle("Zoom/pan as needed, then click on the landing event")
fig.legend(loc="upper right")

cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

if not clicked_time:
    print("No click detected. Exiting.")
    exit()

landing_data_time = clicked_time[0]
landing_video_time = float(input("Enter the landing time in the video (in seconds): "))
offset = landing_video_time - landing_data_time
print(f"Using offset of {offset:.2f} seconds to sync data and video.")

# --- OPEN VIDEO ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = 0

# --- Overlay parameters ---
PLOT_EVERY_N_FRAMES = 1  # OpenCV is fast, so update every frame
plot_win = 5  # seconds before and after current time
plot_width = frame_width // 2
plot_height = frame_height // 3
plot_x = 0
plot_y = frame_height - plot_height

# --- Precompute min/max for scaling ---
alt_min, alt_max = df['Smoothed Altitude MSL (ft)'].min(), df['Smoothed Altitude MSL (ft)'].max()
rod_min, rod_max = df['rate_of_descent_ftps'].min(), df['rate_of_descent_ftps'].max()
acc_min, acc_max = df['Smoothed Accleration (g)'].min(), df['Smoothed Accleration (g)'].max()

with tqdm(total=total_frames, desc="Processing video frames") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        video_time = frame_idx / fps
        data_time = video_time - offset

        # --- Get windowed data ---
        x_min = max(0, data_time - plot_win)
        x_max = data_time + plot_win
        plot_data = df[(df['Time (s)'] >= x_min) & (df['Time (s)'] <= x_max)]

        # --- Draw plot background ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (plot_x, plot_y), (plot_x + plot_width, plot_y + plot_height), (30, 30, 30), -1)
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # --- Draw axes ---
        margin = 40
        axis_color = (200, 200, 200)
        # Y axes for each variable
        cv2.line(frame, (plot_x + margin, plot_y), (plot_x + margin, plot_y + plot_height), axis_color, 1)
        cv2.line(frame, (plot_x + plot_width - margin, plot_y), (plot_x + plot_width - margin, plot_y + plot_height), axis_color, 1)
        # X axis
        cv2.line(frame, (plot_x + margin, plot_y + plot_height - margin), (plot_x + plot_width - margin, plot_y + plot_height - margin), axis_color, 1)

        # --- Draw data lines ---
        if len(plot_data) > 1:
            t = plot_data['Time (s)'].values
            alt = plot_data['Smoothed Altitude MSL (ft)'].values
            rod = plot_data['rate_of_descent_ftps'].values
            acc = plot_data['Smoothed Accleration (g)'].values

            # Normalize time to plot area
            t_norm = ((t - x_min) / (x_max - x_min + 1e-6)) * (plot_width - 2 * margin) + plot_x + margin

            # Altitude (blue)
            alt_norm = (alt - alt_min) / (alt_max - alt_min + 1e-6)
            alt_norm = plot_y + plot_height - margin - alt_norm * (plot_height - 2 * margin)
            for i in range(1, len(t)):
                pt1 = (int(t_norm[i-1]), int(alt_norm[i-1]))
                pt2 = (int(t_norm[i]), int(alt_norm[i]))
                cv2.line(frame, pt1, pt2, (255, 200, 0), 2)

            # ROD (red)
            rod_norm = (rod - rod_min) / (rod_max - rod_min + 1e-6)
            rod_norm = plot_y + plot_height - margin - rod_norm * (plot_height - 2 * margin)
            for i in range(1, len(t)):
                pt1 = (int(t_norm[i-1]), int(rod_norm[i-1]))
                pt2 = (int(t_norm[i]), int(rod_norm[i]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Acceleration (green)
            acc_norm = (acc - acc_min) / (acc_max - acc_min + 1e-6)
            acc_norm = plot_y + plot_height - margin - acc_norm * (plot_height - 2 * margin)
            for i in range(1, len(t)):
                pt1 = (int(t_norm[i-1]), int(acc_norm[i-1]))
                pt2 = (int(t_norm[i]), int(acc_norm[i]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Draw vertical line at current time
            curr_x = int(((data_time - x_min) / (x_max - x_min + 1e-6)) * (plot_width - 2 * margin) + plot_x + margin)
            cv2.line(frame, (curr_x, plot_y + margin), (curr_x, plot_y + plot_height - margin), (255, 255, 255), 2)

        # --- Draw text display ---
        current_row = df.iloc[(df['Time (s)'] - data_time).abs().argsort()[:1]]
        altitude = current_row['Smoothed Altitude MSL (ft)'].values[0]
        rod = current_row['rate_of_descent_ftps'].values[0]
        acc = current_row['Smoothed Accleration (g)'].values[0]
        info_text = f"Alt: {altitude:,.0f} ft  ROD: {rod:,.1f} ft/s  Acc: {acc:,.2f} g"
        cv2.putText(
            frame, info_text, (plot_x + 10, plot_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Y-axis title (vertical, left side)
        cv2.putText(
            frame, "Altitude/ROD/Acc", (plot_x + 5, plot_y + margin - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA
        )

        # X-axis title (bottom center)
        cv2.putText(
            frame, "Time (s)", (plot_x + plot_width // 2 - 40, plot_y + plot_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA
        )

        # Y-axis (left) ticks and labels for Altitude
        for i, val in enumerate(np.linspace(alt_min, alt_max, 5)):
            y = int(plot_y + plot_height - margin - ((val - alt_min) / (alt_max - alt_min + 1e-6)) * (plot_height - 2 * margin))
            cv2.line(frame, (plot_x + margin - 7, y), (plot_x + margin, y), (200, 200, 200), 1)
            cv2.putText(
                frame, f"{int(val):d}", (plot_x + 2, y + 5),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 200, 0), 1, cv2.LINE_AA
            )

        # Y-axis (right) ticks and labels for ROD
        for i, val in enumerate(np.linspace(rod_min, rod_max, 5)):
            y = int(plot_y + plot_height - margin - ((val - rod_min) / (rod_max - rod_min + 1e-6)) * (plot_height - 2 * margin))
            cv2.line(frame, (plot_x + plot_width - margin, y), (plot_x + plot_width - margin + 7, y), (200, 200, 200), 1)
            cv2.putText(
                frame, f"{val:.0f}", (plot_x + plot_width - margin + 10, y + 5),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA
            )

        # X-axis ticks and labels (Time)
        for i, val in enumerate(np.linspace(x_min, x_max, 5)):
            x = int(((val - x_min) / (x_max - x_min + 1e-6)) * (plot_width - 2 * margin) + plot_x + margin)
            y = plot_y + plot_height - margin
            cv2.line(frame, (x, y), (x, y + 7), (200, 200, 200), 1)
            cv2.putText(
                frame, f"{val:.1f}", (x - 15, y + 25),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA
            )

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

cap.release()
out.release()
print("Overlay video saved:", output_path)