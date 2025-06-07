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
    title="Select the FlySight video",
    filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")]
)
output_path = filedialog.asksaveasfilename(
    title="Save output video as",
    defaultextension=".mp4",
    filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
)
if not video_path or not output_path:
    print("Operation cancelled.")
    exit()

# --- LOAD DATA ---

GPSData = ReadRawData.LoadFlysightData("Select the FlySight GPS data file")
min_utc = GPSData["UTC"].min()
GPSData["Elapsed (s)"] = (GPSData["UTC"] - min_utc).dt.total_seconds()

# Calculate total speed
GPSData["Total Speed"] = np.sqrt(
    GPSData["East Velocity"]**2 +
    GPSData["North Velocity"]**2 +
    GPSData["Down Velocity"]**2
)

# --- Let user pick landing in FlySight data by clicking the graph interactively ---
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
ax1.plot(GPSData['Elapsed (s)'], GPSData['Altitude MSL'], label='Altitude (m)', color='b')
ax2 = ax1.twinx()
ax2.plot(GPSData['Elapsed (s)'], GPSData['Total Speed'], label='Total Speed (m/s)', color='g')
ax1.set_xlabel("Elapsed Time (s)")
ax1.set_ylabel("Altitude MSL (m)", color='b')
ax2.set_ylabel("Total Speed (m/s)", color='g')
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
plot_win = 5  # seconds before and after current time
plot_width = int(frame_width * 0.8)
plot_height = int(frame_height * 0.45)
plot_x = 0
plot_y = frame_height - plot_height

# --- Precompute min/max for scaling ---
alt_min, alt_max = GPSData['Altitude MSL'].min(), GPSData['Altitude MSL'].max()
spd_min, spd_max = GPSData['Total Speed'].min(), GPSData['Total Speed'].max()

with tqdm(total=total_frames, desc="Processing video frames") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        video_time = frame_idx / fps
        data_time = video_time - offset

        # Clamp data_time to available data range
        if data_time < GPSData['Elapsed (s)'].min():
            data_time_clamped = GPSData['Elapsed (s)'].min()
        elif data_time > GPSData['Elapsed (s)'].max():
            data_time_clamped = GPSData['Elapsed (s)'].max()
        else:
            data_time_clamped = data_time

        current_row = GPSData.iloc[(GPSData['Elapsed (s)'] - data_time_clamped).abs().argsort()[:1]]
        altitude = current_row['Altitude MSL'].values[0]
        velocity = current_row['Down Velocity'].values[0]

        # --- Get windowed data ---
        x_min = max(GPSData['Elapsed (s)'].min(), data_time_clamped - plot_win)
        x_max = min(GPSData['Elapsed (s)'].max(), data_time_clamped + plot_win)
        plot_data = GPSData[(GPSData['Elapsed (s)'] >= x_min) & (GPSData['Elapsed (s)'] <= x_max)]

        # --- Draw plot background ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (plot_x, plot_y), (plot_x + plot_width, plot_y + plot_height), (30, 30, 30), -1)
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # --- Draw axes ---
        margin = 55
        axis_color = (200, 200, 200)
        cv2.line(frame, (plot_x + margin, plot_y), (plot_x + margin, plot_y + plot_height), axis_color, 1)
        cv2.line(frame, (plot_x + plot_width - margin, plot_y), (plot_x + plot_width - margin, plot_y + plot_height), axis_color, 1)
        cv2.line(frame, (plot_x + margin, plot_y + plot_height - margin), (plot_x + plot_width - margin, plot_y + plot_height - margin), axis_color, 1)

        # --- Draw data lines ---
        if len(plot_data) > 1:
            t = plot_data['Elapsed (s)'].values
            alt = plot_data['Altitude MSL'].values
            spd = plot_data['Total Speed'].values

            t_norm = ((t - x_min) / (x_max - x_min + 1e-6)) * (plot_width - 2 * margin) + plot_x + margin

            # Altitude (blue)
            alt_norm = (alt - alt_min) / (alt_max - alt_min + 1e-6)
            alt_norm = plot_y + plot_height - margin - alt_norm * (plot_height - 2 * margin)
            for i in range(1, len(t)):
                pt1 = (int(t_norm[i-1]), int(alt_norm[i-1]))
                pt2 = (int(t_norm[i]), int(alt_norm[i]))
                cv2.line(frame, pt1, pt2, (255, 200, 0), 2)

            # Total Speed (green)
            spd_norm = (spd - spd_min) / (spd_max - spd_min + 1e-6)
            spd_norm = plot_y + plot_height - margin - spd_norm * (plot_height - 2 * margin)
            for i in range(1, len(t)):
                pt1 = (int(t_norm[i-1]), int(spd_norm[i-1]))
                pt2 = (int(t_norm[i]), int(spd_norm[i]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Draw vertical line at current (clamped) data time
            curr_x = int(((data_time_clamped - x_min) / (x_max - x_min + 1e-6)) * (plot_width - 2 * margin) + plot_x + margin)
            cv2.line(frame, (curr_x, plot_y + margin), (curr_x, plot_y + plot_height - margin), (255, 255, 255), 2)

        # --- Draw text display (upper right) ---
        current_row = GPSData.iloc[(GPSData['Elapsed (s)'] - data_time_clamped).abs().argsort()[:1]]
        altitude = current_row['Altitude MSL'].values[0]
        velocity = current_row['Down Velocity'].values[0]

        info_text = [
            f"Alt: {altitude:,.1f} m",
            f"Vert Spd: {velocity:,.2f} m/s",
        ]
        text_margin = 20
        text_height = 30
        start_x = frame_width - 350
        start_y = text_margin + text_height

        for i, line in enumerate(info_text):
            cv2.putText(
                frame, line,
                (start_x, start_y + i * text_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )

        # X-axis title (bottom center)
        cv2.putText(
            frame, "Time (s)", (plot_x + plot_width // 2 - 40, plot_y + plot_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA
        )

        # Y-axis (left) ticks and labels for Altitude
        for i, val in enumerate(np.linspace(alt_min, alt_max, 5)):
            y = int(plot_y + plot_height - margin - ((val - alt_min) / (alt_max - alt_min + 1e-6)) * (plot_height - 2 * margin))
            cv2.line(frame, (plot_x + margin - 7, y), (plot_x + margin, y), (200, 200, 200), 1)
            cv2.putText(
                frame, f"{int(val):d}", (plot_x + 2, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 1, cv2.LINE_AA
            )
        cv2.putText(
            frame, "Alt (m)", (plot_x + 2, plot_y + margin - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA
        )

        # Y-axis (right) ticks and labels for Total Speed
        for i, val in enumerate(np.linspace(spd_min, spd_max, 5)):
            y = int(plot_y + plot_height - margin - ((val - spd_min) / (spd_max - spd_min + 1e-6)) * (plot_height - 2 * margin))
            cv2.line(frame, (plot_x + plot_width - margin, y), (plot_x + plot_width - margin + 7, y), (200, 200, 200), 1)
            cv2.putText(
                frame, f"{val:.1f}", (plot_x + plot_width - margin + 10, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
            )
        cv2.putText(
            frame, "Spd (m/s)", (plot_x + plot_width - margin + 5, plot_y + margin - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )

        # X-axis ticks and labels (Time)
        for i, val in enumerate(np.linspace(x_min, x_max, 5)):
            x = int(((val - x_min) / (x_max - x_min + 1e-6)) * (plot_width - 2 * margin) + plot_x + margin)
            y = plot_y + plot_height - margin
            cv2.line(frame, (x, y), (x, y + 7), (200, 200, 200), 1)
            cv2.putText(
                frame, f"{val:.1f}", (x - 15, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA
            )

        # Plot title (top center of plot area)
        plot_title = "FlySight Data Overlay"
        title_size = 1.1
        title_thickness = 2
        (title_w, title_h), _ = cv2.getTextSize(plot_title, cv2.FONT_HERSHEY_SIMPLEX, title_size, title_thickness)
        title_x = plot_x + (plot_width - title_w) // 2
        title_y = plot_y + 35
        cv2.putText(
            frame, plot_title, (title_x, title_y),
            cv2.FONT_HERSHEY_SIMPLEX, title_size, (255, 255, 255), title_thickness, cv2.LINE_AA
        )

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

cap.release()
out.release()
print("Overlay video saved:", output_path)