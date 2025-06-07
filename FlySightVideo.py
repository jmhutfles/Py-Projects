import cv2
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
import ReadRawData
import Conversions
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import os

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
GPSData = ReadRawData.LoadFlysightData("Select FLysight Data")
min_utc = GPSData["UTC"].min()
GPSData["Elapsed (s)"] = (GPSData["UTC"] - min_utc).dt.total_seconds()

# Calculate speeds and conversions
GPSData["Altitude_ft"] = GPSData["Altitude MSL"] * 3.28084
GPSData["Down_Vel_mph"] = GPSData["Down Velocity"] * 2.23694
GPSData["Horiz_Speed"] = np.sqrt(GPSData["East Velocity"]**2 + GPSData["North Velocity"]**2)
GPSData["Horiz_Speed_mph"] = GPSData["Horiz_Speed"] * 2.23694
GPSData["Total_Speed"] = np.sqrt(
    GPSData["East Velocity"]**2 +
    GPSData["North Velocity"]**2 +
    GPSData["Down Velocity"]**2
)
GPSData["Total_Speed_mph"] = GPSData["Total_Speed"] * 2.23694
GPSData["Glide_Ratio"] = np.where(
    GPSData["Down Velocity"] != 0,
    GPSData["Horiz_Speed"] / np.abs(GPSData["Down Velocity"]),
    np.nan
)
GPSData = GPSData.dropna(subset=["Altitude_ft", "Down_Vel_mph", "Horiz_Speed_mph", "Total_Speed_mph", "Glide_Ratio", "Elapsed (s)"])

# --- Let user pick landing in FlySight data by clicking the graph interactively ---
clicked_time = []
def on_click(event):
    if plt.get_current_fig_manager().toolbar.mode != '':
        return
    if event.xdata is None or event.ydata is None:
        return
    clicked_time.append(event.xdata)
    print(f"Selected landing time: {event.xdata:.2f} s")
    plt.close()

fig, ax1 = plt.subplots(figsize=(12, 6))

# Left y-axis: Altitude
ax1.plot(GPSData['Elapsed (s)'], GPSData['Altitude_ft'], label='Altitude (ft)', color='b')
ax1.set_ylabel("Altitude (ft)", color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Right y-axis: Speeds
ax2 = ax1.twinx()
ax2.plot(GPSData['Elapsed (s)'], GPSData['Down_Vel_mph'], label='Vert Spd (mph)', color='r')
ax2.plot(GPSData['Elapsed (s)'], GPSData['Horiz_Speed_mph'], label='Horiz Spd (mph)', color='g')
ax2.plot(GPSData['Elapsed (s)'], GPSData['Total_Speed_mph'], label='Total Spd (mph)', color='m')
ax2.set_ylabel("Speed (mph)", color='k')
ax2.tick_params(axis='y', labelcolor='k')

# Far right y-axis: Glide Ratio
ax3 = ax2.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.plot(GPSData['Elapsed (s)'], GPSData['Glide_Ratio'], label='Glide Ratio', color='c')
ax3.set_ylabel("Glide Ratio", color='c')
ax3.tick_params(axis='y', labelcolor='c')

# X-axis
ax1.set_xlabel("Elapsed Time (s)")

# Title
fig.suptitle("Select landing event: Altitude, Speeds, and Glide Ratio")

# Combine legends from all axes
lines, labels = [], []
for ax in [ax1, ax2, ax3]:
    line, label = ax.get_legend_handles_labels()
    lines += line
    labels += label
fig.legend(lines, labels, loc="upper right")

cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.tight_layout()
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
alt_min, alt_max = GPSData['Altitude_ft'].min(), GPSData['Altitude_ft'].max()
down_min, down_max = GPSData['Down_Vel_mph'].min(), GPSData['Down_Vel_mph'].max()
horiz_min, horiz_max = GPSData['Horiz_Speed_mph'].min(), GPSData['Horiz_Speed_mph'].max()
total_min, total_max = GPSData['Total_Speed_mph'].min(), GPSData['Total_Speed_mph'].max()
glide_min, glide_max = np.nanmin(GPSData['Glide_Ratio']), np.nanmax(GPSData['Glide_Ratio'])

with tqdm(total=total_frames, desc="Processing video frames") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        video_time = frame_idx / fps
        data_time = video_time - offset

        has_data = (GPSData['Elapsed (s)'].min() <= data_time <= GPSData['Elapsed (s)'].max())

        if not has_data:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            text = "NO DATA"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            thickness = 6
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
            continue
        else:
            # Clamp data_time to available data range
            if data_time < GPSData['Elapsed (s)'].min():
                data_time_clamped = GPSData['Elapsed (s)'].min()
            elif data_time > GPSData['Elapsed (s)'].max():
                data_time_clamped = GPSData['Elapsed (s)'].max()
            else:
                data_time_clamped = data_time

            current_row = GPSData.iloc[(GPSData['Elapsed (s)'] - data_time_clamped).abs().argsort()[:1]]
            altitude_ft = current_row['Altitude_ft'].values[0]
            down_vel_mph = current_row['Down_Vel_mph'].values[0]
            horiz_speed_mph = current_row['Horiz_Speed_mph'].values[0]
            total_speed_mph = current_row['Total_Speed_mph'].values[0]
            glide_ratio = current_row['Glide_Ratio'].values[0]

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
                down = plot_data['Down_Vel_mph'].values
                horiz = plot_data['Horiz_Speed_mph'].values
                glide = plot_data['Glide_Ratio'].values

                t_norm = ((t - x_min) / (x_max - x_min + 1e-6)) * (plot_width - 2 * margin) + plot_x + margin

                # Auto-scale for speeds in this window
                down_min_win, down_max_win = down.min(), down.max()
                horiz_min_win, horiz_max_win = horiz.min(), horiz.max()
                speed_min = min(down_min_win, horiz_min_win)
                speed_max = max(down_max_win, horiz_max_win)

                # Vertical Speed (red)
                down_norm = (down - speed_min) / (speed_max - speed_min + 1e-6)
                down_norm = plot_y + plot_height - margin - down_norm * (plot_height - 2 * margin)
                for i in range(1, len(t)):
                    pt1 = (int(t_norm[i-1]), int(down_norm[i-1]))
                    pt2 = (int(t_norm[i]), int(down_norm[i]))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

                # Horizontal Speed (green)
                horiz_norm = (horiz - speed_min) / (speed_max - speed_min + 1e-6)
                horiz_norm = plot_y + plot_height - margin - horiz_norm * (plot_height - 2 * margin)
                for i in range(1, len(t)):
                    pt1 = (int(t_norm[i-1]), int(horiz_norm[i-1]))
                    pt2 = (int(t_norm[i]), int(horiz_norm[i]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                # Glide Ratio (cyan), always scale 0-6
                glide_clipped = np.clip(glide, 0, 6)
                glide_norm = (glide_clipped - 0) / 6.0
                glide_norm = plot_y + plot_height - margin - glide_norm * (plot_height - 2 * margin)
                for i in range(1, len(t)):
                    if not np.isnan(glide_norm[i-1]) and not np.isnan(glide_norm[i]):
                        pt1 = (int(t_norm[i-1]), int(glide_norm[i-1]))
                        pt2 = (int(t_norm[i]), int(glide_norm[i]))
                        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

                # Draw vertical line at current (clamped) data time
                curr_x = int(((data_time_clamped - x_min) / (x_max - x_min + 1e-6)) * (plot_width - 2 * margin) + plot_x + margin)
                cv2.line(frame, (curr_x, plot_y + margin), (curr_x, plot_y + plot_height - margin), (255, 255, 255), 2)

            # --- Draw text display (upper right) ---
            info_text = [
                f"Alt: {altitude_ft:,.0f} ft",
                f"Vert Spd: {down_vel_mph:.1f} mph",
                f"Horiz Spd: {horiz_speed_mph:.1f} mph",
                f"Total Spd: {total_speed_mph:.1f} mph",
                f"Glide: {glide_ratio:.2f}"
            ]
            text_margin = 20
            text_height = 30
            start_x = frame_width - 350
            start_y = text_margin + text_height

            # Draw background rectangle for text
            rect_width = 330
            rect_height = len(info_text) * text_height + 20
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (start_x - 20, start_y - text_height),
                (start_x - 20 + rect_width, start_y - text_height + rect_height),
                (30, 30, 30), -1
            )
            alpha = 0.6
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Draw the text
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
                frame, "Alt (ft)", (plot_x + 2, plot_y + margin - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA
            )

            # Y-axis (right) ticks and labels for Speed (mph)
            for i, val in enumerate(np.linspace(total_min, total_max, 5)):
                y = int(plot_y + plot_height - margin - ((val - total_min) / (total_max - total_min + 1e-6)) * (plot_height - 2 * margin))
                cv2.line(frame, (plot_x + plot_width - margin, y), (plot_x + plot_width - margin + 7, y), (200, 200, 200), 1)
                cv2.putText(
                    frame, f"{val:.1f}", (plot_x + plot_width - margin + 10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA
                )
            cv2.putText(
                frame, "Speed (mph)", (plot_x + plot_width - margin + 5, plot_y + margin - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA
            )

            # Y-axis (left) ticks and labels for Speed (mph)
            for i, val in enumerate(np.linspace(speed_min, speed_max, 5)):
                y = int(plot_y + plot_height - margin - ((val - speed_min) / (speed_max - speed_min + 1e-6)) * (plot_height - 2 * margin))
                cv2.line(frame, (plot_x + margin - 7, y), (plot_x + margin, y), (200, 200, 200), 1)
                cv2.putText(
                    frame, f"{val:.1f}", (plot_x + 2, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
                )
            cv2.putText(
                frame, "Speed (mph)", (plot_x + 2, plot_y + margin - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )

            # Y-axis (far right) ticks and labels for Glide Ratio (0-6)
            for i, val in enumerate(np.linspace(0, 6, 7)):
                y = int(plot_y + plot_height - margin - ((val - 0) / 6.0) * (plot_height - 2 * margin))
                cv2.putText(
                    frame, f"{val:.0f}", (plot_x + plot_width - margin + 100, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA
                )
            cv2.putText(
                frame, "Glide", (plot_x + plot_width - margin + 100, plot_y + margin - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
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

# --- OPTIONAL: Add original audio back using ffmpeg ---
def mux_audio(original_video, overlay_video, output_with_audio):
    try:
        # ffmpeg command to copy video from overlay, audio from original
        cmd = [
            "ffmpeg",
            "-y",
            "-i", overlay_video,
            "-i", original_video,
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_with_audio
        ]
        subprocess.run(cmd, check=True)
        print(f"Overlay video with audio saved as: {output_with_audio}")
    except Exception as e:
        print("Could not mux audio with ffmpeg:", e)

# Only run if output_path and video_path are set and ffmpeg is available
if os.path.isfile(output_path) and os.path.isfile(video_path):
    output_with_audio = os.path.splitext(output_path)[0] + "_with_audio.mp4"
    mux_audio(video_path, output_path, output_with_audio)