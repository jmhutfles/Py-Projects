import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tkinter import Tk, filedialog
import ReadRawData
import Conversions
from tqdm import tqdm

# --- FILE DIALOG PROMPTS ---
root = Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select the parachute drop video", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")])
output_path = filedialog.asksaveasfilename(title="Save output video as", defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
plot_window = 10  # Seconds to show in the plot window

if not video_path or not output_path:
    print("Operation cancelled.")
    exit()

# --- LOAD DATA ---
raw_df = ReadRawData.ReadABT("Select the ABT file.")
df = Conversions.format_and_smooth_abt_data(raw_df)

# --- Let user pick landing in ABT data by clicking the graph interactively ---
clicked_time = []

def on_click(event):
    # Ignore clicks while zooming or panning
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

# --- Ask user for landing time in video (in seconds) ---
landing_video_time = float(input("Enter the landing time in the video (in seconds): "))

# --- Calculate offset ---
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

PLOT_EVERY_N_FRAMES = 20  # Only update the plot every 5 frames
last_plot_img = None

with tqdm(total=total_frames, desc="Processing video frames") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        video_time = frame_idx / fps
        data_time = video_time - offset

        # Only update the plot every N frames
        if frame_idx % PLOT_EVERY_N_FRAMES == 0 or last_plot_img is None:
            plot_data = df[df['Time (s)'] <= data_time]

            fig, ax1 = plt.subplots(figsize=(5, 2), dpi=100, facecolor='none')
            fig.patch.set_alpha(0.0)
            ax1.set_facecolor('none')
            ax1.plot(plot_data['Time (s)'], plot_data['Smoothed Accleration (g)'], color='g', label="Acc (g)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Acc (g)", color='g')
            ax1.tick_params(axis='y', labelcolor='g')

            window_half = 5
            x_min = max(0, data_time - window_half)
            x_max = data_time + window_half

            # Include both past and future data in the plot window
            plot_data = df[(df['Time (s)'] >= x_min) & (df['Time (s)'] <= x_max)]

            ax1.set_xlim(x_min, x_max)

            # Draw vertical line at current time
            ax1.axvline(data_time, color='k', linestyle='--', linewidth=2, label='Current Time')

            ax1.set_ylim(df['Smoothed Accleration (g)'].min(), df['Smoothed Accleration (g)'].max())

            ax2 = ax1.twinx()
            ax2.set_facecolor('none')
            ax2.plot(plot_data['Time (s)'], plot_data['Smoothed Altitude MSL (ft)'], color='b', label="Alt (ft)")
            ax2.set_ylabel("Alt (ft)", color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.set_ylim(df['Smoothed Altitude MSL (ft)'].min(), df['Smoothed Altitude MSL (ft)'].max())

            ax3 = ax1.twinx()
            ax3.set_facecolor('none')
            ax3.spines['right'].set_position(('outward', 60))
            ax3.plot(plot_data['Time (s)'], plot_data['rate_of_descent_ftps'], color='r', label="ROD (ft/s)")
            ax3.set_ylabel("ROD (ft/s)", color='r')
            ax3.tick_params(axis='y', labelcolor='r')
            ax3.set_ylim(df['rate_of_descent_ftps'].min(), df['rate_of_descent_ftps'].max())

            fig.tight_layout()
            canvas = FigureCanvas(fig)
            canvas.draw()
            plot_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close(fig)
            # Resize plot image (with alpha channel)
            plot_img = cv2.resize(plot_img, (frame_width // 2, frame_height // 3))

            # Split into color and alpha
            plot_rgb = plot_img[..., :3]
            plot_alpha = plot_img[..., 3] / 255.0  # Normalize alpha to 0-1

            # Region of interest on the frame
            roi = frame[0:plot_img.shape[0], 0:plot_img.shape[1]]

            # Alpha blend
            for c in range(3):
                roi[..., c] = (plot_alpha * plot_rgb[..., c] + (1 - plot_alpha) * roi[..., c]).astype(np.uint8)

            # Place blended ROI back in frame
            frame[0:plot_img.shape[0], 0:plot_img.shape[1]] = roi

            # Save for reuse
            last_plot_img = plot_img
        else:
            # Split into color and alpha
            plot_rgb = last_plot_img[..., :3]
            plot_alpha = last_plot_img[..., 3] / 255.0  # Normalize alpha to 0-1

            # Region of interest on the frame
            roi = frame[0:last_plot_img.shape[0], 0:last_plot_img.shape[1]]

            # Alpha blend
            for c in range(3):
                roi[..., c] = (plot_alpha * plot_rgb[..., c] + (1 - plot_alpha) * roi[..., c]).astype(np.uint8)

            # Place blended ROI back in frame
            frame[0:last_plot_img.shape[0], 0:last_plot_img.shape[1]] = roi

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

cap.release()
out.release()
print("Overlay video saved:", output_path)