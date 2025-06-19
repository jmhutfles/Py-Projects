import cv2
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
import ReadRawData
import Conversions
from tqdm import tqdm
import matplotlib.pyplot as plt

def IMUVideo():

    while True:

        # --- FILE DIALOG PROMPTS ---
        root = Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            title="Select the IMU video",
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
        raw_df, _ = ReadRawData.ReadIMU("Select the IMU file.")
        df = Conversions.format_and_smooth_imu_data(raw_df)  # Use your smoothing function

        # --- Let user pick landing in IMU data by clicking the graph interactively ---
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
        ax2.plot(df['Time (s)'], df['Smoothed Acceleration (g)'], label='Smoothed Acceleration (g)', color='g')
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

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_time = frame_idx / fps

        # --- CREATE VIDEO WRITER ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # --- Overlay parameters ---
        plot_win = 5
        plot_width = int(frame_width * 0.8)
        plot_height = int(frame_height * 0.45)
        plot_x = 0
        plot_y = frame_height - plot_height

        # --- Precompute min/max for scaling ---
        rod_min, rod_max = df['rate_of_descent_ftps'].min(), df['rate_of_descent_ftps'].max()
        acc_min, acc_max = df['Smoothed Acceleration (g)'].min(), df['Smoothed Acceleration (g)'].max()

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

                # --- Auto-scale axes for the current window ---
                if len(plot_data) > 1:
                    rod_min, rod_max = plot_data['rate_of_descent_ftps'].min(), plot_data['rate_of_descent_ftps'].max()
                    acc_min, acc_max = plot_data['Smoothed Acceleration (g)'].min(), plot_data['Smoothed Acceleration (g)'].max()
                else:
                    # fallback to global min/max if not enough data
                    rod_min, rod_max = df['rate_of_descent_ftps'].min(), df['rate_of_descent_ftps'].max()
                    acc_min, acc_max = df['Smoothed Acceleration (g)'].min(), df['Smoothed Acceleration (g)'].max()

                # --- Draw plot background ---
                overlay = frame.copy()
                cv2.rectangle(overlay, (plot_x, plot_y), (plot_x + plot_width, plot_y + plot_height), (30, 30, 30), -1)
                alpha = 0.3
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # --- Draw axes and labels ---
                margin = 55
                axis_color = (200, 200, 200)
                # Left y-axis for ROD (auto-scaled)
                cv2.line(frame, (plot_x + margin, plot_y), (plot_x + margin, plot_y + plot_height), axis_color, 1)
                # Right y-axis for Acceleration
                cv2.line(frame, (plot_x + plot_width - margin, plot_y), (plot_x + plot_width - margin, plot_y + plot_height), axis_color, 1)
                # X axis
                cv2.line(frame, (plot_x + margin, plot_y + plot_height - margin), (plot_x + plot_width - margin, plot_y + plot_height - margin), axis_color, 1)

                # --- Draw axes and labels ---
                margin = 55
                axis_color = (200, 200, 200)
                # ... draw axes code ...

                # --- Draw X axis time scale ---
                num_ticks = 5  # Number of time ticks to show
                tick_length = 8
                for i in range(num_ticks + 1):
                    tick_time = x_min + i * (x_max - x_min) / num_ticks
                    tick_x = int(((tick_time - x_min) / (x_max - x_min + 1e-6)) * (plot_width - 2 * margin) + plot_x + margin)
                    tick_y1 = plot_y + plot_height - margin
                    tick_y2 = tick_y1 + tick_length
                    cv2.line(frame, (tick_x, tick_y1), (tick_x, tick_y2), (220, 220, 220), 2)
                    cv2.putText(
                        frame, f"{tick_time:.1f}s",
                        (tick_x - 18, tick_y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA
                    )
            
                # ROD axis label (auto-scaled)
                rod_axis_x = plot_x + margin - 50
                cv2.putText(frame, f"{rod_max:.1f} ft/s", (rod_axis_x, plot_y + margin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(frame, f"{rod_min:.1f} ft/s", (rod_axis_x, plot_y + plot_height - margin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(frame, "ROD", (rod_axis_x, plot_y + plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                # Acceleration axis label (auto-scaled)
                acc_axis_x = plot_x + plot_width - margin + 10
                cv2.putText(frame, f"{acc_max:.2f} g", (acc_axis_x, plot_y + margin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"{acc_min:.2f} g", (acc_axis_x, plot_y + plot_height - margin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, "Acc", (acc_axis_x, plot_y + plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                # --- Draw data lines ---
                if len(plot_data) > 1:
                    t = plot_data['Time (s)'].values
                    rod = plot_data['rate_of_descent_ftps'].values
                    acc = plot_data['Smoothed Acceleration (g)'].values

                    t_norm = ((t - x_min) / (x_max - x_min + 1e-6)) * (plot_width - 2 * margin) + plot_x + margin

                    # ROD (red, left y-axis)
                    rod_norm = (rod - rod_min) / (rod_max - rod_min + 1e-6)
                    rod_norm = plot_y + plot_height - margin - rod_norm * (plot_height - 2 * margin)
                    for i in range(1, len(t)):
                        pt1 = (int(t_norm[i-1]), int(rod_norm[i-1]))
                        pt2 = (int(t_norm[i]), int(rod_norm[i]))
                        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

                    # Acceleration (green, right y-axis)
                    acc_norm = (acc - acc_min) / (acc_max - acc_min + 1e-6)
                    acc_norm = plot_y + plot_height - margin - acc_norm * (plot_height - 2 * margin)
                    for i in range(1, len(t)):
                        pt1 = (int(t_norm[i-1]), int(acc_norm[i-1]))
                        pt2 = (int(t_norm[i]), int(acc_norm[i]))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                    # Draw vertical line at current time
                    curr_x = int(((data_time - x_min) / (x_max - x_min + 1e-6)) * (plot_width - 2 * margin) + plot_x + margin)
                    cv2.line(frame, (curr_x, plot_y + margin), (curr_x, plot_y + plot_height - margin), (255, 255, 255), 2)

                    # --- Draw legend just above the plot ---
                    legend_y = plot_y - 15
                    cv2.putText(frame, "ROD (ft/s)", (plot_x + margin, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.putText(frame, "Acc (g)", (plot_x + margin + 180, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # --- Draw text display ---
                idx = (df['Time (s)'] - data_time).abs().idxmin()
                current_row = df.iloc[[idx]]
                altitude = current_row['Smoothed Altitude MSL (ft)'].values[0]
                acc = current_row['Smoothed Acceleration (g)'].values[0]
                rod_val = current_row['rate_of_descent_ftps'].values[0]

                # Calculate max acceleration so far
                max_acc_so_far = df[df['Time (s)'] <= data_time]['Smoothed Acceleration (g)'].max()

                info_text = [
                    f"Alt: {altitude:,.0f} ft",
                    f"Acc: {acc:,.2f} g",
                    f"ROD: {rod_val:,.1f} ft/s",
                    f"Max Acc: {max_acc_so_far:.2f} g"
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

                out.write(frame)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()
        print("Overlay video saved:", output_path)

        again_choice = input("Do you want to process another video? (y/n): ").strip().lower()
        if again_choice != 'y':
            break
if __name__ == "__main__":
    IMUVideo()